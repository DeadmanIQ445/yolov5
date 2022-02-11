import torch
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import reshape_as_raster
from rasterio.plot import reshape_as_image
from tqdm import tqdm
import os
import slidingwindow

def inv_affine(row, file_name,meta, label):
    """
    This method removes affine of the polygons so that the boxes could be used for model
    """
    row['xmin'] = (row['minx']-meta['transform'][2])/meta['transform'][0]
    row['xmax'] = (row['maxx']-meta['transform'][2])/meta['transform'][0]
    row['ymin'] = (row['maxy']-meta['transform'][5])/meta['transform'][4]
    row['ymax'] = (row['miny']-meta['transform'][5])/meta['transform'][4]
    row['image_path'] = file_name
    row['label'] = label
    return row


def preprocess_image(image, device):
    """Preprocess a single RGB numpy array as a prediction from channels last, to channels first"""
    image = torch.tensor(image, device=device).unsqueeze(0)
    #     image[0] = image[0] / 255
    #     image[1] = image[1] / 255

    return image


def image_name_from_path(image_path):
    """Convert path to image name for use in indexing."""
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    return image_name


def compute_windows(numpy_image, patch_size, patch_overlap, dim_order=slidingwindow.DimOrder.ChannelHeightWidth):
    """Create a sliding window object from a raster tile.
    Args:
        numpy_image (array): Raster object as numpy array to cut into crops
    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     dim_order,
                                     patch_size, patch_overlap)

    return (windows)


def select_annotations(annotations, windows, index, allow_empty=False):
    """Select annotations that overlap with selected image crop.
    Args:
        image_name (str): Name of the image in the annotations file to lookup.
        annotations_file: path to annotations file in
            the format -> image_path, xmin, ymin, xmax, ymax, label
        windows: A sliding window object (see compute_windows)
        index: The index in the windows object to use a crop bounds
        allow_empty (bool): If True, allow window crops
            that have no annotations to be included
    Returns:
        selected_annotations: a pandas dataframe of annotations
    """

    # Window coordinates - with respect to tile
    window_xmin, window_ymin, w, h = windows[index].getRect()
    window_xmax = window_xmin + w
    window_ymax = window_ymin + h

    # buffer coordinates a bit to grab boxes that might start just against
    # the image edge. Don't allow boxes that start and end after the offset
    offset = 40
    selected_annotations = annotations[(annotations.xmin > (window_xmin - offset)) &
                                       (annotations.xmin < (window_xmax)) &
                                       (annotations.xmax >
                                        (window_xmin)) & (annotations.ymin >
                                                          (window_ymin - offset)) &
                                       (annotations.xmax <
                                        (window_xmax + offset)) & (annotations.ymin <
                                                                   (window_ymax)) &
                                       (annotations.ymax >
                                        (window_ymin)) & (annotations.ymax <
                                                          (window_ymax + offset))]

    # change the image name
    image_name = os.path.splitext("{}".format(annotations.image_path.unique()[0]))[0]
    image_basename = os.path.splitext(image_name)[0]
    selected_annotations.image_path = "{}_{}.tif".format(image_basename, index)

    # If no matching annotations, return a line with the image name, but no
    # records
    if selected_annotations.empty:
        if allow_empty:
            selected_annotations = pd.DataFrame(
                ["{}_{}.tif".format(image_basename, index)], columns=["image_path"])
            selected_annotations["xmin"] = 0
            selected_annotations["ymin"] = 0
            selected_annotations["xmax"] = 0
            selected_annotations["ymax"] = 0
            # Dummy label
            selected_annotations["label"] = annotations.label.unique()[0]
        else:
            return None
    else:
        # update coordinates with respect to origin
        selected_annotations.xmax = (selected_annotations.xmin - window_xmin) + (
                selected_annotations.xmax - selected_annotations.xmin)
        selected_annotations.xmin = (selected_annotations.xmin - window_xmin)
        selected_annotations.ymax = (selected_annotations.ymin - window_ymin) + (
                selected_annotations.ymax - selected_annotations.ymin)
        selected_annotations.ymin = (selected_annotations.ymin - window_ymin)

        # cut off any annotations over the border.
        selected_annotations.loc[selected_annotations.xmin < 0, "xmin"] = 0
        selected_annotations.loc[selected_annotations.xmax > w, "xmax"] = w
        selected_annotations.loc[selected_annotations.ymin < 0, "ymin"] = 0
        selected_annotations.loc[selected_annotations.ymax > h, "ymax"] = h

    return selected_annotations


def save_crop(base_dir, image_name, index, crop):
    """Save window crop as image file to be read by PIL.
    Filename should match the image_name + window index
    """
    # create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    image_basename = os.path.splitext(image_name)[0]
    filename = "{}/{}_{}.tif".format(base_dir, image_basename, index)
    new_dataset = rasterio.open(filename, 'w', height=crop.shape[0], width=crop.shape[1], count=3, dtype=crop.dtype,
                                driver='GTiff')
    new_dataset.write(reshape_as_raster(crop))
    new_dataset.close()

    return filename


def split_raster(annotations_file,
                 path_to_raster=None,
                 numpy_image=None,
                 base_dir=".",
                 patch_size=400,
                 patch_overlap=0.05,
                 allow_empty=False,
                 image_name=None):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.
    Args:
        numpy_image: a numpy object to be used as a raster, usually opened from rasterio.open.read()
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str): Path to annotations file (with column names)
            data in the format -> image_path, xmin, ymin, xmax, ymax, label
        base_dir (str): Where to save the annotations and image
            crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset
        image_name (str): If numpy_image arg is used, what name to give the raster?
    Returns:
        A pandas dataframe with annotations file for training.
    """

    # Load raster as image
    # Load raster as image
    if (numpy_image is None) & (path_to_raster is None):
        raise IOError(
            "supply a raster either as a path_to_raster or if ready from existing in memory numpy object, as numpy_image=")

    if path_to_raster:
        numpy_image = rasterio.open(path_to_raster).read()
        #         numpy_image = np.expand_dims(numpy_image, axis=0)
        numpy_image = np.moveaxis(numpy_image, 0, 2)
    else:
        if image_name is None:
            raise (IOError(
                "If passing an numpy_image, please also specify a image_name to match the column in the annotation.csv file"))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap, slidingwindow.DimOrder.HeightWidthChannel)

    # Get image name for indexing
    if image_name is None:
        image_name = os.path.basename(path_to_raster)

        # Load annotations file and coerce dtype
    annotations = pd.read_csv(annotations_file)

    # open annotations file
    image_annotations = annotations[annotations.image_path == image_name]

    # Sanity checks
    if image_annotations.empty:
        raise ValueError(
            "No image names match between the file:{} and the image_path: {}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    if not all([
        x in annotations.columns
        for x in ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    ]):
        raise ValueError("Annotations file has {} columns, should have "
                         "format image_path, xmin, ymin, xmax, ymax, label".format(
            annotations.shape[1]))

    annotations_files = []
    for index, window in enumerate(windows):

        # Crop image
        crop = numpy_image[windows[index].indices()]

        # skip if empty crop
        if crop.size == 0:
            continue

        # Find annotations, image_name is the basename of the path
        crop_annotations = select_annotations(image_annotations, windows, index,
                                              allow_empty)

        # If empty images not allowed, select annotations returns None
        if crop_annotations is not None:
            # save annotations
            annotations_files.append(crop_annotations)

            # save image crop
            save_crop(base_dir, image_name, index, crop)
    if len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))

    annotations_files = pd.concat(annotations_files)

    # Checkpoint csv files, useful for parallelization
    # Use filename of the raster path to save the annotations
    image_basename = os.path.splitext(image_name)[0]
    file_path = image_basename + ".csv"
    file_path = os.path.join(base_dir, file_path)
    annotations_files.to_csv(file_path, index=False, header=True)

    return annotations_files


def predict_image(model, image, return_plot, device, iou_threshold=0.1):
    """Predict an image with a deepforest model
    Args:
        image: a numpy array of a RGB image ranged from 0-255
        path: optional path to read image from disk instead of passing image arg
        return_plot: Return image with plotted detections
        device: pytorch device of 'cuda' or 'cpu' for gpu prediction. Set internally.
    Returns:
        boxes: A pandas dataframe of predictions (Default)
        img: The input with predictions overlaid (Optional)
    """

    image = preprocess_image(image, device=device)

    with torch.no_grad():
        prediction = model(image)

    # return None for no predictions
    if len(prediction[0]["boxes"]) == 0:
        return None

    # This function on takes in a single image.
    df = visualize.format_boxes(prediction[0])
    df = predict.across_class_nms(df, iou_threshold=iou_threshold)

    if return_plot:
        # Bring to gpu
        if not device.type == "cpu":
            image = image.cpu()

        # Cv2 likes no batch dim, BGR image and channels last, 0-255
        image = np.array(image.squeeze(0))
        image = np.rollaxis(image, 0, 3)
        image = image[:, :, ::-1] * 255
        image = image.astype("uint8")
        image = visualize.plot_predictions(image, df)

        return image
    else:
        return df


def predict_tile(model,
                 device,
                 raster_path=None,
                 image=None,
                 patch_size=400,
                 patch_overlap=0.10,
                 iou_threshold=0.15,
                 return_plot=False,
                 use_soft_nms=False,
                 sigma=0.5,
                 thresh=0.001):
    if image is not None:
        pass
    else:
        # load raster as image
        with rasterio.open(raster_path) as im:
            image = reshape_as_image(im.read())

    # Compute sliding window index
    windows = compute_windows(image, patch_size, patch_overlap)
    # Save images to tempdir
    predicted_boxes = []

    for index, window in enumerate(tqdm(windows)):
        # crop window and predict
        crop = image[windows[index].indices()]
        # crop is RGB channel order, change to BGR?
        boxes = predict_image(model=model, image=crop, return_plot=False, device=device)
        if boxes is not None:
            # transform the coordinates to original system
            xmin, ymin, xmax, ymax = windows[index].getRect()
            boxes.xmin = boxes.xmin + xmin
            boxes.xmax = boxes.xmax + xmin
            boxes.ymin = boxes.ymin + ymin
            boxes.ymax = boxes.ymax + ymin

            predicted_boxes.append(boxes)

    if len(predicted_boxes) == 0:
        print("No predictions made, returning None")
        return None

    predicted_boxes = pd.concat(predicted_boxes)
    # Non-max supression for overlapping boxes among window
    if patch_overlap == 0:
        mosaic_df = predicted_boxes
    else:
        print(
            f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max supression"
        )
        # move prediciton to tensor
        boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
                             dtype=torch.float32)
        scores = torch.tensor(predicted_boxes.score.values, dtype=torch.float32)
        labels = predicted_boxes.label.values

        if not use_soft_nms:
            # Performs non-maximum suppression (NMS) on the boxes according to
            # their intersection-over-union (IoU).
            bbox_left_idx = predict.nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
        else:
            # Performs soft non-maximum suppression (soft-NMS) on the boxes.
            bbox_left_idx = predict.soft_nms(boxes=boxes,
                                             scores=scores,
                                             sigma=sigma,
                                             thresh=thresh)

        bbox_left_idx = bbox_left_idx.numpy()
        new_boxes, new_labels, new_scores = boxes[bbox_left_idx].type(
            torch.int), labels[bbox_left_idx], scores[bbox_left_idx]

        # Recreate box dataframe
        image_detections = np.concatenate([
            new_boxes,
            np.expand_dims(new_labels, axis=1),
            np.expand_dims(new_scores, axis=1)
        ],
            axis=1)

        mosaic_df = pd.DataFrame(
            image_detections, columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

        print(f"{mosaic_df.shape[0]} predictions kept after non-max suppression")

    return mosaic_df



