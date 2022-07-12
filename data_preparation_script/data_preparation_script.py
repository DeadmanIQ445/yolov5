from sklearn.model_selection import train_test_split
from data_preparation_script.utils import *
import fiona
import rasterio.mask
import warnings
import os
import geopandas as gpd
import pandas as pd
import shutil

warnings.filterwarnings("ignore")


def create_train_ds(input_dir, resulting_ds_dir, patch_size=512, use_AOI=False):
    """
    Preparing data for training by splitting big input images into tiles

        Args:
            input_dir: directory, which contains directories with data that will be used for training
            resulting_ds_dir: directory for resulting splitted images
    """
    for p in tqdm(os.listdir(os.path.join(input_dir, 'Raster'))):
        path = os.path.join(input_dir, 'Raster', p)
        gt_path = os.path.join(input_dir, 'Razmetka', p)
        print(path)
        orig = gpd.read_file(gt_path)
        file_3chanel = "three_" + file
        with rasterio.open(os.path.join(path, file)) as rast:
            meta = rast.meta
            out_image = rast.read()
            # if use_AOI:
            #     path_AOI = os.path.join(os.path.join(os.path.split(input_dir)[0], 'AOI'),
            #                             os.path.split(path)[1])
            #     # path_AOI = os.path.join(path_AOI, os.path.split(path)[1].split('_gpx_GT')[0] + "_gpx_AOI.shp")
            #     with fiona.open(path_AOI, "r") as shapefile:
            #         shapes = [feature["geometry"] for feature in shapefile]
            #         out_image, out_transform = rasterio.mask.mask(rast, shapes)
        with rasterio.open(file_3chanel, 'w', **meta) as new_dataset:
            new_dataset.write(out_image)

        original_data = orig['geometry'].bounds.apply(inv_affine, 1, file_name=file_3chanel, meta=meta, label='Tree')
        original_data.to_csv(f"{p}_before_proc.csv")
        train_annotations = split_raster(path_to_raster=file_3chanel,
                                         annotations_file=f"{p}_before_proc.csv",
                                         base_dir=resulting_ds_dir,
                                         patch_size=patch_size,
                                         patch_overlap=0.1, allow_empty=False)

        os.remove(file_3chanel)
        os.remove(f"{p}_before_proc.csv")

    final_df = None
    for csv in os.listdir(resulting_ds_dir):
        if csv.endswith(".csv"):
            processed = pd.read_csv(os.path.join(resulting_ds_dir, csv)).dropna().reset_index().drop(['index'], axis=1)

            if final_df is None:
                final_df = processed
            else:
                final_df = final_df.append(processed)
    final_df.to_csv(os.path.join(resulting_ds_dir, 'final_df.csv'))


def run(data_path='/home/arina/OKS/data/', patch_size=512, use_AOI=True):
    data_path_train = os.path.join(data_path, 'train')
    # data_path_test = os.path.join(data_path, 'test')
    train_samples_dir = os.path.join(data_path, f'summer_treecanopy_{patch_size}')
    # test_samples_dir = os.path.join(data_path, f'test_{patch_size}')
    samples_dir = train_samples_dir
    yolo_base = os.path.join(os.path.dirname(samples_dir), f'coco128')
    create_train_ds(data_path_train, samples_dir, patch_size)

    train_val_test = pd.read_csv(os.path.join(samples_dir, 'final_df.csv'))

    train_val_test['label'] = 'Tree'

    train_paths, val_paths = train_test_split(train_val_test['image_path'].unique(), test_size=0.15, random_state=42)
    train = train_val_test[train_val_test['image_path'].isin(train_paths)]
    val = train_val_test[train_val_test['image_path'].isin(val_paths)]
    train.to_csv(os.path.join(samples_dir, "train.csv"))
    val.to_csv(os.path.join(samples_dir, 'val.csv'))

    create_train_ds(data_path_test, test_samples_dir, patch_size)

    for train_val_label in ["train2017", "val2017"]:
        samples_dir = train_samples_dir
        if 'val' in train_val_label:
            csv = 'val.csv'
        elif 'train' in train_val_label:
            csv = 'train.csv'

        elif 'test' in train_val_label:
            csv = 'final_df.csv'
            samples_dir = test_samples_dir

        df_path = os.path.join(samples_dir, csv)

        a = gpd.read_file(df_path)

        images_base = os.path.join(yolo_base, 'images')
        labels_base = os.path.join(yolo_base, 'labels')
        os.makedirs(yolo_base, exist_ok=True)
        os.makedirs(images_base, exist_ok=True)
        os.makedirs(labels_base, exist_ok=True)

        for name, group in a.groupby(['image_path']):
            base_name = os.path.splitext(os.path.basename(name))[0]
            group['xmin'] = group['xmin'].astype(float)
            group['xmax'] = group['xmax'].astype(float)
            group['ymin'] = group['ymin'].astype(float)
            group['ymax'] = group['ymax'].astype(float)
            x = (group['xmin'] + group['xmax']) / (2.0 * patch_size)
            y = (group['ymin'] + group['ymax']) / (2.0 * patch_size)
            w = (group['xmax'] - group['xmin']) / patch_size
            h = (group['ymax'] - group['ymin']) / patch_size
            res = pd.concat([x * 0, x, y, w, h], axis=1)
            labels_train = os.path.join(labels_base, train_val_label)
            os.makedirs(labels_train, exist_ok=True)
            images_train = os.path.join(images_base, train_val_label)
            os.makedirs(images_train, exist_ok=True)
            res.to_csv(f'{os.path.join(labels_train, base_name)}.txt', header=None, index=None, sep=' ', mode='w+')
            shutil.copy2(os.path.join(samples_dir, name), os.path.join(images_train, name))


if __name__ == "__main__":
    run()
