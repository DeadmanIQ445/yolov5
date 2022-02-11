# %%
from sklearn.model_selection import train_test_split
import geopandas as gpd
from utils import *
import fiona
import rasterio.mask
# %%
import warnings

warnings.filterwarnings("ignore")
# %%
data_path = '/media/deadman445/disk/05/train'
patch_size = 400
samples_dir = f'/media/deadman445/disk/PycharmProjects/data/05/summer_treecanopy_{patch_size}'



use_AOI = True


# %% md
# Processing for training
# %%
def create_train_ds(input_dir, resulting_ds_dir):
    """
    Preparing data for training by splitting big input images into tiles

        Args:
            input_dir: directory, which contains directories with data that will be used for training
            resulting_ds_dir: directory for resulting splitted images
    """
    for p in tqdm(os.listdir(input_dir)):
        path = os.path.join(data_path, p)
        print(path)
        for file in os.listdir(path):
            if file.endswith(".shp"):
                orig = gpd.read_file(os.path.join(path, file))
            if file.endswith('.tif'):
                file_3chanel = "three_" + file
                with rasterio.open(os.path.join(path, file)) as rast:
                    meta = rast.meta
                    out_image = rast.read()
                    if use_AOI:
                        path_AOI = os.path.join(os.path.join(os.path.split(input_dir)[0], 'AOI'), os.path.split(path)[1])
                        path_AOI = os.path.join(path_AOI, os.path.split(path)[1].split('_gpx_GT')[0] + "_gpx_AOI.shp")
                        with fiona.open(path_AOI, "r") as shapefile:
                            shapes = [feature["geometry"] for feature in shapefile]
                            out_image, out_transform = rasterio.mask.mask(rast, shapes)
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
    for csv in os.listdir(samples_dir):
        if csv.endswith(".csv"):
            processed = pd.read_csv(os.path.join(resulting_ds_dir, csv)).dropna().reset_index().drop(['index'], axis=1)

            if final_df is None:
                final_df = processed
            else:
                final_df = final_df.append(processed)
    final_df.to_csv(os.path.join(resulting_ds_dir, 'final_df.csv'))


create_train_ds(data_path, samples_dir)
# %%
# perfroming train/test split by images

train_val_test = pd.read_csv(os.path.join(samples_dir, 'final_df.csv'))

train_val_test['label'] = 'Tree'

train_val_paths, test_paths = train_test_split(train_val_test['image_path'].unique(), test_size=0.15, random_state=42)
train_val = train_val_test[train_val_test['image_path'].isin(train_val_paths)]
test = train_val_test[train_val_test['image_path'].isin(test_paths)]
train_val.to_csv(os.path.join(samples_dir, "train_val.csv"))
test.to_csv(os.path.join(samples_dir, 'test.csv'))