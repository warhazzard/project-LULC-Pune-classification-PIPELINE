import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge
import geopandas as gpd
import xarray as xr
import rioxarray 

xr.set_options(keep_attrs=True, display_expand_data=True)


def mosaic_rasters(*raster_files, output_file_path='mosaic.tif'):
    """
    Mosaics multiple raster files into a single raster.

    Args:
        raster_files: List of paths to raster files - GeoTiff.
        output_file_path: Path to save the mosaiced raster.
    
    Returns:
        Mosaiced raster as GeoTiff.
    """
    if not all(isinstance(file, str) for file in raster_files):
        raise TypeError("All raster files must be strings")
    
    if not all(os.path.exists(file) for file in raster_files):
        raise FileNotFoundError("One or more raster files do not exist")

    try:
        # Open all rasters and merge them
        src_files_to_mosaic = [rasterio.open(fp) for fp in raster_files]
        mosaic, out_transform = merge(src_files_to_mosaic)

        # Copy metadata from first file to update mosaic's metadata
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "count": mosaic.shape[0]
        })

        # Write mosaicked raster to output_file_path and close the files
        with rasterio.open(output_file_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files_to_mosaic:
            src.close()

    except Exception as e:
        raise RuntimeError(f"Error during raster mosaicking: {e}")

    return output_file_path


def reproject(file, projection):
    """
    Reprojects a raster (xarray.DataArray) or vector (GeoDataFrame) to a new CRS.

    Args:
        file: rioxarray dataset
        projection: EPSG code of the target projection

    Returns:
        Reprojected rioxarray dataset
    """

    if isinstance(file, rioxarray.DataArray):
        data_reprojected = file.rio.reproject(f"EPSG:{projection}")
    elif isinstance(file, gpd.GeoDataFrame):
        data_reprojected = file.to_crs(epsg=projection)
    else:
        raise TypeError("Unsupported input type. Must be xarray.DataArray or geopandas.GeoDataFrame")
    
    return data_reprojected



def load_raster_data(raster_file_path: str) -> rioxarray.DataArray:
    """
    Load the input data. 

    Args:
        raster_file_path (str): Path to the input data file - Geotiff.

    Returns:
        rioxarry.Dataset object.
    
    """
    if not isinstance(raster_file_path, str):
        raise TypeError("raster_file_path must be a string")
    
    if not os.path.exists(raster_file_path):
        raise FileNotFoundError(f"File {raster_file_path} does not exist")

    
    # Load the raster data
    try:
        raster_ds = rioxarray.open_rasterio(raster_file_path, masked=True)
        print("raster image is loaded successfully as rioxarray.Dataset") 
    except Exception as e:
        raise ValueError(f"Error loading raster data: {e}")
    
    return raster_ds


def load_training_data( training_data: str) -> gpd.GeoDataFrame:
    """
    Load the training data. 

    Args:
        training_data (str): Path to the training data file - GeoJson.

    Returns:
        gpd.GeoDataFrame object.
    
    """
    
    if not isinstance(training_data, str):
        raise TypeError("training_data must be a string")
    if not os.path.exists(training_data):
        raise FileNotFoundError(f"File {training_data} does not exist")

    # Load the training data
    try:
        gdf = gpd.read_file(training_data)
        if "class" not in gdf.columns and "Class" not in gdf.columns:
            class_col = None 
            for col in gdf.columns:
                if col.lower() in ['class', 'classname', 'lulc', 'landcover', 'land_cover', 'land_use', 'category']:
                    class_col = col
                    break
            if class_col is None:
                print("Warning: No class column found in shapefile. Please ensure your shapefile has a column indicating land cover classes.")
    except Exception as e:
        raise ValueError(f"Error loading training data: {e}")
    
    return gdf, class_col


# def visualize_data(raster_ds1, raster_ds2, gdf):
#     """
#     Visualize the raster data and training data.

#     Args:
#         raster_ds1 (rioxarray.DataArray): First raster dataset.
#         raster_ds2 (rioxarray.DataArray): Second raster dataset.
#         gdf (gpd.GeoDataFrame): GeoDataFrame containing training data.

#     Returns:
#         None
#     """
#     # TO WORK ON !!!!



if __name__ == "__main__":
    print("Data prep module for LULC classification.")


