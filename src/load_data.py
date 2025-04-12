import os
import pandas as pd
import numpy as np
import rasterio
import rasterio.enums
from rasterio.merge import merge
import geopandas as gpd
import xarray as xr
import rioxarray 
import dask
import gc

xr.set_options(keep_attrs=True, display_expand_data=True)


def remove_extra_bands(input_path, output):
  """
  Remove extra bands from a raster file and save it as a new file.
  
    Args:
        input_path (str): Path to the input raster file.
        output (str): Path to save the output raster file.
    Outputs:
        Raster file with 12 bands.
  """
  with rasterio.open(input_path) as src:
      meta = src.meta.copy()
      meta.update(count=12)

      with rasterio.open(output, "w", **meta) as dst:
          for i in range(1, 13):
              band = src.read(i)
              dst.write(band, i)
  print('Done')
  
  return None


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

    return None


def reproject(file, projection, output_path):
    """
    Reprojects a raster (xarray.DataArray) or vector (GeoDataFrame) to a new CRS.

    Args:
        file: xarray.DataArray
        projection: EPSG code of the target projection

    Returns:
        Save reprojected raster on disk or return Reprojected GeoDataFrame.
    """

    if isinstance(file, xr.DataArray):
        file.rio.reproject(f"EPSG:{projection}", resolution=(10,-10), resampling=rasterio.enums.Resampling.bilinear).rio.to_raster(output_path, tiled=True, windowed=True, lock=False)

        # output_bands = []
        # for i in range(file.sizes["band"]):
        #     band = file.isel(band=i)
        #     band_reprojected = band.rio.reproject(f"EPSG:{projection}")
        #     output_bands.append(band_reprojected)

        # Combine reprojected bands back into one DataArray
        # data_reprojected = xr.concat(output_bands, dim="band")

    elif isinstance(file, gpd.GeoDataFrame):
        data_reprojected = file.to_crs(epsg=projection)
        print("Reprojection completed successfully.")
        return data_reprojected
    else:
        raise TypeError("Unsupported input type. Must be xarray.DataArray or geopandas.GeoDataFrame")
    print("Reprojection completed successfully.")
    return None


def resample_raster(raster_ds, target_resolution=None): # need optimization: if -> to be used for other imagery types (other than sentinel2)
    """
    Resample the raster dataset to a target resolution, or to the highest band resolution if target_resolution is None.

    Args:
        raster_ds (xarray.DataArray): Raster dataset to be resampled.
        target_resolution (tuple): Target resolution in the form of (x_res, y_res).

    Returns:
        Resampled xarray.DataArray.
    """
    if target_resolution is None:
        ref_band = raster_ds.sel(band=4)
        target_resolution  = ref_band.rio.resolution()
        target_crs = ref_band.rio.crs

        # Separate 10m and resample the rest
        ten_meter_bands = [x for x in raster_ds.band.values if x in [2, 3, 4, 8]]
        resampled_bands_list = []

        for b in raster_ds.band.values:
            band = raster_ds.sel(band=b)
            if b in ten_meter_bands:
                resampled_bands_list.append(band)
            else:
                resampled = band.rio.reproject(crs=target_crs, resolution=target_resolution, resampling=rasterio.enums.Resampling.bilinear)
                resampled = resampled.compute()
                resampled_bands_list.append(resampled)
                del resampled
                gc.collect()

        resampled_raster = xr.concat(resampled_bands_list, dim="band")
        resampled_raster.rio.write_crs(target_crs, inplace=True)
        resampled_raster = resampled_raster.sortby("band") # just in case :p

    
    else:
        crs = raster_ds.rio.crs
        resampled_raster = raster_ds.rio.reproject(
            crs=crs,
            resolution=target_resolution,
            resampling=rioxarray.rio.warp.Resampling.bilinear,
            )

    return resampled_raster

def load_raster_data(raster_file_path, chunks=False):
    """
    Load the input data. 

    Args:
        raster_file_path (str): Path to the input data file - Geotiff.

    Returns:
        rioxarry.DataArray object.
    """
    if not isinstance(raster_file_path, str):
        raise TypeError("raster_file_path must be a string")
    
    if not os.path.exists(raster_file_path):
        raise FileNotFoundError(f"File {raster_file_path} does not exist")
  
    # Load the raster data
    try:
      if chunks:
        raster_ds = rioxarray.open_rasterio(raster_file_path, masked=True, chunks="auto")
        print("raster image is loaded successfully as rioxarray.Dataset") 
      else:
        raster_ds = rioxarray.open_rasterio(raster_file_path, masked=True)
        print("raster image is loaded successfully as rioxarray.Dataset")
    except Exception as e:
        raise ValueError(f"Error loading raster data: {e}")
    
    return raster_ds


def load_training_data( training_data: str):
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


def show_metadata(raster_path):
    """
    Visualize the raster data and training data.

    Args:
        raster_path (raster path: str): Raster dataset.

    Returns:
        str: Metadata of the raster dataset.
    """
    with rioxarray.open_rasterio(raster_path) as src:
      print(f"Dimensions: \n{src.dims}\n")
      print(f"CRS: \n{src.rio.crs}\n")
      print(f"Transform: \n{src.rio.transform()}\n")
      print(f"Resolution: \n{src.rio.resolution()}\n")
      print(f"{src.coords}\n")
      print(f"Metadata: \n{src.coords['spatial_ref']}\n")

    return None



if __name__ == "__main__":
    print("Data prep module for LULC classification.")


