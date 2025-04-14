import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
import rioxarray
import gc
import dask

xr.set_options(keep_attrs=True, display_expand_data=True)


def calculate_indices(raster_ds, output=None):
    """
    Calculate spectral indices from Sentinel-2 bands.

    Parameters:
    -----------
    raster_ds : xarray.Dataset
        Image dataset containing Sentinel-2 bands
    output : str or None
        If specified, will save the output to the provided path.

    Returns:
    --------
    xarray.Dataset
        Dataset with original bands and added spectral indices
    """
    print("check-1")
    result_ds = raster_ds.copy(deep=True)
    del raster_ds
    gc.collect()
    print("check-2")

    try:
        blue = result_ds.sel(band=2)
        green = result_ds.sel(band=3)
        red = result_ds.sel(band=4)
        nir = result_ds.sel(band=8)
        swir = result_ds.sel(band=11)
        print("check-3")

        # NDVI
        ndvi = ((nir - red) / (nir + red)).clip(min=-1, max=1)
        ndvi.name = "NDVI"
        ndvi.attrs["long_name"] = "Normalized Difference Vegetation Index"
        ndvi.attrs["units"] = "unitless"
        ndvi = ndvi.expand_dims(dim='band')
        ndvi['band'] = [result_ds.sizes['band'] + 1]
        result_ds = xr.concat([result_ds, ndvi], dim='band')
        result_ds.attrs["long_name"] = list(result_ds.attrs.get("long_name", [])) + ["NDVI"]
        del ndvi
        gc.collect()
        print("check-4")

        # SAVI
        L = 0.5  # SAVI soil brightness correction
        savi = (((nir - red) / (nir + red + L)) * (1 + L)).clip(min=-1, max=1)
        savi.name = "SAVI"
        savi.attrs["long_name"] = "Soil Adjusted Vegetation Index"
        savi.attrs["units"] = "unitless"
        savi = savi.expand_dims(dim='band')
        savi['band'] = [result_ds.sizes['band'] + 1]
        result_ds = xr.concat([result_ds, savi], dim='band')
        result_ds.attrs["long_name"] = list(result_ds.attrs.get("long_name", [])) + ["SAVI"]
        del savi
        gc.collect()
        print("check-5")

        # MNDWI
        mndwi = ((green - swir) / (green + swir)).clip(min=-1, max=1)
        mndwi.name = "MNDWI"
        mndwi.attrs["long_name"] = "Modified Normalized Difference Water Index"
        mndwi.attrs["units"] = "unitless"
        mndwi = mndwi.expand_dims(dim='band')
        mndwi['band'] = [result_ds.sizes['band'] + 1]
        result_ds = xr.concat([result_ds, mndwi], dim='band')
        result_ds.attrs["long_name"] = list(result_ds.attrs.get("long_name", [])) + ["MNDWI"]
        del mndwi
        gc.collect()
        print("check-6")

        # NDBI
        ndbi = ((swir - nir) / (swir + nir)).clip(min=-1, max=1)
        ndbi.name = "NDBI"
        ndbi.attrs["long_name"] = "Normalized Difference Built-up Index"
        ndbi.attrs["units"] = "unitless"
        ndbi = ndbi.expand_dims(dim='band')
        ndbi['band'] = [result_ds.sizes['band'] + 1]
        result_ds = xr.concat([result_ds, ndbi], dim='band')
        result_ds.attrs["long_name"] = list(result_ds.attrs.get("long_name", [])) + ["NDBI"]
        del ndbi
        gc.collect()
        print("check-7")

        # BSI
        bsi = (((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))).clip(min=-1, max=1)
        bsi.name = "BSI"
        bsi.attrs["long_name"] = "Bare Soil Index"
        bsi.attrs["units"] = "unitless"
        bsi = bsi.expand_dims(dim='band')
        bsi['band'] = [result_ds.sizes['band'] + 1]
        result_ds = xr.concat([result_ds, bsi], dim='band')
        result_ds.attrs["long_name"] = list(result_ds.attrs.get("long_name", [])) + ["BSI"]
        del bsi
        gc.collect()
        print("check-8")

        if output is not None:
            result_ds.rio.to_raster(output, driver="GTiff") # tiled=True, windowed=True, lock=False
        else:
            return result_ds

    except Exception as e:
        print(f"Error calculating indices: {e}")
        return None


def extract_training_samples(raster_list, gdf, class_col):
    """
    Extract training samples from the raster data using the provided GeoDataFrame.
    
    Args:
        raster_ds (rioxarray.Dataset - list): The list of raster data (rioxarray.Dataset).
        gdf (gpd.GeoDataFrame): The training data as a GeoDataFrame.
        class_col (str): The name of the column in the GeoDataFrame that contains the class labels.
    
    Returns:
        df (pd.DataFrame): A DataFrame containing the extracted training samples.
    """
    # Create a df to store the training samples
    df = pd.DataFrame()

    # Identify and loop through unique classes in the GeoDataFrame
    unique_classes = gdf[class_col].unique()
    print("Unique classes found:", len(unique_classes), f": {unique_classes}")

    for class_value in unique_classes:
        print(f"\n Processing class: {class_value}")

        # Filter and extract the polygons for this class
        class_polygons = gdf[gdf[class_col] == class_value]

        clipped_list = []
        for raster in raster_list:
            # Clip the raster data using the polygons
            clipped = raster.rio.clip(class_polygons.geometry.values, class_polygons.crs, all_touched=True)
            clipped_list.append(clipped)

        # Extract and store the pixels from images from each band, in df with their associated class
        for raster_idx, raster in enumerate(clipped_list):
            # time_suffix = f"_T{raster_idx+1}"

            for band in raster['band']:
                band_base = str(band.values)
                band_name = band_base # + time_suffix

                single_band = raster.sel(band=band)
                values = single_band.values.flatten()
                values = values[~np.isnan(values)]

                # Append to band column in df
                df[band_name] = pd.concat([df.get(band_name, pd.Series(dtype=float)), pd.Series(values)], ignore_index=True)
                df['label'] = pd.concat([df.get('label', pd.Series(dtype=type(class_value))), pd.Series([class_value] * len(values))], ignore_index=True)
                print(f"Extracted {len(values)} pixels for class {class_value} from band {band}")
    
    return df
    
    