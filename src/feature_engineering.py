import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
import rioxarray

xr.set_options(keep_attrs=True, display_expand_data=True)


def calculate_indices(raster_ds):
    """
    Calculate spectral indices from Sentinel-2 bands.
    
    Parameters:
    -----------
    raster_ds : xarray.Dataset
        Image dataset containing Sentinel-2 bands
        
    Returns:
    --------
    xarray.Dataset
        Dataset with original bands and added spectral indices
    """
    # Create a copy of the input dataset
    result_ds = raster_ds.copy(deep=True)
    
    try:
        # Normalized Difference Vegetation Index (NDVI)
        nir = result_ds['B8'].astype(float)
        red = result_ds['B4'].astype(float)

        ndvi = (nir - red) / (nir + red)
        ndvi = ndvi.clip(min=-1, max=1)
        ndvi.name = "NDVI"
        ndvi.attrs["long_name"] = "Normalized Difference Vegetation Index"
        ndvi.attrs["units"] = "unitless"
        result_ds['NDVI'] = ndvi.where(~np.isnan(ndvi), 0)
        
        # Soil Adjusted Vegetation Index (SAVI)
        # L is a soil brightness correction factor, usually 0.5
        L = 0.5
        savi = ((nir - red) / (nir + red + L)) * (1 + L)
        savi = savi.clip(min=-1, max=1)
        savi.name = "SAVI"
        savi.attrs["long_name"] = "Soil Adjusted Vegetation Index"
        savi.attrs["units"] = "unitless"
        result_ds['SAVI'] = savi.where(~np.isnan(savi), 0)
        
        # Modified Normalized Difference Water Index (MNDWI)
        green = result_ds['B3'].astype(float)
        swir = result_ds['B11'].astype(float)

        mndwi = (green - swir) / (green + swir)
        mndwi = mndwi.clip(min=-1, max=1)
        mndwi.name = "MNDWI"
        mndwi.attrs["long_name"] = "Modified Normalized Difference Water Index"
        mndwi.attrs["units"] = "unitless"
        result_ds['MNDWI'] = mndwi.where(~np.isnan(mndwi), 0)
        
        # Normalized Difference Built-up Index (NDBI)
        ndbi = (swir - nir) / (swir + nir)
        ndbi = ndbi.clip(min=-1, max=1)
        ndbi.name = "NDBI"
        ndbi.attrs["long_name"] = "Normalized Difference Built-up Index"
        ndbi.attrs["units"] = "unitless"
        result_ds['NDBI'] = ndbi.where(~np.isnan(ndbi), 0)
        
        # Bare Soil Index (BSI)
        blue = result_ds['B2'].astype(float)

        bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))
        bsi = bsi.clip(min=-1, max=1)
        bsi.name = "BSI"
        bsi.attrs["long_name"] = "Bare Soil Index"
        bsi.attrs["units"] = "unitless"
        result_ds['BSI'] = bsi.where(~np.isnan(bsi), 0)

        return result_ds
    
    except Exception as e:
        print(f"Error calculating indices: {e}")
        return None


def extract_training_samples(raster_ds1, raster_ds2, gdf, class_col):
    """
    Extract training samples from the raster data using the provided GeoDataFrame.
    
    Args:
        raster_ds (rioxarray.Dataset): The raster data as a rioxarray.Dataset.
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
        for raster in ([raster_ds1, raster_ds2]):
            # Clip the raster data using the polygons
            clipped = raster.rio.clip(class_polygons.geometry.values, class_polygons.crs, all_touched=True)
            clipped_list.append(clipped)

        # Extract and store the pixels from images from each band, in df with their associated class
        for raster_idx, raster in enumerate(clipped_list):
            time_suffix = f"_T{raster_idx+1}"

            for band in raster['band']:
                band_base = str(band.values)
                band_name = band_base + time_suffix

                single_band = raster.sel(band=band)
                values = single_band.values.flatten()
                values = values[~np.isnan(values)]

                # Append to band column in df
                df[band_name] = pd.concat([df.get(band_name, pd.Series(dtype=float)), pd.Series(values)], ignore_index=True)
                df['label'] = pd.concat([df.get('label', pd.Series(dtype=type(class_value))), pd.Series([class_value] * len(values))], ignore_index=True)
                print(f"Extracted {len(values)} pixels for class {class_value} from band {band}")
    
    return df
    
    