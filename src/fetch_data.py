import ee

# Autheticate and initialize Earth Engine
def authenticate_gee():
    """
    Authenticate and initialize Google Earth Engine.
    """
    try:
        ee.Initialize(project='ee-hazzard')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='ee-hazzard')


def get_image_projection_info(image):
    """
    Prints the native projection, CRS, and nominal scale of an ee.Image.
    
    Args:
        image (ee.Image): The Earth Engine image.
    """
    proj = image.projection()
    crs = proj.crs().getInfo()
    scale = proj.nominalScale().getInfo()
    
    print(f"🔹 Native CRS: {crs}")
    print(f"🔹 Nominal Scale: {scale} meters per pixel")


def filter_image_collection(imageCollection, csPlusImgCollection, dateFilter, roi, cloudFilter, qa_band, clear_threshold):
    """
    Filter and preprocess the image collection.

    Args:
    - imageCollection: ee.ImageCollection to filter Sentinel-2 images
    - csPlusImgCollection: ee.ImageCollection for Cloud Score+
    - dateFilter: ee.Filter for date range, Sentinel-2
    - roi: ee.Geometry for region of interest
    - cloudFilter: ee.Filter for cloud coverage, Sentinel-2
    - QA_BAND: str, name of the Cloud Score+ QA band
    - Clear_Threshold: float, threshold for cloud score

    Returns:
    - ee.ImageCollection: Filtered image collection
    """
    filtered_image = (imageCollection
        .filter(dateFilter)
        .filterBounds(roi)
        .filter(cloudFilter)
        .linkCollection(csPlusImgCollection, [qa_band])
        .map(lambda img: img.updateMask(img.select(qa_band).gte(clear_threshold)))
        .map(lambda img: img.divide(10000).clip(roi))
    )
    return filtered_image


def create_image_composite(imageCollection):
    """
    Create a composite image from the filtered image collection.

    Args:
        imageCollection (ee.ImageCollection): The filtered image collection.

    Returns:
        ee.Image: The composite image.
    """
    # return imageCollection.select('B.*').median()
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    return imageCollection.select(bands).median()


def clamp_img(image, roi):
    """
    Removes outliers from an image by clamping each band to its 1st and 99th percentiles.

    Args:
        image (ee.Image): The input image.
        roi (ee.Geometry): The region over which to compute percentiles.

    Returns:
        ee.Image: The scaled image with outliers removed.
    """
    # Get band names
    bands = image.bandNames()

    # Compute percentiles for each band
    percentiles = image.reduceRegion(
        reducer=ee.Reducer.percentile([1, 99]),
        geometry=roi,
        scale=10,
        maxPixels=1e13
    )

    def clamp_band(band_name):
        band_name = ee.String(band_name)
        band = image.select([band_name])

        p1 = ee.Number(percentiles.get(band_name.cat('_p1')))
        p99 = ee.Number(percentiles.get(band_name.cat('_p99')))

        clipped_band = band.clamp(p1, p99)

        return clipped_band.rename([band_name])

    # Map over each band and apply clamping
    scaled_bands = bands.map(clamp_band)

    # Combine all single-band images into one multi-band image
    scaled_image = ee.ImageCollection(scaled_bands).toBands()
    
    # Rename to original band names
    return scaled_image.rename(bands)


def resample_image_to_10m(image, band_list, crs='EPSG:4326', scale=10, method='bilinear'):
    """
    Resamples all bands of an image to 10m resolution using the specified method.

    Args:
        image (ee.Image): Input image with mixed resolution bands.
        crs (str): Coordinate Reference System (default: UTM Zone 43N for India).
        scale (int): Desired scale in meters.
        method (str): Resampling method ('nearest', 'bilinear', 'bicubic').

    Returns:
        ee.Image: Resampled image with all bands at 10m resolution.
    """
    bands = image.select(band_list).resample(method).reproject(crs=crs, scale=scale)

    return image.addBands(bands)


def export_image_to_drive(image, region, description, folder, file_name_prefix, scale=10, crs='EPSG:4326', max_pixels=1e13):
    """
    Exports an Earth Engine image to Google Drive.

    Args:
        image (ee.Image): The image to export.
        region (ee.Geometry): Export region.
        description (str): Task description.
        folder (str): Drive folder name.
        file_name_prefix (str): File name prefix.
        scale (int): Pixel resolution (in meters).
        crs (str): Coordinate Reference System.
        max_pixels (int): Max pixels allowed.
    """
    image_select = image.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])

    task = ee.batch.Export.image.toDrive(
        image=image_select,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        # scale=scale,
        scale=image_select.projection().nominalScale(),
        region=region,
        crs=crs,
        maxPixels=max_pixels,
        fileFormat='GeoTIFF',
        )
    
    task.start()
    print(f"Export started: {description}, task id: {task.id}")


def main():
    # Authenticate and initialize Earth Engine  
    authenticate_gee()

    # Define the region of interest (ROI)
    roi = ee.Geometry.Polygon([
        [[73.3933018836958, 18.88730081255972],
        [73.3933018836958, 18.265071030538323],
        [74.1733311805708, 18.265071030538323],
        [74.1733311805708, 18.88730081255972]]
    ])

    # Load datasets
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    # Define date filters
    dateFilter_2019 = ee.Filter.date('2019-01-01', '2019-12-30')
    dateFilter_2024 = ee.Filter.date('2024-01-01', '2024-12-15')

    # Define metadata filters
    cloudFilter = ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 2)

    # Cloud Score+ QA bands
    qa_band = 'cs_cdf'
    clear_threshold = 0.60

    # Get filtered images for 2019 and 2024
    imgCollection_2019 = filter_image_collection(s2, csPlus, dateFilter_2019, roi, cloudFilter, qa_band, clear_threshold)
    imgCollection_2024 = filter_image_collection(s2, csPlus, dateFilter_2024, roi, cloudFilter, qa_band, clear_threshold)

    # Create composite images
    imgComp2019 = create_image_composite(imgCollection_2019)
    imgComp2024 = create_image_composite(imgCollection_2024)

    # Clamp images to remove outliers
    clampedImage2019 = clamp_img(imgComp2019, roi)
    clampedImage2024 = clamp_img(imgComp2024, roi)

    # Resample images to 10m resolution
    # band_list = ['B1', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12']
    # resampledImage2019 = resample_image_to_10m(clampedImage2019, band_list)
    # resampledImage2024 = resample_image_to_10m(clampedImage2024, band_list)

    # Export images to Google Drive
    export_image_to_drive(clampedImage2019, roi, 'year composite 2019', 'GEE_Exports/native', 'img_2019')
    export_image_to_drive(clampedImage2024, roi, 'year composite 2024', 'GEE_Exports/native', 'img_2024')
    print("Export tasks started. Check your Google Drive for the images.")

if __name__ == "__main__":
    main()

