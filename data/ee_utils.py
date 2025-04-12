# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Licensed under the Apache License, Version 2.0

"""Library of Earth Engine utility functions optimized for India."""

import math
import random
import ee

random.seed(123)

DATA_SOURCES = {
    "ELEVATION_SRTM": 'USGS/SRTMGL1_003',
    "VEGETATION_VIIRS": 'MODIS/061/MOD13A2',
    "WEATHER_ERA5": 'ECMWF/ERA5/DAILY',
    "FIRE_MODIS": 'MODIS/061/MOD14A1',
    "POPULATION": 'CIESIN/GPWv411/GPW_Population_Density'
}

DATA_BANDS = {
    "ELEVATION_SRTM": ['elevation'],
    "VEGETATION_VIIRS": ['NDVI'],
    "WEATHER_ERA5": ['mean_2m_air_temperature', 'total_precipitation', 'u_component_of_wind_10m',
                     'v_component_of_wind_10m'],
    "FIRE_MODIS": ['FireMask'],
    "POPULATION": ['population_density']
}

DATA_TIME_SAMPLING = {
    "VEGETATION_VIIRS": 8,
    "WEATHER_ERA5": 1,
    "FIRE_MODIS": 1
}

DETECTION_BAND = 'detection'
DEFAULT_KERNEL_SIZE = 64
DEFAULT_SAMPLING_RESOLUTION = 4000  # 10km for speed
DEFAULT_EVAL_SPLIT = 0.2
DEFAULT_LIMIT_PER_EE_CALL = 30  # Reduced for EE efficiency
DEFAULT_SEED = 123

COORDINATES = {
    'India': [68, 6, 97, 36]
}


def get_image(data_type):
    return ee.Image(DATA_SOURCES[data_type]).select(DATA_BANDS[data_type])


def get_image_collection(data_type):
    return ee.ImageCollection(DATA_SOURCES[data_type]).select(DATA_BANDS[data_type])


def remove_mask(image):
    return image.updateMask(ee.Image(1))


def export_feature_collection(feature_collection, description, folder, bands, file_format='TFRecord'):
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        fileFormat=file_format,
        selectors=bands
    )
    task.start()
    return task


def convert_features_to_arrays(image_list, kernel_size=DEFAULT_KERNEL_SIZE):
    feature_stack = ee.Image.cat(image_list).float()
    kernel_list = ee.List.repeat(1, kernel_size)
    kernel_lists = ee.List.repeat(kernel_list, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_lists)
    return feature_stack.neighborhoodToArray(kernel)


def get_detection_count(detection_image, geometry, sampling_scale=DEFAULT_SAMPLING_RESOLUTION,
                        detection_band=DETECTION_BAND):
    detection_stats = detection_image.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=sampling_scale
    )
    try:
        return int(detection_stats.get(detection_band).getInfo())
    except ee.EEException:
        return -1


def extract_samples(image, detection_count, geometry, sampling_ratio=0, detection_band=DETECTION_BAND,
                    sampling_limit_per_call=DEFAULT_LIMIT_PER_EE_CALL, resolution=DEFAULT_SAMPLING_RESOLUTION,
                    seed=DEFAULT_SEED):
    feature_collection = ee.FeatureCollection([])
    num_per_call = sampling_limit_per_call // (sampling_ratio + 1)
    for _ in range(math.ceil(detection_count / num_per_call)):
        samples = image.stratifiedSample(
            region=geometry,
            numPoints=0,
            classBand=detection_band,
            scale=resolution,
            seed=seed,
            classValues=[0, 1],  # Only positive samples
            classPoints=[num_per_call * sampling_ratio, num_per_call],
            dropNulls=True
        )
        feature_collection = feature_collection.merge(samples)
    return feature_collection


def split_days_into_train_eval_test(start_date, end_date, split_ratio=DEFAULT_EVAL_SPLIT, window_length_days=8):
    num_days = int(ee.Date.difference(end_date, start_date, 'days').getInfo())
    days = list(range(0, num_days, window_length_days))
    random.shuffle(days)
    num_eval = int(len(days) * split_ratio)
    split_days = {}
    split_days['train'] = days[:-2 * num_eval]
    split_days['eval'] = days[-2 * num_eval:-num_eval]
    split_days['test'] = days[-num_eval:]
    return split_days
