# # export_ee_data.py
# """Earth Engine helper functions for India wildfire data export."""
#
# import ee
# from absl import logging
# import ee_utils
#
# logging.set_verbosity(logging.INFO)
#
#
# def _get_all_feature_bands():
#     bands = ('elevation', 'population', 'NDVI', 'mean_2m_air_temperature',
#              'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
#              'PrevFireMask', 'FireMask')
#     logging.info('Feature bands: %s', bands)
#     return bands
#
#
# def _get_all_image_collections():
#     image_collections = {
#         'vegetation': ee_utils.get_image_collection('VEGETATION_VIIRS'),
#         'weather': ee_utils.get_image_collection('WEATHER_ERA5'),
#         'fire': ee_utils.get_image_collection('FIRE_MODIS'),
#     }
#     logging.info('Image collections loaded: %s', list(image_collections.keys()))
#     return image_collections, ee_utils.DATA_TIME_SAMPLING
#
#
# def _get_time_slices(window_start, window, projection, resampling_scale, lag=1):
#     date_str = window_start.format('YYYY-MM-DD').getInfo()
#     image_collections, time_sampling = _get_all_image_collections()
#     window_end = window_start.advance(window, 'day')
#
#     fire_collection = image_collections['fire'].filterDate(window_start, window_end).map(ee_utils.remove_mask)
#     fire_count = fire_collection.size().getInfo()
#     fire = fire_collection.max().gte(7).rename('FireMask') if fire_count > 0 else ee.Image(0).rename(
#         'FireMask').reproject(projection.atScale(resampling_scale))
#     detection = fire_collection.max().clamp(6, 7).subtract(6).rename('detection') if fire_count > 0 else ee.Image(
#         0).rename('detection').reproject(projection.atScale(resampling_scale))
#
#     if ee.Number(ee_utils.get_detection_count(detection, geometry=ee.Geometry.Rectangle(ee_utils.COORDINATES['India']),
#                                               sampling_scale=5000)).getInfo() == 0:
#         return None
#
#     vegetation = image_collections['vegetation'].filterDate(
#         window_start.advance(-lag - time_sampling['VEGETATION_VIIRS'], 'day'),
#         window_start.advance(-lag, 'day')
#     ).median().reproject(projection).resample('bicubic')
#     weather = image_collections['weather'].filterDate(
#         window_start.advance(-lag - time_sampling['WEATHER_ERA5'], 'day'),
#         window_start.advance(-lag, 'day')
#     ).median().reproject(projection.atScale(resampling_scale)).resample('bicubic')
#     prev_fire_collection = image_collections['fire'].filterDate(
#         window_start.advance(-lag - time_sampling['FIRE_MODIS'], 'day'),
#         window_start.advance(-lag, 'day')
#     ).map(ee_utils.remove_mask)
#     prev_fire = prev_fire_collection.max().gte(7).rename(
#         'PrevFireMask') if prev_fire_collection.size().getInfo() > 0 else ee.Image(0).rename('PrevFireMask').reproject(
#         projection.atScale(resampling_scale))
#
#     return vegetation, weather, prev_fire, fire, detection
#
#
# def _export_dataset(folder, prefix, start_date, start_days, geometry, kernel_size, sampling_scale, num_samples_per_file,
#                     lag=1):
#     logging.info('Starting export for dataset: %s', prefix)
#     elevation = ee_utils.get_image("ELEVATION_SRTM").reproject(crs='EPSG:4326', scale=sampling_scale)
#     population = ee_utils.get_image_collection("POPULATION").filterDate(
#         start_date, start_date.advance(1826, 'days')
#     ).median().reproject(crs='EPSG:4326', scale=sampling_scale).rename('population')
#     projection = ee_utils.get_image_collection("WEATHER_ERA5").first().select(('total_precipitation',)).projection()
#     features = _get_all_feature_bands()
#     file_count = 0
#     feature_collection = ee.FeatureCollection([])
#     total_samples = 0
#     days_processed = 0
#
#     for start_day in start_days:
#         days_processed += 1
#         window_start = start_date.advance(start_day, 'days')
#         date_str = window_start.format('YYYY-MM-DD').getInfo()
#         logging.info('Day %d (%s): Processing...', start_day, date_str)
#
#         time_slices = _get_time_slices(window_start, 1, projection, sampling_scale, lag)
#         if time_slices is None:
#             logging.info('Day %d (%s): Fire count = 0, skipping', start_day, date_str)
#             continue
#
#         image_list = (elevation, population) + time_slices[:-1]
#         detection = time_slices[-1]
#         to_sample = detection.addBands(ee.Image.cat(image_list).reproject(projection.atScale(sampling_scale)))
#         tiles = ee_utils.tile_image_to_features(to_sample, geometry, kernel_size, sampling_scale)
#         feature_collection = feature_collection.merge(tiles)
#         size = feature_collection.size().getInfo()
#         total_samples += size
#         logging.info('Day %d: Merged %d tiles (Running total: %d across %d days)', start_day, size,
#                      total_samples, days_processed)
#
#         if size >= num_samples_per_file:
#             export_name = f'{prefix}_{file_count:03d}'
#             logging.info('Exporting %d tiles to %s/%s', size, folder, export_name)
#             ee_utils.export_feature_collection(feature_collection, export_name, folder, features)
#             file_count += 1
#             feature_collection = ee.FeatureCollection([])
#
#     if feature_collection.size().getInfo() > 0:
#         export_name = f'{prefix}_{file_count:03d}'
#         logging.info('Exporting remaining %d tiles to %s/%s', feature_collection.size().getInfo(), folder, export_name)
#         ee_utils.export_feature_collection(feature_collection, export_name, folder, features)
#
#
# def export_ml_datasets(folder, start_date, end_date, prefix='', lag=1, num_samples_per_file=500, kernel_size=64,
#                        eval_split_ratio=0.2, sampling_scale=5000):
#     logging.info('Starting export_ml_datasets from %s to %s...', start_date.format('YYYY-MM-DD').getInfo(),
#                  end_date.format('YYYY-MM-DD').getInfo())
#     split_days = ee_utils.split_days_into_train_eval(start_date, end_date, eval_split_ratio)
#     geometry = ee.Geometry.Rectangle(ee_utils.COORDINATES['India'])
#     for mode in ['train', 'eval']:
#         logging.info('Exporting %s dataset...', mode)
#         _export_dataset(
#             folder=folder,
#             prefix=f'{mode}_{prefix}',
#             start_date=start_date,
#             start_days=split_days[mode],
#             geometry=geometry,
#             kernel_size=kernel_size,
#             sampling_scale=sampling_scale,
#             num_samples_per_file=num_samples_per_file,
#             lag=lag
#         )
#     logging.info('All datasets exported successfully!')
"""Earth Engine helper functions for India wildfire data export."""

# import ee
# from absl import logging
# import ee_utils
#
# logging.set_verbosity(logging.WARNING)
#
#
# def _get_all_feature_bands():
#     bands = ('elevation', 'population', 'NDVI', 'mean_2m_air_temperature',
#              'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
#              'PrevFireMask', 'FireMask')
#     logging.info('Feature bands: %s', bands)
#     return bands
#
#
# def _get_all_image_collections():
#     image_collections = {
#         'vegetation': ee_utils.get_image_collection('VEGETATION_VIIRS'),
#         'weather': ee_utils.get_image_collection('WEATHER_ERA5'),
#         'fire': ee_utils.get_image_collection('FIRE_MODIS'),
#     }
#     logging.info('Image collections loaded: %s', list(image_collections.keys()))
#     return image_collections, ee_utils.DATA_TIME_SAMPLING
#
#
# def _get_time_slices(window_start, window, projection, resampling_scale, lag=1):
#     date_str = window_start.format('YYYY-MM-DD').getInfo()
#     image_collections, time_sampling = _get_all_image_collections()
#     window_end = window_start.advance(window, 'day')
#     fire = image_collections['fire'].filterDate(window_start, window_end).map(ee_utils.remove_mask).max()
#     detection = fire.clamp(6, 7).subtract(6).rename('detection')
#     if ee.Number(ee_utils.get_detection_count(detection, geometry=ee.Geometry.Rectangle(ee_utils.COORDINATES['India'])
#     ,sampling_scale=5000)).getInfo() == 0:
#         return None  # Skip if no fires
#     vegetation = image_collections['vegetation'].filterDate(
#         window_start.advance(-lag - time_sampling['VEGETATION_VIIRS'], 'day'),
#         window_start.advance(-lag, 'day')
#     ).median().reproject(projection).resample('bicubic')
#     weather = image_collections['weather'].filterDate(
#         window_start.advance(-lag - time_sampling['WEATHER_ERA5'], 'day'),
#         window_start.advance(-lag, 'day')
#     ).median().reproject(projection.atScale(resampling_scale)).resample('bicubic')
#     prev_fire = image_collections['fire'].filterDate(
#         window_start.advance(-lag - time_sampling['FIRE_MODIS'], 'day'),
#         window_start.advance(-lag, 'day')
#     ).map(ee_utils.remove_mask).max().rename('PrevFireMask')
#     return vegetation, weather, prev_fire, fire, detection
#
#
# def _export_dataset(folder, prefix, start_date, start_days, geometry, lag=1, num_samples_per_file=1000):
#     logging.info('Starting export for dataset: %s', prefix)
#     elevation = ee_utils.get_image("ELEVATION_SRTM").reproject(crs='EPSG:4326', scale=5000)
#     population = ee_utils.get_image_collection("POPULATION").filterDate(
#         start_date, start_date.advance(1826, 'days')  # 2019-01-01 to 2023-12-31
#     ).median().reproject(crs='EPSG:4326', scale=5000).rename('population')
#     projection = ee_utils.get_image_collection("WEATHER_ERA5").first().select(('total_precipitation',)).projection()
#     resampling_scale = 20000
#     features = _get_all_feature_bands()
#     file_count = 0
#     feature_collection = ee.FeatureCollection([])
#     all_days = range(0, 1826)  # 2019-01-01 to 2023-12-31
#     sampling_scale = 5000  # 5 km
#     logging.info('Processing %d days at %d m resolution', len(all_days), sampling_scale)
#
#     for start_day in all_days:
#         window_start = start_date.advance(start_day, 'days')
#         date_str = window_start.format('YYYY-MM-DD').getInfo()
#         time_slices = _get_time_slices(window_start, 1, projection, resampling_scale, lag)
#         if time_slices is None:
#             logging.info('Day %d (%s): Fire count = 0, skipping', start_day, date_str)
#             continue
#         image_list = (elevation, population) + time_slices[:-1]
#         detection = time_slices[-1]
#         arrays = ee_utils.convert_features_to_arrays(image_list)
#         to_sample = detection.addBands(arrays)
#         fire_count = ee_utils.get_detection_count(detection, geometry=geometry, sampling_scale=sampling_scale)
#         logging.info('Day %d (%s): Fire count = %d', start_day, date_str, fire_count)
#         if fire_count > 0:
#             samples = ee_utils.extract_samples(to_sample, fire_count, geometry)
#             feature_collection = feature_collection.merge(samples)
#             size = feature_collection.size().getInfo()
#             if size >= num_samples_per_file:
#                 export_name = f'{prefix}_{file_count:03d}'
#                 logging.info('Exporting %d samples to %s/%s', size, folder, export_name)
#                 ee_utils.export_feature_collection(feature_collection, export_name, folder, features)
#                 file_count += 1
#                 feature_collection = ee.FeatureCollection([])
#     size = feature_collection.size().getInfo()
#     if size > 0:
#         export_name = f'{prefix}_{file_count:03d}'
#         logging.info('Exporting final %d samples to %s/%s', size, folder, export_name)
#         ee_utils.export_feature_collection(feature_collection, export_name, folder, features)
#     logging.info('Export completed for %s!', prefix)
#
#
# def export_ml_datasets(folder, start_date, end_date, prefix='', lag=1, num_samples_per_file=1000):
#     logging.info('Starting export_ml_datasets from %s to %s...', start_date.format('YYYY-MM-DD').getInfo(),
#                  end_date.format('YYYY-MM-DD').getInfo())
#     split_days = {'train': range(0, 1826)}  # 2019-01-01 to 2023-12-31
#     logging.info('Days split into train: %d days', len(split_days['train']))
#     geometry = ee.Geometry.Rectangle(ee_utils.COORDINATES['India'])
#     logging.info('Using geometry for India: %s', geometry.getInfo()['coordinates'])
#     logging.info('Exporting train dataset...')
#     _export_dataset(folder, f'train_{prefix}', start_date, split_days['train'], geometry, lag, num_samples_per_file)
#     logging.info('All datasets exported successfully!'
#
#


# Dataset/Data/Export_ee_data/export_ee_data.py
# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Licensed under the Apache License, Version 2.0

"""Earth Engine helper functions for India wildfire data export."""

import ee
import logging
import ee_utils

logging.basicConfig(level=logging.INFO)


def _get_all_feature_bands():
    bands = ('elevation', 'population', 'NDVI', 'mean_2m_air_temperature',
             'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
             'PrevFireMask', 'FireMask')
    logging.info('Feature bands: %s', bands)
    return bands


def _get_all_image_collections():
    image_collections = {
        'vegetation': ee_utils.get_image_collection('VEGETATION_VIIRS'),
        'weather': ee_utils.get_image_collection('WEATHER_ERA5'),
        'fire': ee_utils.get_image_collection('FIRE_MODIS'),
    }
    logging.info('Image collections loaded: %s', list(image_collections.keys()))
    return image_collections, ee_utils.DATA_TIME_SAMPLING


def _get_time_slices(window_start, window, projection, resampling_scale, lag=1):
    date_str = window_start.format('YYYY-MM-DD').getInfo()
    image_collections, time_sampling = _get_all_image_collections()
    window_end = window_start.advance(window, 'day')
    fire = image_collections['fire'].filterDate(window_start, window_end).map(ee_utils.remove_mask).max()
    detection = fire.clamp(6, 7).subtract(6).rename('detection')
    if ee.Number(ee_utils.get_detection_count(detection, geometry=ee.Geometry.Rectangle(ee_utils.COORDINATES['India']),
                                              sampling_scale=4000)).getInfo() == 0:
        return None
    vegetation = image_collections['vegetation'].filterDate(
        window_start.advance(-lag - time_sampling['VEGETATION_VIIRS'], 'day'),
        window_start.advance(-lag, 'day')
    ).median().reproject(projection).resample('bicubic')
    weather = image_collections['weather'].filterDate(
        window_start.advance(-lag - time_sampling['WEATHER_ERA5'], 'day'),
        window_start.advance(-lag, 'day')
    ).median().reproject(projection.atScale(resampling_scale)).resample('bicubic')
    prev_fire = image_collections['fire'].filterDate(
        window_start.advance(-lag - time_sampling['FIRE_MODIS'], 'day'),
        window_start.advance(-lag, 'day')
    ).map(ee_utils.remove_mask).max().rename('PrevFireMask')
    return vegetation, weather, prev_fire, fire, detection


def _export_dataset(folder, prefix, start_date, start_days, geometry, lag=1, num_samples_per_file=500):
    logging.info('Starting export for dataset: %s', prefix)
    elevation = ee_utils.get_image("ELEVATION_SRTM").reproject(crs='EPSG:4326', scale=4000)
    population = ee_utils.get_image_collection("POPULATION").filterDate(
        start_date, start_date.advance(1826, 'days')
    ).median().reproject(crs='EPSG:4326', scale=4000).rename('population')
    projection = ee_utils.get_image_collection("WEATHER_ERA5").first().select(('total_precipitation',)).projection()
    resampling_scale = 20000
    features = _get_all_feature_bands()
    file_count = 0
    feature_collection = ee.FeatureCollection([])
    all_days = range(0, 1826)
    sampling_scale = 4000
    total_samples = 0  # Track cumulative samples
    days_processed = 0  # Track days

    logging.info('Processing %d days at %d m resolution', len(all_days), sampling_scale)

    for start_day in all_days:
        days_processed += 1
        window_start = start_date.advance(start_day, 'days')
        date_str = window_start.format('YYYY-MM-DD').getInfo()
        time_slices = _get_time_slices(window_start, 1, projection, resampling_scale, lag)
        if time_slices is None:
            logging.info('Day %d (%s): Fire count = 0, skipping (Total samples so far: %d)', start_day, date_str,
                         total_samples)
            continue
        image_list = (elevation, population) + time_slices[:-1]
        detection = time_slices[-1]
        arrays = ee_utils.convert_features_to_arrays(image_list)
        to_sample = detection.addBands(arrays)
        fire_count = ee_utils.get_detection_count(detection, geometry=geometry, sampling_scale=sampling_scale)
        logging.info('Day %d (%s): Fire count = %d', start_day, date_str, fire_count)
        if fire_count > 0:
            samples = ee_utils.extract_samples(to_sample, fire_count, geometry)
            feature_collection = feature_collection.merge(samples)
            size = feature_collection.size().getInfo()
            total_samples += size  # Update total
            logging.info('Day %d: Merged %d samples (Running total: %d across %d days)', start_day, size, total_samples,
                         days_processed)
            if size >= num_samples_per_file:
                export_name = f'{prefix}_{file_count:03d}'
                logging.info('Exporting %d samples to %s/%s (Total exported so far: %d)', size, folder, export_name,
                             total_samples)
                ee_utils.export_feature_collection(feature_collection, export_name, folder, features)
                file_count += 1
                feature_collection = ee.FeatureCollection([])
                logging.info('Reset feature collection, continuing...')
    size = feature_collection.size().getInfo()
    if size > 0:
        total_samples += size
        export_name = f'{prefix}_{file_count:03d}'
        logging.info('Exporting final %d samples to %s/%s (Total exported: %d)', size, folder, export_name,
                     total_samples)
        ee_utils.export_feature_collection(feature_collection, export_name, folder, features)
    logging.info('Export completed for %s! Total samples: %d across %d days', prefix, total_samples, days_processed)


def export_ml_datasets(folder, start_date, end_date, prefix='', lag=1, num_samples_per_file=500,
                       eval_split_ratio=0.2):
    logging.info('Starting export_ml_datasets from %s to %s...',
                 start_date.format('YYYY-MM-DD').getInfo(),
                 end_date.format('YYYY-MM-DD').getInfo())

    # Split days into train, eval, and test sets
    split_days = ee_utils.split_days_into_train_eval_test(
        start_date, end_date, split_ratio=eval_split_ratio, window_length_days=8
    )

    logging.info('Days split - Train: %d, Eval: %d, Test: %d',
                 len(split_days['train']), len(split_days['eval']), len(split_days['test']))

    geometry = ee.Geometry.Rectangle(ee_utils.COORDINATES['India'])
    logging.info('Using geometry for India: %s', geometry.getInfo()['coordinates'])

    # Export datasets
    for mode in ['train', 'eval', 'test']:
        logging.info('Exporting %s dataset...', mode)
        _export_dataset(folder, f'{mode}_{prefix}', start_date, split_days[mode], geometry, lag, num_samples_per_file)

    logging.info('All datasets exported successfully!')
