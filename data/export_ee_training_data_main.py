# # export_ee_training_data_main.py
# # coding=utf-8
# # Copyright 2024 The Google Research Authors.
# # Licensed under the Apache License, Version 2.0
#
# """Main program to export training data for India wildfire ML project."""
#
# from absl import app
# from absl import flags
# from absl import logging
# import ee
# import export_ee_data
#
# FLAGS = flags.FLAGS
# flags.DEFINE_string('folder', 'india_wildfire_spread', 'Output folder on Google Drive.')
# flags.DEFINE_string('start_date', '2019-01-01', 'Start date (YYYY-MM-DD).')
# flags.DEFINE_string('end_date', '2023-12-31', 'End date (YYYY-MM-DD).')
# flags.DEFINE_string('prefix', 'wildfire_ind', 'File prefix for output files.')
# flags.DEFINE_integer('lag', 1, 'Days prior for PrevFireMask.')
# flags.DEFINE_integer('num_samples_per_file', 500, 'Approximate number of samples per TFRecord file.')
# flags.DEFINE_integer('kernel_size', 64, 'Size of the exported tiles in pixels (square).')
# flags.DEFINE_float('eval_split_ratio', 0.2, 'Proportion of dataset for evaluation (0 to 1).')
# flags.DEFINE_integer('sampling_scale', 5000, 'Resolution at which to export the data (in meters).')
#
#
# def main(argv):
#     if len(argv) > 10:
#         raise app.UsageError('Too many command-line arguments.')
#
#     logging.info('Starting India wildfire EE export job...')
#     logging.info(
#         'Configuration: folder=%s, start_date=%s, end_date=%s, prefix=%s, lag=%d, num_samples_per_file=%d, kernel_size=%d, eval_split_ratio=%f, sampling_scale=%d',
#         FLAGS.folder, FLAGS.start_date, FLAGS.end_date, FLAGS.prefix, FLAGS.lag, FLAGS.num_samples_per_file,
#         FLAGS.kernel_size, FLAGS.eval_split_ratio, FLAGS.sampling_scale)
#
#     try:
#         logging.info('Initializing Earth Engine with project=ee-khush22b...')
#         ee.Initialize(project='ee-khush22b')
#         logging.info('EE authenticated successfully!')
#
#         start_date = ee.Date(FLAGS.start_date)
#         end_date = ee.Date(FLAGS.end_date)
#         logging.info('Converted dates: start_date=%s, end_date=%s', start_date.format('YYYY-MM-DD').getInfo(),
#                      end_date.format('YYYY-MM-DD').getInfo())
#         logging.info('Calling export_ml_datasets...')
#         export_ee_data.export_ml_datasets(
#             folder=FLAGS.folder,
#             start_date=start_date,
#             end_date=end_date,
#             prefix=FLAGS.prefix,
#             lag=FLAGS.lag,
#             num_samples_per_file=FLAGS.num_samples_per_file,
#             kernel_size=FLAGS.kernel_size,
#             eval_split_ratio=FLAGS.eval_split_ratio,
#             sampling_scale=FLAGS.sampling_scale
#         )
#         logging.info('Export process completed! Check Earth Engine Tasks for progress.')
#     except Exception as e:
#         logging.error('Export process failed: %s', str(e))
#         raise
#
#
# if __name__ == '__main__':
#     logging.use_python_logging()
#     app.run(main)

# Dataset/Data/Export_ee_training_data_main.py
# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Licensed under the Apache License, Version 2.0

"""Main program to export training data for India wildfire ML project."""

from absl import app
from absl import flags
from absl import logging
import ee
import export_ee_data

# FLAGS = flags.FLAGS
# flags.DEFINE_string('folder', 'india_wildfire_spread', 'Output folder on Google Drive.')
# flags.DEFINE_string('start_date', '2019-01-01', 'Start date (YYYY-MM-DD).')
# flags.DEFINE_string('end_date', '2023-12-31', 'End date (YYYY-MM-DD).')
# flags.DEFINE_string('prefix', 'wildfire', 'File prefix for output files.')  # Updated to 'wildfire'
# flags.DEFINE_integer('lag', 1, 'Days prior for PrevFireMask.')
# flags.DEFINE_integer('num_samples_per_file', 1000,
#                      'Approximate number of samples per TFRecord file.')  # Updated to 1000
#
#
# def main(argv):
#     if len(argv) > 6:
#         raise app.UsageError('Too many command-line arguments.')
#
#     logging.info('Starting India wildfire EE export job...')
#     logging.info('Configuration: folder=%s, start_date=%s, end_date=%s, prefix=%s, lag=%d, num_samples_per_file=%d',
#                  FLAGS.folder, FLAGS.start_date, FLAGS.end_date, FLAGS.prefix, FLAGS.lag, FLAGS.num_samples_per_file)
#
#     try:
#         logging.info('Initializing Earth Engine with project=ee-khush22b...')
#         ee.Initialize(project='ee-khush22b')
#         logging.info('EE authenticated successfully!')
#
#         start_date = ee.Date(FLAGS.start_date)
#         end_date = ee.Date(FLAGS.end_date)
#         logging.info('Converted dates: start_date=%s, end_date=%s', start_date.format('YYYY-MM-DD').getInfo(),
#                      end_date.format('YYYY-MM-DD').getInfo())
#         logging.info('Calling export_ml_datasets...')
#         export_ee_data.export_ml_datasets(
#             folder=FLAGS.folder,
#             start_date=start_date,
#             end_date=end_date,
#             prefix=FLAGS.prefix,
#             lag=FLAGS.lag,
#             num_samples_per_file=FLAGS.num_samples_per_file
#         )
#         logging.info('Export process completed! Check Earth Engine Tasks for progress.')
#     except Exception as e:
#         logging.error('Export process failed: %s', str(e))
#         raise
#
#
# if __name__ == '__main__':
#     logging.use_python_logging()
#     app.run(main)


# Dataset/Data/Export_ee_training_data_main.py
# coding=utf-8
# Copyright 2024 The Google Research Authors.
# Licensed under the Apache License, Version 2.0

"""Main program to export training data for India wildfire ML project."""

from absl import app
from absl import flags
import logging
import ee
import export_ee_data

FLAGS = flags.FLAGS
flags.DEFINE_string('folder', 'wildfire', 'Output folder on Google Drive.')
flags.DEFINE_string('start_date', '2019-01-01', 'Start date (YYYY-MM-DD).')
flags.DEFINE_string('end_date', '2023-12-31', 'End date (YYYY-MM-DD).')
flags.DEFINE_string('prefix', 'india_spread', 'File prefix for output files.')
flags.DEFINE_integer('lag', 1, 'Days prior for PrevFireMask.')
flags.DEFINE_integer('num_samples_per_file', 500, 'Approximate number of samples per TFRecord file.')


def main(argv):
    if len(argv) > 6:
        raise app.UsageError('Too many command-line arguments.')

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting India wildfire EE export job...')
    logging.info('Configuration: folder=%s, start_date=%s, end_date=%s, prefix=%s, lag=%d, num_samples_per_file=%d',
                 FLAGS.folder, FLAGS.start_date, FLAGS.end_date, FLAGS.prefix, FLAGS.lag, FLAGS.num_samples_per_file)

    try:
        logging.info('Initializing Earth Engine with project=ee-krax...')
        ee.Initialize(project='ee-krax')
        logging.info('EE authenticated successfully!')

        start_date = ee.Date(FLAGS.start_date)
        end_date = ee.Date(FLAGS.end_date)
        logging.info('Converted dates: start_date=%s, end_date=%s', start_date.format('YYYY-MM-DD').getInfo(),
                     end_date.format('YYYY-MM-DD').getInfo())
        logging.info('Calling export_ml_datasets...')
        export_ee_data.export_ml_datasets(
            folder=FLAGS.folder,
            start_date=start_date,
            end_date=end_date,
            prefix=FLAGS.prefix,
            lag=FLAGS.lag,
            num_samples_per_file=FLAGS.num_samples_per_file
        )
        logging.info('Export process completed! Check Earth Engine Tasks for progress.')
    except Exception as e:
        logging.error('Export process failed: %s', str(e))
        raise


if __name__ == '__main__':
    app.run(main)
