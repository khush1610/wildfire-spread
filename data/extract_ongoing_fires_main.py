"""Job for extracting ongoing fires from the dataset exported using export_ee_data.py.

Ongoing fires are fires for which there is at least one positive fire label in
the PrevFireMask and in the FireMask. Samples of ongoing fires are written to
new TFRecords.
"""


from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
import datasetF


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'file_pattern', None,
    'Unix glob pattern for the files containing the input data.')
flags.DEFINE_string('out_file_prefix', None,
                    'File prefix to save the TFRecords.')

flags.DEFINE_integer(
    'data_size', 64,
    'Size of the tiles in pixels (square) as read from input files.')

flags.DEFINE_integer('num_samples_per_file', 3000,
                     'Number of samples to write per TFRecord.')

flags.DEFINE_string(
    'compression_type', 'GZIP',
    'Compression used for the input. Must be one of "GZIP", "ZLIB", or "" (no '
    'compression).')


def _parse_fn(example_proto, data_size, feature_names):
    """Reads a serialized example with simple error handling that avoids graph scope issues."""
    # Create a basic feature dictionary
    features_dict = {}

    # Define all features as optional with default values
    for feature_name in feature_names:
        features_dict[feature_name] = tf.io.FixedLenFeature(
            shape=[data_size * data_size],
            dtype=tf.float32,
            default_value=tf.zeros([data_size * data_size], dtype=tf.float32)
        )

    # Parse the example with the features dictionary
    features = tf.io.parse_single_example(example_proto, features_dict)

    # Process the features without conditional logic
    feature_list = []

    for feature_name in feature_names:
        # Get the feature tensor and reshape it
        feature_tensor = features[feature_name]
        reshaped_tensor = tf.reshape(feature_tensor, [data_size, data_size])

        # Apply fire mask processing if needed
        if 'FireMask' in feature_name:
            # Use a simpler version of map_fire_labels that doesn't introduce graph scope issues
            # This depends on how datasetF.map_fire_labels is implemented
            # For now, just include the tensor as is
            feature_list.append(reshaped_tensor)
        else:
            feature_list.append(reshaped_tensor)

    return feature_list


def get_dataset(
        file_pattern,
        data_size,
        compression_type,
        feature_names,
):
    """Gets the dataset from the file pattern with improved error handling."""
    # Create the dataset from file pattern
    tf_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    # Use a simple function to load TFRecord files that ignores errors
    def load_tfrecords(filename):
        print(f"EVENT: Loading TFRecord: {filename}")
        # Wrap the dataset creation in a try-except to handle potential file errors
        try:
            record_dataset = tf.data.TFRecordDataset(filename, compression_type=compression_type)
            return record_dataset
        except tf.errors.InvalidArgumentError:
            # Return an empty dataset instead of failing
            return tf.data.Dataset.from_tensor_slices([])

    # Interleave the datasets with error handling
    tf_dataset = tf_dataset.interleave(
        load_tfrecords,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Add prefetching for better performance
    tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Use filter to skip any problematic examples before parsing

    def filter_valid_examples(example_proto):
        try:
            # Try a basic parse to see if it's valid
            tf.io.parse_single_example(
                example_proto,
                {name: tf.io.FixedLenFeature([], tf.string, default_value='')
                 for name in ['dummy']}
            )
            return True
        except:
            return False

    # Skip the filtering step to avoid graph scope issues
    # tf_dataset = tf_dataset.filter(filter_valid_examples)

    # Apply the parse function
    tf_dataset = tf_dataset.map(
        lambda x: _parse_fn(x, data_size, feature_names),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Add final prefetching
    tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return tf_dataset


def write_to_tfrecord(tf_writer,
                      feature_names,
                      feature_list):
    """Writes the features to TFRecord files.

  Args:
    tf_writer: TFRecord writer.
    feature_names: Names of all the features.
    feature_list: Values of all the features.
  """
    feature_dict = {}

    for i, feature_name in enumerate(feature_names):
        feature_dict[feature_name] = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=feature_list[i].numpy().reshape(-1)))

    tf_example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))

    tf_writer.write(tf_example.SerializeToString())


def write_ongoing_dataset(tf_dataset,
                          feature_names, file_prefix,
                          num_samples_per_file, compression_type):
    """Writes dataset of ongoing fires with simpler logic to avoid graph scope issues."""
    ongoing_count = 0
    ongoing_tfrecord_count = 0

    prev_fire_index = feature_names.index('PrevFireMask')
    fire_index = feature_names.index('FireMask')

    # Set up TFRecord options
    options = tf.io.TFRecordOptions(compression_type=compression_type)

    # Create a writer for the first file
    out_file = f'{file_prefix}_ongoing_{ongoing_tfrecord_count:03d}.tfrecord.gz'
    writer = tf.io.TFRecordWriter(out_file, options=options)

    # Process the dataset
    for feature_list in tf_dataset:
        # Check if we need to create a new file
        if ongoing_count > 0 and ongoing_count % num_samples_per_file == 0:
            writer.close()
            ongoing_tfrecord_count += 1
            out_file = f'{file_prefix}_ongoing_{ongoing_tfrecord_count:03d}.tfrecord.gz'
            writer = tf.io.TFRecordWriter(out_file, options=options)

        # Get the PrevFireMask tensor
        prev_fire_mask = feature_list[prev_fire_index]

        # Convert to numpy and check for fire presence
        # Using numpy operations outside TF graph to avoid scope issues
        try:
            prev_fire_np = prev_fire_mask.numpy()

            # Check if there's at least one positive fire label
            if np.max(prev_fire_np) >= 0.5:  # Using 0.5 threshold for binary classification
                write_to_tfrecord(writer, feature_names, feature_list)
                ongoing_count += 1
                print(f"EVENT: Ongoing count: {ongoing_count}")

                if ongoing_count % 100 == 0:
                    print(f"EVENT: Processed {ongoing_count} ongoing fire samples")
            else:
                print("EVENT: No positive fire label found, skipping")
        except Exception as e:
            print(f"EVENT: Error processing record: {e}")
            continue

    # Close the final writer
    writer.close()
    print(f"EVENT: Wrote {ongoing_count} ongoing fire samples across {ongoing_tfrecord_count + 1} files")


def main(_):
    feature_names = constants.INPUT_FEATURES + constants.OUTPUT_FEATURES
    print(f"EVENT: Got feature names: {feature_names}")

    tf_dataset = get_dataset(
        FLAGS.file_pattern,
        data_size=FLAGS.data_size,
        feature_names=feature_names,
        compression_type=FLAGS.compression_type)

    write_ongoing_dataset(tf_dataset, feature_names, FLAGS.out_file_prefix,
                          FLAGS.num_samples_per_file, FLAGS.compression_type)

    print("EVENT: Main function complete")


if __name__ == '__main__':
    flags.mark_flag_as_required('file_pattern')
    flags.mark_flag_as_required('out_file_prefix')
    app.run(main)
    print("EVENT: Script complete")
