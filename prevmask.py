import tensorflow as tf
import numpy as np

file_pattern = "./kaggle/next_day_wildfire_spread_eval_00.tfrecord"
dataset = tf.data.TFRecordDataset(file_pattern, compression_type=None)
for record in dataset.take(10):  # Inspect first 10 records
    features = tf.io.parse_single_example(record, {
        'PrevFireMask': tf.io.FixedLenFeature([64 * 64], tf.float32)
    })
    prev_fire_np = features['PrevFireMask'].numpy()
    if np.any(prev_fire_np == 1):
        print("Found PrevFireMask == 1 in sample")
        print(f"Unique values: {np.unique(prev_fire_np)}")