"""Hyperparameters for Next Day Wildfire Spread model (Tuned for 29-30% AUC-PR, <9 hours)."""

import constants


class HParams:
    def __init__(self):
        self.train_path = "./kaggle/next_day_wildfire_spread_train_*.tfrecord"
        self.eval_path = "./kaggle/next_day_wildfire_spread_eval_*.tfrecord"
        self.test_path = "./kaggle/next_day_wildfire_spread_test_*.tfrecord"
        self.data_sample_size = 64
        self.sample_size = 64
        self.output_sample_size = 64
        self.batch_size = 16
        self.shuffle_buffer_size = 2500  # ~17% of train
        self.compression_type = None
        self.input_sequence_length = 1
        self.output_sequence_length = 1
        self.input_features = constants.INPUT_FEATURES
        self.output_features = constants.OUTPUT_FEATURES
        self.random_flip = True
        self.random_rotate = True
        self.random_crop = False
        self.downsample_threshold = 0.0
        self.binarize_output = True
        self.azimuth_in_channel = constants.INPUT_FEATURES.index("th")
        self.azimuth_out_channel = None
        self.encoder_layers = [32, 64, 128, 256]
        self.decoder_layers = [256, 128, 64]
        self.encoder_pools = [2, 2, 2, 2]
        self.decoder_pools = [2, 2, 2]
        self.num_out_channels = 1
        self.dropout = 0.2
        self.batch_norm = "all"
        self.l1_regularization = 0.0
        self.l2_regularization = 0.01
        self.learning_rate = 0.0001
        self.epochs = 450  # ~7.5h
        self.output_dir = "/kaggle/working/model_tuned"
