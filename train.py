
# coding=utf-8
"""Training script for Next Day Wildfire Spread dataset."""

import os
import math
import tensorflow as tf
from tensorflow import keras
import cnn_autoencoder_model as cnn_autoencoder
import losses as custom_losses
import metrics as custom_metrics
import model_utils
import dataset
from params import HParams

def build_model(hparams):
    input_shape = (hparams.sample_size, hparams.sample_size, len(hparams.input_features))
    input_tensor = keras.Input(shape=input_shape)
    model = cnn_autoencoder.create_model(
        input_tensor=input_tensor,
        num_out_channels=hparams.num_out_channels,
        encoder_layers=hparams.encoder_layers,
        decoder_layers=hparams.decoder_layers,
        encoder_pools=hparams.encoder_pools,
        decoder_pools=hparams.decoder_pools,
        dropout=hparams.dropout,
        batch_norm=hparams.batch_norm,
        l1_regularization=hparams.l1_regularization,
        l2_regularization=hparams.l2_regularization
    )
    outputs = tf.keras.layers.Resizing(64, 64)(model.output)
    return tf.keras.Model(inputs=input_tensor, outputs=outputs)

def train_model():
    hparams = HParams()
    train_dataset = dataset.make_dataset(hparams, "train").shuffle(hparams.shuffle_buffer_size).repeat().prefetch(1)
    eval_dataset = dataset.make_dataset(hparams, "eval").prefetch(1)

    for inputs, labels in train_dataset.take(1):
        print("Input shape:", inputs.shape)
        print("Label shape:", labels.shape)

    model = build_model(hparams)
    print("Model output shape:", model.output.shape)
    
    checkpoint_dir = os.path.join(hparams.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, "model_epoch_{epoch:03d}.weights.h5"),
        save_weights_only=True,
        save_freq=1500*25,  # Every 25 epochs
        verbose=1
    )
    csv_logger = keras.callbacks.CSVLogger(os.path.join(hparams.output_dir, "training_log.csv"), append=True)
    
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        print(f"Loaded model from {latest_checkpoint}")
    else:
        print("Starting fresh model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate),
        loss=custom_losses.weighted_cross_entropy_with_logits_with_masked_class(pos_weight=5.0),
        metrics=[
            custom_metrics.AUCWithMaskedClass(curve="PR", with_logits=True, name="auc_pr"),
            custom_metrics.PrecisionWithMaskedClass(with_logits=True, thresholds=0.2, name="precision"),
            custom_metrics.RecallWithMaskedClass(with_logits=True, thresholds=0.2, name="recall")
        ]
    )

    os.makedirs(hparams.output_dir, exist_ok=True)
    best_model_exporter = model_utils.BestModelExporter(
        metric_key="val_auc_pr",
        min_or_max="max",
        output_dir=hparams.output_dir,
        use_h5=True
    )

    steps_per_epoch = 1500  # ~1.64 passes of train
    validation_steps = math.ceil(1847 / 16)

    model.fit(
        train_dataset,
        epochs=hparams.epochs,
        validation_data=eval_dataset,
        callbacks=[checkpoint, csv_logger, best_model_exporter],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    final_model_path = os.path.join(hparams.output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    train_model()
