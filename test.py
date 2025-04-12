# test.py
# coding=utf-8
"""Testing script for Next Day Wildfire Spread dataset."""

import tensorflow as tf
from tensorflow.compat.v2 import keras
import cnn_autoencoder_model as cnn_autoencoder
import losses as custom_losses
import metrics as custom_metrics
import dataset
from params import HParams
import os


# Define the model (same as training for consistency)
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


# Testing function
def test_model():
    hparams = HParams()
    test_dataset = dataset.make_dataset(hparams=hparams, mode="predict")
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # Build model structure
    model = build_model(hparams)

    # Compile with same settings as training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams.learning_rate),
        loss=custom_losses.weighted_cross_entropy_with_logits_with_masked_class(pos_weight=3.0),
        metrics=[
            custom_metrics.AUCWithMaskedClass(curve="PR", with_logits=True, name="auc_pr"),
            custom_metrics.PrecisionWithMaskedClass(with_logits=True, thresholds=0.5, name="precision"),
            custom_metrics.RecallWithMaskedClass(with_logits=True, thresholds=0.5, name="recall")
        ]
    )

    # Load the trained weights
    model_path = os.path.join(hparams.output_dir, "final_model.keras")
    model.load_weights(model_path)
    print(f"Loaded model from: {model_path}")

    # Evaluate on test set
    results = model.evaluate(test_dataset, return_dict=True)
    print(
        f"AUC PR: {results['auc_pr'] * 100:.1f}%, Precision: {results['precision'] * 100:.1f}%, Recall: {results['recall'] * 100:.1f}%")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    test_model()
