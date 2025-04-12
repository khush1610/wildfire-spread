
"""Custom loss functions for TensorFlow."""

import tensorflow as tf


def weighted_cross_entropy_with_logits_with_masked_class(
    pos_weight = 3.0):
  """Wrapper function for masked weighted cross-entropy with logits.

  This loss function ignores the classes with negative class id.

  Args:
    pos_weight: A coefficient to use on the positive examples.

  Returns:
    A weighted cross-entropy with logits loss function that ignores classes
    with negative class id.
    with negative class id.
  """

  def masked_weighted_cross_entropy_with_logits(y_true,
                                                logits):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    loss = tf.math.reduce_mean(mask * tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=logits, pos_weight=pos_weight))
    return loss

  return masked_weighted_cross_entropy_with_logits