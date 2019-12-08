import tensorflow as tf

from optimization.adamw import AdamWeightDecay
from optimization.warmup import WarmUp


def adamw_polynomial_decay_warmup(decay_steps, warmup_steps,
                                  learning_rate=2e-5, end_learning_rate=0.0,
                                  fp16=False):
  # lr
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=decay_steps,
    end_learning_rate=end_learning_rate,
  )

  learning_rate_fn = WarmUp(
    initial_learning_rate=learning_rate,
    decay_schedule_fn=learning_rate_fn,
    warmup_steps=warmup_steps,
  )

  # optimizer
  optimizer = AdamWeightDecay(
      learning_rate=learning_rate_fn,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=['layer_norm', 'bias'])

  if fp16:
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
      optimizer)

  return optimizer