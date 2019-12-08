import tensorflow as tf


def common_settings():
  assert tf.version.VERSION.startswith('2.')

  gpus = tf.config.experimental.list_physical_devices('gpu')
  if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

