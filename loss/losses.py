from loss.cross_entropy import get_cross_entropy_loss
from loss.focal import get_focal_loss_multi_v1
from loss.sparse_categorical_cross_entropy import get_sparse_categorical_cross_entropy_loss


def get_loss_fn(**config):
  if config.get('loss') == 'entropy':
    return get_cross_entropy_loss(config)
  elif config.get('loss') == 'focal':
    return get_focal_loss_multi_v1(config)
  else:
    return get_sparse_categorical_cross_entropy_loss(config)

