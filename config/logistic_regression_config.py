from easydict import EasyDict

from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.normalization

# training
cfg.weights_init_type = WeightsInitType.uniform
cfg.weights_init_kwargs = {'sigma': 1, 'epsilon': 1}

cfg.gamma = 0.01
cfg.gd_stopping_criteria = GDStoppingCriteria.difference_norm
cfg.nb_epoch = 100
cfg.min_difference_norm = 1e-2
