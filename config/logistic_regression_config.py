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
cfg.min_metric_difference = 1e-8
cfg.min_gradient_norm = 2

# how many iterations to do if criteria is not satisfied
cfg.nb_metric_value = 5
cfg.nb_difference_norm = 40
