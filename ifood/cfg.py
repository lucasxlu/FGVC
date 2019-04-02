from collections import OrderedDict

cfg = OrderedDict()

cfg['image_dir'] = '/home/xulu/DataSet/iFood'
cfg['batch_size'] = 64
cfg['weight_decay'] = 1e-4
cfg['init_lr'] = 1e-2
cfg['decay_step'] = 50
cfg['epoch'] = 300