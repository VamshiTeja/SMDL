import numpy as np
from easydict import EasyDict as edict

root = edict()
cfg = root

root.run_label = 'smile'
root.gpu_ids = '0'

root.seed = 99

root.repeat_rounds = 1
root.class_per_episode = 100
root.use_all_exemplars = False
root.sampling_strategy = 'random'   # Can be either: 'random', 'submodular'
root.load_class_list_from_file = False

root.model = 'ResNet32'

root.epochs = 2

root.batch_size = 512
root.batch_size_test = 100

root.learning_rate = 0.1
root.momentum = 0.9
root.weight_decay = 1e-4

root.timestamp = 'placeholder'  # Will be updated at runtime
root.output_dir = 'placeholder'

root.use_custom_batch_selection = True
root.override_submodular_sampling = False
root.num_of_partitions = 10

# Dataset Details
root.dataset = edict()
root.dataset.name = 'CIFAR'
root.dataset.total_num_classes = 100
root.dataset.memory_budget = 2000

root.ltl_log_ep = 5                  # log(1/eps) : "Lazier Than Lazy Greedy, Mirzasoleiman et al. AAAI 2015"

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, root)

