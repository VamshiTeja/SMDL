import numpy as np
import argparse
import subprocess
import os
import yaml
import cPickle as pickle

from lib.config import cfg, cfg_from_file


def main():
    """
    This function:
        - Creates a random class-list that will be used by all the methods,
        - Saves it to file,
        - Creates yaml files with run configurations,
        - Executes each of the runs.
    :return: None
    """

    # Retrieving the arguments
    parser = argparse.ArgumentParser(description="SMILe: SubModular Incremental Learning")
    parser.add_argument("--cfg", dest='cfg_file', default='./config/smile.yml', type=str, help="An optional config file"
                                                                                               " to be loaded")
    args = parser.parse_args()

    # Updating the configuration object
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # Creating the class list
    class_list = []
    classes = range(0, cfg.dataset.total_num_classes)
    for i in range(cfg.repeat_rounds):
        classes = np.random.permutation(classes)
        class_list.append(classes)

    # Saving the class list
    output_dir = './run_sandbox/'
    cfg.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/class_list.pkl', 'w') as f:
        pickle.dump(class_list, f)

    # Start Smiling :)
    gpus = [5, 6, 7]
    current_run_label = cfg['run_label']
    cfg['load_class_list_from_file'] = True

    # YAML for Random Run
    cfg['gpu_ids'] = str(gpus[1])
    cfg['run_label'] = 'random_' + current_run_label
    cfg['sampling_strategy'] = 'random'
    with open(output_dir + '/random.yml', 'w') as outfile:
        yaml.dump(dict(cfg), outfile, default_flow_style=False)

    # YAML for SubModular Run
    cfg['gpu_ids'] = str(gpus[0])
    cfg['run_label'] = 'submodular_' + current_run_label
    cfg['sampling_strategy'] = 'submodular'
    with open(output_dir + '/submodular.yml', 'w') as outfile:
        yaml.dump(dict(cfg), outfile, default_flow_style=False)

    # YAML for Full-Dataset Run
    cfg['gpu_ids'] = str(gpus[2])
    cfg['run_label'] = 'full_dataset_' + current_run_label
    cfg['use_all_exemplars'] = True
    with open(output_dir + '/full_dataset.yml', 'w') as outfile:
        yaml.dump(dict(cfg), outfile, default_flow_style=False)

    p1 = subprocess.Popen(['/raid/joseph/il/opy27/bin/python', 'smile.py','--cfg', './run_sandbox/full_dataset.yml'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['/raid/joseph/il/opy27/bin/python', 'smile.py','--cfg', './run_sandbox/random.yml'], stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['/raid/joseph/il/opy27/bin/python', 'smile.py','--cfg', './run_sandbox/submodular.yml'], stdout=subprocess.PIPE)

    print 'All processes started normally.'

    p1.communicate()
    p2.communicate()
    p3.communicate()

    print 'Finishing Launcher.'


if __name__ == '__main__':
    main()
