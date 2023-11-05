import argparse, pprint
from datetime import datetime
from torch import optim
import torch.nn as nn

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--subject_num', type=int, default=4)
    parser.add_argument('--label_type', type=str, default='A')

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=6)

    parser.add_argument('--cls_weight', type=float, default=1)
    parser.add_argument('--diff_weight', type=float, default=0.1)
    parser.add_argument('--sim_weight', type=float, default=0.002)
    parser.add_argument('--recon_weight', type=float, default=0.04)

    parser.add_argument('--subject_recon_weight', type=float, default=0.05)
    parser.add_argument('--subject_diff_weight', type=float, default=0.1)

    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='elu')

    # Model
    parser.add_argument('--model', type=str,
                        default='MISA', help='one of {MISA, }')

    # Data
    parser.add_argument('--data', type=str, default='deap')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print(kwargs.data)

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
