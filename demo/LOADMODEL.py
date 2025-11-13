from model.model_forecastangle import MotionMamba as polar_angle_model
from model.model_concatangle import MotionMamba as conc_angle_model
import os
import json
import torch
import yaml
from easydict import EasyDict as edict
from typing import Any, IO


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def loadweightmodel():

    args1 = get_config("configs/h36m/first.yaml")
    args2 = get_config("configs/h36m/large.yaml")
    first = polar_angle_model(args1)
    second = conc_angle_model(args2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        first = torch.nn.DataParallel(first)
        second = torch.nn.DataParallel(second)
    first.to(device)
    second.to(device)

    first_checkpoint = torch.load("checkpoint/stage1/best_epoch.pth.tr", map_location=lambda storage, loc: storage)
    second_checkpoint = torch.load("checkpoint/stage2/base/best_epoch.pth.tr", map_location=lambda storage, loc: storage)
    first.load_state_dict(first_checkpoint['model'], strict=True)
    second.load_state_dict(second_checkpoint['model'], strict=True)

    return first, second

