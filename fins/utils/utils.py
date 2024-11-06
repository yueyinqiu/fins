import yaml
from easydict import EasyDict as ed
import typing


def load_config(file_path) -> typing.Any:
    with open(file_path, encoding="utf-8") as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    return ed(contents)
