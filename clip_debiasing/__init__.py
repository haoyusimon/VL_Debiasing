from typing import Any

class Dotdict(dict):
    def __getattr__(self, __name: str) -> Any:
        return super().get(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        return super().__delitem__(__name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

# change the dataset directory 
DATA_PATH = "/data"
FAIRFACE_DATA_PATH = "/data/fairface/fairface"
PROMPT_DATA_PATH = "/data/prompt_templates.csv"
FACET_DATA_PATH = "/FACET"
UTKFACE_DATA_PATH = "/utkface"

from .models import *
from .measure_bias import measure_bias
