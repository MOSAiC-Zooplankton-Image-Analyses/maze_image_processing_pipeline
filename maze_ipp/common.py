import sys
from typing import Any, ClassVar, Mapping
import numpy as np
from pydantic import BaseModel, model_validator


def convert_img_dtype(image, dtype: np.dtype):
    image = np.asarray(image)

    if dtype.kind == "f":
        if image.dtype.kind == "u":
            factor = np.array(1.0 / np.iinfo(image.dtype).max, dtype=dtype)
            return np.multiply(image, factor)

        if image.dtype.kind == "f":
            return np.asarray(image, dtype)

    raise ValueError(f"Can not convert {image.dtype} to {dtype}.")


class DefaultModel(BaseModel):
    __default_field__: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def parse_shortform(cls, data: Any):
        if not isinstance(data, Mapping):
            return {cls.__default_field__: data}
        return data


class TrueToDefaultsModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def parse_shortform(cls, data: Any):
        if data is True:
            return {}
        return data


def add_note(err: BaseException, msg: str) -> None:
    if sys.version_info < (3, 11):
        err.__notes__ = getattr(err, "__notes__", []) + [msg]
    else:
        err.add_note(msg)
