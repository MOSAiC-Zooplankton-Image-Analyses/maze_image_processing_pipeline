import sys
from typing import Mapping
import numpy as np


def convert_img_dtype(image, dtype: np.dtype):
    image = np.asarray(image)

    if dtype.kind == "f":
        if image.dtype.kind == "u":
            factor = np.array(1.0 / np.iinfo(image.dtype).max, dtype=dtype)
            return np.multiply(image, factor)

        if image.dtype.kind == "f":
            return np.asarray(image, dtype)

    raise ValueError(f"Can not convert {image.dtype} to {dtype}.")


def add_note(err: BaseException, msg: str) -> None:
    if sys.version_info < (3, 11):
        err.__notes__ = getattr(err, "__notes__", []) + [msg]
    else:
        err.add_note(msg)


def recursive_update(left, right):
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError(
            f"left and right must be Mappings, got {type(left)} / {type(right)}"
        )

    return {
        right_k: (
            recursive_update(left[right_k], right_v)
            if isinstance(right_v, Mapping) and isinstance(left.get(right_k), Mapping)
            else right_v
        )
        for right_k, right_v in right.items()
    }
