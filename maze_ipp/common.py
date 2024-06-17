import sys
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
