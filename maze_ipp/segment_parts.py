import argparse
import contextlib
from morphocut.batch import BatchedPipeline
from morphocut.stream import Progress
from morphocut.tiles import TiledPipeline
import skimage.color
import skimage.util
import torch
from torchvision.transforms import Compose, ToTensor, Pad
import numpy as np

from morphocut.contrib.ecotaxa import EcotaxaObject, EcotaxaReader, EcotaxaWriter
from morphocut.core import Call, Pipeline, Variable
from morphocut.torch import PyTorch


class Crop:
    def __init__(self, left, top, right, bottom) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __call__(self, img: np.ndarray):
        return img[self.top : -self.bottom, self.left : -self.right]


class ClearMargins:
    def __init__(self, left=0, top=0, right=0, bottom=0, cval=0) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.cval = cval

    def __call__(self, img: np.ndarray):
        if self.left == 0 and self.top == 0 and self.right == 0 and self.bottom == 0:
            return img

        img = img.copy()
        if self.top > 0:
            img[: self.top] = self.cval
        if self.bottom > 0:
            img[-self.bottom :] = self.cval
        if self.left > 0:
            img[:, : self.left] = self.cval
        if self.right > 0:
            img[:, -self.right :] = self.cval

        return img


class PadTo:
    def __init__(self, min_size, cval=0) -> None:
        self.min_size = min_size
        self.cval = cval

    def __call__(self, img: np.ndarray):
        img_shape = np.array(img.shape[:2])
        pad_before = np.ceil((self.min_size - img_shape).clip(0)).astype(int)
        pad_after = (self.min_size - img_shape - pad_before).clip(0)
        pad_width = tuple(zip(pad_before, pad_after))

        if img.ndim == 3:
            pad_width = pad_width + ((0, 0),)

        img = np.pad(
            img,
            pad_width,
            mode="constant",
            constant_values=self.cval,
        )

        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_fn", required=True)
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--batch-size", dest="batch_size")
    parser.add_argument(
        "--clear-margins",
        dest="clear_margins",
        nargs=4,
        help="(left, top, right, bottom)",
        default=None,
        type=int,
    )
    parser.add_argument("input_archive_fn")
    parser.add_argument("output_archive_fn")
    args = parser.parse_args()

    model = torch.jit.load(args.model_fn, map_location=args.device)

    with Pipeline() as p:
        et_object: Variable[EcotaxaObject] = EcotaxaReader(args.input_archive_fn)

        # Get rank 0 image
        image = Call(lambda et_object: et_object.image, et_object)
        meta = Call(lambda et_object: et_object.meta, et_object)

        if args.clear_margins is not None:
            image = Call(ClearMargins(*args.clear_margins), image)

        with TiledPipeline(
            (1024, 1024), image, tile_stride=(896, 896)
        ), contextlib.ExitStack() as exit_stack:
            if args.batch_size:
                exit_stack.enter_context(BatchedPipeline(args.batch_size))

            # TODO: Unstack batch stuff
            labels = PyTorch(
                model,
                image,
                device=args.device,
                output_key=0,
                pre_transform=ToTensor(),
                post_transform=lambda labels: labels.cpu().numpy().transpose((1, 2, 0)),
                pin_memory=args.device.startswith("cuda"),
            )

        # TODO: Measure areas

        segment_image = Call(
            lambda labels, image: skimage.util.img_as_ubyte(
                skimage.color.label2rgb(labels, image, bg_label=0, bg_color=None)
            ),
            labels,
            image,
        )

        EcotaxaWriter(
            args.output_archive_fn,
            [
                (meta["object_id"] + "_overlay.png", segment_image),
            ],
            meta,
        )

        Progress()

    p.run()


if __name__ == "__main__":
    main()
