import contextlib
import datetime
import glob
import gzip
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import parse
import skimage.color
import skimage.exposure
import skimage.filters
import skimage.morphology
import yaml
from morphocut import Pipeline
from morphocut.batch import BatchedPipeline
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.core import (
    Call,
    Node,
    Output,
    ReturnOutputs,
    Stream,
    Variable,
    closing_if_closable,
)
from morphocut.file import Glob
from morphocut.image import ExtractROI, FindRegions, ImageProperties, ImageReader
from morphocut.pipelines import DataParallelPipeline, MergeNodesPipeline
from morphocut.stream import Filter, Progress, Slice, StreamBuffer, Unpack
from morphocut.tiles import TiledPipeline
from morphocut.torch import PyTorch
from morphocut.utils import StreamEstimator, stream_groupby
from skimage.feature.orb import ORB
import skimage.measure
from skimage.measure._regionprops import RegionProperties

import loki
import segmenter
from maze_dl.merge_labels import merge_labels


def find_data_roots(project_root):
    print("Detecting project folders...")
    for root, dirs, _ in os.walk(project_root):
        if "Pictures" in dirs and "Telemetrie" in dirs:
            yield root
            # Do not descend further
            dirs[:] = []


LOG2META = {
    "sample_date": "DATE",
    "sample_time": "TIME",
    "acq_instrument_name": "DEVICE",
    "acq_instrument_serial": "LOKI",
    "sample_cruise": "CRUISE",
    "sample_station": "STATION",
    "sample_station_no": "STATION_NR",
    "sample_haul": "HAUL",
    "sample_user": "USER",
    "sample_vessel": "SHIP",
    "sample_gps_src": "GPS_SRC",
    "sample_latitude": "FIX_LAT",
    "sample_longitude": "FIX_LON",
}


def read_log(data_root: str, meta):
    # Find singular log filename
    (log_fn,) = glob.glob(os.path.join(data_root, "Log", "LOKI*.log"))

    log = loki.read_log(log_fn)

    # Return merge of existing and new meta
    return {**meta, **{ke: log[kl] for ke, kl in LOG2META.items()}}


TMD2META = {
    "object_longitude": "GPS_LON",
    "object_latitude": "GPS_LAT",
    "object_pressure": "PRESS",
    "object_temperature": "TEMP",  # or OXY_TEMP?
    "object_oxygen_concentration": "OXY_CON",
    "object_oxygen_saturation": "OXY_SAT",
    "object_temperature_oxsens": "OXY_TEMP",
    "object_conductivity": "COND_COND",
    # "COND_TEMP",
    "object_salinity": "COND_SALY",
    # "COND_DENS",
    # "COND_SSPEED"
    # "FLOUR_1",
    # "FLOUR_CR",
    # "FLOUR_CV",
    # "FLOUR_TR",
    # "FLOUR_TD",
    # "ROLL",
    # "PITCH",
    # "NICK",
    # "LOKI_REC",
    # "LOKI_PIC",
    # "LOKI_FRAME",
    # "CAM_STAT",
    # "HOUSE_STAT",
    # "HOUSE_T1",
    # "HOUSE_T2",
    # "HOUSE_VOLT",
}

tmd_fn_pat = "{:04d}{:02d}{:02d} {:02d}{:02d}{:02d}"
telemetry_fn_parser = parse.compile(tmd_fn_pat)


def parse_telemetry_fn(tmd_fn):
    tmd_basename = os.path.basename(tmd_fn)
    r: parse.Result = telemetry_fn_parser.search(tmd_basename)  # type: ignore
    if r is None:
        raise ValueError(f"Could not parse telemetry filename: {tmd_basename}")

    return datetime.datetime(*r)


def _read_tmd(tmd_fn, ignore_errors=False):
    dt = parse_telemetry_fn(tmd_fn)

    try:
        tmd = loki.read_tmd(tmd_fn)
    except:
        print(f"Error reading {tmd_fn}")
        if not ignore_errors:
            raise
        return dt, {}

    return dt, {k_et: tmd[k_loki] for k_et, k_loki in TMD2META.items() if k_loki in tmd}


def _read_dat(dat_fn, ignore_errors=False):
    dt = parse_telemetry_fn(dat_fn)

    try:
        dat = loki.read_dat(dat_fn)
    except Exception as exc:
        print(f"Error reading {dat_fn}")
        if ignore_errors:
            print(exc)
        else:
            raise
        return dt, {}

    return dt, {k_et: dat[k_loki] for k_et, k_loki in TMD2META.items() if k_loki in dat}


def read_all_telemetry(data_root: str, ignore_errors=False):
    print("Reading telemetry...")

    tmd_pat = os.path.join(data_root, "Telemetrie", "*.tmd")
    telemetry = pd.DataFrame.from_dict(
        dict(
            _read_tmd(tmd_fm, ignore_errors=ignore_errors)
            for tmd_fm in glob.iglob(tmd_pat)
        ),
        orient="index",
    )

    # Also read .dat files
    dat_pat = os.path.join(data_root, "Telemetrie", "*.dat")
    dat = pd.DataFrame.from_dict(
        dict(
            _read_dat(dat_fm, ignore_errors=ignore_errors)
            for dat_fm in glob.iglob(dat_pat)
        ),
        orient="index",
    )
    dat = dat[~dat.index.isin(telemetry.index)]
    if not dat.empty:
        telemetry = pd.concat((telemetry, dat))

    return telemetry.sort_index()


def merge_telemetry(meta: Dict, telemetry: pd.DataFrame):
    # Construct tmd_fn and extract date
    tmd_fn = "{object_date} {object_time}.tmd".format_map(meta)
    dt = parse_telemetry_fn(tmd_fn)

    (idx,) = telemetry.index.get_indexer([dt], method="nearest")
    return {**meta, **telemetry.iloc[idx].to_dict()}


REQUIRED_SAMPLE_META = [
    "sample_bottomdepth",
    "sample_region",
    "sample_detail_location",
    "sample_vessel",
    "sample_latitude",
    "sample_longitude",
    "sample_station",
    "sample_haul",
    "acq_instrument",
]


class MissingMetaError(Exception):
    pass


def update_and_validate_sample_meta(data_root: str, meta: Dict):
    """Read and validate metadata.

    Make sure that all required fields are included and generate sample_id and acq_id.
    """

    meta_fn = os.path.join(data_root, "meta.yaml")

    # Make a copy
    meta = dict(meta)

    # Update with additional metadata
    if os.path.isfile(meta_fn):
        with open(meta_fn) as f:
            meta.update(yaml.unsafe_load(f))

    missing_keys = set(REQUIRED_SAMPLE_META) - set(meta.keys())
    if missing_keys:
        missing_keys_str = ", ".join(sorted(missing_keys))

        Path(meta_fn).touch()

        raise MissingMetaError(
            f"The following fields are missing: {missing_keys_str}.\nSupply them in {meta_fn}"
        )

    meta["sample_id"] = "{sample_station}_{sample_haul}".format_map(meta)
    meta["acq_id"] = "{acq_instrument}_{sample_id}".format_map(meta)

    return meta


objid_pattern = "{object_date} {object_time}  {object_milliseconds}  {object_sequence:06d} {object_posx:04d} {object_posy:04d}"
objid_parser = parse.compile(objid_pattern)


def parse_object_id(object_id, meta):
    result = objid_parser.parse(object_id)
    if result is None:
        raise ValueError(f"Can not parse object ID: {object_id}")

    object_frame_id = "{object_date} {object_time}  {object_milliseconds}".format_map(
        result.named
    )

    return {
        **meta,
        "object_id": object_id,
        "object_frame_id": object_frame_id,
        **result.named,
    }


def stretch_contrast(image, low, high):
    low_, high_ = np.quantile(image, (low, high))
    return skimage.exposure.rescale_intensity(image, (low_, high_))


def build_stored_segmentation(stored_config, image, meta):
    pickle_fn = stored_config["pickle_fn"]

    with gzip.open(pickle_fn, "rb") as f:
        segmenter_: segmenter.Segmenter = pickle.load(f)
    segmenter_.configure(n_jobs=1, verbose=0)

    def _predict_scores(image, segmenter_):
        features = segmenter_.extract_features(image)
        mask = segmenter_.preselect(image)
        scores = segmenter_.predict_pixels(features, mask)

        return scores

    scores = Call(_predict_scores, image, segmenter_)

    stitched = Stitch(
        image,
        scores,
        groupby=meta["object_frame_id"],
        x=meta["object_posx"],
        y=meta["object_posy"],
        skip_single=stored_config["skip_single"],
    )

    image_large, scores_large = stitched.unpack(2)

    labels_large = Call(segmenter_.postprocess, scores_large, image_large)

    if stored_config["full_frame_archive_fn"] is not None:
        segment_image = Call(
            lambda labels, image: skimage.util.img_as_ubyte(
                skimage.color.label2rgb(labels, image, bg_label=0, bg_color=None)
            ),
            labels_large,
            image_large,
        )

        score_image = Call(
            skimage.util.img_as_ubyte,
            scores_large,
        )

        full_frame_archive_fn = Call(
            str.format_map, stored_config["full_frame_archive_fn"], meta
        )

        #  Store stitched result
        EcotaxaWriter(
            full_frame_archive_fn,
            [
                # Input image
                ("img/" + meta["object_frame_id"] + ".png", image_large),
                # Image overlayed with labeled segments
                ("overlay/" + meta["object_frame_id"] + ".png", segment_image),
                ("score/" + meta["object_frame_id"] + ".png", score_image),
                # # Mask (255=foreground)
                # (
                #     "mask/" + meta["object_frame_id"] + ".png",
                #     Call(lambda labels: (labels > 0).astype("uint8")*255, labels_large),
                # ),
            ],
        )

    # Extract individual objects
    region = FindRegions(labels_large, image_large, padding=75)

    image = ExtractROI(image_large, region)

    def recalc_metadata(region: RegionProperties, meta):
        meta = meta.copy()

        (y0, x0, x1, y1) = region.bbox

        meta["object_posx"] = x0
        meta["object_posy"] = y0
        meta["object_sequence"] = region.label
        meta["object_width"] = x1 - x0
        meta["object_height"] = y1 - y0

        meta["object_id"] = objid_pattern.format_map(meta)

        return meta

    # Re-calculate metadata (object_id, posx, posy)
    meta = Call(recalc_metadata, region, meta)

    # Re-calculate features
    meta = CalculateZooProcessFeatures(region, meta, prefix="object_")
    return image, meta


def assert_compatible_shape(label, image):
    try:
        assert (
            image.shape[: label.ndim] == label.shape
        ), f"{label.shape} vs. {image.shape}"
    except:
        print(f"{label.shape} vs. {image.shape}")
        raise


def convert_img_dtype(image, dtype: np.dtype):
    image = np.asarray(image)

    if dtype.kind == "f":
        if image.dtype.kind == "u":
            factor = np.array(1.0 / np.iinfo(image.dtype).max, dtype=dtype)
            return np.multiply(image, factor)

        if image.dtype.kind == "f":
            return np.asarray(image, dtype)

    raise ValueError(f"Can not convert {image.dtype} to {dtype}.")


def build_segmentation_postprocessing(config: Mapping, foreground_pred):
    with contextlib.ExitStack() as exit_stack:
        if config["n_threads"] > 1:
            # Post-processing task is CPU-bound
            exit_stack.enter_context(DataParallelPipeline(executor=config["n_threads"]))

        # Convert to bool
        foreground_pred = Call(np.asarray, foreground_pred, dtype=bool)

        # Opening (remove small details)
        if config["opening_radius"] > 0:
            foreground_pred = Call(
                skimage.morphology.binary_opening,
                foreground_pred,
                skimage.morphology.disk(config["opening_radius"]),
            )

        # Closing (close small gaps)
        if config["closing_radius"] > 0:
            foreground_pred = Call(
                skimage.morphology.binary_closing,
                foreground_pred,
                skimage.morphology.disk(config["closing_radius"]),
            )

        # Label the image
        labels = Call(
            skimage.measure.label,
            foreground_pred,
        )

        # Remove objects below area threshold
        if config["min_area"] > 0:
            labels = Call(
                skimage.morphology.remove_small_objects,
                labels,
                min_size=config["min_area"],
                out=labels,
            )

        # Merge neighboring labels
        if config["merge_labels"] > 0:
            labels = Call(
                merge_labels,
                labels,
                max_distance=config["merge_labels"],
                labels_out=labels,
            )

    return foreground_pred, labels


def build_pytorch_segmentation(config: Mapping, image, meta):
    import torch
    import torch.jit
    import torchvision.transforms.functional as ttf

    if config["stitch"]:
        # Buffer to have enough ROIs immediately available for stitching
        StreamBuffer(16)

        # Stitch individual ROIs together to build a full-frame image
        image = Stitch(
            image,
            groupby=meta["object_frame_id"],
            x=meta["object_posx"],
            y=meta["object_posy"],
            skip_single=config["skip_single"],
        ).unpack(1)

    # Deep Learning Segmentation
    device = config["device"]
    dtype = config["dtype"]

    if config.get("model_fn"):
        model = torch.load(config.get("model_fn"), map_location=device)
    elif config.get("jit_model_fn"):
        model = torch.jit.load(config.get("jit_model_fn"), map_location=device)
    else:
        raise ValueError("No model fn")

    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module"

    # Convert model to the specified dtype
    torch_dtype = getattr(torch, dtype)
    np_dtype = np.dtype(dtype)
    model = model.to(torch_dtype)

    def pre_transform(img):
        """Ensure RGB image, convert to specified dtype and transpose."""
        if img.ndim == 2:
            img = skimage.color.gray2rgb(img)

        img = img.transpose((2, 0, 1))

        img = convert_img_dtype(img, np_dtype)

        return torch.from_numpy(img).contiguous()

    with TiledPipeline((1024, 1024), image, tile_stride=(896, 896)):
        # # Skip empty tiles
        # Filter(lambda obj: (obj[image] > 0).any())
        # TODO: Rework Filter
        # Filter(lambda image: (image > 0).any(), image)
        Filter(Call(lambda image: (image > 0).any(), image))

        with contextlib.ExitStack() as exit_stack:
            if config["batch_size"]:
                exit_stack.enter_context(BatchedPipeline(config["batch_size"]))

            if config["n_threads"] > 1:
                exit_stack.enter_context(
                    DataParallelPipeline(executor=config["n_threads"])
                )

            foreground_pred = PyTorch(
                model,
                image,
                device=device,
                output_key=0,
                pre_transform=pre_transform,
                pin_memory=True,
                autocast=config["autocast"],
            )

            # # Dummy for speed optimization
            # foreground_pred = Call(
            #     lambda image: np.zeros(image.shape[:2], dtype=bool),
            #     image,
            # )

    # Postprocessing
    foreground_pred, labels = build_segmentation_postprocessing(
        config["postprocess"], foreground_pred
    )

    # DEBUG: Store segmentation output
    if config["full_frame_archive_fn"] is not None:
        Call(assert_compatible_shape, labels, image)
        segment_image = Call(
            lambda labels, image: skimage.util.img_as_ubyte(
                skimage.color.label2rgb(labels, image, bg_label=0, bg_color=None)
            ),
            labels,
            image,
        )

        score_image = Call(
            skimage.util.img_as_ubyte,
            foreground_pred,
        )

        full_frame_archive_fn = Call(
            str.format_map, config["full_frame_archive_fn"], meta
        )

        #  Store stitched result
        EcotaxaWriter(
            full_frame_archive_fn,
            [
                # Input image
                ("img/" + meta["object_frame_id"] + ".png", image),
                # Image overlayed with labeled segments
                ("overlay/" + meta["object_frame_id"] + ".png", segment_image),
                ("score/" + meta["object_frame_id"] + ".png", score_image),
            ],
        )

        StreamBuffer(2)

    # Extract individual objects
    region = FindRegions(labels, image, padding=75)

    image = ExtractROI(image, region)

    def recalc_metadata(region: RegionProperties, meta):
        meta = meta.copy()

        (y0, x0, x1, y1) = region.bbox

        meta["object_posx"] = x0
        meta["object_posy"] = y0
        meta["object_sequence"] = region.label
        meta["object_width"] = x1 - x0
        meta["object_height"] = y1 - y0

        meta["object_id"] = objid_pattern.format_map(meta)

        return meta

    # Re-calculate metadata (object_id, posx, posy)
    meta = Call(recalc_metadata, region, meta)

    # Re-calculate features
    meta = CalculateZooProcessFeatures(region, meta, prefix="object_")
    return image, meta


def build_segmentation(segmentation_config, image, meta) -> Tuple[Variable, Variable]:
    if segmentation_config is None:
        return image, meta

    if "threshold" in segmentation_config:
        threshold_config = segmentation_config["threshold"]
        mask = image > threshold_config["threshold"]

        Filter(lambda obj: obj[mask].any())

        props = ImageProperties(mask, image)
        meta = CalculateZooProcessFeatures(props, meta, prefix="object_")
        return image, meta

    if "stored" in segmentation_config:
        return build_stored_segmentation(segmentation_config["stored"], image, meta)

    if "pytorch" in segmentation_config:
        return build_pytorch_segmentation(segmentation_config["pytorch"], image, meta)

    raise ValueError(f"Unknown segmentation config: {segmentation_config}")


def detector_extractor(img):
    # img = skimage.transform.rescale(img, 0.75, preserve_range=True)
    img = skimage.filters.gaussian(img, 0.5, preserve_range=True)

    detector_extractor = ORB(
        downscale=1.2,
        fast_n=12,
        fast_threshold=0.84,
        harris_k=0.02,
        n_keypoints=100,
        n_scales=8,
    )
    detector_extractor.detect_and_extract(img)

    return detector_extractor.keypoints, detector_extractor.descriptors


def calc_overlap(xy0, wh0, xy1, wh1):
    """overlap_xy OR (keypoint_score AND overlap_y)"""

    l0 = xy0[0]
    r0 = xy0[0] + wh0[0]

    l1 = xy1[0]
    r1 = xy1[0] + wh1[0]

    t0 = xy0[1]
    b0 = xy0[1] + wh0[1]

    t1 = xy1[1]
    b1 = xy1[1] + wh1[1]

    w0, h0 = wh0
    w1, h1 = wh0

    intersect_x = max(0, min(r0, r1) - max(l0, l1))
    intersect_y = max(0, min(b0, b1) - max(t0, t1))

    union_x = max(1, max(r0, r1) - min(l0, l1))
    union_y = max(1, max(b0, b1) - min(t0, t1))

    overlap_x = intersect_x / union_x
    overlap_y = intersect_y / union_y

    intersect_xy = intersect_x * intersect_y
    overlap_xy = intersect_xy / (w0 * h0 + w1 * h1 - intersect_xy)

    return overlap_x, overlap_y, overlap_xy


def score_fn_simple(meta0, meta1):
    xy0 = meta0["object_posx"], meta0["object_posy"]
    xy1 = meta1["object_posx"], meta1["object_posy"]
    wh0 = meta0["object_width"], meta0["object_height"]
    wh1 = meta1["object_width"], meta1["object_height"]

    overlap_x, overlap_y, overlap_xy = calc_overlap(xy0, wh0, xy1, wh1)

    return overlap_xy


@ReturnOutputs
@Output("stitched")
class Stitch(Node):
    def __init__(self, *input, groupby, x, y, skip_single=False) -> None:
        super().__init__()

        self.input = input
        self.groupby = groupby
        self.x = x
        self.y = y
        self.skip_single = skip_single

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()

            for _, group in stream_groupby(stream, self.groupby):
                group = list(group)

                if self.skip_single and len(group) < 2:
                    continue

                with stream_estimator.consume(
                    group[0].n_remaining_hint, est_n_emit=1, n_consumed=len(group)
                ) as incoming_group:

                    data = pd.DataFrame(
                        [self.prepare_input(obj, ("input", "x", "y")) for obj in group],
                        columns=["input", "x", "y"],
                    )

                    data[["h", "w"]] = data["input"].apply(
                        lambda input: pd.Series(input[0].shape[:2])
                    )

                    data["xmax"] = data["x"] + data["w"]
                    data["ymax"] = data["y"] + data["h"]

                    first_input = data["input"].iloc[0]

                    target_shape = (data["ymax"].max(), data["xmax"].max())

                    # Create accumulator arrays
                    # Use float to avoid overflows
                    stitched = [
                        np.zeros(target_shape + inp.shape[2:], dtype=float)
                        for inp in first_input
                    ]

                    for row in data.itertuples():
                        for i, inp in enumerate(row.input):
                            sl = (
                                slice(row.y, row.y + row.h),
                                slice(row.x, row.x + row.w),
                            )
                            np.maximum(stitched[i][sl], inp, out=stitched[i][sl])

                    stitched = [
                        st.astype(inp.dtype) for st, inp in zip(stitched, first_input)
                    ]

                    yield self.prepare_output(
                        group[0].copy(),
                        stitched,
                        n_remaining_hint=incoming_group.emit(),
                    )


def build_pipeline(input, segmentation, output):
    """
    Args:
        input_path (str): Path to LOKI data (directory containing "Pictures" and "Telemetrie").

    Example:
        python pipeline.py /media/mschroeder/LOKI-Daten/LOKI/PS101/0059_PS101-59/
    """

    meta = input.get("meta", {})

    # Set defaults
    meta.setdefault("acq_instrument", "LOKI")

    # List directories that contain "Pictures" and "Telemetrie" folders
    data_roots = list(find_data_roots(input["path"]))

    print(f"Found {len(data_roots):d} input directories below {input['path']}")

    with Pipeline() as p:
        data_root = Unpack(data_roots)

        Progress(data_root)

        # Load LOG metadata
        meta = Call(read_log, data_root, meta)

        # Load *all* telemetry data
        telemetry = Call(read_all_telemetry, data_root, ignore_errors=True)

        # Load additional metadata
        meta = Call(update_and_validate_sample_meta, data_root, meta)

        target_archive_fn = Call(
            lambda meta: os.path.join(
                output["path"],
                "LOKI_{sample_station}_{sample_haul}.zip".format_map(meta),
            ),
            meta,
        )

        input_meta_archive_fn = Call(
            lambda meta: os.path.join(
                output["path"],
                "LOKI_{sample_station}_{sample_haul}_input_meta.zip".format_map(meta),
            ),
            meta,
        )

        Call(print, meta, "\n")

        # Find images
        pictures_pat = Call(os.path.join, data_root, "Pictures", "*", "*.bmp")
        # img_fn = Find(pictures_path, [".bmp"])
        img_fn = Glob(pictures_pat, sorted=True)

        slice_ = input.get("slice", None)
        if slice_ is not None:
            print(f"Only processing the first {slice_} objects.")
            Slice(slice_)

        Progress()

        # Extract object ID
        object_id = Call(
            lambda img_fn: os.path.splitext(os.path.basename(img_fn))[0], img_fn
        )

        # Parse object ID and update metadata
        meta = Call(parse_object_id, object_id, meta)

        # TODO: Skip frames where no annotated objects exist

        def error_handler(exc, img_fn):
            traceback.print_exc(file=sys.stderr)
            print("img_fn:", img_fn, file=sys.stderr)

        # Skip object if image can not be loaded
        with MergeNodesPipeline(on_error=error_handler, on_error_args=(img_fn,)):
            # Read image
            image = ImageReader(img_fn, "L")

        meta = Call(
            lambda image, meta: {
                **meta,
                "object_height": image.shape[0],
                "object_width": image.shape[1],
            },
            image,
            meta,
        )

        # Write input metadata
        EcotaxaWriter(input_meta_archive_fn, [], meta)

        # Merge telemetry
        meta = Call(merge_telemetry, meta, telemetry)

        image, meta = build_segmentation(segmentation, image, meta)

        # # Detect duplicates
        # dupset_id = DetectDuplicatesSimple(
        #     meta["object_frame_id"],
        #     meta["object_id"],
        #     score_fn=score_fn_simple,
        #     score_arg=meta,
        #     min_similarity=0.98,
        #     max_age=1,
        #     verbose=False,
        # )

        # # Only keep unique objects
        # Filter(dupset_id == meta["object_id"])

        # Filter(meta["object_mc_area"] > 500)

        StreamBuffer(8)

        target_image_fn = Call(
            output.get("image_fn", "{object_id}.jpg").format_map, meta
        )

        # # # Image enhancement: Stretch contrast
        # # image = Call(stretch_contrast, image, 0.01, 0.99)

        EcotaxaWriter(target_archive_fn, (target_image_fn, image), meta)

    p.run()


if __name__ == "__main__":
    import sys

    from schema import PipelineSchema

    if len(sys.argv) != 2:
        print("You need to supply a task file", file=sys.stderr)

    task_fn = sys.argv[1]

    sys.path.insert(0, os.path.realpath(os.curdir))

    os.chdir(os.path.dirname(task_fn))

    with open(task_fn) as f:
        config = PipelineSchema().load(yaml.safe_load(f))
        p = build_pipeline(**config)
