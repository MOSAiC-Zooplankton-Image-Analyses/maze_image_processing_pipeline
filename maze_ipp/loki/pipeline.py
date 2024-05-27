import contextlib
import datetime
import fnmatch
import glob
import itertools
import logging
import os
import pathlib
import sys
import warnings
from typing import Collection, Dict, Iterator, Mapping, Sequence, Tuple, Union

import exceptiongroup
import lokidata
import maze_ipp
import morphocut
import natsort as ns
import numpy as np
import pandas as pd
import parse
import skimage.color
import skimage.exposure
import skimage.filters
import skimage.measure
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
    RawOrVariable,
    ReturnOutputs,
    Stream,
    StreamObject,
    Variable,
    closing_if_closable,
)
from morphocut.image import ExtractROI, FindRegions, ImageProperties, ImageReader
from morphocut.pipelines import (
    AggregateErrorsPipeline,
    DataParallelPipeline,
    MergeNodesPipeline,
)
from morphocut.scalebar import DrawScalebar
from morphocut.stitch import Stitch
from morphocut.stream import Filter
from morphocut.stream import Progress as LiveProgress
from morphocut.stream import Slice, StreamBuffer, Unpack
from morphocut.tiles import TiledPipeline
from morphocut.torch import PyTorch
from morphocut.utils import StreamEstimator
from omni_archive import Archive
from pathlib_abc import PathBase, PurePathBase
from pyecotaxa.archive import read_tsv
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from skimage.feature.orb import ORB
from skimage.measure._regionprops import RegionProperties
from tqdm import tqdm

from ..merge_labels import merge_labels
from .config_schema import (
    DetectDuplicatesModelOrFalse,
    EcoTaxaOutputConfig,
    LokiInputConfig,
    PytorchSegmentationConfig,
    SegmentationConfig,
    SegmentationPipelineConfig,
    SegmentationPostprocessingConfig,
    ThresholdSegmentationConfig,
)
from .zoomie2 import DetectDuplicatesSimple

logging.captureWarnings(True)
logger = logging.getLogger(__name__)

if sys.stdout.isatty():
    Progress = LiveProgress
else:
    from functools import partial

    from ..log_progress import LogProgress

    # Set log_interval to 3min
    Progress = partial(LogProgress, log_interval=3 * 60)


class FilterEval(Node):
    """
    Filter the stream using a boolean expression.
    """

    def __init__(self, expression: str, data: RawOrVariable[Mapping]):
        super().__init__()

        self._compiled_expr = compile(expression, "<string>", "eval")

        self.data = data

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()
            for obj in stream:
                with stream_estimator.consume(obj.n_remaining_hint) as incoming:
                    try:
                        data: Mapping = self.prepare_input(obj, "data")  # type: ignore

                        if not eval(self._compiled_expr, None, data):
                            continue
                    except Exception as exc:
                        raise type(exc)(*exc.args, f"{self}")

                    obj.n_remaining_hint = incoming.emit()
                    yield obj


def read_log_and_yaml_meta(data_root: PathBase, meta: Mapping):
    # Find singular log filename
    (log_fn,) = (data_root / "Log").glob("LOKI*.log")
    meta_fn = data_root / "meta.yaml"

    # Return combination of initial meta, LOKI log metadata and yaml meta
    return {
        **meta,
        **lokidata.read_log(log_fn, remap_fields=lokidata.LOG_FIELDS_TO_ECOTAXA),
        **lokidata.read_yaml(meta_fn),
    }


TMD2META = {
    "object_lon": "GPS_LON",
    "object_lat": "GPS_LAT",
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


def parse_telemetry_fn(tmd_fn: PurePathBase):
    r: parse.Result = telemetry_fn_parser.search(tmd_fn.name)  # type: ignore
    if r is None:
        raise ValueError(f"Could not parse telemetry filename: {tmd_fn.name}")

    return datetime.datetime(*r)


def _read_tmd(tmd_fn: PathBase, ignore_errors=False):
    dt = parse_telemetry_fn(tmd_fn)

    try:
        tmd = lokidata.read_tmd(tmd_fn)
    except:
        logger.error(f"Error reading {tmd_fn}", exc_info=True)
        if not ignore_errors:
            raise
        return dt, {}

    return dt, {k_et: tmd[k_loki] for k_et, k_loki in TMD2META.items() if k_loki in tmd}


def _read_dat(dat_fn: PathBase, ignore_errors=False):
    dt = parse_telemetry_fn(dat_fn)

    try:
        dat = lokidata.read_dat(dat_fn)
    except Exception:
        logger.error(f"Error reading {dat_fn}", exc_info=True)
        if not ignore_errors:
            raise
        return dt, {}

    return dt, {k_et: dat[k_loki] for k_et, k_loki in TMD2META.items() if k_loki in dat}


class Telemetry:
    def __init__(
        self,
        data_root: PathBase,
        ignore_errors=False,
        tolerance: Union[None, str, pd.Timedelta] = None,
    ):
        self.telemetry = self._read_all_telemetry(data_root, ignore_errors)

        median_timedelta = pd.Series(self.telemetry.index).diff().median()

        logger.info(
            f"Read telemetry for {data_root}. Median time delta is {median_timedelta}."
        )

        if isinstance(tolerance, str):
            tolerance = pd.Timedelta(tolerance)

        self.tolerance = tolerance

        self._not_found_dt = set()

    @staticmethod
    def _read_all_telemetry(data_root: PathBase, ignore_errors=False) -> pd.DataFrame:
        logger.info(f"Reading telemetry in {data_root}...")

        telemetry_path = data_root / "Telemetrie"

        tmd_fns = list(telemetry_path.glob("*.tmd"))
        tmd_names = set(tmd_fn.stem for tmd_fn in tmd_fns)

        logger.info(f"Found {len(tmd_fns)} *.tmd files")
        telemetry = pd.DataFrame.from_dict(
            dict(
                _read_tmd(tmd_fn, ignore_errors=ignore_errors)
                for tmd_fn in tqdm(tmd_fns, desc=f"{telemetry_path}/*.tmd")
            ),
            orient="index",
        )

        # Also read .dat files
        dat_fns = list(telemetry_path.glob("*.dat"))
        logger.info(f"Found {len(dat_fns)} *.dat files")
        dat = pd.DataFrame.from_dict(
            dict(
                _read_dat(dat_fn, ignore_errors=ignore_errors)
                for dat_fn in tqdm(dat_fns, desc=f"{telemetry_path}/*.dat")
                if dat_fn.stem not in tmd_names
            ),
            orient="index",
        )
        dat = dat[~dat.index.isin(telemetry.index)]
        if not dat.empty:
            telemetry = pd.concat((telemetry, dat))

        if telemetry.empty:
            files = list(p.name for p in itertools.islice(telemetry_path.iterdir(), 10))
            if len(files) == 10:
                files[-1] = "..."

            msg = f"{data_root}/{telemetry_path} contains no readable telemetry files, just {', '.join(files)}"
            if ignore_errors:
                logger.error(msg)
            else:
                raise ValueError(msg)

        telemetry.index = telemetry.index.astype("datetime64[ns]")

        return telemetry.sort_index()

    def merge_telemetry(
        self,
        meta: Dict,
    ):
        # Construct tmd_fn and extract date
        tmd_fn = "{object_date} {object_time}.tmd".format_map(meta)
        dt = parse_telemetry_fn(pathlib.PurePosixPath(tmd_fn))

        (idx,) = self.telemetry.index.get_indexer(
            [dt], method="nearest", tolerance=self.tolerance
        )

        # Missing values in the target are marked by -1.
        if idx == -1:
            if dt not in self._not_found_dt:
                logger.warn(f"No telemetry found for {dt}")
                self._not_found_dt.add(dt)

            return meta

        return {**meta, **self.telemetry.iloc[idx].to_dict()}


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


def update_and_validate_sample_meta(data_root: PurePathBase, meta: Dict):
    """
    Validate metadata.

    Make sure that all required fields are included and generate sample_id and acq_id.
    """

    missing_keys = set(REQUIRED_SAMPLE_META) - set(meta.keys())
    if missing_keys:
        missing_keys_str = ", ".join(sorted(missing_keys))

        meta_fn = data_root / "meta.yaml"
        raise MissingMetaError(
            f"The following fields are missing: {missing_keys_str}.\nSupply them in {meta_fn}"
        )

    meta["sample_id"] = "{sample_station}_{sample_haul}".format_map(meta)
    meta["acq_id"] = "{acq_instrument}_{sample_id}".format_map(meta)

    # These are per sample_id
    meta["process_datetime"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["process_id"] = "{acq_id}_{process_datetime}".format_map(meta)

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


# def rescale_intensity(image, q_low: Optional[float], q_high: Optional[float]):
#     low: float
#     high: float
#     if q_low is None:
#         if q_high is None:
#             return image

#         low = skimage.exposure.exposure.intensity_range(image, "dtype")[0]
#         high = np.quantile(image, q_high, method="median_unbiased")

#     else:  # q_low is not None
#         if q_high is None:
#             low = np.quantile(image, q_low, method="median_unbiased")
#             high = skimage.exposure.exposure.intensity_range(image, "dtype")[1]
#         else:  # q_high is not None
#             low, high = np.quantile(image, (q_low, q_high), method="median_unbiased")

#     return skimage.exposure.rescale_intensity(image, (low, high))


def rescale_max_intensity(image: np.ndarray):
    return skimage.exposure.rescale_intensity(image, (0, image.max()))


def assert_compatible_shape(label, image):
    try:
        assert (
            image.shape[: label.ndim] == label.shape
        ), f"{label.shape} vs. {image.shape}"
    except:
        logger.error(f"{label.shape} vs. {image.shape}", exc_info=True)
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


def build_segmentation_postprocessing(
    config: SegmentationPostprocessingConfig, foreground_pred
):
    with contextlib.ExitStack() as exit_stack:
        if config.n_threads > 1:
            # Post-processing task is CPU-bound
            exit_stack.enter_context(DataParallelPipeline(executor=config.n_threads))

        # Convert to bool
        foreground_pred = Call(np.asarray, foreground_pred, dtype=bool)

        # Opening (remove small details)
        if config.opening_radius > 0:
            foreground_pred = Call(
                skimage.morphology.binary_opening,
                foreground_pred,
                skimage.morphology.disk(
                    config.opening_radius,
                    decomposition="crosses",
                ),
            )

        # Closing (close small gaps)
        if config.closing_radius > 0:
            foreground_pred = Call(
                skimage.morphology.binary_closing,
                foreground_pred,
                skimage.morphology.disk(
                    config.closing_radius,
                    decomposition="crosses",
                ),
            )

        # Label the image
        labels = Call(
            skimage.measure.label,
            foreground_pred,
        )

        if config.clear_border:
            Call(
                lambda labels: skimage.segmentation.clear_border(labels, out=labels),
                labels,
            )

        # Remove objects below area threshold
        if config.min_area > 0:
            labels = Call(
                skimage.morphology.remove_small_objects,
                labels,
                min_size=config.min_area,
                out=labels,
            )

        # Merge neighboring labels
        if config.merge_segments_distance > 0:
            labels = Call(
                merge_labels,
                labels,
                max_distance=config.merge_segments_distance,
                labels_out=labels,
            )

    return foreground_pred, labels


def build_pytorch_segmentation(
    config: PytorchSegmentationConfig,
    target_dir,
    image: Variable[np.ndarray],
    meta,
    process_meta: Dict,
):
    import torch
    import torch.jit

    if config.stitch:
        # Buffer to have enough ROIs immediately available for stitching
        StreamBuffer(16)

        # Stitch individual ROIs together to build a full-frame image
        image = Stitch(
            image,
            groupby=meta["object_frame_id"],
            offset=(meta["object_posy"], meta["object_posx"]),
        )

        if config.stitch.skip_single:
            keep = Call(lambda image: image.n_regions > 1, image)
            Filter(keep)
    else:
        process_meta["process_segmentation_pytorch_stitch"] = False

    # Deep Learning Segmentation
    device = config.device
    dtype = config.dtype

    model = torch.jit.load(config.model_fn, map_location=device)

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
            if config.batch_size:
                exit_stack.enter_context(BatchedPipeline(config.batch_size))

            if config.n_threads > 1:
                exit_stack.enter_context(
                    DataParallelPipeline(executor=config.n_threads)
                )

            foreground_pred = PyTorch(
                model,
                image,
                device=device,
                output_key=0,
                pre_transform=pre_transform,
                pin_memory=device.startswith("cuda"),
                autocast=config.autocast,
            )

            # # Dummy for speed optimization
            # foreground_pred = Call(
            #     lambda image: np.zeros(image.shape[:2], dtype=bool),
            #     image,
            # )

    # Postprocessing
    if config.postprocess:
        foreground_pred, labels = build_segmentation_postprocessing(
            config.postprocess, foreground_pred
        )

    # DEBUG: Store segmentation output
    if config.full_frame_archive_fn is not None:
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
            lambda meta: os.path.join(
                target_dir, config.full_frame_archive_fn.format_map(meta)
            ),
            meta,
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
    region = FindRegions(
        labels,
        image,
        padding=config.padding,
        min_intensity=config.min_intensity,
    )

    image = ExtractROI(
        image,
        region,
        alpha=1 if config.apply_mask else 0,
        bg_color=config.background_color,
        keep_background=config.keep_background,
    )

    def recalc_metadata(region: RegionProperties, meta):
        meta = meta.copy()

        (y0, x0, x1, y1) = region.bbox

        meta["object_posx"] = x0
        meta["object_posy"] = y0
        meta["object_sequence"] = region.label
        meta["object_width"] = x1 - x0
        meta["object_height"] = y1 - y0

        meta["object_id"] = objid_pattern.format_map(meta)

        meta["object_frac_invalid"] = (region.image_intensity[region.image] == 0).mean()

        return meta

    # Re-calculate metadata (object_id, posx, posy)
    meta = Call(recalc_metadata, region, meta)

    # Re-calculate features
    meta = CalculateZooProcessFeatures(region, meta, prefix="object_")

    # # Restore regular ndarray image
    # image = Call(lambda image: image.data, image)

    return image, meta, region.image


def one_of(
    config: Mapping, keys: Sequence
) -> Union[Tuple[None, None], Tuple[str, Mapping]]:
    matches = [k for k in keys if k in config]

    if not matches:
        return (None, None)

    if len(matches) == 1:
        k = matches[0]
        return (k, config[k])

    raise ValueError(f"Multiple matches for {keys} in {config}")


def build_threshold_segmentation(config: ThresholdSegmentationConfig, image, meta):
    mask = image > config.threshold_brighter

    Filter(lambda obj: obj[mask].any())

    props = ImageProperties(mask, image)
    meta = CalculateZooProcessFeatures(props, meta, prefix="object_")

    return image, meta


def build_segmentation(
    config: SegmentationConfig, target_dir: str, image, meta, process_meta: Dict
) -> Tuple[Variable, Variable, RawOrVariable]:
    mask = None

    if config is None:
        return image, meta, mask

    if config.threshold is not None:
        image, meta = build_threshold_segmentation(config.threshold, image, meta)
    elif config.pytorch is not None:
        image, meta, mask = build_pytorch_segmentation(
            config.pytorch, target_dir, image, meta, process_meta
        )
    else:
        raise ValueError(f"Unknown segmentation config: {config}")

    if config.filter_expr is not None:
        logger.info(
            f"Filtering segmentation results by expression {config.filter_expr!r}"
        )
        FilterEval(config.filter_expr, meta)

    return image, meta, mask


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
    w1, h1 = wh1

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


def build_object_frame_id_filter(
    valid_frame_id_fn: str | None, meta: Variable[Mapping]
):
    if valid_frame_id_fn is None:
        return

    logger.info(f"Filtering object_frame_id from {valid_frame_id_fn}...")

    index = pd.Index(
        pd.read_csv(valid_frame_id_fn, header=None, index_col=False).squeeze()
    )

    Filter(lambda obj: obj[meta]["object_frame_id"] in index)


def _find_files_glob(pattern: str, ignore_patterns: Collection | None = None):
    for fn in glob.iglob(pattern):
        if ignore_patterns is not None and any(
            fnmatch.fnmatch(fn, pat) for pat in ignore_patterns
        ):
            logger.info(f"Ignoring {fn}.")
            continue

        yield fn


def build_input(
    input_config: LokiInputConfig,
    output_config: EcoTaxaOutputConfig,
    meta: Variable,
    process_meta: Dict,
):
    # Insert input_meta into meta
    default_meta = input_config.default_meta
    default_meta.setdefault("acq_instrument", "LOKI")
    meta = Call(lambda meta, default_meta: {**meta, **default_meta}, meta, default_meta)

    # Discover samples
    _sample_roots: Iterator[str] = _find_files_glob(
        input_config.path, input_config.ignore_patterns
    )

    if input_config.discover:

        def _find_sample_roots(sample_roots):
            for sample_root in sample_roots:
                yield from lokidata.find_data_roots(
                    input_config.path, input_config.ignore_patterns
                )

        _sample_roots = _find_sample_roots(_sample_roots)

    sample_roots = [Archive(fn) for fn in _sample_roots]

    logger.info(f"Found {len(sample_roots):d} input directories")

    sample_root = Unpack(ns.natsorted(sample_roots, alg=ns.PATH | ns.IGNORECASE))

    Progress(sample_root)

    # Load LOG metadata
    meta = Call(read_log_and_yaml_meta, sample_root, meta)

    # Preload all metadata (to avoid later validation errores)
    with AggregateErrorsPipeline():
        # Generate additional metadata and validate
        meta = Call(update_and_validate_sample_meta, sample_root, meta)

        if input_config.merge_telemetry is not None:
            telemetry_config = input_config.merge_telemetry
            logger.info(f"Merging telemetry: {telemetry_config}")

            # Load *all* telemetry data
            telemetry = Call(
                Telemetry,
                sample_root,
                ignore_errors=True,
                **telemetry_config.model_dump(),
            )
        else:
            telemetry = None

        # Unload archives so that they don't stack up in memory while aggregating metadata and telemetry
        Call(
            lambda data_root: (
                data_root.close() if hasattr(data_root, "close") else None
            ),
            sample_root,
        )

    os.makedirs(output_config.target_dir, exist_ok=True)

    target_archive_fn = Call(
        lambda meta: os.path.join(
            output_config.target_dir,
            "LOKI_{sample_station}_{sample_haul}.zip".format_map(meta),
        ),
        meta,
    )

    if output_config.skip_existing:

        def check_not_exists(target_archive_fn):
            if not os.path.exists(target_archive_fn):
                return True

            logger.info(f"Skipping target '{target_archive_fn}'.")
            return False

        Filter(Call(check_not_exists, target_archive_fn))

    if input_config.save_meta:
        input_meta_archive_fn = Call(
            lambda meta: os.path.join(
                output_config.target_dir,
                "LOKI_{sample_station}_{sample_haul}_input_meta.zip".format_map(meta),
            ),
            meta,
        )

    # Buffer per-data_root processing (telemetry, metadata)
    StreamBuffer(1)

    # Call(print, meta, "\n")

    # Find images
    picture_fns = Call(
        lambda data_root: sorted(
            path
            for path in (data_root / "Pictures").glob("*/*.*")
            if path.suffix in [".jpg", ".bmp", ".png"]
        ),
        sample_root,
    )

    Call(
        lambda picture_fns, data_root: logger.info(
            f"{len(picture_fns)} input images in {data_root}."
        ),
        picture_fns,
        sample_root,
    )

    picture_fn = Unpack(picture_fns)

    # Extract object ID
    object_id = Call(
        lambda picture_fn: picture_fn.stem,
        picture_fn,
    )

    # Parse object ID and update metadata
    meta = Call(parse_object_id, object_id, meta)

    # Skip frames where no annotated objects exist (if configured)
    build_object_frame_id_filter(input_config.valid_frame_id_fn, meta)

    # DEBUG: Slice
    if input_config.slice is not None:
        logger.warning(f"Only processing the first {input_config.slice} input objects.")
        Slice(input_config.slice)

    def error_handler(exc, img_fn):
        exceptiongroup.print_exc(file=sys.stderr)
        logger.error(f"Could not read image: {img_fn}", exc_info=True)

    # Skip object if image can not be loaded
    with MergeNodesPipeline(on_error=error_handler, on_error_args=(picture_fn,)):
        # Read image
        image = ImageReader(picture_fn, "L")

    meta = Call(
        lambda image, meta: {
            **meta,
            "object_height": image.shape[0],
            "object_width": image.shape[1],
            "object_bounding_box_area": image.shape[0] * image.shape[1],
        },
        image,
        meta,
    )

    # Filter based on expression
    if input_config.filter_expr is not None:
        logger.info(f"Filtering input by expression {input_config.filter_expr!r}")
        process_meta["process_input_filter"] = input_config.filter_expr
        FilterEval(input_config.filter_expr, meta)

    # Detect duplicates
    build_duplicate_detection(
        input_config.detect_duplicates, image, meta, "input", process_meta
    )

    if input_config.save_meta:
        # Write input metadata
        EcotaxaWriter(input_meta_archive_fn, [], meta)

    if telemetry is not None:
        # Merge telemetry
        meta = Call(Telemetry.merge_telemetry, telemetry, meta)

    return image, meta, target_archive_fn


def build_duplicate_detection(
    detect_duplicates_config: DetectDuplicatesModelOrFalse,
    image,
    meta,
    where: str,
    process_meta,
):
    if detect_duplicates_config is None:
        return

    logger.info(
        f"Duplicate detection ({where}) is active ({detect_duplicates_config})."
    )

    # Detect duplicates
    dupset_id = DetectDuplicatesSimple(
        meta["object_frame_id"],
        meta["object_id"],
        score_fn=score_fn_simple,
        score_arg=meta,
        min_similarity=detect_duplicates_config.min_similarity,
        max_age=detect_duplicates_config.max_age,
    )

    # Only keep the first object of each duplicate set
    def keep_duplicate(dupset_id, meta):
        if dupset_id == meta["object_id"]:
            return True

        logger.info(f"Dropping duplicate ({where}): {meta['object_id']} of {dupset_id}")
        return False

    Filter(Call(keep_duplicate, dupset_id, meta))


@ReturnOutputs
@Output("meta")
class MergeAnnotations(Node):
    def __init__(
        self,
        meta,
        *,
        annotations_fn: str,
        min_overlap=0.5,
        min_validated_overlap=0.8,
    ):
        super().__init__()

        self.meta = meta

        self.min_overlap = min_overlap
        self.min_validated_overlap = min_validated_overlap

        annotations = read_tsv(annotations_fn)

        missing_columns = {
            "object_width",
            "object_height",
            "object_posx",
            "object_posy",
            "object_frame_id",
        } - set(annotations.columns)
        if missing_columns:
            raise ValueError(
                f"The following columns are missing: {sorted(missing_columns)}"
            )

        self._annotations_by_frame_id = annotations.groupby("object_frame_id")
        self._annotation_columns = [
            c for c in annotations.columns if c.startswith("object_annotation")
        ]

    def transform(self, meta: dict) -> dict:
        object_frame_id = meta["object_frame_id"]

        try:
            frame_annotations: pd.DataFrame = self._annotations_by_frame_id.get_group(
                object_frame_id
            )
        except KeyError:
            return meta

        if not frame_annotations.size:
            return meta

        def _calc_overlap_xy(annotation: pd.Series) -> float:
            meta0 = annotation.to_dict()

            return score_fn_simple(meta0, meta)

        # Match object by overlap
        overlap_xy = frame_annotations.apply(_calc_overlap_xy, axis=1)
        best_idx = overlap_xy.sort_values(ascending=False).index[0]

        best_overlap = overlap_xy[best_idx]

        meta["object_annotation_merge_overlap"] = best_overlap

        if best_overlap > self.min_overlap:
            # A match was found
            annotation_meta = frame_annotations.loc[
                best_idx, self._annotation_columns
            ].to_dict()

            # Downgrade annotation status if match is imperfect
            if best_overlap < self.min_validated_overlap and annotation_meta[
                "object_annotation_status"
            ] in ("validated", "dubious"):
                annotation_meta["object_annotation_status"] = "predicted"

            annotation_meta["object_annotation_merge_src"] = frame_annotations.at[
                best_idx, "object_id"
            ]
        else:
            # No match was found
            annotation_meta = {k: "" for k in self._annotation_columns}

        meta.update(annotation_meta)

        return meta


def build_and_run_pipeline(pipeline_config: SegmentationPipelineConfig):
    """
    Args:
        input_path (str): Path to LOKI data (directory containing "Pictures" and "Telemetrie").

    Example:
        python pipeline.py /media/mschroeder/LOKI-Daten/LOKI/PS101/0059_PS101-59/
    """

    with Pipeline() as p:
        process_meta_var = Variable("process_meta", p)
        process_meta = {}

        process_meta["process_morphocut_version"] = morphocut.__version__
        process_meta["process_loki_pipeline_version"] = maze_ipp.__version__
        process_meta["process_skimage_version"] = skimage.__version__
        # TODO: More process metadata

        image, meta, target_archive_fn = build_input(
            pipeline_config.input,
            pipeline_config.output,
            process_meta_var,
            process_meta,
        )

        Progress("Input objects")

        image, meta, mask = build_segmentation(
            pipeline_config.segmentation,
            pipeline_config.output.target_dir,
            image,
            meta,
            process_meta,
        )

        StreamBuffer(8)

        # Post-Processing
        postprocess_config = pipeline_config.postprocess

        build_duplicate_detection(
            postprocess_config.detect_duplicates,
            image,
            meta,
            "output",
            process_meta,
        )

        process_meta["process_rescale_max_intensity"] = (
            postprocess_config.rescale_max_intensity
        )

        if postprocess_config.rescale_max_intensity:
            logger.info(f"Rescaling intensity of output images")
            # Image enhancement: Stretch contrast
            image = Call(rescale_max_intensity, image)

        if postprocess_config.scalebar is not None:
            scalebar_config = postprocess_config.scalebar
            process_meta["process_scalebar_px_per_mm"] = scalebar_config.px_per_mm

            logger.info("Drawing scalebar")
            image = DrawScalebar(
                image,
                length_in_unit=1,
                px_per_unit=scalebar_config.px_per_mm,
                unit="mm",
                fg_color=255,
                bg_color=0,
            )

        if postprocess_config.merge_annotations is not None:
            logger.info(f"Merging annotations: {postprocess_config.merge_annotations}")
            meta = MergeAnnotations(
                meta, **postprocess_config.merge_annotations.model_dump()
            )

        # DEBUG: Slice
        if postprocess_config.slice is not None:
            logger.warning(
                f"Only processing the first {postprocess_config.slice} output objects."
            )
            Slice(postprocess_config.slice)

        if postprocess_config.filter_expr is not None:
            logger.info(f"Filtering output by expression {output_config.filter_expr!r}")

            FilterEval(output_config.filter_expr, meta)

        ## Output
        output_config = pipeline_config.output

        target_image_fn = Call(output_config.image_fn.format_map, meta)
        output_images = [(target_image_fn, image)]
        if output_config.store_mask:
            target_mask_fn = Call(filename_suffix, target_image_fn, "_mask")
            output_images.append((target_mask_fn, mask))

        EcotaxaWriter(
            target_archive_fn,
            output_images,
            meta,
            store_types=output_config.type_header,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Only one label was provided to `remove_small_objects`",
            UserWarning,
        )

        # Inject pipeline metadata into the stream
        obj = StreamObject(n_remaining_hint=1)
        obj[process_meta_var] = process_meta
        p.run(iter([obj]))


def filename_suffix(fn: str, suffix: str):
    stem, ext = os.path.splitext(fn)
    return stem + suffix + ext


def main(task_fn: str):
    # Setup logging
    root_logger = logging.getLogger()
    stream_handler = RichHandler(highlighter=NullHighlighter())
    stream_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)

    sys.path.insert(0, os.path.realpath(os.curdir))

    os.chdir(os.path.dirname(task_fn))

    task_name = os.path.splitext(os.path.basename(task_fn))[0]
    task_fn_modified = datetime.datetime.fromtimestamp(os.stat(task_fn).st_mtime)

    log_fn = os.path.abspath(
        f'{task_name}-{datetime.datetime.now().isoformat(timespec="seconds")}.log'
    )
    print(f"Logging to {log_fn}.")
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Also capture exceptions

    def log_except_hook(*exc_info):
        root_logger.error("Unhandled exception", exc_info=exc_info)  # type: ignore

    sys.excepthook = log_except_hook

    # # Also capture py.warnings
    # warnings_logger = logging.getLogger("py.warnings")
    # warnings_logger.addHandler(file_handler)
    # warnings_logger.addHandler(stream_handler)
    # warnings_logger.setLevel(logging.INFO)

    root_logger.info(
        f"Loading pipeline config from {task_fn} ({task_fn_modified.isoformat(timespec='seconds')})"
    )

    # logging.getLogger("zoomie2").setLevel(logging.DEBUG)

    log_levels = {
        name: logging.getLevelName(logging.getLogger(name).getEffectiveLevel())
        for name in sorted(root_logger.manager.loggerDict)
    }
    root_logger.info(f"Log levels: {log_levels}")

    with open(task_fn) as f:
        try:
            segmentation_pipeline_cfg = SegmentationPipelineConfig.model_validate(
                yaml.safe_load(f)
            )
        except pydantic.ValidationError as exc:
            logger.error(str(exc))
        else:
            build_and_run_pipeline(segmentation_pipeline_cfg)  # type: ignore
