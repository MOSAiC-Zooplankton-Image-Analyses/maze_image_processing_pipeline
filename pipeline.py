import datetime
import glob
import os
from pathlib import Path
from shutil import ReadError
from tabnanny import verbose
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import parse
import PIL.Image
import skimage.exposure
from skimage.feature.orb import ORB
from timer_cm import Timer
import yaml
from morphocut import Pipeline
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.core import Call, Variable
from morphocut.file import Glob
from morphocut.image import ImageProperties, ImageReader
from morphocut.stream import Filter, Progress, StreamBuffer, Unpack
import skimage.filters

import loki
from zoomie2 import DetectDuplicates, StoreDupsets


def find_data_roots(project_root):
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


def _read_tmd(tmd_fn):
    try:
        tmd = loki.read_tmd(tmd_fn)
    except:
        print(f"Error reading {tmd_fn}")
        raise

    dt = parse_telemetry_fn(tmd_fn)

    return dt, {ke: tmd[kl] for ke, kl in TMD2META.items()}


def _read_dat(dat_fn):
    try:
        dat = loki.read_dat(dat_fn)
    except:
        print(f"Error reading {dat_fn}")
        raise

    dt = parse_telemetry_fn(dat_fn)

    return dt, {ke: dat[kl] for ke, kl in TMD2META.items()}


def read_all_telemetry(data_root: str):
    print("Reading telemetry...")

    tmd_pat = os.path.join(data_root, "Telemetrie", "*.tmd")
    telemetry = pd.DataFrame.from_dict(
        dict(_read_tmd(tmd_fm) for tmd_fm in glob.iglob(tmd_pat)), orient="index"
    )

    # Also read .dat files
    dat_pat = os.path.join(data_root, "Telemetrie", "*.dat")
    dat = pd.DataFrame.from_dict(
        dict(_read_dat(dat_fm) for dat_fm in glob.iglob(dat_pat)), orient="index"
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

    return {**meta, "object_id": object_id, **result.named}


def stretch_contrast(image, low, high):
    low_, high_ = np.quantile(image, (low, high))
    return skimage.exposure.rescale_intensity(image, (low_, high_))


def build_segmentation(segmentation_config, image, meta) -> Tuple[Variable, Variable]:
    if segmentation_config is None:
        meta = Call(
            lambda image, meta: {
                **meta,
                "object_height": image.shape[0],
                "object_width": image.shape[1],
            },
            image,
            meta,
        )
        return image, meta

    if "threshold" in segmentation_config:
        threshold_config = segmentation_config["threshold"]
        mask = image > threshold_config["threshold"]

        Filter(lambda obj: obj[mask].any())

        props = ImageProperties(mask, image)
        meta = CalculateZooProcessFeatures(props, meta, prefix="object_")
        return image, meta


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


def score_fn(score, meta0, meta1):
    """overlap_xy OR (keypoint_score AND overlap_y)"""

    xy0 = meta0["object_posx"], meta0["object_posy"]
    xy1 = meta1["object_posx"], meta1["object_posy"]
    wh0 = meta0["object_width"], meta0["object_height"]
    wh1 = meta1["object_width"], meta1["object_height"]

    overlap_x, overlap_y, overlap_xy = calc_overlap(xy0, wh0, xy1, wh1)

    return max(overlap_xy, min(score, overlap_y))


def build_pipeline(input, output):
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

        # # Load *all* telemetry data
        # telemetry = Call(read_all_telemetry, data_root)

        # Load additional metadata
        meta = Call(update_and_validate_sample_meta, data_root, meta)

        target_archive_fn = Call(
            lambda meta: os.path.join(
                output["path"],
                "LOKI_{sample_station}_{sample_haul}.zip".format_map(meta),
            ),
            meta,
        )

        duplicate_dir = Call(
            lambda meta: os.path.join(
                output["path"],
                f"{datetime.datetime.now():%y-%m-%d}-"
                + "LOKI_{sample_station}_{sample_haul}".format_map(meta),
                "duplicates",
            ),
            meta,
        )

        Call(print, meta, "\n")

        # Find images
        pictures_pat = Call(os.path.join, data_root, "Pictures", "*", "*.bmp")
        # img_fn = Find(pictures_path, [".bmp"])
        img_fn = Glob(pictures_pat, sorted=True)

        # Extract object ID
        object_id = Call(
            lambda img_fn: os.path.splitext(os.path.basename(img_fn))[0], img_fn
        )

        # Parse object ID and update metadata
        meta = Call(parse_object_id, object_id, meta)

        # # Merge telemetry
        # meta = Call(merge_telemetry, meta, telemetry)

        # Read image
        image = ImageReader(img_fn)

        image, meta = build_segmentation(input.get("segmentation"), image, meta)

        # Filter(meta["object_mc_area"] > 500)

        frame_id = Call(
            lambda meta: "{object_date} {object_time}  {object_milliseconds}".format_map(
                meta
            ),
            meta,
        )

        StreamBuffer(8)

        # Detect Duplicates (ZOOMIE2)

        dupset_id = DetectDuplicates(
            object_id,
            image,
            frame_id,
            score_fn=score_fn,
            score_arg=meta,
            detector_extractor=detector_extractor,
            verbose=True,
            min_similarity=0.5,
            n_workers=8,
            # n_workers=1,
            pre_score_thr=0.75,
            max_age=24,
        )

        StoreDupsets(
            object_id, dupset_id, image, frame_id, duplicate_dir, save_singletons=True
        )

        # # if segmentation == "threshold":
        # #     # Simple segmentation
        # #     mask = ...
        # #     props = ImageProperties(mask, image)
        # #     meta = CalculateZooProcessFeatures(props, prefix="object_mc_")
        # # else:
        # #     # Assemble and resegment
        # #     image, meta = Resegment()

        # target_image_fn = Call(
        #     output.get("image_fn", "{object_id}.jpg").format_map, meta
        # )

        # # # Image enhancement: Stretch contrast
        # # image = Call(stretch_contrast, image, 0.01, 0.99)

        # EcotaxaWriter(target_archive_fn, (target_image_fn, image), meta)

        Progress()

    p.run()


if __name__ == "__main__":
    import sys

    from schema import PipelineSchema

    with open(sys.argv[1]) as f:
        config = PipelineSchema().load(yaml.safe_load(f))
        p = build_pipeline(**config)
