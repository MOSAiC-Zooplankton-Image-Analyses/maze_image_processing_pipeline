import datetime
import glob
import os
from typing import Dict

import numpy as np
import pandas as pd
import parse
import skimage.exposure
import yaml
from morphocut import Pipeline
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.core import Call
from morphocut.file import Glob
from morphocut.image import ImageProperties, ImageReader
from morphocut.stream import Progress, Unpack

import loki


def find_data_roots(project_root):
    for root, dirs, _ in os.walk(project_root):
        if "Pictures" in dirs and "Telemetrie" in dirs:
            yield root
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
    "sample_longitude": "FIX_LONG",
}


def read_log(data_root: str, meta):
    # Find singular log filename
    (log_fn,) = glob.glob(os.path.join(data_root, "Log", "LOKI*.log"))

    log = loki.read_log(log_fn)

    # Return merge of existing and new meta
    return {**meta, **{ke: log[kl] for ke, kl in LOG2META.items()}}


TMD2META = {
    "object_longitude": "GPS_LONG",
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

tmd_fn_pat = "{:04d}{:02d}{:02d} {:02d}{:02d}{:02d}.tmd"
tmd_fn_parser = parse.compile(tmd_fn_pat)


def parse_tmd_fn(tmd_fn):
    tmd_basename = os.path.basename(tmd_fn)
    r: parse.Result = tmd_fn_parser.parse(tmd_basename)  # type: ignore
    if r is None:
        raise ValueError(f"Could not parse tmd filename: {tmd_basename}")

    return datetime.datetime(*r)


def _read_tmd(tmd_fn):
    tmd = loki.read_tmd(tmd_fn)

    dt = parse_tmd_fn(tmd_fn)

    return dt, {ke: tmd[kl] for ke, kl in TMD2META.items()}


def read_all_telemetry(data_root: str):
    tmd_pat = os.path.join(data_root, "Telemetrie", "*.tmd")

    print("Reading telemetry...")

    telemetry = pd.DataFrame.from_dict(
        dict(_read_tmd(tmd_fm) for tmd_fm in glob.iglob(tmd_pat)), orient="index"
    ).sort_index()

    print(telemetry)

    return telemetry


def merge_telemetry(meta: Dict, telemetry: pd.DataFrame):
    # Construct tmd_fn and extract date
    tmd_fn = "{object_date} {object_time}.tmd".format_map(meta)
    dt = parse_tmd_fn(tmd_fn)

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

        raise MissingMetaError(
            f"The following fields are missing: {missing_keys_str}.\nSupply them in {meta_fn}"
        )

    meta["sample_id"] = "{sample_station}_{sample_haul}".format_map(meta)
    meta["acq_id"] = "{acq_instrument}_{sample_id}".format_map(meta)

    return meta


objid_pattern = "{object_date} {object_time}  {object_milliseconds:03d}  {object_sequence:06d} {object_posx:04d} {object_posy:04d}"
objid_parser = parse.compile(objid_pattern)


def parse_object_id(object_id, meta):
    result = objid_parser.parse(object_id)
    if result is None:
        raise ValueError(f"Can not parse object ID: {object_id}")

    return {**meta, "object_id": object_id, **result.named}


def stretch_contrast(image, low, high):
    low_, high_ = np.quantile(image, (low, high))
    return skimage.exposure.rescale_intensity(image, (low_, high_))


def build_segmentation(segmentation_config, image, meta):
    if segmentation_config is None:
        return image, meta

    if segmentation_config == "threshold":
        mask = image < threshold
        props = ImageProperties(mask, image)
        return image, CalculateZooProcessFeatures(props, meta, prefix="object_mc_")


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

    print(f"Found {len(data_roots):d} input directories.")

    with Pipeline() as p:
        data_root = Unpack(data_roots)

        Progress(data_root)

        # Load LOG metadata
        meta = Call(read_log, data_root, meta)

        # Load *all* telemetry data
        telemetry = Call(read_all_telemetry, data_root)

        # Load additional metadata
        meta = Call(update_and_validate_sample_meta, data_root, meta)

        target_archive_fn = Call(
            lambda meta: os.path.join(
                output["path"],
                "LOKI_{sample_station}_{sample_haul}.zip".format_map(meta),
            ),
            meta,
        )

        Call(print, meta, "\n")

        # Find images
        pictures_pat = Call(os.path.join, data_root, "Pictures", "*", "*.bmp")
        # img_fn = Find(pictures_path, [".bmp"])
        img_fn = Glob(pictures_pat, prefetch=True)

        # Extract object ID
        object_id = Call(
            lambda img_fn: os.path.splitext(os.path.basename(img_fn))[0], img_fn
        )

        # Parse object ID and update metadata
        meta = Call(parse_object_id, object_id, meta)
        del object_id

        # Merge telemetry
        meta = Call(merge_telemetry, meta, telemetry)

        # Read image
        image = ImageReader(img_fn)

        image, meta = build_segmentation(input.get("segmentation"), image, meta)

        # if segmentation == "threshold":
        #     # Simple segmentation
        #     mask = ...
        #     props = ImageProperties(mask, image)
        #     meta = CalculateZooProcessFeatures(props, prefix="object_mc_")
        # else:
        #     # Assemble and resegment
        #     image, meta = Resegment()

        target_image_fn = Call(lambda meta: f"{meta['object_id']}.jpg", meta)

        # # Image enhancement: Stretch contrast
        # image = Call(stretch_contrast, image, 0.01, 0.99)

        EcotaxaWriter(target_archive_fn, (target_image_fn, image), meta)

        Progress()

    p.run()


if __name__ == "__main__":
    import sys

    from schema import PipelineSchema

    with open(sys.argv[1]) as f:
        config = PipelineSchema().load(yaml.safe_load(f))
        p = build_pipeline(**config)
