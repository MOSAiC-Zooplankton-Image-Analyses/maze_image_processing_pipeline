import logging
from typing import Any, Collection, Tuple, Mapping
from tqdm.auto import tqdm
import fnmatch
import os
import yaml

logger = logging.getLogger(__name__)

def german_float(s: str):
    return float(s.replace(",", "."))


TMD_FIELDS = {
    1: ("DEVICE", None),  # Loki-Name
    5: ("GPS_LON", german_float),  # Longitude (GPS or fix),	[+E, -W]
    6: ("GPS_LAT", german_float),  # Latitude (GPS or fix),	[+N, -S]
    10: ("PRESS", german_float),  # Aandera 4017D, Pressure	[kPa]
    11: ("TEMP", german_float),  # Aandera 4017D, Temperature	[°C]
    20: ("OXY_CON", german_float),  # Aandera 4330F, Oxygen concentration	[mg*l^-1]
    21: ("OXY_SAT", german_float),  # Aandera 4330F, Oxygen saturation	[%]
    22: ("OXY_TEMP", german_float),  # Aandera 4330F, Oxygen temperature	[°C]
    30: ("COND_COND", german_float),  # Aandera 3919 A/W, Conductivity	[mS/cm]
    31: ("COND_TEMP", german_float),  # Aandera 3919 A/W, Temperature	[°C]
    32: ("COND_SALY", german_float),  # Aandera 3919 A/W, Salinity	[PSU]
    33: ("COND_DENS", german_float),  # Aandera 3919 A/W, Density	[kg/m^3]
    34: ("COND_SSPEED", german_float),  # Aandera 3919 A/W, Soundspeed	[m/s]
    40: ("FLOUR_1", german_float),  # Flourescence
    41: ("FLOUR_CR", german_float),  # HAARDT Flourescence, Clorophyll Range
    42: ("FLOUR_CV", german_float),  # HAARDT Flourescence, Clorophyll Value
    43: ("FLOUR_TR", german_float),  # HAARDT Flourescence, Turbidity Range
    44: ("FLOUR_TD", german_float),  # HAARDT Flourescence, Turbidity Val
    200: ("ROLL", german_float),  # ISITEC, Roll	[°]
    201: ("PITCH", german_float),  # ISITEC, Pitch	[°]
    202: ("NICK", german_float),  # ISITEC, Nick	[°]
    230: ("LOKI_REC", None),  # LOKI Recorder status
    231: ("LOKI_PIC", int),  # Loki Recorder consecutive picture number
    232: ("LOKI_FRAME", german_float),  # Loki Recorder frame rate	[fps]
    235: ("CAM_STAT", None),  # Camera status
    240: ("HOUSE_STAT", None),  # Housekeeping status
    241: ("HOUSE_T1", german_float),  # Housekeeping temperature 1	[°C]
    242: ("HOUSE_T2", german_float),  # Housekeeping temperature 2	[°C]
    243: ("HOUSE_VOLT", german_float),  # Housekeeping voltage	[V]
}

DAT_FIELDS = {
    1: ("FW_REV", None),  # Firmware version
    2: ("COND_SSPEED", float),  # Speed of sound
    3: ("COND_DENS", float),  # Density
    4: ("COND_TEMP", float),  # Temperature
    5: ("COND_COND", float),  # Conductivity
    6: ("COND_SALY", float),  # Salinity
    7: ("OXY_CON", float),  # Oxygen concentration
    8: ("OXY_SAT", float),  # Oxygen saturation
    9: ("OXY_TEMP", float),  # Temperature
    10: ("HOUSE_T1", float),  # Housekeeping temperature
    11: ("HOUSE_VOLT", float),  # Housekeeping  voltage
    16: ("FLOUR_1", float),  # Fluorescence
    17: ("UNKNOWN", None),  # ??
    18: ("UNKNOWN", None),  # ??
    19: ("UNKNOWN", None),  # ??
    20: ("PRESS", float),  # Pressure
    21: ("TEMP", float),  # Temperature
    22: ("UNKNOWN", None),  # ??
    23: ("LOKI_REC", None),  # Recorder status
    24: ("LOKI_PIC", None),  # Picture #
    25: ("LOKI_FRAME", None),  # Frame rate
    26: ("GPS_LAT", float),  # Position Latitute
    27: ("GPS_LON", float),  # Position Longitude
}

LOG_FIELDS = {
    1: ("DATE", None),  # Startdate	UTC
    2: ("TIME", None),  # Starttime	UTC
    3: ("PICTURE#", int),  # Aktuelle Bildnummer VPR-Recorder
    4: ("DEVICE", None),  # Loki-Name
    5: ("LOKI", None),  # Loki-Serial
    6: ("FW_REV", None),  # Firmwareversion
    7: ("SW_REV", None),  # Softwareversion
    8: ("CRUISE", None),  # Cruise Name
    9: ("STATION", None),  # Station
    10: ("STATION_NR", None),  # Stationsnumber
    11: ("HAUL", None),  # Haul
    12: ("USER", None),  # Investigator
    13: ("SHIP", None),  # Ship name
    14: ("SHIP_PORT", None),  # Port of Registry
    15: ("SHIP_STAT", None),  # State of Registry
    16: ("SHIP_AFF", None),  # Ship affiliation
    17: ("GPS_SRC", None),  # GPS Source (0 = NoGPS, 1 = Fixed, 2 = Ext.)
    18: ("FIX_LON", german_float),  # Fixed Longitude	[+E, -W]
    19: ("FIX_LAT", german_float),  # Fixed Latitude	[+N, -S]
    20: ("TEMP_INDEX", None),  # Temperature Sensor Index for calculation
    61: ("ERROR", None),  # Any Error Message
    62: ("WAKEUP", None),  # AnyWakeUp Controller Message
    63: ("STOP_DATE", None),  # Stopdate	UTC
    64: ("STOP_TIME", None),  # Stoptime	UTC
}


def _parse_tmd_line(line: str, fields) -> Tuple[str, Any]:
    try:
        idx, value = line.rstrip("\n").split(";", 1)
    except:
        print("Offending line:", line)
        raise

    name, converter = fields[int(idx)]
    if converter is not None:
        try:
            value = converter(value)
        except Exception as exc:
            raise type(exc)(*exc.args, f"Field {name}") from exc

    return name, value


def _parse_dat_line(idx: int, line: str, fields) -> Tuple[str, Any]:
    value = line.rstrip("\n")

    name, converter = fields[idx]

    if converter is not None:
        try:
            value = converter(value)
        except Exception as exc:
            raise type(exc)(*exc.args, f"Field {name}") from exc

    return name, value


def read_tmd(fn):
    with open(fn, "r") as f:
        return dict(_parse_tmd_line(l, TMD_FIELDS) for l in f)


def read_dat(fn):
    with open(fn, "r") as f:
        # FIXME: Sometimes, a .dat contains multiple lines.
        # Here, we only use the first one.
        contents = f.readline()

    fields = contents.split("\t")

    return dict(
        _parse_dat_line(i, f, DAT_FIELDS)
        for i, f in enumerate(fields, 1)
        if i in DAT_FIELDS
    )

_Log_to_EcoTaxa = {
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

def read_log(fn, format="raw"):
    with open(fn, "r") as f:
        data = dict(_parse_tmd_line(l, LOG_FIELDS) for l in f)

    if format == "raw":
        return data

    if format == "ecotaxa":
        return {ke: data[kl] for ke, kl in _Log_to_EcoTaxa.items()}

    raise ValueError(f"Unknown format: {format!r}")

def read_yaml_meta(meta_fn: str) -> Mapping[str, Any]:
    if not os.path.isfile(meta_fn):
        return {}

    with open(meta_fn) as f:
        value = yaml.unsafe_load(f)

        if not isinstance(value, Mapping):
            raise ValueError(f"Unexpected content in {meta_fn}: {value}")

        return value


def find_data_roots(project_root, ignore_patterns: Collection | None = None, progress=True):
    logger.info("Detecting project folders...")
    with tqdm(leave=False, disable=not progress) as progress_bar:
        for root, dirs, _ in os.walk(project_root):
            progress_bar.set_description(root, refresh=False)
            progress_bar.update(1)

            if ignore_patterns is not None:
                if any(fnmatch.fnmatch(root, pat) for pat in ignore_patterns):
                    # Do not descend further
                    dirs[:] = []
                    continue

            if "Pictures" in dirs and "Telemetrie" in dirs:
                yield root
                # Do not descend further
                dirs[:] = []
