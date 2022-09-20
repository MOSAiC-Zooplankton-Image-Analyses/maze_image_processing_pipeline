from typing import Any, Generator, Iterable, Tuple


def german_float(s: str):
    return float(s.replace(",", "."))


TMD_FIELDS = {
    1: ("DEVICE", None),  # Loki-Name
    5: ("GPS_LONG", None),  # Longitude (GPS or fix),	[+E, -W]
    6: ("GPS_LAT", None),  # Latitude (GPS or fix),	[+N, -S]
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
    18: ("FIX_LONG", german_float),  # Fixed Longitude	[+E, -W]
    19: ("FIX_LAT", german_float),  # Fixed Latitude	[+N, -S]
    20: ("TEMP_INDEX", None),  # Temperature Sensor Index for calculation
    61: ("ERROR", None),  # Any Error Message
    62: ("WAKEUP", None),  # AnyWakeUp Controller Message
    63: ("STOP_DATE", None),  # Stopdate	UTC
    64: ("STOP_TIME", None),  # Stoptime	UTC
}


def _parse_line(line: str, fields) -> Tuple[str, Any]:
    idx, value = line.rstrip("\n").split(";", 1)
    name, converter = fields[int(idx)]
    if converter is not None:
        try:
            value = converter(value)
        except Exception as exc:
            raise type(exc)(*exc.args, f"Field {name}")

    return name, value


def read_tmd(fn):
    with open(fn, "r") as f:
        return dict(_parse_line(l, TMD_FIELDS) for l in f)


def read_log(fn):
    with open(fn, "r") as f:
        return dict(_parse_line(l, LOG_FIELDS) for l in f)
