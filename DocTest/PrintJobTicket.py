"""Normalized print-job tickets: the effective print settings of every page.

A ticket combines PJL job defaults with the page-description language
(PCL 5, PCL XL or PostScript) that overrides them, carries settings forward
from page to page, and normalizes the values so jobs in different languages
can be compared.

Value tables were checked against the primary documents:

- PCL 5: HP PCL 5 Technical Reference Manual (1992) p. 4-5, 4-12, 5-2, 5-4;
  PCL 5 Comparison Guide (1999) tables 1-1A/1-1C and p. 2-78..2-81 (media by
  name); HP PCL Implementor's Guide 6.0 ch. 7.2 (additional page sizes)
- PCL XL: Feature Reference Class 2.0 r1.8 appendix G, Class 2.1 supplement
  (p. 10: DuplexPageMode is relative to the page orientation) and Class 3.0
  supplement
- PJL: PJL Technical Reference Manual ed. 10, chapter 6 and table A-2
  (SET/DEFAULT and PJL reset conditions, variable values)
- PostScript: PLRM 3 section 6.2 (Tumble is relative to default user space;
  the Orientation page-device key rotates roll-fed media and is not used),
  PPD 4.3 (*Duplex, *InputSlot, *OutputBin, *MediaType), DSC 3.0
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

from DocTest.PrintJobParsing import (
    PclCommand,
    PclObserver,
    PclPages,
    capture_postscript_page_device,
    iter_pclxl,
    read_pcl5_pages,
    read_postscript,
    split_jobs,
)

UNSPECIFIED = "unspecified"
PAGE_SETTINGS = ("paper_size", "orientation", "duplex", "copies", "input_tray", "output_bin", "media_type")
JOB_SETTINGS = ("resolution", "color_mode")
MAPPABLE_SETTINGS = ("input_tray", "output_bin", "media_type")

# --- paper sizes ---------------------------------------------------------------

# name -> (width, height) in PostScript points, portrait
PAPER_DIMENSIONS = {
    "A3": (842, 1191), "A4": (595, 842), "A5": (420, 595), "A6": (298, 420),
    "B5": (499, 709), "JIS B4": (729, 1032), "JIS B5": (516, 729),
    "Letter": (612, 792), "Legal": (612, 1008), "Executive": (522, 756), "Ledger": (792, 1224),
    "Com10": (297, 684), "Monarch": (279, 540), "DL": (312, 624), "C5": (459, 649),
    "JIS B6": (363, 516), "Hagaki": (283, 420), "Oufuku-Hagaki": (567, 420), "JIS Exec": (612, 935),
    "JIS 8K": (737, 1043), "JIS 16K": (522, 737),
}
# names describe the sheet size: B5 paper and the B5 envelope are both 176x250 mm
PCL5_PAPER = {"1": "Executive", "2": "Letter", "3": "Legal", "6": "Ledger", "10": "JIS Exec", "11": "Ledger",
              "24": "A6", "25": "A5", "26": "A4", "27": "A3", "44": "JIS B6", "45": "JIS B5", "46": "JIS B4",
              "65": "B5", "71": "Hagaki", "72": "Oufuku-Hagaki", "80": "Monarch", "81": "Com10",
              "90": "DL", "91": "C5", "100": "B5"}
PCLXL_PAPER = {0: "Letter", 1: "Legal", 2: "A4", 3: "Executive", 4: "Ledger", 5: "A3", 6: "Com10",
               7: "Monarch", 8: "C5", 9: "DL", 10: "JIS B4", 11: "JIS B5", 12: "B5", 13: "B5",
               14: "Hagaki", 15: "Oufuku-Hagaki", 16: "A5", 17: "A6", 18: "JIS B6", 19: "JIS 8K",
               20: "JIS 16K", 21: "JIS Exec", 96: UNSPECIFIED}
# PJL PAPER values and PCL XL / PPD media names (case-insensitive)
PAPER_NAMES = {
    "LETTER": "Letter", "LEGAL": "Legal", "A4": "A4", "A3": "A3", "A5": "A5", "A6": "A6",
    "EXECUTIVE": "Executive", "LEDGER": "Ledger", "TABLOID": "Ledger", "11X17": "Ledger",
    "COM10": "Com10", "MONARCH": "Monarch", "DL": "DL", "C5": "C5", "B5": "B5",
    "JISB4": "JIS B4", "JISB5": "JIS B5", "JISB6": "JIS B6", "B4": "JIS B4", "JISEXEC": "JIS Exec",
    "JPOST": "Hagaki", "JPOSTD": "Oufuku-Hagaki", "HAGAKI": "Hagaki",
}

# --- orientation / duplex ----------------------------------------------------------

ORIENTATIONS = {0: "portrait", 1: "landscape", 2: "reverse_portrait", 3: "reverse_landscape"}
PCLXL_ORIENTATIONS = {**ORIENTATIONS, 4: UNSPECIFIED}          # 4 = eDefaultOrientation
LANDSCAPE = ("landscape", "reverse_landscape")
PCL5_DUPLEX = {"0": "simplex", "1": "long_edge", "2": "short_edge"}
# PCL XL binding is relative to the page: vertical = long edge on portrait pages
PCLXL_DUPLEX = {0: "short_edge", 1: "long_edge"}      # 0 horizontal, 1 vertical binding (portrait)
SWAP_EDGE = {"long_edge": "short_edge", "short_edge": "long_edge"}
DSC_DUPLEX = {"NONE": "simplex", "DUPLEXNOTUMBLE": "long_edge", "DUPLEXTUMBLE": "short_edge",
              "SIMPLEX": "simplex"}

# --- trays, bins, media (generic names only where the specs define them) ---------------

PCL5_TRAYS = {"2": "manual", "3": "manual_envelope", "6": "envelope", "7": "auto"}
PCLXL_TRAYS = {0: UNSPECIFIED, 1: "auto", 2: "manual", 3: "multipurpose", 4: "upper", 5: "lower",
               6: "envelope", 7: "third"}
PCL5_BINS = {"0": UNSPECIFIED, "1": "face_down", "2": "face_up"}
PCLXL_BINS = {0: UNSPECIFIED, 1: "face_down", 2: "face_up", 3: "job_offset"}
PJL_BINS = {"UPPER": "face_down", "LOWER": "face_up"}
PCL5_MEDIA = {"0": "plain", "1": "bond", "2": "special", "3": "glossy", "4": "transparency"}
SOURCE_DEFAULT = "default"


def _media_name(name: str) -> str:
    name = " ".join(str(name).split()).lower()
    return name or "plain"                     # PCL XL: an empty media type is plain paper


def paper_from_points(width: float, height: float) -> str:
    """Name of a paper size given in points (either orientation, ±2 pt)."""
    for name, (w, h) in PAPER_DIMENSIONS.items():
        for a, b in ((w, h), (h, w)):
            if abs(width - a) <= 2 and abs(height - b) <= 2:
                return name
    return f"{round(width * 25.4 / 72)}x{round(height * 25.4 / 72)}mm"


def paper_from_name(name: str, language: str) -> str:
    key = str(name).strip().strip('"').upper().replace(" ", "").replace("_", "")
    for suffix in ("PAPER", "SMALL"):
        if key.endswith(suffix) and key[:-len(suffix)] in PAPER_NAMES:
            key = key[:-len(suffix)]
    return PAPER_NAMES.get(key, f"{language}:{str(name).strip()}")


# --- ticket model ------------------------------------------------------------------

@dataclass
class Setting:
    value: str = UNSPECIFIED
    source: str = UNSPECIFIED
    raw: Optional[str] = None


def _blank_settings(names: Iterable[str]) -> Dict[str, Setting]:
    return {name: Setting() for name in names}


@dataclass
class PageTicket:
    page: int
    settings: Dict[str, Setting]

    def as_dict(self) -> Dict:
        entry: Dict = {"page": self.page}
        entry.update({name: s.value for name, s in self.settings.items()})
        entry["sources"] = {name: s.source for name, s in self.settings.items()}
        entry["raw"] = {name: s.raw for name, s in self.settings.items() if s.raw is not None}
        return entry


@dataclass
class JobTicket:
    languages: List[str] = field(default_factory=list)
    pages: List[PageTicket] = field(default_factory=list)
    job: Dict[str, Setting] = field(default_factory=lambda: _blank_settings(JOB_SETTINGS))
    pjl: Dict[str, str] = field(default_factory=dict)
    settings_source: str = ""

    def as_dict(self) -> Dict:
        return {
            "languages": list(self.languages),
            "page_count": len(self.pages),
            "settings_source": self.settings_source,
            "job": {name: s.value for name, s in self.job.items()},
            "job_sources": {name: s.source for name, s in self.job.items()},
            "pjl": dict(self.pjl),
            "pages": [page.as_dict() for page in self.pages],
        }


# --- PJL -------------------------------------------------------------------------------

def _parse_pjl_line(line: str) -> Optional[Tuple[str, str, str]]:
    """``(command, variable, value)`` for SET/DEFAULT lines, ``(command, '', '')`` otherwise."""
    words = line.split(None, 2)
    if len(words) < 2 or words[0].upper() != "@PJL":
        return None
    command = words[1].upper()
    if command not in ("SET", "DEFAULT") or len(words) < 3:
        return command, "", ""
    assignment = words[2]
    if assignment.upper().startswith("LPARM:"):
        parts = assignment.split(None, 1)
        assignment = parts[1] if len(parts) > 1 else ""
    variable, sep, value = assignment.partition("=")
    if not sep:
        return command, "", ""
    return command, variable.strip().upper(), value.strip().strip('"')


def _pjl_page_settings(variables: Dict[str, str]) -> Dict[str, Setting]:
    settings: Dict[str, Setting] = {}

    def put(name, value, raw):
        settings[name] = Setting(value, "pjl", raw)

    if "PAPER" in variables:
        put("paper_size", paper_from_name(variables["PAPER"], "pjl"), variables["PAPER"])
    if "ORIENTATION" in variables:
        orientation = variables["ORIENTATION"].upper()
        put("orientation", {"PORTRAIT": "portrait", "LANDSCAPE": "landscape"}.get(
            orientation, f"pjl:{variables['ORIENTATION']}"), variables["ORIENTATION"])
    if "DUPLEX" in variables:
        duplex = variables["DUPLEX"].upper()
        binding = variables.get("BINDING", "LONGEDGE").upper()
        value = "simplex" if duplex == "OFF" else (
            "short_edge" if binding == "SHORTEDGE" else "long_edge") if duplex == "ON" else f"pjl:{variables['DUPLEX']}"
        put("duplex", value, f"DUPLEX={variables['DUPLEX']} BINDING={variables.get('BINDING', '')}".strip())
    for variable in ("COPIES", "QTY"):
        if variable in variables:
            put("copies", variables[variable], variables[variable])
            break
    if "MEDIASOURCE" in variables:
        put("input_tray", f"pjl:{variables['MEDIASOURCE']}", variables["MEDIASOURCE"])
    if variables.get("MANUALFEED", "").upper() == "ON":
        put("input_tray", "manual", "MANUALFEED=ON")
    if "OUTBIN" in variables:
        raw = variables["OUTBIN"]
        put("output_bin", PJL_BINS.get(raw.upper(), f"pjl:{raw}"), raw)
    if "MEDIATYPE" in variables:
        raw = variables["MEDIATYPE"]
        put("media_type", "plain" if raw.upper() == "PAPER" else _media_name(raw), raw)
    return settings


def _pjl_job_settings(variables: Dict[str, str]) -> Dict[str, Setting]:
    settings = _blank_settings(JOB_SETTINGS)
    if "RESOLUTION" in variables:
        settings["resolution"] = Setting(variables["RESOLUTION"], "pjl", variables["RESOLUTION"])
    if "RENDERMODE" in variables:
        mode = variables["RENDERMODE"].upper()
        settings["color_mode"] = Setting({"COLOR": "color", "GRAYSCALE": "grayscale"}.get(
            mode, f"pjl:{variables['RENDERMODE']}"), "pjl", variables["RENDERMODE"])
    return settings


class _PjlEnvironment:
    """PJL ``DEFAULT`` writes the user defaults, ``SET`` the current environment.

    The current environment is reloaded from the defaults at every PJL reset:
    a UEL outside ``JOB``..``EOJ``, ``JOB``, ``EOJ``, ``RESET`` and
    ``INITIALIZE`` (which also clears the defaults).
    """

    def __init__(self):
        self.defaults: Dict[str, str] = {}
        self.current: Dict[str, str] = {}
        self.all: Dict[str, str] = {}
        self.in_job = False

    def start_segment(self):
        if not self.in_job:
            self.current = dict(self.defaults)

    def apply(self, lines: List[str]):
        for line in lines:
            parsed = _parse_pjl_line(line)
            if parsed is None:
                continue
            command, variable, value = parsed
            if command == "JOB":
                self.in_job = True
                self.current = dict(self.defaults)
            elif command == "EOJ":
                self.in_job = False
                self.current = dict(self.defaults)
            elif command == "RESET":
                self.current = dict(self.defaults)
            elif command == "INITIALIZE":
                self.defaults = {}
                self.current = {}
            elif variable:
                if command == "DEFAULT":
                    self.defaults[variable] = value
                else:
                    self.current[variable] = value
                self.all[variable] = value


# --- per-language state machines -----------------------------------------------------------

def _language_defaults(language: str) -> Dict[str, Setting]:
    settings = _blank_settings(PAGE_SETTINGS)
    settings["copies"] = Setting("1", SOURCE_DEFAULT)
    if language in ("PCL5", "PCLXL"):
        settings["duplex"] = Setting("simplex", SOURCE_DEFAULT)
        settings["orientation"] = Setting("portrait", SOURCE_DEFAULT)
    return settings


def _initial_state(language: str, pjl: Dict[str, str]) -> Dict[str, Setting]:
    state = _language_defaults(language)
    state.update(_pjl_page_settings(pjl))
    return state


class _Pcl5TicketObserver(PclObserver):
    def __init__(self, pjl: Dict[str, str], pages: List[PageTicket], first_page: int):
        self.pjl = pjl
        self.pages = pages
        self.first_page = first_page
        self.state = _initial_state("PCL5", pjl)

    def command(self, command: PclCommand, data: bytes) -> None:
        key, value = command.key, command.value.lstrip("+")

        def put(name, normalized):
            self.state[name] = Setting(normalized, "pcl5", f"{key}{value}")

        if key == "&lA":
            put("paper_size", PCL5_PAPER.get(value, f"pcl5:{value}"))
        elif key == "&lO":
            put("orientation", ORIENTATIONS.get(int(value), f"pcl5:{value}") if value.isdigit() else f"pcl5:{value}")
        elif key == "&lS":
            put("duplex", PCL5_DUPLEX.get(value, f"pcl5:{value}"))
        elif key == "&lX":
            put("copies", value)
        elif key == "&lH" and value not in ("0", ""):
            put("input_tray", PCL5_TRAYS.get(value, f"pcl5:{value}"))
        elif key == "&lG":
            put("output_bin", PCL5_BINS.get(value, f"pcl5:{value}"))
        elif key == "&lM":
            put("media_type", PCL5_MEDIA.get(value, f"pcl5:{value}"))
        elif key == "&nW" and command.payload > 1 and command.payload_offset >= 0:
            payload = data[command.payload_offset:command.payload_offset + command.payload]
            if payload[:1] == b"d":                       # operation 100: media type by name
                name = payload[1:].decode("latin-1").strip()
                self.state["media_type"] = Setting(_media_name(name), "pcl5", f"&n#Wd{name}")

    def page_end(self, page: int) -> None:
        self.pages.append(PageTicket(self.first_page + page,
                                     {k: Setting(s.value, s.source, s.raw) for k, s in self.state.items()}))

    def reset(self) -> None:
        self.state = _initial_state("PCL5", self.pjl)


def _read_pcl5(data: bytes, pjl: Dict[str, str], pages: List[PageTicket]) -> None:
    observer = _Pcl5TicketObserver(pjl, pages, len(pages))
    read_pcl5_pages(data, PclPages(), observer)


def _read_pclxl(data: bytes, pjl: Dict[str, str], pages: List[PageTicket]) -> None:
    state: Optional[Dict[str, Setting]] = None
    for operator, attributes in iter_pclxl(data):
        if operator == 0x43:                                  # BeginPage
            state = _initial_state("PCLXL", pjl)
            size = attributes.get(0x25)
            if isinstance(size, int):
                state["paper_size"] = Setting(PCLXL_PAPER.get(size, f"pclxl:{size}"),
                                              "pclxl" if size != 96 else UNSPECIFIED, f"MediaSize={size}")
            elif isinstance(size, str):
                state["paper_size"] = Setting(paper_from_name(size, "pclxl"), "pclxl", f"MediaSize={size}")
            if isinstance(value := attributes.get(0x28), int):
                state["orientation"] = Setting(PCLXL_ORIENTATIONS.get(value, f"pclxl:{value}"), "pclxl",
                                               f"Orientation={value}")
            if isinstance(value := attributes.get(0x26), int):
                state["input_tray"] = Setting(PCLXL_TRAYS.get(value, f"pclxl:{value}"), "pclxl", f"MediaSource={value}")
            if 0x27 in attributes:
                value = attributes[0x27]
                state["media_type"] = Setting(_media_name(value) if isinstance(value, str) else f"pclxl:{value}",
                                              "pclxl", f"MediaType={value}")
            if 0x24 in attributes:
                value = attributes[0x24]
                bin_name = PCLXL_BINS.get(value, f"pclxl:{value}") if isinstance(value, int) else f"pclxl:{value}"
                state["output_bin"] = Setting(bin_name, "pclxl", f"MediaDestination={value}")
            if isinstance(value := attributes.get(0x35), int):
                edge = PCLXL_DUPLEX.get(value, f"pclxl:{value}")
                if state["orientation"].value in LANDSCAPE:       # binding is relative to the page
                    edge = SWAP_EDGE.get(edge, edge)
                state["duplex"] = Setting(edge, "pclxl", f"DuplexPageMode={value}")
            if not isinstance(attributes.get(0x35), int) and 0x34 in attributes:
                state["duplex"] = Setting("simplex", "pclxl", f"SimplexPageMode={attributes[0x34]}")
        elif operator == 0x44:                                # EndPage
            state = state or _initial_state("PCLXL", pjl)
            if 0x31 in attributes:
                state["copies"] = Setting(str(attributes[0x31]), "pclxl", f"PageCopies={attributes[0x31]}")
            pages.append(PageTicket(len(pages) + 1, state))
            state = None


_PS_SOURCE_DSC = "ps-dsc"
_PS_SOURCE_DEVICE = "ps-page-device"


def _apply_ps_feature(state: Dict[str, Setting], key: str, option: str) -> None:
    feature = key.lstrip("*")
    raw = f"*{feature} {option}"
    if feature == "PageSize" or feature == "PageRegion":
        state["paper_size"] = Setting(paper_from_name(option, "ps"), _PS_SOURCE_DSC, raw)
    elif feature == "Duplex":
        state["duplex"] = Setting(DSC_DUPLEX.get(option.upper(), f"ps:{option}"), _PS_SOURCE_DSC, raw)
    elif feature == "InputSlot":
        state["input_tray"] = Setting(f"ps:{option}", _PS_SOURCE_DSC, raw)
    elif feature == "ManualFeed" and option.lower() == "true":
        state["input_tray"] = Setting("manual", _PS_SOURCE_DSC, raw)
    elif feature == "OutputBin":
        state["output_bin"] = Setting(f"ps:{option}", _PS_SOURCE_DSC, raw)
    elif feature == "MediaType":
        state["media_type"] = Setting(_media_name(option), _PS_SOURCE_DSC, raw)


def _ps_value(text: str) -> str:
    return text[1:-1] if text.startswith("(") and text.endswith(")") else text.lstrip("/")


def _apply_page_device(state: Dict[str, Setting], request: Dict[str, str]) -> None:
    """Apply one requested page-device dictionary.

    Keys starting with ``_`` hold raw device state (Duplex, Tumble, page
    shape) needed to derive the normalized settings; they are not reported.
    """
    source = _PS_SOURCE_DEVICE
    if "PageSize" in request:
        numbers = request["PageSize"].strip("[]").split()
        try:
            width, height = float(numbers[0]), float(numbers[1])
            state["paper_size"] = Setting(paper_from_points(width, height), source, f"PageSize={request['PageSize']}")
            state["_orientation"] = Setting("landscape" if width > height else "portrait", source,
                                            f"PageSize={request['PageSize']}")
        except (IndexError, ValueError):
            pass
    for key in ("Duplex", "Tumble"):
        if request.get(key) in ("true", "false"):
            state[f"_{key.lower()}"] = Setting(request[key])
    if "Duplex" in request or "Tumble" in request or "PageSize" in request:
        duplex = state.get("_duplex")
        if duplex is not None:
            tumble = state.get("_tumble", Setting("false")).value
            if duplex.value == "false":
                value = "simplex"
            else:
                # Tumble is relative to default user space, i.e. to the page shape
                value = "short_edge" if tumble == "true" else "long_edge"
                if state.get("_orientation", Setting("portrait")).value == "landscape":
                    value = SWAP_EDGE[value]
            state["duplex"] = Setting(value, source, f"Duplex={duplex.value} Tumble={tumble}")
    if request.get("NumCopies", "null") != "null":
        state["copies"] = Setting(request["NumCopies"].split(".")[0], source, f"NumCopies={request['NumCopies']}")
    if request.get("MediaPosition", "null") != "null":
        state["input_tray"] = Setting(f"ps:position{request['MediaPosition']}", source,
                                      f"MediaPosition={request['MediaPosition']}")
    if request.get("ManualFeed") == "true":
        state["input_tray"] = Setting("manual", source, "ManualFeed=true")
    if "MediaType" in request and request["MediaType"] != "null":
        state["media_type"] = Setting(_media_name(_ps_value(request["MediaType"])), source,
                                      f"MediaType={request['MediaType']}")
    if "OutputType" in request and request["OutputType"] != "null":
        state["output_bin"] = Setting(f"ps:{_ps_value(request['OutputType'])}", source,
                                      f"OutputType={request['OutputType']}")


def _dsc_orientation(value: str) -> Optional[str]:
    return {"PORTRAIT": "portrait", "LANDSCAPE": "landscape"}.get(value.strip().upper())


def _read_postscript(path: str, data: bytes, pjl: Dict[str, str], ticket: JobTicket) -> None:
    job = read_postscript(data)
    capture = capture_postscript_page_device(path)
    if capture.error:
        ticket.settings_source = f"dsc (page-device requests unavailable: {capture.error})"
        requests: List[Tuple[int, Dict[str, str]]] = []
    else:
        ticket.settings_source = "dsc+page-device"
        requests = capture.requests
    page_count = job.page_count or (capture.page_count if not capture.error else None)
    if not page_count:
        raise AssertionError(
            f"Cannot determine the page count of {path}: the job has no DSC page structure "
            "(%%Page comments) and Ghostscript could not interpret it.")
    document = _initial_state("POSTSCRIPT", pjl)
    for entry in job.header:
        if entry["property"] == "Orientation" and _dsc_orientation(entry["value"]):
            document["orientation"] = Setting(_dsc_orientation(entry["value"]), _PS_SOURCE_DSC,
                                              f"%%Orientation: {entry['value']}")
    for feature in job.features:
        _apply_ps_feature(document, feature["property"], feature["value"])
    device_state: Dict[str, Setting] = {}
    request_index = 0
    for page in range(1, page_count + 1):
        state = {k: Setting(s.value, s.source, s.raw) for k, s in document.items()}
        for entry in (e for e in job.pages if e["page"] == str(page)):
            if entry["property"].startswith("*"):
                _apply_ps_feature(state, entry["property"], entry["value"])
            elif entry["property"] == "PageOrientation" and _dsc_orientation(entry["value"]):
                state["orientation"] = Setting(_dsc_orientation(entry["value"]), _PS_SOURCE_DSC,
                                               f"%%PageOrientation: {entry['value']}")
        while request_index < len(requests) and requests[request_index][0] <= page:
            _apply_page_device(device_state, requests[request_index][1])
            request_index += 1
        for name, setting in device_state.items():          # interpreted requests take precedence
            if not name.startswith("_"):
                state[name] = setting
        if state["orientation"].source != _PS_SOURCE_DSC and "_orientation" in device_state:
            state["orientation"] = device_state["_orientation"]     # DSC orientation comments win
        ticket.pages.append(PageTicket(page, {name: state[name] for name in PAGE_SETTINGS}))


# --- mapping ---------------------------------------------------------------------------------

Mapping = Dict[str, Dict[str, str]]


def load_mapping(mapping: Union[None, str, Dict]) -> Mapping:
    """Read a mapping given as a dictionary or a JSON file path."""
    if mapping in (None, "", {}):
        return {}
    if isinstance(mapping, str):
        if not os.path.isfile(mapping):
            raise ValueError(f"Mapping file not found: {mapping}")
        with open(mapping, encoding="utf-8") as handle:
            mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError("The mapping must be a dictionary or a JSON object.")
    unknown = sorted(set(mapping) - set(MAPPABLE_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown mapping section(s) {', '.join(unknown)}. "
                         f"Allowed sections: {', '.join(MAPPABLE_SETTINGS)}.")
    return {section: {str(k): str(v) for k, v in values.items()} for section, values in mapping.items()}


def _apply_mapping(ticket: JobTicket, mapping: Mapping) -> None:
    for page in ticket.pages:
        for section, translations in mapping.items():
            setting = page.settings[section]
            if setting.value in translations:
                page.settings[section] = Setting(translations[setting.value], setting.source,
                                                 setting.raw if setting.raw is not None else setting.value)


# --- reading ---------------------------------------------------------------------------------

def read_ticket(path: str, data: bytes, mapping: Union[None, str, Dict] = None) -> JobTicket:
    """Build the normalized ticket of a PCL 5, PCL XL or PostScript job."""
    translations = load_mapping(mapping)
    ticket = JobTicket()
    environment = _PjlEnvironment()
    segments = split_jobs(data)
    languages = {s.language for s in segments if s.language}
    if "POSTSCRIPT" in languages and len(languages) > 1:
        raise ValueError(f"Print jobs mixing PostScript with {', '.join(sorted(languages - {'POSTSCRIPT'}))} "
                         "are not supported.")
    sources = set()
    for segment in segments:
        environment.start_segment()
        environment.apply(segment.pjl)
        if segment.language is None or not segment.data:
            continue
        ticket.languages.append(segment.language)
        pjl = dict(environment.current)
        if segment.language == "PCL5":
            _read_pcl5(segment.data, pjl, ticket.pages)
            sources.add("pcl5")
        elif segment.language == "PCLXL":
            _read_pclxl(segment.data, pjl, ticket.pages)
            sources.add("pclxl")
        elif segment.language == "POSTSCRIPT":
            _read_postscript(path, data, pjl, ticket)
            break
    ticket.pjl = dict(environment.all)
    job_pjl = dict(environment.defaults)
    job_pjl.update(environment.all)
    ticket.job = _pjl_job_settings(job_pjl)
    if not ticket.settings_source:
        ticket.settings_source = "+".join(["pjl"] * bool(ticket.pjl) + sorted(sources))
    _apply_mapping(ticket, translations)
    return ticket


# --- comparison --------------------------------------------------------------------------------

def _flatten(ticket: Dict) -> Dict[Tuple[str, str], str]:
    entries: Dict[Tuple[str, str], str] = {("job", "page_count"): str(ticket["page_count"])}
    for name, value in ticket["job"].items():
        entries[("job", name)] = str(value)
    for name, value in ticket["pjl"].items():
        entries[("pjl", name)] = str(value)
    for page in ticket["pages"]:
        for name in PAGE_SETTINGS:
            entries[(f"page {page['page']}", name)] = str(page[name])
    return entries


def _is_ignored(scope: str, name: str, ignored: Iterable[str]) -> bool:
    ignored = set(ignored)
    if scope == "pjl" and "pjl" in ignored:
        return True
    return name.lower() in ignored


def ticket_differences(reference: Dict, candidate: Dict, ignored: Iterable[str]) -> List[str]:
    """Human-readable differences between two ticket dictionaries."""
    ignored = set(ignored)
    ref, cand = _flatten(reference), _flatten(candidate)
    differences: List[str] = []
    page_scopes_ref = {s for s, _ in ref if s.startswith("page ")}
    page_scopes_cand = {s for s, _ in cand if s.startswith("page ")}

    def order(key):
        scope, name = key
        rank = 0 if scope == "job" else 1 if scope == "pjl" else 2
        number = int(scope.split()[1]) if scope.startswith("page ") else 0
        return rank, number, name

    for key in sorted(set(ref) | set(cand), key=order):
        scope, name = key
        if _is_ignored(scope, name, ignored):
            continue
        if scope.startswith("page ") and (scope not in page_scopes_ref or scope not in page_scopes_cand):
            continue
        if ref.get(key) != cand.get(key):
            differences.append(f"{scope}: {name} {ref.get(key, 'missing')} → {cand.get(key, 'missing')}")
    for scope in sorted(page_scopes_ref ^ page_scopes_cand, key=lambda s: int(s.split()[1])):
        where = "candidate" if scope in page_scopes_ref else "reference"
        differences.append(f"{scope}: missing in {where}")
    return differences


def ticket_mismatches(ticket: Dict, expected: Dict[str, str], page: Optional[int] = None) -> List[str]:
    """Settings of ``ticket`` that differ from ``expected`` (case-insensitive)."""
    unknown = sorted(set(expected) - set(PAGE_SETTINGS) - set(JOB_SETTINGS) - {"page_count"})
    if unknown:
        raise ValueError(f"Unknown ticket setting(s) {', '.join(unknown)}. Known settings: "
                         f"{', '.join(PAGE_SETTINGS + JOB_SETTINGS + ('page_count',))}.")
    mismatches: List[str] = []
    for name, value in expected.items():
        if name == "page_count" or name in JOB_SETTINGS:
            actual = ticket["page_count"] if name == "page_count" else ticket["job"][name]
            if str(actual).casefold() != str(value).casefold():
                mismatches.append(f"job: {name} is {actual}, expected {value}")
    pages = ticket["pages"] if page is None else [p for p in ticket["pages"] if p["page"] == int(page)]
    if page is not None and not pages:
        raise AssertionError(f"The print job has no page {page} (it has {ticket['page_count']} page(s)).")
    for entry in pages:
        for name, value in expected.items():
            if name in PAGE_SETTINGS and str(entry[name]).casefold() != str(value).casefold():
                mismatches.append(f"page {entry['page']}: {name} is {entry[name]}, expected {value}")
    return mismatches
