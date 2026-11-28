"""Byte-level readers for print jobs: PJL, PCL 5, PCL XL and PostScript (DSC).

The readers follow the published grammars (HP PJL Technical Reference, HP
PCL 5 Technical Reference, HP PCL XL Feature Reference 2.0/2.1/3.0, Adobe
DSC 3.0) and work on raw bytes, so binary payloads such as downloaded fonts
or raster rows are skipped by their declared length instead of being
misread as commands or page breaks.
"""

import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

UEL = b"\x1b%-12345X"
ESC = 0x1B
FORM_FEED = 0x0C


# --------------------------------------------------------------------------
# PJL and job segmentation
# --------------------------------------------------------------------------

@dataclass
class JobSegment:
    """One page-description-language section of a print job."""

    language: Optional[str]          # 'PCL5', 'PCLXL', 'POSTSCRIPT' or None (PJL only)
    pjl: List[str] = field(default_factory=list)
    data: bytes = b""


_LANGUAGES = {"PCL": "PCL5", "PCL5": "PCL5", "PCL5E": "PCL5", "PCL5C": "PCL5",
              "PCLXL": "PCLXL", "PCL6": "PCLXL", "POSTSCRIPT": "POSTSCRIPT", "PS": "POSTSCRIPT"}
_ENTER_LANGUAGE = re.compile(rb"@PJL\s+ENTER\s+LANGUAGE\s*=\s*([A-Za-z0-9]+)", re.IGNORECASE)


def _sniff_language(data: bytes) -> Optional[str]:
    head = data.lstrip(b"\x00\x04\r\n\t ")[:256]
    if not head:
        return None
    if head.startswith((b") HP-PCL XL", b"( HP-PCL XL", b"` HP-PCL XL")):
        return "PCLXL"
    if head.startswith(b"%!"):
        return "POSTSCRIPT"
    return "PCL5"


def split_jobs(data: bytes) -> List[JobSegment]:
    """Split a print job at Universal Exit Language sequences.

    Each part starts with optional ``@PJL`` lines; an ``ENTER LANGUAGE`` line
    (or the data itself) determines the language of the bytes that follow.
    """
    segments: List[JobSegment] = []
    parts = data.split(UEL)
    for index, part in enumerate(parts):
        pjl: List[str] = []
        pos = 0
        language = None
        if index > 0 or part.lstrip().startswith(b"@PJL"):
            while True:
                rest = part[pos:]
                stripped = pos + len(rest) - len(rest.lstrip(b"\r\n\t "))
                if not part.startswith(b"@PJL", stripped):
                    break
                end = part.find(b"\n", stripped)
                end = len(part) if end < 0 else end + 1
                line = part[stripped:end].decode("latin-1").strip()
                pjl.append(line)
                pos = end
                match = _ENTER_LANGUAGE.match(line.encode("latin-1"))
                if match:
                    language = _LANGUAGES.get(match.group(1).decode().upper(), match.group(1).decode().upper())
                    break
        payload = part[pos:]
        if language is None and payload.strip(b"\x00\x04\r\n\t "):
            language = _sniff_language(payload)
        if not payload.strip(b"\x00\x04\r\n\t "):
            payload = b""
            if language is not None and not pjl:
                language = None
        if pjl or payload:
            segments.append(JobSegment(language if payload else None, pjl, payload))
    return segments


# --------------------------------------------------------------------------
# PCL 5
# --------------------------------------------------------------------------

# Commands followed by a counted binary payload: (parameter, group, terminator)
PCL5_PAYLOAD_COMMANDS = frozenset({
    ("*", "b", "W"), ("*", "b", "V"), ("(", "s", "W"), (")", "s", "W"), ("(", "f", "W"),
    ("*", "c", "W"), ("*", "v", "W"), ("&", "n", "W"), ("&", "p", "X"), ("*", "m", "W"),
    ("*", "l", "W"), ("*", "i", "W"), ("*", "g", "W"), ("&", "b", "W"),
})
_PCL5_VALUE = re.compile(rb"([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)([\x40-\x5e\x60-\x7e])")
_HPGL_MARKING = re.compile(rb"(?i)(PD|LB|PE|RR|RA|EP|FP|WG|EW|CI|AA|AR|BR|BZ)")


@dataclass(frozen=True)
class PclCommand:
    parameter: str
    group: str
    terminator: str
    value: str
    payload: int = 0          # length of the binary payload that followed
    payload_offset: int = field(default=-1, compare=False)   # where that payload starts

    @property
    def key(self) -> str:
        return f"{self.parameter}{self.group}{self.terminator}"


def iter_pcl5(data: bytes) -> Iterator[Tuple]:
    """Tokenize PCL 5 bytes.

    Yields ``("cmd", PclCommand)``, ``("esc2", char)`` for two-character
    commands such as ``ESC E``, ``("text", start, end)`` for printable data,
    ``("ff",)`` for form feeds, ``("hpgl", start, end)`` for HP-GL/2 blocks
    and ``("uel",)``.
    """
    n = len(data)
    i = 0
    while i < n:
        j = data.find(b"\x1b", i)
        if j < 0:
            j = n
        k = i
        while k < j:                       # text and form feeds up to the escape
            ff = data.find(b"\x0c", k, j)
            end = j if ff < 0 else ff
            if end > k:
                yield ("text", k, end)
            if ff < 0:
                break
            yield ("ff",)
            k = ff + 1
        if j >= n - 1:
            break
        c = data[j + 1]
        if data.startswith(UEL, j):
            yield ("uel",)
            i = j + len(UEL)
            continue
        if 0x21 <= c <= 0x2F:
            parameter = chr(c)
            pos = j + 2
            if c == 0x25:                  # ESC % # A / ESC % # B
                match = _PCL5_VALUE.match(data, pos)
                if not match:
                    i = j + 2
                    continue
                pos = match.end()
                if match.group(2) == b"B":
                    stop = data.find(b"\x1b%", pos)
                    stop = n if stop < 0 else stop
                    yield ("hpgl", pos, stop)
                    i = stop
                    continue
                yield ("cmd", PclCommand("%", "", match.group(2).decode(), match.group(1).decode()))
                i = pos
                continue
            group = ""
            if pos < n and 0x60 <= data[pos] <= 0x7E:
                group = chr(data[pos])
                pos += 1
            while pos < n:
                match = _PCL5_VALUE.match(data, pos)
                if not match:
                    break
                value = match.group(1).decode()
                terminator_byte = match.group(2)[0]
                terminator = chr(terminator_byte).upper()
                pos = match.end()
                payload = 0
                if (parameter, group, terminator) in PCL5_PAYLOAD_COMMANDS:
                    try:
                        payload = max(0, int(float(value or "0")))
                    except ValueError:
                        payload = 0
                    payload = min(payload, n - pos)
                yield ("cmd", PclCommand(parameter, group, terminator, value, payload, pos))
                pos += payload
                if 0x40 <= terminator_byte <= 0x5E:
                    break
            i = max(pos, j + 2)
            continue
        if 0x30 <= c <= 0x7E:
            yield ("esc2", chr(c))
            i = j + 2
            continue
        i = j + 1


# Page-setup commands reported per page
PCL5_PAGE_SETUP = {
    "&lA": "paper_format",
    "&lH": "paper_source",
    "&lO": "page_orientation",
    "&lS": "duplex",
    "&lX": "copies",
}


@dataclass
class PclPages:
    """Pages and per-page settings of a PCL job."""

    page_count: int = 0
    commands: List[Dict[str, str]] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    overlay_macro: bool = False


def _is_marking_text(data: bytes, start: int, end: int) -> bool:
    return any(b > 0x20 for b in data[start:end])


class PclObserver:
    """Receives the PCL 5 command stream together with the page model's
    decisions, so other views (e.g. job tickets) reuse the page-eject rules."""

    def command(self, command: "PclCommand", data: bytes) -> None:
        pass

    def page_end(self, page: int) -> None:
        pass

    def reset(self) -> None:
        pass


def read_pcl5_pages(data: bytes, result: PclPages, observer: Optional[PclObserver] = None) -> None:
    """Count pages and attribute page-setup commands of one PCL 5 segment.

    A form feed always ejects a page; ``ESC&l0H``, ``ESC E``, a UEL and the
    end of the data eject a page only when it carries marks (text, raster,
    rectangle fills, HP-GL/2 drawing or macro calls).
    """
    marks = False
    pending: List[Dict[str, str]] = []
    observer = observer or PclObserver()

    def end_page():
        nonlocal marks, pending
        result.page_count += 1
        result.commands.extend(pending)
        pending = []
        marks = False
        observer.page_end(result.page_count)

    for token in iter_pcl5(data):
        kind = token[0]
        if kind == "text":
            if not marks and _is_marking_text(data, token[1], token[2]):
                marks = True
        elif kind == "ff":
            end_page()
        elif kind == "hpgl":
            if _HPGL_MARKING.search(data, token[1], token[2]):
                marks = True
        elif kind in ("uel",) or (kind == "esc2" and token[1] == "E"):
            if marks:
                end_page()
            if kind == "esc2":
                observer.reset()
        elif kind == "cmd":
            command: PclCommand = token[1]
            observer.command(command, data)
            key = command.key
            if key == "&lH" and command.value.lstrip("+") in ("0", ""):
                if marks:
                    end_page()
                continue
            name = PCL5_PAGE_SETUP.get(key)
            if name is not None:
                pending.append({"page": str(result.page_count + 1), "property": name,
                                "value": command.value.lstrip("+")})
            elif key in ("*bW", "*bV") and command.payload > 0:
                marks = True
            elif key == "*cP" or (key == "&pX" and command.payload > 0):
                marks = True
            elif key == "&fX":
                if command.value in ("2", "3"):
                    marks = True
                elif command.value == "4":
                    result.overlay_macro = True
    if marks:
        end_page()
    # settings issued after the last page never reach paper; leave them out
    # pending is discarded on purpose


# --------------------------------------------------------------------------
# PCL XL
# --------------------------------------------------------------------------

_XL_SCALAR = {0xC0: ("B", 1), 0xC1: ("H", 2), 0xC2: ("I", 4), 0xC3: ("h", 2), 0xC4: ("i", 4), 0xC5: ("f", 4)}
_XL_ARRAY = {0xC8: ("B", 1), 0xC9: ("H", 2), 0xCA: ("I", 4), 0xCB: ("h", 2), 0xCC: ("i", 4), 0xCD: ("f", 4)}
_XL_XY = {0xD0: ("B", 1), 0xD1: ("H", 2), 0xD2: ("I", 4), 0xD3: ("h", 2), 0xD4: ("i", 4), 0xD5: ("f", 4)}
_XL_BOX = {0xE0: ("B", 1), 0xE1: ("H", 2), 0xE2: ("I", 4), 0xE3: ("h", 2), 0xE4: ("i", 4), 0xE5: ("f", 4)}
_XL_WHITESPACE = frozenset({0x00, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20})
XL_BEGIN_PAGE, XL_END_PAGE = 0x43, 0x44
XL_ATTRIBUTES = {0x24: "MediaDestination", 0x25: "MediaSize", 0x26: "MediaSource", 0x27: "MediaType", 0x28: "Orientation",
                 0x31: "PageCopies", 0x34: "SimplexPageMode", 0x35: "DuplexPageMode"}
_XL_DUPLEX = {0: "duplex_horizontal", 1: "duplex_vertical"}


class PclXlError(ValueError):
    """A PCL XL stream that cannot be decoded."""


def iter_pclxl(data: bytes) -> Iterator[Tuple[int, Dict[int, object]]]:
    """Decode a PCL XL stream into ``(operator, {attribute_id: value})``."""
    header = data.find(b"HP-PCL XL")
    if header < 1:
        raise PclXlError("PCL XL stream header not found")
    binding = data[header - 2:header - 1]
    if binding == b"`":
        raise PclXlError("PCL XL ASCII binding is not supported")
    endian = "<" if binding == b")" else ">" if binding == b"(" else None
    if endian is None:
        raise PclXlError(f"Unknown PCL XL binding {binding!r}")
    i = data.find(b"\n", header)
    if i < 0:
        return
    i += 1
    n = len(data)
    attributes: Dict[int, object] = {}
    value: object = None
    try:
        while i < n:
            tag = data[i]
            i += 1
            if tag in _XL_SCALAR:
                fmt, size = _XL_SCALAR[tag]
                value = struct.unpack_from(endian + fmt, data, i)[0]
                i += size
            elif tag in _XL_ARRAY:
                fmt, size = _XL_ARRAY[tag]
                length_tag = data[i]
                i += 1
                if length_tag == 0xC0:
                    count = data[i]
                    i += 1
                elif length_tag == 0xC1:
                    count = struct.unpack_from(endian + "H", data, i)[0]
                    i += 2
                else:
                    raise PclXlError(f"invalid array length type at offset {i - 1}")
                raw = data[i:i + count * size]
                if len(raw) < count * size:
                    raise PclXlError(f"truncated array at offset {i}")
                value = raw.decode("latin-1") if tag == 0xC8 else list(
                    struct.unpack_from(f"{endian}{count}{fmt}", data, i))
                i += count * size
            elif tag in _XL_XY:
                fmt, size = _XL_XY[tag]
                value = struct.unpack_from(f"{endian}2{fmt}", data, i)
                i += 2 * size
            elif tag in _XL_BOX:
                fmt, size = _XL_BOX[tag]
                value = struct.unpack_from(f"{endian}4{fmt}", data, i)
                i += 4 * size
            elif tag == 0xF8:
                attributes[data[i]] = value
                i += 1
            elif tag == 0xF9:
                attributes[struct.unpack_from(endian + "H", data, i)[0]] = value
                i += 2
            elif tag == 0xFA:
                i += 4 + struct.unpack_from(endian + "I", data, i)[0]
            elif tag == 0xFB:
                i += 1 + data[i]
            elif tag in _XL_WHITESPACE:
                continue
            elif 0x41 <= tag <= 0xBF:
                yield tag, attributes
                attributes = {}
            else:
                raise PclXlError(f"unexpected byte 0x{tag:02x} at offset {i - 1}")
    except (struct.error, IndexError) as exc:
        raise PclXlError(f"truncated PCL XL stream near offset {i}") from exc


def read_pclxl_pages(data: bytes, result: PclPages) -> None:
    """Count pages and report BeginPage/EndPage settings of one PCL XL segment."""
    for operator, attributes in iter_pclxl(data):
        if operator == XL_BEGIN_PAGE:
            page = str(result.page_count + 1)
            for attribute_id, name in ((0x25, "paper_format"), (0x26, "paper_source"), (0x28, "page_orientation")):
                if attribute_id in attributes:
                    result.commands.append({"page": page, "property": name,
                                            "value": str(attributes[attribute_id])})
            if 0x35 in attributes:
                mode = attributes[0x35]
                duplex = _XL_DUPLEX.get(mode, f"duplex_{mode}") if isinstance(mode, int) else f"duplex_{mode}"
                result.commands.append({"page": page, "property": "duplex", "value": duplex})
            elif 0x34 in attributes:
                result.commands.append({"page": page, "property": "duplex", "value": "simplex"})
        elif operator == XL_END_PAGE:
            result.page_count += 1
            if 0x31 in attributes:
                result.commands.append({"page": str(result.page_count), "property": "copies",
                                        "value": str(attributes[0x31])})


def read_pcl_pages(data: bytes) -> PclPages:
    """Pages, per-page settings and languages of a PCL 5 / PCL XL job."""
    result = PclPages()
    for segment in split_jobs(data):
        if segment.language is None or not segment.data:
            continue
        result.languages.append(segment.language)
        if segment.language == "PCLXL":
            read_pclxl_pages(segment.data, result)
        elif segment.language == "PCL5":
            read_pcl5_pages(segment.data, result)
        else:
            raise ValueError(f"Unsupported language {segment.language} in a PCL job")
    return result


# --------------------------------------------------------------------------
# PostScript (DSC)
# --------------------------------------------------------------------------

@dataclass
class PostscriptJob:
    pjl_commands: List[str] = field(default_factory=list)
    header: List[Dict[str, str]] = field(default_factory=list)
    features: List[Dict[str, str]] = field(default_factory=list)
    pages: List[Dict[str, str]] = field(default_factory=list)
    trailer: List[Dict[str, str]] = field(default_factory=list)
    page_count: Optional[int] = None


_LINE_END = re.compile(rb"\r\n|\r|\n")
_BEGIN_DATA = re.compile(r"(\d+)(?:\s+\S+)?(?:\s+(Lines|Bytes))?", re.IGNORECASE)


def _iter_lines(data: bytes) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(line_start, line_end, next_line_start)`` for every line."""
    pos = 0
    n = len(data)
    for match in _LINE_END.finditer(data):
        yield pos, match.start(), match.end()
        pos = match.end()
    if pos < n:
        yield pos, n, n


def _dsc_comment(line: str) -> Tuple[str, str]:
    body = line[2:]
    name, sep, value = body.partition(":")
    if not sep:
        return body.strip(), ""
    return name.strip(), value.strip()


_HEADER_PAGE_COMMENTS = frozenset({"Pages", "PageOrder"})


def _is_body_comment(name: str) -> bool:
    """DSC comments that can only appear after the header."""
    if name in _HEADER_PAGE_COMMENTS:
        return False
    return name.startswith(("Begin", "End", "Page", "Include")) or name in ("Trailer", "EOF")


def _scan_dsc(data: bytes, job: PostscriptJob) -> None:
    start = data.find(b"%!")
    if start < 0:
        return
    data = data[start:]
    lines = list(_iter_lines(data))
    in_header = True
    depth = 0
    current_page: Optional[int] = None
    in_trailer = False
    skip_until = 0
    previous: Optional[Dict[str, str]] = None
    pages = job.page_count or 0
    first = True
    for line_start, line_end, next_start in lines:
        if line_start < skip_until:
            continue
        raw = data[line_start:line_end]
        line = raw.decode("latin-1")
        if first:
            first = False
            if line.startswith("%!"):
                job.header.append({"property": "Version", "value": line[2:].strip()})
            continue
        if in_header:
            if line.startswith("%%+") and previous is not None:
                previous["value"] = (previous["value"] + " " + line[3:].strip()).strip()
                continue
            if line.startswith("%%") and not line.startswith("%%EndComments"):
                name, value = _dsc_comment(line)
                if _is_body_comment(name):
                    in_header = False      # body comment without %%EndComments
                else:
                    previous = {"property": name, "value": value}
                    job.header.append(previous)
                    continue
            else:
                in_header = False
                if line.startswith("%%EndComments"):
                    continue
        if not line.startswith("%%"):
            continue
        name, value = _dsc_comment(line)
        if name in ("BeginData", "BeginBinary"):
            match = _BEGIN_DATA.match(value)
            if match:
                amount = int(match.group(1))
                unit = (match.group(2) or "Bytes").lower() if name == "BeginData" else "bytes"
                if unit == "lines":
                    index = [k for k, entry in enumerate(lines) if entry[0] == next_start]
                    if index and index[0] + amount < len(lines):
                        skip_until = lines[index[0] + amount][0]
                    else:
                        skip_until = len(data)
                else:
                    skip_until = next_start + amount
            continue
        if name == "BeginDocument":
            depth += 1
            continue
        if name == "EndDocument":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if name == "Page":
            pages += 1
            current_page = pages
            continue
        if name == "Trailer":
            in_trailer = True
            current_page = None
            continue
        if name in ("BeginFeature", "IncludeFeature"):
            key, _, option = value.partition(" ")
            entry = {"property": key.strip(), "value": option.strip()}
            if current_page is None:
                job.features.append(entry)
            else:
                job.pages.append({"page": str(current_page), **entry})
            continue
        if in_trailer:
            if name != "EOF":
                job.trailer.append({"property": name, "value": value})
            continue
        if current_page is not None and name.startswith("Page") and name not in ("PageTrailer",):
            job.pages.append({"page": str(current_page), "property": name, "value": value})
    job.page_count = pages
    # resolve (atend) header values from the trailer
    trailer_values = {entry["property"]: entry["value"] for entry in job.trailer}
    for entry in job.header:
        if entry["value"] == "(atend)" and entry["property"] in trailer_values:
            entry["value"] = trailer_values[entry["property"]]


def read_postscript(data: bytes) -> PostscriptJob:
    """Read PJL commands and DSC structure of a PostScript job.

    ``page_count`` is the number of top-level ``%%Page:`` comments, or
    ``None`` when the job has no DSC page structure.
    """
    job = PostscriptJob()
    segments = split_jobs(data)
    postscript = [s for s in segments if s.language == "POSTSCRIPT"]
    for segment in segments:
        job.pjl_commands.extend(segment.pjl)
    if not postscript and segments and segments[0].language is None:
        return job
    if not postscript:
        postscript = [JobSegment("POSTSCRIPT", [], data)]
    for segment in postscript:
        if job.header:
            # a second PostScript job: its pages continue the count
            extra = PostscriptJob(page_count=job.page_count)
            _scan_dsc(segment.data, extra)
            job.pages.extend(extra.pages)
            job.features.extend(extra.features)
            job.trailer.extend(extra.trailer)
            job.page_count = extra.page_count
        else:
            _scan_dsc(segment.data, job)
    if not job.page_count:
        job.page_count = None
    return job


GHOSTSCRIPT_COMMANDS = ("gs", "gswin64c", "gswin32c", "ghostscript")
_PAGE_MARKER = "DOCTEST-PAGE"
_COUNT_PAGES = ("<< /EndPage { exch pop 2 ne dup { (" + _PAGE_MARKER + "\\n) print flush } if } bind >> "
                "setpagedevice")


def find_ghostscript() -> Optional[str]:
    for name in GHOSTSCRIPT_COMMANDS:
        path = shutil.which(name)
        if path:
            return path
    return None


def count_postscript_pages(path: str) -> int:
    """Count the pages a PostScript job produces by interpreting it."""
    gs = find_ghostscript()
    if gs is None:
        raise AssertionError(
            f"Cannot determine the page count of {path}: the job has no DSC page "
            "structure (%%Page comments) and Ghostscript is required to count its "
            "pages, but no ghostscript executable was found in path.")
    try:
        result = subprocess.run(
            [gs, "-q", "-dSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=nullpage",
             "-c", _COUNT_PAGES, "-f", str(path)],
            capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AssertionError(f"Cannot determine the page count of {path}: {exc}") from exc
    output = (result.stdout + result.stderr).decode("latin-1", errors="replace")
    if result.returncode != 0:
        raise AssertionError(
            f"Cannot determine the page count of {path}: Ghostscript exited with "
            f"{result.returncode}: {output.strip()[-800:]}")
    return output.count(_PAGE_MARKER)


@dataclass
class PageDeviceCapture:
    """Page-device settings a PostScript job requested while it ran."""

    requests: List[Tuple[int, Dict[str, str]]] = field(default_factory=list)   # (page, {key: value})
    page_count: int = 0
    error: Optional[str] = None


CAPTURED_PAGE_DEVICE_KEYS = ("PageSize", "Duplex", "Tumble", "NumCopies", "MediaPosition",
                             "MediaType", "ManualFeed", "OutputType")
_REQUEST_MARKER = "DOCTEST-SPD"
# EndPage marks every printed page; the redefined setpagedevice prints each
# requested dictionary before calling the original operator. Page numbers are
# derived from the order of the markers, not from a PostScript counter, because
# per-page save/restore would undo such a counter.
_CAPTURE_PAGE_DEVICE = (
    "/__doctest_spd /setpagedevice load def "
    "<< /EndPage { exch pop 2 ne dup { (" + _PAGE_MARKER + "\n) print flush } if } bind >> __doctest_spd "
    "/setpagedevice { dup (" + _REQUEST_MARKER + ") print "
    "{ exch ( ) print =only (=) print ==only } forall (\n) print flush __doctest_spd } bind def"
)
_REQUEST_VALUE = re.compile(r" ([A-Za-z]+)=(\[[^\]]*\]|\([^)]*\)|\S+)")


def capture_postscript_page_device(path: str) -> PageDeviceCapture:
    """Run a PostScript job in Ghostscript and record its page-device requests.

    Never raises for interpreter problems: the reason is returned in ``error``
    so callers can fall back to DSC comments.
    """
    gs = find_ghostscript()
    if gs is None:
        return PageDeviceCapture(error="Ghostscript not found")
    try:
        result = subprocess.run(
            [gs, "-q", "-dSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=nullpage",
             "-c", _CAPTURE_PAGE_DEVICE, "-f", str(path)],
            capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return PageDeviceCapture(error=f"Ghostscript could not run: {exc}")
    output = result.stdout.decode("latin-1", errors="replace")
    capture = PageDeviceCapture()
    for line in output.splitlines():
        if line.startswith(_PAGE_MARKER):
            capture.page_count += 1
        elif line.startswith(_REQUEST_MARKER):
            values = {key: value for key, value in _REQUEST_VALUE.findall(line[len(_REQUEST_MARKER):])
                      if key in CAPTURED_PAGE_DEVICE_KEYS}
            if values:
                capture.requests.append((capture.page_count + 1, values))
    if result.returncode != 0:
        errors = (result.stderr.decode("latin-1", errors="replace") or output).strip()
        capture.error = f"Ghostscript exited with {result.returncode}: {errors[-400:]}"
    return capture
