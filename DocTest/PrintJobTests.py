import logging
from pprint import pformat
import re
from typing import Dict, List, Optional, Union
from deepdiff import DeepDiff
from robot.api import logger as robot_logger
from robot.api.deco import keyword, not_keyword
from DocTest.Downloader import is_url, download_file_from_url
from DocTest.PrintJobParsing import count_postscript_pages, read_pcl_pages, read_postscript
from DocTest.PrintJobTicket import read_ticket, ticket_differences, ticket_mismatches

logger = logging.getLogger(__name__)

ROBOT_AUTO_KEYWORDS = False

# Values that differ on every run of the same job (DSC comments and PJL
# variables / key=value attributes); compared case-insensitively.
DEFAULT_IGNORED_PROPERTIES = (
    "CreationDate", "Title", "For", "Routing", "Invocation",
    "NAME", "JOBNAME", "USERNAME", "USERID", "HOSTID", "HOSTNAME",
    "LHOSTID", "LHOSTJOBID", "UPN", "DATE", "TIME",
)

_PJL_PAIR = re.compile(r'\b([A-Za-z][A-Za-z0-9_:]*)\s*=\s*("[^"]*"|\S+)')
_EMPTY_PJL = re.compile(r"(?i)^@PJL(\s+(SET|DEFAULT))?$")


class PrintJob(object):


    def __init__(self, jobtype, properties):
        self.jobtype = jobtype
        self.properties = properties

@not_keyword
def chop(text, prefix=None, suffix=None):
    if prefix is not None:
        text=re.sub(prefix, '', text)
    if suffix is not None:
        text=re.sub(suffix, '', text)
    return text


def _read_bytes(filename):
    if is_url(filename):
        filename = download_file_from_url(filename)
    with open(filename, "rb") as f:
        return filename, f.read()


@keyword
def get_pcl_print_job(filename):
    """
    Parses a pcl file and returns a PrintJob object.

    ``filename`` is a pcl file. It can be a local file or a url.

    Both PCL 5 and PCL XL (PCL 6) jobs are supported, including jobs that
    contain several PJL-delimited sections. The returned properties are:

    - ``pcl_commands``: page-setup commands per page, as dictionaries with
      ``page``, ``property`` and ``value``. The properties are
      ``paper_format``, ``paper_source``, ``page_orientation``, ``duplex``
      and ``copies``. Values are reported as the job states them: PCL 5
      command values (e.g. paper format ``26`` for A4), PCL XL attribute
      values (e.g. media size ``2`` for A4) and duplex ``simplex``,
      ``duplex_horizontal`` or ``duplex_vertical`` for PCL XL.
    - ``page_count``: number of pages the job prints.
    - ``language``: page-description language of every section
      (``PCL5`` or ``PCLXL``).
    """
    _, data = _read_bytes(filename)
    pages = read_pcl_pages(data)
    return PrintJob('pcl', [
        {'property': 'pcl_commands', 'value': pages.commands},
        {'property': 'page_count', 'value': pages.page_count},
        {'property': 'language', 'value': pages.languages},
    ])

@keyword
def get_postscript_print_job(filename):
    """
    Parses a postscript file and returns a PrintJob object.

    ``filename`` is a postscript file. It can be a local file or a url.

    Jobs wrapped in PJL, with any line endings and with or without the
    optional DSC sections are supported. The returned properties are:

    - ``pjl_commands``: the ``@PJL`` lines around the job.
    - ``header``: the header comments (``Version`` holds the ``%!`` line).
    - ``features``: requested document features, e.g. ``*Duplex``.
    - ``pages``: page comments and page features per page.
    - ``trailer``: the trailer comments.
    - ``page_count``: number of pages. A job without ``%%Page`` comments
      is interpreted with Ghostscript to count its pages; without
      Ghostscript such a job raises an error.
    """
    path, data = _read_bytes(filename)
    job = read_postscript(data)
    page_count = job.page_count
    if page_count is None:
        page_count = count_postscript_pages(path)

    properties = []
    properties.append({'property':'pjl_commands', 'value':job.pjl_commands})
    properties.append({'property':'header', 'value':job.header})
    properties.append({'property':'features', 'value':job.features})
    properties.append({'property':'pages', 'value':job.pages})
    properties.append({'property':'trailer', 'value':job.trailer})
    properties.append({'property':'page_count', 'value':page_count})
    postscript_printjob = PrintJob('postscript', properties)
    return postscript_printjob

@keyword
def compare_print_jobs(type, reference_file, test_file, ignore_properties: Union[List[str], str, None] = None):
    """Compares several print job metadata/properties of ``reference_file`` and ``test_file``.

        ``type`` can either be ``pcl`` or ``ps`` (postscript).

        The compared properties depend on the ``type``.

            PCL (PCL 5 and PCL XL)

                - Paper Source
                - Copies
                - Duplex
                - Page Orientation
                - Paper Format
                - Page Count
                - Language

            Postscript

                - PJL Commands
                - Header
                - Features
                - Pages
                - Trailer
                - Page Count

        Values that change on every run of the same job are ignored by
        default: the DSC comments ``CreationDate``, ``Title``, ``For``,
        ``Routing`` and ``Invocation``, and the PJL values ``NAME``,
        ``JOBNAME``, ``USERNAME``, ``USERID``, ``HOSTID``, ``HOSTNAME``,
        ``LHOSTID``, ``LHOSTJOBID``, ``UPN``, ``DATE`` and ``TIME``.

        ``ignore_properties`` names further properties to ignore: a list or
        a comma-separated string of property groups (e.g. ``trailer``), DSC
        comment or feature names (e.g. ``Creator``, ``*InputSlot``) or PJL
        variables. Names are matched case-insensitively and extend the
        defaults.

        Result is passed if all properties are equal.

        ``reference_file`` and ``test_file`` may be .ps or .pcl files


        Examples:
        | `Compare Print Jobs`    pcl    reference.pcl    candidate.pcl    #Performs a property comparison of both files
        | `Compare Print Jobs`    ps    reference.ps    candidate.ps    #Performs a property comparison of both files
        | `Compare Print Jobs`    ps    reference.ps    candidate.ps    ignore_properties=Creator,trailer    #Also ignores the Creator comment and the trailer


        """
    if type=='ps':
        reference_print_job = get_postscript_print_job(reference_file)
        test_print_job = get_postscript_print_job(test_file)
    elif type=='pcl':
        reference_print_job = get_pcl_print_job(reference_file)
        test_print_job = get_pcl_print_job(test_file)
    else:
        raise ValueError(
            f"Unsupported print job type '{type}'. Supported types are: 'pcl', 'ps'."
        )
    compare_properties(reference_print_job, test_print_job, ignore_properties)


def _normalize_name(name):
    return str(name).strip().lstrip("*").lower()


def _ignored_names(ignore_properties):
    names = set(_normalize_name(n) for n in DEFAULT_IGNORED_PROPERTIES)
    if ignore_properties:
        if isinstance(ignore_properties, str):
            ignore_properties = ignore_properties.split(",")
        names.update(_normalize_name(n) for n in ignore_properties if str(n).strip())
    return names


def _filter_pjl(line, ignored):
    filtered = _PJL_PAIR.sub(
        lambda m: "" if _normalize_name(m.group(1)) in ignored else m.group(0), line)
    filtered = re.sub(r"\s+", " ", filtered).strip()
    return None if _EMPTY_PJL.match(filtered) else filtered


def _filter_entry(entry, ignored):
    if isinstance(entry, dict):
        name = entry.get("property", entry.get("feature"))
        if name is not None and _normalize_name(name) in ignored:
            return None
        return entry
    if isinstance(entry, str) and entry.upper().startswith("@PJL"):
        return _filter_pjl(entry, ignored)
    return entry


def _apply_ignores(properties, ignored):
    result = []
    for item in properties:
        if _normalize_name(item["property"]) in ignored:
            continue
        value = item["value"]
        if isinstance(value, list):
            value = [e for e in (_filter_entry(e, ignored) for e in value) if e is not None]
        result.append({"property": item["property"], "value": value})
    return result


def _as_list(value):
    return value if isinstance(value, list) else [value]


def compare_properties(reference_print_job, test_print_job, ignore_properties=None):
    properties_are_equal = True
    list_difference = []
    ignored = _ignored_names(ignore_properties)
    reference_properties = _apply_ignores(reference_print_job.properties, ignored)
    test_properties = _apply_ignores(test_print_job.properties, ignored)
    robot_logger.info("Reference File:")
    robot_logger.info(pformat(reference_properties))

    robot_logger.info("Candidate File:")
    robot_logger.info(pformat(test_properties))

    robot_logger.info(pformat(DeepDiff(reference_properties, test_properties, verbose_level=2)))
    for reference_property_item in reference_properties:
        test_property_item = next((item for item in test_properties if item["property"] == reference_property_item["property"]), None)
        if test_property_item is None:
            properties_are_equal = False
            for x in _as_list(reference_property_item['value']):
                list_difference.append({'file':'reference', 'property':reference_property_item['property'], 'value':x})
            continue
        if reference_property_item['value']!=test_property_item['value']:
            properties_are_equal = False
            reference_values = _as_list(reference_property_item['value'])
            test_values = _as_list(test_property_item['value'])
            before = len(list_difference)

            for x in reference_values:
                if x not in test_values:
                    list_difference.append({'file':'reference', 'property':reference_property_item['property'], 'value':x})

            for x in test_values:
                if x not in reference_values:
                    list_difference.append({'file':'test', 'property':test_property_item['property'], 'value':x})

            if len(list_difference) == before:
                list_difference.append({'file':'both', 'property':reference_property_item['property'],
                                        'value':'same entries in a different order or multiplicity'})

    reference_property_names = {item["property"] for item in reference_properties}
    for test_property_item in test_properties:
        if test_property_item["property"] not in reference_property_names:
            properties_are_equal = False
            for x in _as_list(test_property_item['value']):
                list_difference.append({'file':'test', 'property':test_property_item['property'], 'value':x})

    if not properties_are_equal:
        robot_logger.info(pformat(list_difference))
        raise AssertionError('The compared print jobs are different.')

@keyword
def check_print_job_property(print_job, property, value):
    """Checks if one specific ``property`` of a ``print_job`` file has an expected ``value``.

        ``print_job`` is a print job returned by `Get Pcl Print Job` or
        `Get Postscript Print Job`.

        The possible ``property`` depends on the type of the print job.

            PCL

                - pcl_commands (Paper Source, Copies, Duplex, Page Orientation, Paper Format per page)
                - page_count
                - language

            Postscript

                - pjl_commands
                - header
                - features
                - pages
                - trailer
                - page_count

        For list properties the check passes if ``value`` is one of the
        entries. For ``page_count`` the check passes if the count equals
        ``value``.

        Examples:

        | ${value1} =     Create Dictionary     page=1     property=copies     value=1
        | ${value2} =     Create Dictionary     page=1     property=paper_source     value=2
        | `Check Print Job Property`     ${pcl_print_job}     pcl_commands     ${value1}
        | `Check Print Job Property`     ${pcl_print_job}     pcl_commands     ${value2}
        | `Check Print Job Property`     ${pcl_print_job}     page_count     2

        | ${value1} =     Create Dictionary     property=Copyright     value=ExampleCompany Inc.
        | ${value2} =     Create Dictionary     property=LanguageLevel     value=2
        | `Check Print Job Property`     ${postscript_print_job}     header     ${value1}
        | `Check Print Job Property`     ${postscript_print_job}     header     ${value2}

        """

    test_property_item = next((item for item in print_job.properties if item["property"] == property), None)
    if not test_property_item:
        raise AssertionError('The property does not exist:', property, value)
    actual = test_property_item['value']
    matches = value in actual if isinstance(actual, list) else str(value) == str(actual)
    if not matches:
        robot_logger.info(f"Expected property: {property}\n\nExpected value:\n{pformat(value)}\n\nActual value:\n{pformat(test_property_item['value'])}")
        raise AssertionError('The print job property check failed.')

def _ticket(print_job, mapping=None):
    if isinstance(print_job, dict):
        return print_job
    path, data = _read_bytes(print_job)
    return read_ticket(path, data, mapping).as_dict()


@keyword
def get_print_job_ticket(print_job, mapping: Union[Dict, str, None] = None):
    """Returns the normalized print settings ("job ticket") of a PCL 5, PCL XL or PostScript job.

    ``print_job`` is a print file (local path or url).

    The ticket is a dictionary:

    - ``languages``: page-description languages of the job (``PCL5``, ``PCLXL``, ``POSTSCRIPT``)
    - ``page_count``: number of pages
    - ``job``: ``resolution`` and ``color_mode``
    - ``pjl``: every PJL ``SET``/``DEFAULT`` variable of the job
    - ``pages``: one entry per page with ``paper_size``, ``orientation``, ``duplex``,
      ``copies``, ``input_tray``, ``output_bin`` and ``media_type``, plus the
      ``sources`` and ``raw`` values each setting was taken from
    - ``settings_source``: which inputs were used; for PostScript it states
      whether the page-device requests could be read with Ghostscript

    Each page shows the settings in effect when it prints: PJL defaults,
    overridden by the commands of the page-description language, carried
    forward from page to page until changed or reset.

    Values are normalized so jobs in different languages are comparable:
    paper sizes as names (``A4``, ``Letter``, ...) or ``<w>x<h>mm``,
    orientation ``portrait``/``landscape``/``reverse_portrait``/``reverse_landscape``,
    duplex ``simplex``/``long_edge``/``short_edge``. Trays, bins and media use
    generic names where the language defines them (``auto``, ``manual``,
    ``envelope``) and otherwise a language-qualified raw value such as
    ``pcl5:4`` or ``ps:Tray2``. A setting the job leaves to the printer is
    ``unspecified``.

    ``mapping`` translates printer-specific values into your own names: a
    dictionary or the path of a JSON file with the sections ``input_tray``,
    ``output_bin`` and ``media_type``, e.g.
    ``{"input_tray": {"pcl5:4": "drawer2", "ps:Tray2": "drawer2"}}``.

    Examples:
    | ${ticket}    `Get Print Job Ticket`    invoice.pcl
    | Should Be Equal    ${ticket}[pages][0][duplex]    long_edge
    """
    return _ticket(print_job, mapping)


@keyword
def compare_print_job_tickets(reference, candidate, ignore_properties: Union[List[str], str, None] = None,
                              mapping: Union[Dict, str, None] = None):
    """Compares the normalized print settings of two print jobs, page by page.

    ``reference`` and ``candidate`` are print files (or tickets returned by
    `Get Print Job Ticket`). They may use different languages, e.g. a PCL 5
    job and its PostScript or PCL XL rendition.

    Compared are the page count, ``resolution``, ``color_mode``, every PJL
    variable and all page settings. The keyword fails with one line per
    difference, e.g. ``page 3: input_tray tray2 → tray1``.

    The run-specific PJL values ignored by `Compare Print Jobs` are ignored
    here too. ``ignore_properties`` names more: settings (``input_tray``),
    ``pjl`` for all PJL variables, or single PJL variables.

    Different drivers emit different PJL, so cross-language comparisons
    usually ignore the job-level values:

    Examples:
    | `Compare Print Job Tickets`    reference.pcl    candidate.pcl
    | `Compare Print Job Tickets`    invoice.pcl    invoice.ps    ignore_properties=pjl,resolution,color_mode
    """
    reference_ticket = _ticket(reference, mapping)
    candidate_ticket = _ticket(candidate, mapping)
    robot_logger.info("Reference ticket:\n" + pformat(reference_ticket))
    robot_logger.info("Candidate ticket:\n" + pformat(candidate_ticket))
    differences = ticket_differences(reference_ticket, candidate_ticket, _ignored_names(ignore_properties))
    if differences:
        robot_logger.info("\n".join(differences))
        shown = differences[:20]
        if len(differences) > 20:
            shown.append(f"... and {len(differences) - 20} more")
        raise AssertionError("The compared print job tickets are different:\n" + "\n".join(shown))


@keyword
def print_job_ticket_should_contain(print_job, page: Optional[int] = None,
                                    mapping: Union[Dict, str, None] = None, **expected):
    """Checks expected print settings of a print job without a reference file.

    ``print_job`` is a print file or a ticket from `Get Print Job Ticket`.
    Name the expected settings as arguments: page settings (``paper_size``,
    ``orientation``, ``duplex``, ``copies``, ``input_tray``, ``output_bin``,
    ``media_type``) are checked on every page, or only on ``page``; job
    settings (``resolution``, ``color_mode``, ``page_count``) on the job.
    Values are compared case-insensitively.

    Examples:
    | `Print Job Ticket Should Contain`    statement.pcl    duplex=long_edge    paper_size=A4
    | `Print Job Ticket Should Contain`    statement.pcl    page=1    input_tray=auto
    """
    if not expected:
        raise ValueError("Name at least one expected setting, e.g. duplex=long_edge.")
    ticket = _ticket(print_job, mapping)
    mismatches = ticket_mismatches(ticket, expected, page)
    if mismatches:
        raise AssertionError("The print job ticket does not contain the expected settings:\n"
                             + "\n".join(mismatches))


def printTable(myDict, colList=None):
   """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
   If column names (colList) aren't specified, they will show in random order.
   Author: Thierry Husson - Use it as you want but don't blame me.
   """
   if type(myDict) is not list:
       myDict = [myDict]
   if not colList:
       colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict:
       myList.append([str(item[col] if item[col] is not None else '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList, strict=False)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   for item in myList:
       robot_logger.info(formatStr.format(*item))
