import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
import uuid

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("http", "https", "ftp")
DEFAULT_MAX_SIZE = 100 * 1024 * 1024  # 100 MB


def is_url(url, allowed_schemes=None):
    """
    Check if the provided string is a valid URL.

    Args:
        url: The string to check.
        allowed_schemes: Optional tuple/list of allowed URL schemes.
            If provided, the URL's scheme must be in this collection.
            If None, any scheme with a netloc is accepted (backward compatible).
    """
    try:
        result = urllib.parse.urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        if allowed_schemes is not None and result.scheme not in allowed_schemes:
            return False
        return True
    except (ValueError, AttributeError):
        return False


def convert_github_blob_to_raw(url):
    """
    Convert a GitHub blob URL to a raw content URL.

    For example:
        https://github.com/user/repo/blob/main/path/file.png
    becomes:
        https://raw.githubusercontent.com/user/repo/main/path/file.png

    If the URL is not a GitHub blob URL, returns the original URL unchanged.
    """
    pattern = r"^https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$"
    match = re.match(pattern, url)
    if match:
        user = match.group(1)
        repo = match.group(2)
        rest = match.group(3)
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{rest}"
        logger.info("Converted GitHub blob URL to raw URL: %s", raw_url)
        return raw_url
    return url


def download_file_from_url(url, directory=None, filename=None, max_size=None):
    """
    Download the file from the url and save it in the provided directory.
    If directory is None, save it in a temp directory.
    Save the file with the provided filename.
    If the filename is None, derive a filename from the URL using a UUID
    and the file extension from the URL path.
    GitHub blob URLs are automatically converted to raw content URLs.

    Args:
        url: The URL to download the file from.
        directory: The directory to save the file in. Defaults to temp dir.
        filename: The filename to save the file as. Defaults to UUID + ext.
        max_size: Maximum allowed file size in bytes. Defaults to
            DEFAULT_MAX_SIZE (100 MB). If the downloaded file exceeds this
            size, it is deleted and a ValueError is raised.

    Raises:
        ValueError: If the URL scheme is not in ALLOWED_SCHEMES or if the
            downloaded file exceeds max_size.
    """
    # Validate URL scheme
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Allowed schemes: {', '.join(ALLOWED_SCHEMES)}"
        )

    if max_size is None:
        max_size = DEFAULT_MAX_SIZE

    if directory is None:
        directory = tempfile.gettempdir()
    if filename is None:
        _, ext = os.path.splitext(os.path.basename(parsed.path))
        if ext:
            filename = str(uuid.uuid4()) + ext
        else:
            logger.warning(
                "Could not determine file extension from URL: %s", url
            )
            filename = str(uuid.uuid4())
    file_path = os.path.join(directory, filename)
    download_url = convert_github_blob_to_raw(url)
    urllib.request.urlretrieve(download_url, file_path)

    # Validate downloaded file size
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        os.remove(file_path)
        raise ValueError(
            f"Downloaded file size ({file_size} bytes) exceeds maximum "
            f"allowed size ({max_size} bytes). File has been deleted."
        )

    # Sniff for HTML content when we likely expected a document/image
    _warn_if_html_content(file_path, url)

    return file_path


def _warn_if_html_content(file_path, url):
    """
    Check if the downloaded file appears to be HTML when we likely expected
    something else (e.g., an image, PDF, or other binary document).

    Logs a warning if HTML markers are detected but does not raise an error,
    since the content could be legitimately HTML.
    """
    _, ext = os.path.splitext(file_path)
    html_extensions = {".html", ".htm"}
    if ext.lower() in html_extensions:
        return  # HTML was expected, no warning needed

    try:
        with open(file_path, "rb") as f:
            header = f.read(512)
        # Look for common HTML markers in the first 512 bytes
        header_lower = header.lstrip().lower()
        if header_lower.startswith(b"<!doctype") or header_lower.startswith(
            b"<html"
        ):
            logger.warning(
                "Downloaded file from %s appears to contain HTML content "
                "but was expected to be a different file type (extension: %s). "
                "This may indicate a redirect to a login page or error page.",
                url,
                ext if ext else "(none)",
            )
    except OSError:
        pass  # If we can't read the file for sniffing, skip the check


def get_filename_from_url(url):
    """
    Check if the URL contains a valid filename.
    If the URL contains a valid filename, return the filename.
    If the URL does not contain a filename, return None.

    Uses proper URL parsing to strip query parameters and fragments.
    """
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    basename = os.path.basename(parsed.path)
    if not basename or "." not in basename:
        return None
    return basename
