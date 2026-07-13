"""Unit tests for Downloader module."""

import io
import os
import logging
from unittest.mock import patch

import pytest

from DocTest.Downloader import (
    ALLOWED_SCHEMES,
    DEFAULT_MAX_SIZE,
    _warn_if_html_content,
    convert_github_blob_to_raw,
    download_file_from_url,
    get_filename_from_url,
    is_url,
)


def _payload_opener(payload=b"%PDF-1.4 test content"):
    """Return a fake for DocTest.Downloader._open_url serving ``payload``."""

    def _open(_url):
        return io.BytesIO(payload)

    return _open


class TestIsUrl:
    """Test cases for is_url function."""

    def test_valid_http_url(self):
        """Test with valid HTTP URL."""
        assert is_url("http://example.com") is True
        assert is_url("http://example.com/file.pdf") is True

    def test_valid_https_url(self):
        """Test with valid HTTPS URL."""
        assert is_url("https://example.com") is True
        assert is_url("https://example.com/path/file.txt") is True

    def test_valid_ftp_url(self):
        """Test with valid FTP URL."""
        assert is_url("ftp://ftp.example.com/file.zip") is True

    def test_invalid_url_no_scheme(self):
        """Test with invalid URL missing scheme."""
        assert is_url("example.com") is False
        assert is_url("www.example.com") is False

    def test_invalid_url_no_netloc(self):
        """Test with invalid URL missing netloc."""
        assert is_url("http://") is False
        assert is_url("https://") is False

    def test_invalid_url_local_path(self):
        """Test with local file paths."""
        assert is_url("/path/to/file.txt") is False
        assert is_url("C:\\path\\to\\file.txt") is False
        assert is_url("./relative/path.txt") is False

    def test_invalid_url_empty_string(self):
        """Test with empty string."""
        assert is_url("") is False

    def test_invalid_url_none(self):
        """Test with None."""
        assert is_url(None) is False

    def test_invalid_url_malformed(self):
        """Test with malformed URLs."""
        assert is_url("not a url") is False
        assert is_url("://missing-scheme") is False
        assert is_url("http:///missing-netloc") is False


class TestDownloadFileFromUrl:
    """Test cases for download_file_from_url function."""

    @patch("DocTest.Downloader._warn_if_html_content")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_with_default_params(
        self, mock_uuid, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test download with default directory and filename."""
        monkeypatch.setattr(
            "DocTest.Downloader.tempfile.gettempdir", lambda: str(tmp_path)
        )
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf"

        result = download_file_from_url(url)

        expected_path = os.path.join(str(tmp_path), "test-uuid.pdf")
        assert result == expected_path
        assert os.path.exists(expected_path)

    @patch("DocTest.Downloader._warn_if_html_content")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_with_custom_directory(
        self, mock_uuid, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test download with custom directory."""
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf"
        directory = str(tmp_path)

        result = download_file_from_url(url, directory=directory)

        expected_path = os.path.join(directory, "test-uuid.pdf")
        assert result == expected_path
        assert os.path.exists(expected_path)

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_download_with_custom_filename(self, mock_warn_html, tmp_path, monkeypatch):
        """Test download with custom filename."""
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        url = "https://example.com/test.pdf"
        directory = str(tmp_path)
        filename = "custom_name.pdf"

        result = download_file_from_url(url, directory=directory, filename=filename)

        expected_path = os.path.join(directory, filename)
        assert result == expected_path
        assert os.path.exists(expected_path)

    @patch("DocTest.Downloader._warn_if_html_content")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_url_without_extension(
        self, mock_uuid, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test download with URL without file extension.

        When the URL path has no extension, the filename should be a UUID
        with no extension (not a broken path fragment like 'com/path/file').
        """
        monkeypatch.setattr(
            "DocTest.Downloader.tempfile.gettempdir", lambda: str(tmp_path)
        )
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/path/file"

        result = download_file_from_url(url)

        expected_path = os.path.join(str(tmp_path), "test-uuid")
        assert result == expected_path

    @patch("DocTest.Downloader._warn_if_html_content")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_url_with_query_params(
        self, mock_uuid, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test download with URL containing query parameters.

        Query parameters should be stripped; only the clean extension
        should be used in the filename.
        """
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf?param=value"
        directory = str(tmp_path)

        result = download_file_from_url(url, directory=directory)

        expected_path = os.path.join(directory, "test-uuid.pdf")
        assert result == expected_path

    @patch("DocTest.Downloader._warn_if_html_content")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_url_with_fragment(
        self, mock_uuid, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test download with URL containing fragment identifier.

        Fragments should be stripped; only the clean extension
        should be used in the filename.
        """
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf#page=5"
        directory = str(tmp_path)

        result = download_file_from_url(url, directory=directory)

        expected_path = os.path.join(directory, "test-uuid.pdf")
        assert result == expected_path

    @patch("DocTest.Downloader._warn_if_html_content")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_github_blob_url_converts_to_raw(
        self, mock_uuid, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test that GitHub blob URLs are converted to raw URLs for download."""
        requested_urls = []

        def _open(url):
            requested_urls.append(url)
            return io.BytesIO(b"data")

        monkeypatch.setattr("DocTest.Downloader._open_url", _open)
        mock_uuid.return_value = "test-uuid"
        url = "https://github.com/user/repo/blob/main/path/file.png"
        directory = str(tmp_path)

        result = download_file_from_url(url, directory=directory)

        expected_path = os.path.join(directory, "test-uuid.png")
        assert result == expected_path
        expected_raw_url = (
            "https://raw.githubusercontent.com/user/repo/main/path/file.png"
        )
        assert requested_urls == [expected_raw_url]


class TestGetFilenameFromUrl:
    """Test cases for get_filename_from_url function."""

    def test_url_with_filename(self):
        """Test URL with valid filename."""
        url = "https://example.com/path/document.pdf"
        result = get_filename_from_url(url)
        assert result == "document.pdf"

    def test_url_with_complex_filename(self):
        """Test URL with complex filename."""
        url = "https://example.com/path/to/my_document_v1.2.pdf"
        result = get_filename_from_url(url)
        assert result == "my_document_v1.2.pdf"

    def test_url_without_filename(self):
        """Test URL without filename (ends with directory)."""
        url = "https://example.com/path/directory"
        result = get_filename_from_url(url)
        assert result is None

    def test_url_with_trailing_slash(self):
        """Test URL with trailing slash."""
        url = "https://example.com/path/"
        result = get_filename_from_url(url)
        assert result is None

    def test_url_root_domain(self):
        """Test URL with root domain only.

        When the URL has no path component, the basename of the path is empty,
        so the function should return None (not the domain like 'example.com').
        """
        url = "https://example.com"
        result = get_filename_from_url(url)
        assert result is None

    def test_url_with_query_params(self):
        """Test URL with query parameters.

        Query parameters should be stripped from the filename.
        """
        url = "https://example.com/document.pdf?download=true"
        result = get_filename_from_url(url)
        assert result == "document.pdf"

    def test_url_with_fragment(self):
        """Test URL with fragment identifier.

        Fragments should be stripped from the filename.
        """
        url = "https://example.com/document.pdf#page=1"
        result = get_filename_from_url(url)
        assert result == "document.pdf"

    def test_none_url(self):
        """Test with None URL."""
        result = get_filename_from_url(None)
        assert result is None

    def test_empty_string_url(self):
        """Test with empty string URL."""
        result = get_filename_from_url("")
        assert result is None

    def test_url_with_multiple_dots(self):
        """Test URL with multiple dots in filename."""
        url = "https://example.com/archive.tar.gz"
        result = get_filename_from_url(url)
        assert result == "archive.tar.gz"

    def test_url_with_encoded_characters(self):
        """Test URL with encoded characters."""
        url = "https://example.com/my%20document.pdf"
        result = get_filename_from_url(url)
        assert result == "my%20document.pdf"


class TestConvertGithubBlobToRaw:
    """Test cases for convert_github_blob_to_raw function."""

    def test_github_blob_url(self):
        """Test conversion of a standard GitHub blob URL."""
        url = "https://github.com/user/repo/blob/main/path/to/file.py"
        result = convert_github_blob_to_raw(url)
        assert result == (
            "https://raw.githubusercontent.com/user/repo/main/path/to/file.py"
        )

    def test_github_blob_url_with_branch(self):
        """Test conversion with a specific branch name."""
        url = "https://github.com/user/repo/blob/feature-branch/src/app.js"
        result = convert_github_blob_to_raw(url)
        assert result == (
            "https://raw.githubusercontent.com/user/repo/feature-branch/src/app.js"
        )

    def test_github_blob_url_with_commit_sha(self):
        """Test conversion with a commit SHA."""
        url = "https://github.com/user/repo/blob/abc123def/README.md"
        result = convert_github_blob_to_raw(url)
        assert result == (
            "https://raw.githubusercontent.com/user/repo/abc123def/README.md"
        )

    def test_non_github_url_unchanged(self):
        """Test that non-GitHub URLs are returned unchanged."""
        url = "https://example.com/path/to/file.pdf"
        result = convert_github_blob_to_raw(url)
        assert result == url

    def test_github_non_blob_url_unchanged(self):
        """Test that GitHub URLs without /blob/ are returned unchanged."""
        url = "https://github.com/user/repo/tree/main/src"
        result = convert_github_blob_to_raw(url)
        assert result == url

    def test_github_raw_url_unchanged(self):
        """Test that already-raw GitHub URLs are returned unchanged."""
        url = "https://raw.githubusercontent.com/user/repo/main/file.py"
        result = convert_github_blob_to_raw(url)
        assert result == url

    def test_github_blob_url_http(self):
        """Test conversion of HTTP (not HTTPS) GitHub blob URL."""
        url = "http://github.com/user/repo/blob/main/file.txt"
        result = convert_github_blob_to_raw(url)
        assert result == (
            "https://raw.githubusercontent.com/user/repo/main/file.txt"
        )


class TestSchemeValidation:
    """Test cases for URL scheme validation in download_file_from_url."""

    def test_file_scheme_rejected(self):
        """Test that file:// URLs are rejected."""
        with pytest.raises(ValueError, match="scheme 'file' is not allowed"):
            download_file_from_url("file:///etc/passwd")

    def test_data_scheme_rejected(self):
        """Test that data: URLs are rejected."""
        with pytest.raises(ValueError, match="scheme 'data' is not allowed"):
            download_file_from_url("data:text/html,<h1>hello</h1>")

    def test_javascript_scheme_rejected(self):
        """Test that javascript: URLs are rejected."""
        with pytest.raises(ValueError, match="scheme 'javascript' is not allowed"):
            download_file_from_url("javascript:alert(1)")

    def test_empty_scheme_rejected(self):
        """Test that URLs with no scheme are rejected."""
        with pytest.raises(ValueError, match="scheme '' is not allowed"):
            download_file_from_url("//example.com/test.pdf")

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_http_scheme_allowed(self, mock_warn_html, tmp_path, monkeypatch):
        """Test that http:// URLs are allowed."""
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        result = download_file_from_url(
            "http://example.com/test.pdf",
            directory=str(tmp_path),
            filename="test.pdf",
        )
        assert result == os.path.join(str(tmp_path), "test.pdf")

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_https_scheme_allowed(self, mock_warn_html, tmp_path, monkeypatch):
        """Test that https:// URLs are allowed."""
        monkeypatch.setattr("DocTest.Downloader._open_url", _payload_opener())
        result = download_file_from_url(
            "https://example.com/test.pdf",
            directory=str(tmp_path),
            filename="test.pdf",
        )
        assert result == os.path.join(str(tmp_path), "test.pdf")

    def test_ftp_scheme_rejected(self, tmp_path):
        """Test that ftp:// URLs are rejected (hardened default)."""
        with pytest.raises(ValueError, match="scheme 'ftp' is not allowed"):
            download_file_from_url(
                "ftp://ftp.example.com/test.zip",
                directory=str(tmp_path),
                filename="test.zip",
            )

    @pytest.mark.parametrize(
        "location,expected",
        [
            # ftp is allowed by the stdlib redirect handler; our handler must
            # reject it because it is no longer in ALLOWED_SCHEMES.
            ("ftp://internal.example.com/secret.zip", ValueError),
            # file is rejected by the stdlib itself (HTTPError); either way
            # the transfer must abort and leave no file behind.
            ("file:///etc/passwd", Exception),
        ],
    )
    def test_redirect_to_disallowed_scheme_rejected(self, tmp_path, location, expected):
        """A redirect hop to a non-allowlisted scheme must abort the download."""
        import http.server
        import threading

        class _RedirectHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", location)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/test.pdf"
            with pytest.raises(expected):
                download_file_from_url(
                    url, directory=str(tmp_path), filename="redirected.pdf"
                )
            assert not (tmp_path / "redirected.pdf").exists()
        finally:
            server.shutdown()
            server.server_close()


class TestIsUrlWithAllowedSchemes:
    """Test cases for is_url with the allowed_schemes parameter."""

    def test_default_no_scheme_filter(self):
        """Test backward compatibility: without allowed_schemes, any scheme works."""
        assert is_url("ftp://ftp.example.com") is True
        assert is_url("http://example.com") is True
        assert is_url("https://example.com") is True
        assert is_url("custom://some.host") is True

    def test_allowed_schemes_filter(self):
        """Test that allowed_schemes restricts accepted schemes."""
        assert is_url("https://example.com", allowed_schemes=("https",)) is True
        assert is_url("http://example.com", allowed_schemes=("https",)) is False
        assert is_url("ftp://ftp.example.com", allowed_schemes=("https",)) is False

    def test_allowed_schemes_multiple(self):
        """Test with multiple allowed schemes."""
        schemes = ("http", "https")
        assert is_url("http://example.com", allowed_schemes=schemes) is True
        assert is_url("https://example.com", allowed_schemes=schemes) is True
        assert is_url("ftp://ftp.example.com", allowed_schemes=schemes) is False

    def test_allowed_schemes_with_module_constant(self):
        """Test with the module-level ALLOWED_SCHEMES constant."""
        assert is_url("http://example.com", allowed_schemes=ALLOWED_SCHEMES) is True
        assert is_url("https://example.com", allowed_schemes=ALLOWED_SCHEMES) is True
        assert is_url("ftp://ftp.example.com", allowed_schemes=ALLOWED_SCHEMES) is False
        assert is_url("custom://some.host", allowed_schemes=ALLOWED_SCHEMES) is False

    def test_invalid_url_still_rejected_with_schemes(self):
        """Test that invalid URLs are still rejected even with allowed_schemes."""
        assert is_url("not-a-url", allowed_schemes=ALLOWED_SCHEMES) is False
        assert is_url("", allowed_schemes=ALLOWED_SCHEMES) is False
        assert is_url(None, allowed_schemes=ALLOWED_SCHEMES) is False


class TestFileSizeValidation:
    """Test cases for file size validation in download_file_from_url."""

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_oversized_file_rejected(self, mock_warn_html, tmp_path, monkeypatch):
        """Test that downloads exceeding max_size abort and leave no file."""
        monkeypatch.setattr(
            "DocTest.Downloader._open_url", _payload_opener(b"x" * 2048)
        )
        url = "https://example.com/large_file.bin"

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            download_file_from_url(
                url, directory=str(tmp_path), filename="large.bin", max_size=1024
            )

        # The partial file should have been deleted
        assert not (tmp_path / "large.bin").exists()

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_file_within_default_max_size_accepted(
        self, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test that files within DEFAULT_MAX_SIZE are accepted."""
        monkeypatch.setattr(
            "DocTest.Downloader._open_url", _payload_opener(b"x" * 4096)
        )
        url = "https://example.com/file.pdf"

        result = download_file_from_url(
            url, directory=str(tmp_path), filename="file.pdf"
        )
        assert result == os.path.join(str(tmp_path), "file.pdf")

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_custom_max_size(self, mock_warn_html, tmp_path, monkeypatch):
        """Test that a custom max_size is respected."""
        monkeypatch.setattr(
            "DocTest.Downloader._open_url", _payload_opener(b"x" * 2048)
        )
        url = "https://example.com/file.pdf"

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            download_file_from_url(
                url, directory=str(tmp_path), filename="file.pdf", max_size=1024
            )

    @patch("DocTest.Downloader._warn_if_html_content")
    def test_file_at_exact_max_size_accepted(
        self, mock_warn_html, tmp_path, monkeypatch
    ):
        """Test that a file exactly at max_size is accepted (not exceeded)."""
        monkeypatch.setattr(
            "DocTest.Downloader._open_url", _payload_opener(b"x" * 1024)
        )
        url = "https://example.com/file.pdf"

        result = download_file_from_url(
            url, directory=str(tmp_path), filename="file.pdf", max_size=1024
        )
        assert result == os.path.join(str(tmp_path), "file.pdf")

    def test_default_max_size_constant(self):
        """Test that DEFAULT_MAX_SIZE is 100 MB."""
        assert DEFAULT_MAX_SIZE == 100 * 1024 * 1024


class TestHtmlContentDetection:
    """Test cases for HTML content detection warning."""

    def test_html_doctype_detected(self, tmp_path):
        """Test that <!DOCTYPE html> content triggers a warning."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"<!DOCTYPE html><html><body>Error</body></html>")

        with patch("DocTest.Downloader.logger") as mock_logger:
            _warn_if_html_content(str(test_file), "https://example.com/test.pdf")
            mock_logger.warning.assert_called_once()
            assert "appears to contain HTML" in mock_logger.warning.call_args[0][0]

    def test_html_tag_detected(self, tmp_path):
        """Test that <html> content triggers a warning."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"<html><head><title>Login</title></head></html>")

        with patch("DocTest.Downloader.logger") as mock_logger:
            _warn_if_html_content(str(test_file), "https://example.com/test.pdf")
            mock_logger.warning.assert_called_once()

    def test_html_with_leading_whitespace_detected(self, tmp_path):
        """Test that HTML content with leading whitespace is still detected."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"  \n  <!DOCTYPE html><html><body></body></html>")

        with patch("DocTest.Downloader.logger") as mock_logger:
            _warn_if_html_content(str(test_file), "https://example.com/test.pdf")
            mock_logger.warning.assert_called_once()

    def test_non_html_content_no_warning(self, tmp_path):
        """Test that binary/non-HTML content does not trigger a warning."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 some pdf content here")

        with patch("DocTest.Downloader.logger") as mock_logger:
            _warn_if_html_content(str(test_file), "https://example.com/test.pdf")
            mock_logger.warning.assert_not_called()

    def test_html_extension_no_warning(self, tmp_path):
        """Test that .html files do not trigger a warning even with HTML content."""
        test_file = tmp_path / "page.html"
        test_file.write_bytes(b"<!DOCTYPE html><html><body>OK</body></html>")

        with patch("DocTest.Downloader.logger") as mock_logger:
            _warn_if_html_content(str(test_file), "https://example.com/page.html")
            mock_logger.warning.assert_not_called()

    def test_htm_extension_no_warning(self, tmp_path):
        """Test that .htm files do not trigger a warning even with HTML content."""
        test_file = tmp_path / "page.htm"
        test_file.write_bytes(b"<html><body>OK</body></html>")

        with patch("DocTest.Downloader.logger") as mock_logger:
            _warn_if_html_content(str(test_file), "https://example.com/page.htm")
            mock_logger.warning.assert_not_called()

    def test_nonexistent_file_no_error(self):
        """Test that a nonexistent file does not raise an error."""
        _warn_if_html_content("/nonexistent/path/file.pdf", "https://example.com/file.pdf")


class TestModuleConstants:
    """Test cases for module-level constants."""

    def test_allowed_schemes(self):
        """Test that ALLOWED_SCHEMES contains expected values."""
        assert "http" in ALLOWED_SCHEMES
        assert "https" in ALLOWED_SCHEMES
        assert "ftp" not in ALLOWED_SCHEMES
        assert "file" not in ALLOWED_SCHEMES
        assert "data" not in ALLOWED_SCHEMES

    def test_default_max_size_is_100mb(self):
        """Test that DEFAULT_MAX_SIZE is 100 MB."""
        assert DEFAULT_MAX_SIZE == 104857600  # 100 * 1024 * 1024
