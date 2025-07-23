"""Unit tests for Downloader module."""

import os
from unittest.mock import patch

from DocTest.Downloader import download_file_from_url, get_filename_from_url, is_url


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

    @patch("DocTest.Downloader.urllib.request.urlretrieve")
    @patch("DocTest.Downloader.tempfile.gettempdir")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_with_default_params(
        self, mock_uuid, mock_gettempdir, mock_urlretrieve
    ):
        """Test download with default directory and filename."""
        mock_gettempdir.return_value = "/tmp"
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf"

        result = download_file_from_url(url)

        expected_path = os.path.join("/tmp", "test-uuid.pdf")
        assert result == expected_path
        mock_urlretrieve.assert_called_once_with(url, expected_path)

    @patch("DocTest.Downloader.urllib.request.urlretrieve")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_with_custom_directory(self, mock_uuid, mock_urlretrieve):
        """Test download with custom directory."""
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf"
        directory = "/custom/dir"

        result = download_file_from_url(url, directory=directory)

        expected_path = os.path.join(directory, "test-uuid.pdf")
        assert result == expected_path
        mock_urlretrieve.assert_called_once_with(url, expected_path)

    @patch("DocTest.Downloader.urllib.request.urlretrieve")
    def test_download_with_custom_filename(self, mock_urlretrieve):
        """Test download with custom filename."""
        url = "https://example.com/test.pdf"
        directory = "/custom/dir"
        filename = "custom_name.pdf"

        result = download_file_from_url(url, directory=directory, filename=filename)

        expected_path = os.path.join(directory, filename)
        assert result == expected_path
        mock_urlretrieve.assert_called_once_with(url, expected_path)

    @patch("DocTest.Downloader.urllib.request.urlretrieve")
    @patch("DocTest.Downloader.tempfile.gettempdir")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_url_without_extension(
        self, mock_uuid, mock_gettempdir, mock_urlretrieve
    ):
        """Test download with URL without file extension."""
        mock_gettempdir.return_value = "/tmp"
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/path/file"

        result = download_file_from_url(url)

        # The implementation splits by '.' and takes the last part as extension
        # For "https://example.com/path/file", the last part after splitting by '.' is "com/path/file"
        expected_path = os.path.join("/tmp", "test-uuid.com/path/file")
        assert result == expected_path

    @patch("DocTest.Downloader.urllib.request.urlretrieve")
    @patch("DocTest.Downloader.uuid.uuid4")
    def test_download_url_with_query_params(self, mock_uuid, mock_urlretrieve):
        """Test download with URL containing query parameters."""
        mock_uuid.return_value = "test-uuid"
        url = "https://example.com/test.pdf?param=value"
        directory = "/test/dir"

        result = download_file_from_url(url, directory=directory)

        # The implementation splits by '.' and takes the last part as extension
        # For the URL with query params, it would be "pdf?param=value"
        expected_path = os.path.join(directory, "test-uuid.pdf?param=value")
        assert result == expected_path


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
        """Test URL with root domain only."""
        url = "https://example.com"
        result = get_filename_from_url(url)
        # The function splits by '/' and takes the last part, which would be "example.com"
        # Since it contains a dot, it's considered a valid filename
        assert result == "example.com"

    def test_url_with_query_params(self):
        """Test URL with query parameters."""
        url = "https://example.com/document.pdf?download=true"
        result = get_filename_from_url(url)
        assert result == "document.pdf?download=true"

    def test_url_with_fragment(self):
        """Test URL with fragment identifier."""
        url = "https://example.com/document.pdf#page=1"
        result = get_filename_from_url(url)
        assert result == "document.pdf#page=1"

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
