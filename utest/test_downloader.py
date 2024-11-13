from DocTest.Downloader import download_file_from_url, get_filename_from_url

def test_download_file():
    url="https://github.com/manykarim/robotframework-doctestlibrary/blob/main/utest/testdata/birthday_left.png"
    path = download_file_from_url(url)
    assert path.endswith(".png")

def test_get_filename_from_url():
    url="https://github.com/manykarim/robotframework-doctestlibrary/blob/main/utest/testdata/birthday_left.png"
    filename = get_filename_from_url(url)
    assert filename == "birthday_left.png"

def test_download_file_with_filename():
    url="https://github.com/manykarim/robotframework-doctestlibrary/blob/main/utest/testdata/birthday_left.png"
    filename = get_filename_from_url(url)
    path = download_file_from_url(url, filename=filename)
    assert path.endswith("birthday_left.png")