import shutil
from pathlib import Path

from DocTest.Downloader import download_file_from_url, get_filename_from_url


def _fake_urlretrieve(monkeypatch, source: Path):
    def _copy(_, destination):
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)
        return str(destination_path), None

    monkeypatch.setattr("DocTest.Downloader.urllib.request.urlretrieve", _copy)


def test_download_file(monkeypatch, tmp_path, testdata_dir):
    source = testdata_dir / "birthday_left.png"
    _fake_urlretrieve(monkeypatch, source)
    monkeypatch.setattr("DocTest.Downloader.tempfile.gettempdir", lambda: str(tmp_path))

    url = "https://github.com/manykarim/robotframework-doctestlibrary/blob/main/utest/testdata/birthday_left.png"
    path = Path(download_file_from_url(url))

    assert path.suffix == ".png"
    assert path.exists()
    assert path.read_bytes() == source.read_bytes()


def test_get_filename_from_url():
    url = "https://github.com/manykarim/robotframework-doctestlibrary/blob/main/utest/testdata/birthday_left.png"
    filename = get_filename_from_url(url)
    assert filename == "birthday_left.png"


def test_download_file_with_filename(monkeypatch, tmp_path, testdata_dir):
    source = testdata_dir / "birthday_left.png"
    _fake_urlretrieve(monkeypatch, source)

    url = "https://github.com/manykarim/robotframework-doctestlibrary/blob/main/utest/testdata/birthday_left.png"
    filename = get_filename_from_url(url)

    path = Path(
        download_file_from_url(url, directory=str(tmp_path), filename=filename)
    )

    assert path.name == "birthday_left.png"
    assert path.exists()
    assert path.read_bytes() == source.read_bytes()
