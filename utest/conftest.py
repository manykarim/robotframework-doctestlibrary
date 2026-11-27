import  pytest
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


TESTDATA_DIR = Path(__file__).parent.resolve() / "testdata"


@pytest.fixture
def testdata_dir():
    testdata_directory = TESTDATA_DIR
    return  testdata_directory


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="session")
def testdata_url():
    """Base URL serving ``utest/testdata`` over local HTTP.

    URL tests exercise the real download path without depending on GitHub,
    whose transient 5xx responses made them flaky in CI."""
    handler = functools.partial(_QuietHandler, directory=str(TESTDATA_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
