import  pytest
from pathlib import Path


@pytest.fixture
def testdata_dir():
    testdata_directory = Path(__file__).parent.resolve() / "testdata"
    return  testdata_directory