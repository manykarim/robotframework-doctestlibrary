from DocTest.VisualTest import VisualTest
import pytest
from pathlib import Path
import cProfile
import pstats
import os

@pytest.fixture
def get_junitxml_directory(request):
    # Access pytest config object
    config = request.config
    # Get the value of --junitxml option, if it exists
    junitxml_path = config.getoption("--junitxml", default=None)
    
    if junitxml_path:
        # Extract and return the directory from the full path
        junitxml_dir = os.path.dirname(junitxml_path)
        return junitxml_dir
    else:
        return 'results'
    return None

def test_profiler_ocr_text_small(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "text_small.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()


def test_profiler_ocr_text_big(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "text_big.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_text_medium(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "text_medium.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_pdf(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "sample_1_page.pdf")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_east_text_small(testdata_dir, request, get_junitxml_directory):
    img = VisualTest(ocr_engine='east')
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "text_small.png", )', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_east_text_big(testdata_dir, request, get_junitxml_directory):
    img = VisualTest(ocr_engine='east')
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "text_big.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_east_text_medium(testdata_dir, request, get_junitxml_directory):
    img = VisualTest(ocr_engine='east')
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "text_medium.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_east_birthday(testdata_dir, request, get_junitxml_directory):
    img = VisualTest(ocr_engine='east')
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "birthday_1080_date_id.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_pytesseract_birthday(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "birthday_1080_date_id.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_ocr_pytesseract_big_file(testdata_dir, request, get_junitxml_directory):

    import cv2
    low_res_image = cv2.imread(str(testdata_dir / 'birthday_1080_date_id.png'))
    # resize image to 10x bigger
    low_res_image = cv2.resize(low_res_image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(testdata_dir / 'birthday_1080_date_id_10x.png'), low_res_image)
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.get_text(testdata_dir / "birthday_1080_date_id_10x.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_east_pytesseract_big_file(testdata_dir, request, get_junitxml_directory):
    
        import cv2
        low_res_image = cv2.imread(str(testdata_dir / 'birthday_1080_date_id.png'))
        # resize image to 10x bigger
        low_res_image = cv2.resize(low_res_image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(testdata_dir / 'birthday_1080_date_id_10x.png'), low_res_image)
        img = VisualTest(ocr_engine='east')
        testname = request.node.name
        results_dir = get_junitxml_directory
        prof = cProfile.Profile()
        prof.runctx('img.get_text(testdata_dir / "birthday_1080_date_id_10x.png")', globals(), locals())
        prof.dump_stats(f'{results_dir}/{testname}.prof')
        stream = open(f'{results_dir}/{testname}.txt', 'w')
        stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
        stats.sort_stats('cumtime')
        stats.print_stats()


def test_profiler_compare_two_images_with_differences(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    with pytest.raises(AssertionError):
        prof.runctx('img.compare_images(testdata_dir / "birthday_left.png", testdata_dir / "birthday_right.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_compare_two_images_without_differences(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.compare_images(testdata_dir / "birthday_left.png", testdata_dir / "birthday_left.png")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()

def test_profiler_compare_two_images_with_masks_and_no_differences(testdata_dir, request, get_junitxml_directory):
    img = VisualTest()
    testname = request.node.name
    results_dir = get_junitxml_directory
    prof = cProfile.Profile()
    prof.runctx('img.compare_images(testdata_dir / "birthday_1080_date_id.png", testdata_dir / "birthday_1080.png", placeholder_file=testdata_dir / "pattern_mask.json")', globals(), locals())
    prof.dump_stats(f'{results_dir}/{testname}.prof')
    stream = open(f'{results_dir}/{testname}.txt', 'w')
    stats = pstats.Stats(f'{results_dir}/{testname}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()