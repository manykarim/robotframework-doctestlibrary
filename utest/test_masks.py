from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.DocumentRepresentation import Page
import pytest
from pathlib import Path
import numpy as np

def test_image_area_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ignore_area_file=testdata_dir / 'area_mask.json')
    assert len(img.abstract_ignore_areas)==1
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()


def test_image_text_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ignore_area_file=testdata_dir / 'pattern_mask.json')
    assert len(img.abstract_ignore_areas)==2
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_image_text_mask_with_east(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ignore_area_file=testdata_dir / 'pattern_mask.json', ocr_engine='east')
    assert len(img.abstract_ignore_areas)>=1
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_pdf_area_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area_file=testdata_dir / 'pdf_area_mask.json')
    assert len(img.abstract_ignore_areas)==1
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_pdf_text_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area_file=testdata_dir / 'pdf_pattern_mask.json')
    assert len(img.abstract_ignore_areas)==2
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_pdf_word_pattern_mask_dimensions(testdata_dir):
    mask = {
        'page': 'all',
        'type': 'word_pattern',
        'pattern': '12345678901234'
    }
    doc = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area=mask)
    area = doc.pages[0].pixel_ignore_areas[0]
    assert area['width'] == 233
    assert area['height'] == 31

def test_pdf_pattern_mask_dimensions(testdata_dir):
    mask = {
        'page': 'all',
        'type': 'pattern',
        'pattern': '.*RTMOE.*'
    }
    doc = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area=mask)
    area = doc.pages[0].pixel_ignore_areas[0]
    assert area['width'] == 233
    assert area['height'] == 31
@pytest.mark.parametrize("unit,value,dpi,expected", [
    ('mm', 25.4, 200, 200),   # regression: was 196 due to pre-conversion truncation
    ('mm', 10, 200, 79),      # 78.74 rounds to 79, not truncates to 78
    ('mm', 10.5, 300, 124),   # fractional mm preserved
    ('cm', 2.54, 200, 200),
    ('cm', 1, 96, 38),        # 37.79 -> 38
    ('pt', 72, 200, 200),
    ('pt', 36.5, 144, 73),
    ('px', 50, 200, 50),
    ('px', 50.6, 200, 51),
])
def test_convert_to_pixels_rounds_after_conversion(unit, value, dpi, expected):
    image = np.zeros((50, 200, 3), dtype=np.uint8)
    page = Page(image, page_number=1, dpi=dpi)
    area = {'x': value, 'y': value, 'width': value, 'height': value}
    x, y, w, h = page._convert_to_pixels(area, unit)
    assert (x, y, w, h) == (expected, expected, expected, expected)


def test_coordinate_mask_fractional_mm_resolves_exactly():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    page = Page(image, page_number=1, dpi=200)
    page._process_coordinates_ignore_area(
        {'page': 'all', 'type': 'coordinates', 'x': 0, 'y': 0,
         'width': 25.4, 'height': 25.4, 'unit': 'mm'})
    assert page.pixel_ignore_areas == [{'x': 0, 'y': 0, 'height': 200, 'width': 200}]


def _page_with_ocr_tokens(tokens):
    image = np.zeros((50, 400, 3), dtype=np.uint8)
    page = Page(image, page_number=1, dpi=200)
    page.ocr_performed = True
    page.ocr_text_data = {
        'text': tokens,
        'left': [10 * (i + 1) for i in range(len(tokens))],
        'top': [5] * len(tokens),
        'width': [40] * len(tokens),
        'height': [15] * len(tokens),
        'conf': ['95'] * len(tokens),
    }
    return page


@pytest.mark.parametrize("pattern,expected_matches", [
    ('.*Robot.*', 1),       # mixed-case pattern matches original-case token
    ('.*Extending.*', 1),
    ('Robot', 1),
    ('R', 2),                # prefix match hits Robot and Remote
    ('.*ROBOT.*', 1),        # legacy uppercase-target patterns keep working
    ('(?i).*robot.*', 1),    # explicit case-insensitive flag works
    ('.*robot.*', 0),        # lowercase literal matches neither casing
    ('.*Missing.*', 0),
])
def test_ocr_pattern_matches_original_case_tokens(pattern, expected_matches):
    """Regression: OCR tokens were uppercased before matching, so mixed-case
    patterns like '.*Robot.*' never matched (while 'R' did)."""
    page = _page_with_ocr_tokens(['Robot', 'Extending', 'Remote', 'framework'])
    page._process_pattern_ignore_area_from_ocr({'type': 'pattern', 'pattern': pattern})
    assert len(page.pixel_ignore_areas) == expected_matches


def _page_with_ocr_lines(lines):
    """Synthetic tesseract-style data: lines is a list of token lists."""
    tokens, lefts, line_nums = [], [], []
    for line_no, line_tokens in enumerate(lines, start=1):
        for word_no, token in enumerate(line_tokens):
            tokens.append(token)
            lefts.append(10 + word_no * 50)
            line_nums.append(line_no)
    image = np.zeros((200, 600, 3), dtype=np.uint8)
    page = Page(image, page_number=1, dpi=200)
    page.ocr_performed = True
    page.ocr_text_data = {
        'text': tokens,
        'left': lefts,
        'top': [20 * n for n in line_nums],
        'width': [45] * len(tokens),
        'height': [15] * len(tokens),
        'conf': ['95'] * len(tokens),
        'block_num': [1] * len(tokens),
        'par_num': [1] * len(tokens),
        'line_num': line_nums,
    }
    return page


OCR_LINES = [
    ['Robot', 'Framework', 'User', 'Guide'],
    ['Getting', 'Started'],
    ['Extending', 'Robot', 'Framework'],
]


@pytest.mark.parametrize("mask,expected", [
    # phrases (whitespace in pattern) are searched anywhere within each line
    ({'type': 'pattern', 'pattern': 'Robot Framework'}, 2),       # lines 1 and 3
    ({'type': 'pattern', 'pattern': '.*Robot Framework.*'}, 2),
    ({'type': 'pattern', 'pattern': 'Getting Started'}, 1),
    ({'type': 'pattern', 'pattern': 'User Guide'}, 1),            # mid-line phrase
    ({'type': 'pattern', 'pattern': 'ROBOT FRAMEWORK'}, 2),       # legacy uppercase
    ({'type': 'pattern', 'pattern': 'No Such Phrase'}, 0),
    # single-token patterns stay word-level for type 'pattern'
    ({'type': 'pattern', 'pattern': '.*Robot.*'}, 2),
    # line_pattern matches whole lines, anchored like the PDF path
    ({'type': 'line_pattern', 'pattern': '.*Framework.*'}, 2),
    ({'type': 'line_pattern', 'pattern': 'Getting.*'}, 1),
    # word_pattern never matches phrases
    ({'type': 'word_pattern', 'pattern': 'Robot Framework'}, 0),
    ({'type': 'word_pattern', 'pattern': 'Robot'}, 2),
])
def test_ocr_phrase_and_line_matching(mask, expected):
    """Phrases with spaces can only exist at line level (OCR tokens are
    single words); line_pattern is line-based like the PDF path."""
    page = _page_with_ocr_lines(OCR_LINES)
    page._process_pattern_ignore_area_from_ocr(mask)
    assert len(page.pixel_ignore_areas) == expected


def test_ocr_phrase_masks_only_matched_words():
    """An exact phrase masks just the words its match span covers."""
    page = _page_with_ocr_lines(OCR_LINES)
    page._process_pattern_ignore_area_from_ocr(
        {'type': 'pattern', 'pattern': 'Robot Framework'})
    # line 1: words 1-2 ("Robot Framework"), not "User Guide"
    assert page.pixel_ignore_areas[0] == {'x': 10, 'y': 20, 'width': 50 + 45, 'height': 15}
    # line 3: words 2-3 ("Robot Framework") after "Extending"
    assert page.pixel_ignore_areas[1] == {'x': 60, 'y': 60, 'width': 50 + 45, 'height': 15}


def test_ocr_phrase_mid_line_span():
    page = _page_with_ocr_lines(OCR_LINES)
    page._process_pattern_ignore_area_from_ocr(
        {'type': 'pattern', 'pattern': 'Framework User'})
    # line 1 words 2-3 only
    assert page.pixel_ignore_areas == [{'x': 60, 'y': 20, 'width': 50 + 45, 'height': 15}]


def test_ocr_wrapped_phrase_masks_whole_line():
    """Wrapping the phrase in .* extends the match span over the whole line."""
    page = _page_with_ocr_lines(OCR_LINES)
    page._process_pattern_ignore_area_from_ocr(
        {'type': 'pattern', 'pattern': '.*User Guide.*'})
    assert page.pixel_ignore_areas == [{'x': 10, 'y': 20, 'width': 3 * 50 + 45, 'height': 15}]


def test_pattern_mask_handles_umlauts_and_symbols():
    image = np.zeros((50, 200, 3), dtype=np.uint8)
    page = Page(image, page_number=1, dpi=200)
    page.ocr_performed = True
    page.ocr_text_data = {
        'text': ['Änderung', 'Café-123#'],
        'left': [10, 60],
        'top': [5, 10],
        'width': [40, 80],
        'height': [15, 18],
        'conf': ['95', '95'],
    }

    umlaut_mask = {
        'type': 'pattern',
        'pattern': '(?i).*ÄNDERUNG.*',
        'xoffset': 2,
        'yoffset': 3,
    }
    symbol_mask = {
        'type': 'pattern',
        'pattern': '(?i).*CAFÉ-123#.*',
        'xoffset': 1,
        'yoffset': 1,
    }

    page._process_pattern_ignore_area_from_ocr(umlaut_mask)
    page._process_pattern_ignore_area_from_ocr(symbol_mask)

    assert len(page.pixel_ignore_areas) == 2
    first, second = page.pixel_ignore_areas
    assert first['x'] <= 12 and first['width'] >= 40
    assert second['x'] <= 61 and second['width'] >= 80
