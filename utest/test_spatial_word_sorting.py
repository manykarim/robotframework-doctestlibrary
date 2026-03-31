"""Unit tests for build_page_structure_from_words() and the spatial_word_sorting config flag."""

import pytest

from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
    build_page_structure,
    build_page_structure_from_words,
    flatten_document_words,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_word_tuple(text, x0, y0, x1, y1, block_no=0, line_no=0, word_no=0):
    """Return a tuple in PyMuPDF ``get_text('words')`` format.

    Format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    """
    return (x0, y0, x1, y1, text, block_no, line_no, word_no)


# ---------------------------------------------------------------------------
# 1. Empty / None inputs
# ---------------------------------------------------------------------------


def test_empty_words_list():
    """Empty input returns PageStructure with no blocks."""
    page = build_page_structure_from_words(0, [], page_width=612.0, page_height=792.0)
    assert isinstance(page, PageStructure)
    assert page.page_number == 0
    assert page.blocks == []
    assert page.width == 612.0
    assert page.height == 792.0


def test_none_words_list():
    """None input returns PageStructure with no blocks."""
    page = build_page_structure_from_words(0, None, page_width=612.0, page_height=792.0)
    assert isinstance(page, PageStructure)
    assert page.blocks == []


# ---------------------------------------------------------------------------
# 2. Single word
# ---------------------------------------------------------------------------


def test_single_word():
    """One word produces one block with one line."""
    words = [_make_word_tuple("hello", 10.0, 100.0, 50.0, 112.0)]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 1
    block = page.blocks[0]
    assert block.line_count == 1
    assert block.lines[0].text == "hello"
    assert len(block.lines[0].spans) == 1
    assert block.lines[0].spans[0].text == "hello"


# ---------------------------------------------------------------------------
# 3. Single line, multiple words
# ---------------------------------------------------------------------------


def test_single_line_multiple_words():
    """Multiple words at the same Y position produce one line, sorted by x0."""
    words = [
        _make_word_tuple("world", 60.0, 100.0, 110.0, 112.0, word_no=1),
        _make_word_tuple("hello", 10.0, 100.0, 50.0, 112.0, word_no=0),
        _make_word_tuple("!", 120.0, 100.0, 130.0, 112.0, word_no=2),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "hello world !"


# ---------------------------------------------------------------------------
# 4. Multiple lines
# ---------------------------------------------------------------------------


def test_multiple_lines():
    """Words at different Y positions produce separate lines sorted top-to-bottom."""
    words = [
        # Second line (y ~ 200)
        _make_word_tuple("second", 10.0, 200.0, 80.0, 212.0),
        # First line (y ~ 100)
        _make_word_tuple("first", 10.0, 100.0, 60.0, 112.0),
        # Third line (y ~ 300)
        _make_word_tuple("third", 10.0, 300.0, 70.0, 312.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 3
    texts = [b.lines[0].text for b in page.blocks]
    assert texts == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# 5. Multi-column layout
# ---------------------------------------------------------------------------


def test_multi_column_layout():
    """Three columns at the same Y range produce words interleaved by Y row.

    This is the key scenario: words from different columns that share the
    same vertical position should be grouped into the same line, ordered
    left-to-right.
    """
    # Row 1 (y=100..112): three columns
    words = [
        _make_word_tuple("C1R1", 10.0, 100.0, 60.0, 112.0),
        _make_word_tuple("C2R1", 210.0, 100.0, 260.0, 112.0),
        _make_word_tuple("C3R1", 410.0, 100.0, 460.0, 112.0),
        # Row 2 (y=130..142): three columns
        _make_word_tuple("C1R2", 10.0, 130.0, 60.0, 142.0),
        _make_word_tuple("C2R2", 210.0, 130.0, 260.0, 142.0),
        _make_word_tuple("C3R2", 410.0, 130.0, 460.0, 142.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 2
    assert page.blocks[0].lines[0].text == "C1R1 C2R1 C3R1"
    assert page.blocks[1].lines[0].text == "C1R2 C2R2 C3R2"


# ---------------------------------------------------------------------------
# 6. Mixed font sizes (adaptive tolerance)
# ---------------------------------------------------------------------------


def test_mixed_font_sizes():
    """Words with different heights at similar Y are grouped using adaptive tolerance.

    Tolerance is min(min_height, word_height) * 0.5.  Words that are close
    enough vertically should be merged into one line.
    """
    # Two words with different heights but overlapping Y midpoints.
    # Word A: height 12, midpoint = 106
    # Word B: height 20, midpoint = 110
    # min_height = 12, tolerance = 12 * 0.5 = 6.0
    # |106 - 110| = 4.0 < 6.0 => same line
    words = [
        _make_word_tuple("small", 10.0, 100.0, 60.0, 112.0),   # height=12, mid=106
        _make_word_tuple("big", 70.0, 100.0, 140.0, 120.0),    # height=20, mid=110
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "small big"


def test_mixed_font_sizes_separate_lines():
    """Words whose midpoints differ more than the adaptive tolerance form separate lines."""
    # Word A: height 10, midpoint = 105
    # Word B: height 10, midpoint = 120
    # tolerance = 10 * 0.5 = 5.0
    # |105 - 120| = 15.0 > 5.0 => different lines
    words = [
        _make_word_tuple("line1", 10.0, 100.0, 60.0, 110.0),   # mid=105
        _make_word_tuple("line2", 10.0, 115.0, 60.0, 125.0),   # mid=120
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 2


# ---------------------------------------------------------------------------
# 7. Text normalization
# ---------------------------------------------------------------------------


def test_text_normalization_applied():
    """Whitespace collapsing, ligature normalization, and strip edges all work."""
    config = StructureExtractionConfig(
        collapse_whitespace=True,
        strip_line_edges=True,
        normalize_ligatures=True,
    )
    # "\ufb01" is the fi ligature
    words = [
        _make_word_tuple("  hello  ", 10.0, 100.0, 60.0, 112.0),
        _make_word_tuple("\ufb01nd", 70.0, 100.0, 120.0, 112.0),
    ]
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0
    )

    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "hello find"


# ---------------------------------------------------------------------------
# 8. Config hash includes spatial_word_sorting
# ---------------------------------------------------------------------------


def test_config_hash_includes_spatial():
    """Two configs differing only in spatial_word_sorting hash differently."""
    c1 = StructureExtractionConfig(spatial_word_sorting=False)
    c2 = StructureExtractionConfig(spatial_word_sorting=True)
    assert hash(c1) != hash(c2)


def test_config_hash_same_when_equal():
    """Configs with identical settings hash the same."""
    c1 = StructureExtractionConfig(spatial_word_sorting=True)
    c2 = StructureExtractionConfig(spatial_word_sorting=True)
    assert hash(c1) == hash(c2)


# ---------------------------------------------------------------------------
# 9. Page dimensions from explicit args
# ---------------------------------------------------------------------------


def test_page_dimensions_from_args():
    """page_width and page_height params are used directly."""
    page = build_page_structure_from_words(
        0, [], page_width=500.0, page_height=700.0
    )
    assert page.width == 500.0
    assert page.height == 700.0


# ---------------------------------------------------------------------------
# 10. Page dimensions from image_shape + dpi
# ---------------------------------------------------------------------------


def test_page_dimensions_from_image_shape():
    """When page_width=0, falls back to image_shape + dpi calculation."""
    # image_shape: (height_px, width_px, channels)
    # width = 720 * 72 / 72 = 720.0
    # height = 1080 * 72 / 72 = 1080.0
    page = build_page_structure_from_words(
        0,
        [],
        page_width=0.0,
        page_height=0.0,
        dpi=72,
        image_shape=(1080, 720, 3),
    )
    assert page.width == 720.0
    assert page.height == 1080.0


def test_page_dimensions_from_image_shape_with_higher_dpi():
    """Verify the DPI scaling formula: page_pt = px * 72 / dpi."""
    # 1440px wide at 144 DPI => 1440 * 72 / 144 = 720 points
    page = build_page_structure_from_words(
        0,
        [],
        page_width=0.0,
        page_height=0.0,
        dpi=144,
        image_shape=(2160, 1440, 3),
    )
    assert page.width == 720.0
    assert page.height == 1080.0


# ---------------------------------------------------------------------------
# 11. Drop empty lines
# ---------------------------------------------------------------------------


def test_drop_empty_lines():
    """Empty words after normalization are dropped when drop_empty_lines=True."""
    config = StructureExtractionConfig(drop_empty_lines=True, strip_line_edges=True)
    words = [
        _make_word_tuple("   ", 10.0, 100.0, 60.0, 112.0),   # becomes empty after strip
        _make_word_tuple("real", 10.0, 200.0, 60.0, 212.0),
    ]
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0
    )

    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "real"


def test_keep_empty_lines_when_disabled():
    """When drop_empty_lines=False, whitespace-only words still produce lines."""
    config = StructureExtractionConfig(
        drop_empty_lines=False,
        collapse_whitespace=False,
        strip_line_edges=False,
    )
    words = [
        _make_word_tuple("   ", 10.0, 100.0, 60.0, 112.0),
        _make_word_tuple("real", 10.0, 200.0, 60.0, 212.0),
    ]
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0
    )

    assert len(page.blocks) == 2


# ---------------------------------------------------------------------------
# 12. Bbox is union of word bboxes
# ---------------------------------------------------------------------------


def test_bbox_is_union_of_word_bboxes():
    """Line bbox is the union of all word bboxes in that line."""
    words = [
        _make_word_tuple("left", 10.0, 100.0, 50.0, 112.0),
        _make_word_tuple("right", 200.0, 98.0, 260.0, 115.0),
    ]
    page = build_page_structure_from_words(
        0, words, config=StructureExtractionConfig(round_precision=None),
        page_width=612.0, page_height=792.0,
    )

    assert len(page.blocks) == 1
    bbox = page.blocks[0].lines[0].bbox
    # x0 = min(10.0, 200.0) = 10.0
    assert bbox[0] == 10.0
    # y0 = min(100.0, 98.0) = 98.0
    assert bbox[1] == 98.0
    # x1 = max(50.0, 260.0) = 260.0
    assert bbox[2] == 260.0
    # y1 = max(112.0, 115.0) = 115.0
    assert bbox[3] == 115.0


# ---------------------------------------------------------------------------
# 13. Round precision applied
# ---------------------------------------------------------------------------


def test_round_precision_applied():
    """Bboxes are rounded per config.round_precision."""
    words = [
        _make_word_tuple("word", 10.12345, 100.6789, 50.99999, 112.11111),
    ]
    config = StructureExtractionConfig(round_precision=2)
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0,
    )

    bbox = page.blocks[0].lines[0].bbox
    assert bbox == (10.12, 100.68, 51.0, 112.11)


def test_round_precision_none_no_rounding():
    """When round_precision is None, coordinates are not rounded."""
    words = [
        _make_word_tuple("word", 10.12345, 100.6789, 50.99999, 112.11111),
    ]
    config = StructureExtractionConfig(round_precision=None)
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0,
    )

    bbox = page.blocks[0].lines[0].bbox
    assert bbox == (10.12345, 100.6789, 50.99999, 112.11111)


# ---------------------------------------------------------------------------
# 14. Words sorted left to right within a line
# ---------------------------------------------------------------------------


def test_words_sorted_left_to_right_within_line():
    """Even if words are added out of order, they come out sorted by x0."""
    words = [
        _make_word_tuple("C", 200.0, 100.0, 220.0, 112.0),
        _make_word_tuple("A", 10.0, 100.0, 30.0, 112.0),
        _make_word_tuple("B", 100.0, 100.0, 120.0, 112.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "A B C"


# ---------------------------------------------------------------------------
# 15. Spatial vs block: same simple text
# ---------------------------------------------------------------------------


def test_spatial_vs_block_same_simple_text():
    """For a simple single-column document, build_page_structure and
    build_page_structure_from_words produce the same word sequence when flattened.
    """
    # Simulate a simple PDF dict for build_page_structure
    pdf_dict = {
        "width": 612.0,
        "height": 792.0,
        "blocks": [
            {
                "type": 0,
                "bbox": (10.0, 100.0, 200.0, 145.0),
                "lines": [
                    {
                        "bbox": (10.0, 100.0, 200.0, 112.0),
                        "spans": [
                            {"text": "hello world", "font": "Arial", "size": 12.0}
                        ],
                    },
                    {
                        "bbox": (10.0, 130.0, 200.0, 142.0),
                        "spans": [
                            {"text": "foo bar", "font": "Arial", "size": 12.0}
                        ],
                    },
                ],
            }
        ],
    }

    # Simulate equivalent word tuples for build_page_structure_from_words
    word_tuples = [
        _make_word_tuple("hello", 10.0, 100.0, 50.0, 112.0, 0, 0, 0),
        _make_word_tuple("world", 55.0, 100.0, 100.0, 112.0, 0, 0, 1),
        _make_word_tuple("foo", 10.0, 130.0, 40.0, 142.0, 0, 1, 0),
        _make_word_tuple("bar", 45.0, 130.0, 80.0, 142.0, 0, 1, 1),
    ]

    config = StructureExtractionConfig()
    page_block = build_page_structure(0, pdf_dict, config=config)
    page_spatial = build_page_structure_from_words(
        0, word_tuples, config=config, page_width=612.0, page_height=792.0,
    )

    # Extract words from both
    def _extract_words(page):
        words = []
        for block in page.blocks:
            for line in block.lines:
                words.extend(line.text.split())
        return words

    block_words = _extract_words(page_block)
    spatial_words = _extract_words(page_spatial)
    assert block_words == spatial_words


# ---------------------------------------------------------------------------
# 16. Integration with flatten_document_words
# ---------------------------------------------------------------------------


def test_integration_with_flatten_document_words():
    """Build a DocumentStructure from spatial pages and verify flatten_document_words works."""
    words_page1 = [
        _make_word_tuple("page", 10.0, 100.0, 50.0, 112.0),
        _make_word_tuple("one", 55.0, 100.0, 90.0, 112.0),
    ]
    words_page2 = [
        _make_word_tuple("page", 10.0, 100.0, 50.0, 112.0),
        _make_word_tuple("two", 55.0, 100.0, 90.0, 112.0),
    ]

    config = StructureExtractionConfig()
    page1 = build_page_structure_from_words(
        0, words_page1, config=config, page_width=612.0, page_height=792.0,
    )
    page2 = build_page_structure_from_words(
        1, words_page2, config=config, page_width=612.0, page_height=792.0,
    )

    doc = DocumentStructure(pages=[page1, page2], config=config)

    flat_words, tokens = flatten_document_words(doc)
    assert flat_words == ["page", "one", "page", "two"]
    assert len(tokens) == 4
    assert tokens[0].source_page == 0
    assert tokens[2].source_page == 1
    assert tokens[0].word_index == 0
    assert tokens[3].word_index == 3


# ---------------------------------------------------------------------------
# 17. Character replacements applied
# ---------------------------------------------------------------------------


def test_character_replacements_applied():
    """Character replacements are applied to word text during normalization."""
    config = StructureExtractionConfig(
        character_replacements={"\u00A0": " ", "\u2013": "-"},
    )
    # Non-breaking space within a word, en-dash in another
    words = [
        _make_word_tuple("hello\u00A0world", 10.0, 100.0, 100.0, 112.0),
        _make_word_tuple("2020\u20132021", 110.0, 100.0, 200.0, 112.0),
    ]
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0,
    )

    assert len(page.blocks) == 1
    line_text = page.blocks[0].lines[0].text
    # NBSP replaced with space, then words joined
    # "hello world" becomes two parts after collapse_whitespace: "hello" "world"
    # so the full text depends on how the joining works
    assert "\u00A0" not in line_text
    assert "\u2013" not in line_text
    assert "2020-2021" in line_text


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_block_index_and_line_index_increment():
    """Block index and global line index are sequential."""
    words = [
        _make_word_tuple("line1", 10.0, 100.0, 60.0, 112.0),
        _make_word_tuple("line2", 10.0, 200.0, 60.0, 212.0),
        _make_word_tuple("line3", 10.0, 300.0, 60.0, 312.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 3
    for i, block in enumerate(page.blocks):
        assert block.index == i
        assert block.lines[0].index == i


def test_page_number_is_preserved():
    """The page_number argument is stored in the result."""
    page = build_page_structure_from_words(42, [], page_width=612.0, page_height=792.0)
    assert page.page_number == 42


def test_block_bbox_equals_line_bbox():
    """Since each block has exactly one line, the block bbox should match the line bbox."""
    words = [
        _make_word_tuple("hello", 10.0, 100.0, 50.0, 112.0),
        _make_word_tuple("world", 55.0, 100.0, 100.0, 112.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 1
    assert page.blocks[0].bbox == page.blocks[0].lines[0].bbox


def test_line_count_property():
    """PageStructure.line_count aggregates across all blocks."""
    words = [
        _make_word_tuple("a", 10.0, 100.0, 30.0, 112.0),
        _make_word_tuple("b", 10.0, 200.0, 30.0, 212.0),
        _make_word_tuple("c", 10.0, 300.0, 30.0, 312.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert page.line_count == 3


def test_spans_contain_full_line_text():
    """Each line has exactly one span whose text matches the line text."""
    words = [
        _make_word_tuple("alpha", 10.0, 100.0, 60.0, 112.0),
        _make_word_tuple("beta", 70.0, 100.0, 120.0, 112.0),
    ]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    line = page.blocks[0].lines[0]
    assert len(line.spans) == 1
    assert line.spans[0].text == line.text
    assert line.spans[0].font is None
    assert line.spans[0].size == 0.0


def test_fonts_set_is_empty():
    """Spatial word extraction does not have font info, so fonts set is empty."""
    words = [_make_word_tuple("test", 10.0, 100.0, 50.0, 112.0)]
    page = build_page_structure_from_words(0, words, page_width=612.0, page_height=792.0)

    assert page.blocks[0].lines[0].fonts == set()


def test_whitespace_replacement_used():
    """The whitespace_replacement from config is used to join words."""
    config = StructureExtractionConfig(whitespace_replacement="|")
    words = [
        _make_word_tuple("a", 10.0, 100.0, 30.0, 112.0),
        _make_word_tuple("b", 40.0, 100.0, 60.0, 112.0),
    ]
    page = build_page_structure_from_words(
        0, words, config=config, page_width=612.0, page_height=792.0,
    )

    assert page.blocks[0].lines[0].text == "a|b"


def test_default_config_used_when_none():
    """When config is None, a default StructureExtractionConfig is used."""
    words = [_make_word_tuple("hello", 10.0, 100.0, 50.0, 112.0)]
    page = build_page_structure_from_words(0, words, config=None, page_width=612.0, page_height=792.0)

    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "hello"
