from pathlib import Path
import io
import re
import logging
import os
import fitz


class PdfDoc(object):

    def __init__(self, filename):
        self.pdf_content = PdfDoc.get_pdf_content(filename)

    @staticmethod
    def get_doc(filename):
        if (os.path.isfile(filename) is False):
            raise AssertionError('The PDF file does not exist: {}'.format(filename))
        # Open a PDF file.
        doc = fitz.open(filename)
        return doc

    @staticmethod
    def get_sig_flags(filename):
        doc = fitz.open(filename)
        try:
            return doc.catalog['AcroForm'].get('SigFlags', None)
        except:
            print("No signature was found in file")
            return None
        
    @staticmethod
    def get_pdf_content(filename):
        doc = fitz.open(filename)
        words_list = []
        for page in doc.pages:

            words = page.get_text("words")
            words = make_text(words)
            words_list.append(words)
        return words_list

    @staticmethod
    def get_pdf_text(filename):
        doc = fitz.open(filename)
        words_list = []
        for page in doc.pages:

            words = page.get_text("words")
            words = make_text(words)
            words_list.append(words)
        return words_list

def make_text(words):
    """Return textstring output of get_text("words").
    Word items are sorted for reading sequence left to right,
    top to bottom.
    """
    line_dict = {}  # key: vertical coordinate, value: list of words
    words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
    for w in words:  # fill the line dictionary
        y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
        word = w[4]  # the text of the word
        line = line_dict.get(y1, [])  # read current line content
        line.append(word)  # append new word
        line_dict[y1] = line  # write back to dict
    lines = list(line_dict.items())
    lines.sort()  # sort vertically
    return "\n".join([" ".join(line[1]) for line in lines])