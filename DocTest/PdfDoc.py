from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pathlib import Path
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.converter import TextConverter
from pdfminer.pdftypes import resolve1
import io
import re
import logging
import os
import fitz


class PdfDoc(object):

    logging.getLogger('pdfminer.pdfinterp').disabled=True
    logging.getLogger('pdfminer.pdfdocument').disabled=True
    logging.getLogger('pdfminer.pdfpage').disabled=True

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
    def get_signature(filename, ignore_time_of_signing=True, ignore_signature_content=True):
        doc = fitz.open(filename)
        try:
            fields = resolve1(doc.catalog['AcroForm'])['Fields']
            for field in fields:
                item = resolve1(field)
                if item['FT'].name=='Sig':
                    signature = resolve1(item.get('V'))
                    if ignore_time_of_signing:
                        signature.pop('M', None)
                    if ignore_signature_content:
                        signature.pop('Contents', None)
                        signature.pop('ByteRange', None)
                return signature
        except:
            print("No signature was found in file")
            return None

    
    @staticmethod
    def get_output_intents(filename):
        doc = fitz.open(filename)
        if doc.catalog.get('OutputIntents', None) is not None:
            output_intents = resolve1(doc.catalog['OutputIntents'])[0]
            return output_intents['S'], output_intents['OutputConditionIdentifier']
        else:
            return None
        
    @staticmethod
    def parse_lt_objs (lt_objs, page_number, text_content=[], page_height=801, dpi_calculation_factor=300/(72)):
        """Iterate through the list of LT* objects and capture the text or image data contained in each"""
        for lt_obj in lt_objs:
            if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine) or isinstance(lt_obj, LTTextBoxHorizontal):
                # text
                text_content.append({"class":lt_obj.__class__.__name__, "page":page_number, "x0":int(lt_obj.x0*dpi_calculation_factor), "y0":int((page_height - lt_obj.y0)*dpi_calculation_factor), "x1":int(lt_obj.x1*dpi_calculation_factor), "y1":int((page_height - lt_obj.y1)*dpi_calculation_factor), "height":int(lt_obj.height*300/72), "width":int(lt_obj.width*300/72),  "text":lt_obj.get_text()})
            # elif isinstance(lt_obj, LTFigure):
            # # LTFigure objects are containers for other LT* objects, so recurse through the children
            #     text_content.append(parse_lt_objs(lt_obj._objs, page_number, text_content, page_height, dpi_calculation_factor))
        if text_content!=[]:
            return text_content
        else:
            return

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
    def get_items_with_matching_text(lt_objs, pattern, **kwargs):
        objecttype = kwargs.pop('objecttype', "textbox")
        results = kwargs.pop('results', None)
        page_height = kwargs.pop('page_height', 801)
        dpi_calculation_factor = kwargs.pop('dpi_calculation_factor', 300/72)
        search_pattern = re.compile(pattern)

        if results == None:
            results = []

        for lt_obj in lt_objs:
            
            if objecttype=="textbox":
                if (isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextBoxHorizontal)) and search_pattern.match(lt_obj.get_text().strip()):
                    results.append({"class":lt_obj.__class__.__name__,  "x":int(lt_obj.x0*dpi_calculation_factor), "y":int((page_height - lt_obj.y1)*dpi_calculation_factor), "width":int(lt_obj.width*dpi_calculation_factor),  "height":int(lt_obj.height*dpi_calculation_factor), "text":lt_obj.get_text()})
            elif objecttype=="textline":
                if (isinstance(lt_obj, LTTextLine) or isinstance(lt_obj, LTTextLineHorizontal)) and search_pattern.match(lt_obj.get_text().strip()):
                    results.append({"class":lt_obj.__class__.__name__,  "x":int(lt_obj.x0*dpi_calculation_factor), "y":int((page_height - lt_obj.y1)*dpi_calculation_factor), "width":int(lt_obj.width*dpi_calculation_factor), "height":int(lt_obj.height*dpi_calculation_factor), "text":lt_obj.get_text()})
                if (isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextBoxHorizontal)) and search_pattern.match(re.sub(r'[\t\n\r\f\v]+', '', lt_obj.get_text())):
                    PdfDoc.get_items_with_matching_text(lt_obj, pattern, objecttype=objecttype, results=results, page_height=page_height, dpi_calculation_factor=dpi_calculation_factor)
        return results

    def get_own_items_with_matching_text(self, page, pattern, objecttype="textbox",  dpi_calculation_factor=300/(72)):
        
        results = PdfDoc.get_items_with_matching_text(self.pdf_content[page-1], pattern, objecttype=objecttype, page_height=self.pdf_content[page-1].height, dpi_calculation_factor=dpi_calculation_factor)
        return results

    @staticmethod
    def get_items_in_area(lt_objs, x, y, w, h, dpi_calculation_factor=300/72 ):
        items_in_area = []
        page_height = lt_objs.height
        for lt_obj in lt_objs:           
            if ((isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextBoxHorizontal)) and not((lt_obj.x0*dpi_calculation_factor > x+w) or (lt_obj.x1*dpi_calculation_factor < x) or ((page_height-lt_obj.y1)*dpi_calculation_factor > y+h) or ((page_height-lt_obj.y0)*dpi_calculation_factor < y))):
                items_in_area.append({"class":lt_obj.__class__.__name__,  "x":int(lt_obj.x0*dpi_calculation_factor), "y":int((page_height - lt_obj.y1)*dpi_calculation_factor), "width":int(lt_obj.width*dpi_calculation_factor),  "height":int(lt_obj.height*dpi_calculation_factor), "text":lt_obj.get_text()})


        # items_in_area = [ item for item in items if \
        #     (item['page'] == page and \
        #         ((item['x0'] <= x and item['x1'] >=x+w and item['y1'] <= y and item['y0'] >= y+h) or \
        #         (item['x0'] >= x and item['x1'] <=x+w and item['y1'] >= y and item['y0'] <= y+h)))]
        return items_in_area
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