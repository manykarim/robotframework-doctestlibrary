from inspect import signature
from DocTest.PdfDoc import PdfDoc
from pprint import pprint
from deepdiff import DeepDiff
from robot.api.deco import keyword, library
import fitz

ROBOT_AUTO_KEYWORDS = False

class PdfTest(object):
    
    
    def __init__(self, **kwargs):
        pass
    
    @keyword
    def compare_pdf_documents(self, reference_document, candidate_document, **kwargs):
        """Compares some PDF metadata/properties of ``reference_document`` and ``candidate_document``.
        
        ``reference_document`` and ``candidate_document`` shall both be path to ``PDF`` files.

        The compared properties are mostly related to digital signatures and are:
               
            - Signature
            - Output Intents
            - SigFlags

        Result is passed if all properties are equal. 
        
        ``reference_document`` and ``candidate_document`` are both .pdf files


        Examples:
        | = Keyword =    |  = reference_document =  | = candidate_document =       |  = comment = |
        | Compare Pdf Documents | reference.pdf | candidate.pdf | #Performs a property comparison of both files |       
        
        """

        ref_doc = fitz.open(reference_document)
        cand_doc = fitz.open(candidate_document)
        reference = {}
        reference['pages'] = []
        #reference['metadata']=ref_doc.metadata
        reference['page_count']=ref_doc.page_count
        reference['sigflags']=ref_doc.get_sigflags()
        for i, page in enumerate(ref_doc.pages()):
            signature_list = []
            text = page.get_text("text").splitlines()
            for widget in page.widgets():
                if widget.is_signed:
                    signature_list.append(list((widget.field_name, widget.field_label, widget.field_value)))
            reference['pages'].append(dict([('number',i+1), ('fonts', page.get_fonts()), ('images', page.get_images()), ('rotation', page.rotation), ('mediabox', page.mediabox), ('signatures', signature_list),('text', text)]))


        candidate = {}
        candidate['pages'] = []
        #candidate['metadata']=cand_doc.metadata
        candidate['page_count']=cand_doc.page_count
        candidate['sigflags']=cand_doc.get_sigflags()
        for i, page in enumerate(cand_doc.pages()):
            signature_list = []
            text = page.get_text("text").splitlines()
            for widget in page.widgets():
                if widget.is_signed:
                    signature_list.append(list((widget.field_name, widget.field_label, widget.field_value)))
            candidate['pages'].append(dict([('number',i+1), ('fonts', page.get_fonts()), ('images', page.get_images()), ('rotation', page.rotation), ('mediabox', page.mediabox), ('signatures', signature_list),('text', text)]))

        if reference!=candidate:
            pprint(DeepDiff(reference, candidate, verbose_level=2), width=200)
            print("Reference Document:")
            pprint(reference)
            print("Candidate Document:")
            pprint(candidate)           
            raise AssertionError('The compared PDF Document Data is different.')

    @keyword
    def check_text_content(self, expected_text_list, candidate_document):
        """Checks if each item provided in the list ``expected_text_list`` appears in the PDF File ``candidate_document``.
        
        ``expected_text_list`` is a list of strings, ``candidate_document`` is the path to a PDF File.
        
        Examples:

        | @{strings}= | Create List | One String | Another String |
        | Check Text Content | ${strings} | candidate.pdf |
        
        """
        doc = fitz.open(candidate_document)
        missing_text_list = []
        all_texts_were_found = None
        for text_item in expected_text_list:
            text_found_in_page = False
            for page in doc.pages():
                if any(text_item in s for s in page.get_text("text").splitlines()):
                    text_found_in_page = True
                    break
            if text_found_in_page:
                continue
            all_texts_were_found = False
            missing_text_list.append({'text':text_item, 'document':candidate_document})
        if all_texts_were_found is False:
            print(missing_text_list)
            raise AssertionError('Some expected texts were not found in document')
    
    @keyword
    def PDF_should_contain_strings(self, expected_text_list, candidate_document):
        """Checks if each item provided in the list ``expected_text_list`` appears in the PDF File ``candidate_document``.
        
        ``expected_text_list`` is a list of strings, ``candidate_document`` is the path to a PDF File.
        
        Examples:

        | @{strings}= | Create List | One String | Another String |
        | PDF Should Contain Strings | ${strings} | candidate.pdf |
        
        """
        doc = fitz.open(candidate_document)
        missing_text_list = []
        all_texts_were_found = None
        for text_item in expected_text_list:
            text_found_in_page = False
            for page in doc.pages():
                if any(text_item in s for s in page.get_text("text").splitlines()):
                    text_found_in_page = True
                    break
            if text_found_in_page:
                continue
            all_texts_were_found = False
            missing_text_list.append({'text':text_item, 'document':candidate_document})
        if all_texts_were_found is False:
            print(missing_text_list)
            raise AssertionError('Some expected texts were not found in document')