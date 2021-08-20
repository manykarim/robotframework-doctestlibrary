from inspect import signature
from DocTest.PdfDoc import PdfDoc
from pprint import pprint
from deepdiff import DeepDiff
from robot.api.deco import keyword, library
import fitz
import re

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
        mask = kwargs.pop('mask', None)
        check_pdf_text = bool(kwargs.pop('check_pdf_text', False))
        compare = (kwargs.pop('compare', "all"))
        compare = [x.strip() for x in compare.split(',')]
        ref_doc = fitz.open(reference_document)
        cand_doc = fitz.open(candidate_document)
        reference = {}
        reference['pages'] = []
        reference['metadata']=ref_doc.metadata
        reference['page_count']=ref_doc.page_count
        reference['sigflags']=ref_doc.get_sigflags()
        for i, page in enumerate(ref_doc.pages()):
            signature_list = []
            text = [x for x in page.get_text("text").splitlines() if not is_masked(x, mask)]
            for widget in page.widgets():
                if widget.is_signed:
                    signature_list.append(list((widget.field_name, widget.field_label, widget.field_value)))
            reference['pages'].append(dict([('number',i+1), ('fonts', page.get_fonts()), ('images', page.get_images()), ('rotation', page.rotation), ('mediabox', page.mediabox), ('signatures', signature_list),('text', text)]))


        candidate = {}
        candidate['pages'] = []
        candidate['metadata']=cand_doc.metadata
        candidate['page_count']=cand_doc.page_count
        candidate['sigflags']=cand_doc.get_sigflags()
        for i, page in enumerate(cand_doc.pages()):
            signature_list = []
            text = [x for x in page.get_text("text").splitlines() if not is_masked(x, mask)]
            for widget in page.widgets():
                if widget.is_signed:
                    signature_list.append(list((widget.field_name, widget.field_label, widget.field_value)))
            candidate['pages'].append(dict([('number',i+1), ('fonts', page.get_fonts()), ('images', page.get_images()), ('rotation', page.rotation), ('mediabox', page.mediabox), ('signatures', signature_list),('text', text)]))

        differences_detected=False

        if 'metadata' in compare or 'all' in compare:
            diff = DeepDiff(reference['metadata'], candidate['metadata'])
            if diff != {}:
                differences_detected=True
                print("Different metadata")
                pprint(diff, width=200)
        if 'signatures' in compare or 'all' in compare:
            diff = DeepDiff(reference['sigflags'], candidate['sigflags'])
            if diff != {}:
                differences_detected=True
                print("Different signature")
                pprint(diff, width=200)
        for ref_page, cand_page in zip(reference['pages'], candidate['pages']):
            diff = DeepDiff(ref_page['rotation'], cand_page['rotation'])
            if diff != {}:
                differences_detected=True
                print("Different rotation")
                pprint(diff, width=200)
            diff = DeepDiff(ref_page['mediabox'], cand_page['mediabox'])
            if diff != {}:
                differences_detected=True
                print("Different mediabox")
                pprint(diff, width=200)
            if 'text' in compare or 'all' in compare:
                diff = DeepDiff(ref_page['text'], cand_page['text'])
                if diff != {}:
                    differences_detected=True
                    print("Different text")
                    pprint(diff, width=200)
            if 'fonts' in compare or 'all' in compare:
                diff = DeepDiff(ref_page['fonts'], cand_page['fonts'])
                if diff != {}:
                    differences_detected=True
                    print("Different fonts")
                    pprint(diff, width=200)
            if 'images' in compare or 'all' in compare:
                diff = DeepDiff(ref_page['images'], cand_page['images'])
                if diff != {}:
                    differences_detected=True
                    print("Different images")
                    pprint(diff, width=200)
            if 'signatures' in compare or 'all' in compare:
                diff = DeepDiff(ref_page['signatures'], cand_page['signatures'])
                if diff != {}:
                    differences_detected=True
                    print("Different signatures")
                    pprint(diff, width=200)

        if differences_detected:             
            raise AssertionError('The compared PDF Document Data is different.')
        # if reference!=candidate:
        #     pprint(DeepDiff(reference, candidate), width=200)
        #     print("Reference Document:")
        #     pprint(reference)
        #     print("Candidate Document:")
        #     pprint(candidate)           
        #     raise AssertionError('The compared PDF Document Data is different.')

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

def is_masked(text, mask):
    if isinstance(mask, str):
        mask = [mask]
    if isinstance(mask, list):
        for single_mask in mask:
            if re.match(single_mask, text):
                return True
    return  False



