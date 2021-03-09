from DocTest.PdfDoc import PdfDoc
from pprint import pprint
from deepdiff import DeepDiff
from robot.api.deco import keyword, library

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

        reference = {}
        reference['Signature']=PdfDoc.get_signature(reference_document)
        reference['Output Intents']=PdfDoc.get_output_intents(reference_document)
        reference['SigFlags']=PdfDoc.get_sig_flags(reference_document)
        candidate = {}
        candidate['Signature']=PdfDoc.get_signature(candidate_document)
        candidate['Output Intents']=PdfDoc.get_output_intents(candidate_document)
        candidate['SigFlags']=PdfDoc.get_sig_flags(candidate_document)
        if reference!=candidate:
            pprint(DeepDiff(reference, candidate, verbose_level=2), width=200)
            print("Reference Document:")
            print(reference)
            print("Candidate Document:")
            print(candidate)           
            raise AssertionError('The compared PDF Document Data is different.')

    @keyword
    def check_text_content(self, expected_text_list, candidate_document):
        """Checks if each item provided in the list ``expected_text_list`` appears in the PDF File ``candidate_document``.
        
        ``expected_text_list`` is a list of strings, ``candidate_document`` is the path to a PDF File.
        
        Examples:

        | @{strings}= | Create List | One String | Another String |
        | Check Text Content | ${strings} | candidate.pdf |
        
        """
        all_texts_were_found = None
        missing_text_list = []
        pdf_content = PdfDoc.get_pdf_content(candidate_document)
        for item in expected_text_list:
            for i in range(len(pdf_content)):
                results = PdfDoc.get_items_with_matching_text(pdf_content[i], item, objecttype='textbox', page_height=pdf_content[i].height)
                for textline in results:
                    (x, y, w, h) = (textline['x'], textline['y'], textline['width'], textline['height'])
                if not results:
                    all_texts_were_found = False
                    missing_text_list.append({'text':item, 'document':candidate_document, 'page':i+1})
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
        pdf_text= PdfDoc.get_pdf_text(candidate_document)
        missing_text_list = []
        all_texts_were_found = None
        for text_item in expected_text_list:
            if not any(text_item in s for s in pdf_text.splitlines()):
                all_texts_were_found = False
                missing_text_list.append({'text':text_item, 'document':candidate_document})
        if all_texts_were_found is False:
            print(missing_text_list)
            raise AssertionError('Some expected texts were not found in document')