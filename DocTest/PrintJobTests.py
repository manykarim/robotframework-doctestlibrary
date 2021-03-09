from parsimonious.grammar import Grammar, NodeVisitor
import re
from pathlib import Path
from pprint import pprint
from deepdiff import DeepDiff
from robot.api.deco import keyword, library

ROBOT_AUTO_KEYWORDS = False

class PrintJob(object):

    
    def __init__(self, jobtype, properties):
        self.jobtype = jobtype
        self.properties = properties

class PclVisitor(NodeVisitor):

    def __init__(self):
        self.pcl_commands = []
        self.page = 1

    
    def visit_page_command_sequence(self, node, visited_children):
        escape, page_sequence_start, page_command = visited_children
        #Case1: Only one page_command item (No sequence): The children contain paper_source, copies, ..
        #Case2: Multiple page_command items (It's a sequence): The children of items contain paper_source, copies, .
        if type(page_command) is list:
            for item in page_command:
                self.add_page_property(item.children)
        else:
            self.add_page_property(page_command.children)

     
    def add_page_property(self, node):
        value = node[0].text
        if value != '':
            expr_name = node[0].expr_name
            if expr_name == 'paper_source_command':
                value = chop(value, suffix='(h|H)$')     
                self.pcl_commands.append({'page':str(self.page), 'property':'paper_source', 'value':value})    
            elif expr_name == 'copies_command':
                value = chop(value, suffix='(x|X)$')     
                self.pcl_commands.append({'page':str(self.page), 'property':'copies', 'value':value})        
            elif expr_name == 'duplex_command':
                value = chop(value, suffix='(s|S)$')     
                self.pcl_commands.append({'page':str(self.page), 'property':'duplex', 'value':value})   
            elif expr_name == 'page_orientation_command':
                value = chop(value, suffix='(o|O)$')     
                self.pcl_commands.append({'page':str(self.page), 'property':'page_orientation', 'value':value})   
            elif expr_name == 'paper_format_command':
                value = chop(value, suffix='(a|A)$')     
                self.pcl_commands.append({'page':str(self.page), 'property':'paper_format', 'value':value})    

    
    def visit_formfeed(self, node, visited_children):
        self.page += 1

    def generic_visit(self, node, visited_children):
        if not node.expr_name and node.children:
            if len(visited_children) == 1:
                return visited_children[0]
            return visited_children
        return node

class PostscriptVisitor(NodeVisitor):
    def __init__(self):
        self.comments=[]
        self.pages=[]
        self.features=[]
        self.setup = []
        self.header=[]
        self.trailer=[]
        self.pjl_commands=[]

    # def visit_file(self, node, visited_children):
    #     result = {'comments': [], 'pages': [], 'setup': []}
    #     for i in visited_children:
    #         try:
    #             result.update(dict(i))
    #         except Exception as e:
    #             print(i)
    #             print(e)
    #             pass
    #     return result

    def visit_pages(self, node, visited_children):
        page, page_sections, _, document, _, page_trailer = visited_children
        page_number = page[1].strip().split()
        for item in page_sections:
            if item!=None:
                item['page']=page_number[0]
                self.pages.append(item)
        pass


    def visit_page_sections(self, node, visited_children):
        if visited_children[0]!=None:

            if node.children[0].expr_name=='page_setup':
                return {'property':'feature', 'value':visited_children}
            else:
                name, value = visited_children[0].children
                name = re.sub('(%%|[\s]|:)', '', name.text)
                value = value.text.strip()
            return {'property':name, 'value':value}

    def visit_page(self, node, visited_children):
        page, value = visited_children
        return page.text, value.text

    def visit_comments(self, node, visited_children):
        comment, value = visited_children
        comment = re.sub('(%%|[\s]|:)', '', comment.text)
        value = value.text.strip()
        return {'property':comment, 'value':value}
        pass

    def visit_pjl_content(self, node, visited_children):
        self.pjl_commands.append(node.text.strip())
        pass

    def visit_pjl_commands(self, node, visited_children):
        pass

    def visit_feature(self, node, visited_children):
        begin_feature, ppd_feature, _ = visited_children 
        name = begin_feature.children[0]
        feature = begin_feature.children[1]
        feature = feature.text.strip()
        ppd_feature = ppd_feature.text.strip()
        self.features.append({'feature': feature, 'value': ppd_feature})
        return {'feature': feature, 'value': ppd_feature}

    def visit_page_setup(self, node, visited_children):
        begin_page_setup, _, feature, _, _ = visited_children
        if isinstance(feature, dict):
            return feature
    def visit_header(self, node, visited_children):
        _, comments, _ = visited_children
        for comment in comments:
            if comment!=None:
                self.header.append(comment)
        return [comment for comment in comments if comment!=None]

    def visit_document_trailer(self, node, visited_children):
        _, comments = visited_children
        for comment in comments:
            if comment!=None:
                self.trailer.append(comment)
        return [comment for comment in comments if comment!=None]

    def generic_visit(self, node, visited_children):
        if not node.expr_name and node.children:
            if len(visited_children) == 1:
                return visited_children[0]
            return visited_children
        return node
       

def chop(text, prefix=None, suffix=None):
    if prefix!=None:
        text=re.sub(prefix, '', text)
    if suffix!=None:
        text=re.sub(suffix, '', text)
    return text


def get_pcl_print_job(filename):

    grammar_pcl_commands = Grammar(
        r"""
        file = command*
        command     = (pcl_command / no_pcl_command / ";" / space / linefeed/ formfeed/ carriage_return / other_data)
        reset       = escape "E"
        no_pcl_command = ~r"[A-Z]{2}[0-9.,-]*"
        pcl_command = raster_image_command_sequence / page_command_sequence / other_pcl_command_sequence
        raster_image_command_sequence = escape ~r"\*r.*?(0|1)(a|A)[\S\s]*?\*r(B|C)"
        other_pcl_command_sequence = escape other_pcl_command_sequence_start other_pcl_command
        page_command_sequence = escape page_command_sequence_start page_command+
        other_pcl_command_sequence_start = ~r"[&\(\)\*%]{1}[a-lm-z0-9]{1}"
        other_pcl_command = ~r".+?(?=([\x1b]|[\x0c]))"
        page_command = paper_source_command / copies_command / duplex_command / page_orientation_command / paper_format_command / other_page_command
        page_command_sequence_start = ~r"&l"
        paper_source_command = ~r"[0-6]{1}(h|H)" 
        copies_command = ~r"(\d)+(x|X)"
        duplex_command = ~r"[0-2]{1}(s|S)"
        page_orientation_command = ~r"[0-3]{1}(o|O)"
        paper_format_command = ~r"(\d)+(a|A)"
        other_page_command = ~r"(\d)+[a-zA-Z]{1}"
        pcl_special_code = "=" / "9" / "E" / "Y" / "Z" / "%-12345X" 
        pcl_code    = ~r"[&\(\)\*%]{1}.*([A-Z]|@|ID){1}" &escape
        whitespace = ~r"\s"
        other_data     = ~r".+?"
        escape      = ~r"[\x1b]"
        other_pcl   = ~r"[\x08\x0d\x0c\x0a\x0f\x0e\x09\x03]"
        space       = " "
        linefeed      = ~r"[\x0a]"
        formfeed    = ~r"[\x0c]"
        carriage_return = ~r"[\x0d]"
        """
    )


    with open(filename, encoding="utf8", errors="ignore") as f:
        content = f.read()

    tree = grammar_pcl_commands.parse(content)

    pv = PclVisitor()
    pv.visit(tree)
    
    pclPrintJob = PrintJob('pcl', [{'property':'pcl_commands', 'value':pv.pcl_commands}])
    return pclPrintJob

def get_postscript_print_job(filename):
    grammar_postscript_commands = Grammar(
        r"""
        file = header defaults? procedure_definitions document_setup pages+ document_trailer eof emptyline?
        header = header_start (comments/pjl_commands)* end_comments
        header_start = "%!PS-Adobe-3.0" linefeed
        comments = comment_type dataline
        comment_type = title_comment / copyright_comment / creator_comment / creation_date_comment / bounding_box_comment / orientation_comment / pages_comment / document_needed_resources_comment / document_supplied_resources_comment / document_data_comment / language_level_comment
        title_comment = "%%Title:"
        copyright_comment = "%%Copyright:"
        creator_comment = "%%Creator:"
        creation_date_comment = "%%CreationDate:"
        bounding_box_comment = "%%BoundingBox:"
        orientation_comment = "%%Orientation:"
        pages_comment = "%%Pages:"
        document_needed_resources_comment = "%%DocumentNeededResources:"
        document_supplied_resources_comment = "%%DocumentSuppliedResources:"
        document_data_comment = "%%DocumentData:"
        language_level_comment = "%%LanguageLevel:"
        end_comments= "%%EndComments" linefeed
        pjl_commands = begin_pjl pjl_content*
        pjl_content = (pjl_prefix _ dataline)
        pjl_prefix = "@PJL"
        begin_pjl = ~r"[\x1b]%-12345X"
        defaults = begin_defaults dataline* end_defaults
        begin_defaults = "%%BeginDefaults:"
        end_defaults = "%%EndDefaults" linefeed
        procedure_definitions = begin_prolog dataline* resource* dataline* end_prolog
        begin_prolog = "%%BeginProlog" linefeed
        end_prolog = "%%EndProlog" linefeed
        resource = begin_resource dataline* end_resource
        begin_resource = "%%BeginResource:" anything linefeed
        end_resource = "%%EndResource" linefeed
        document_setup = begin_setup dataline* end_setup
        begin_setup = "%%BeginSetup" linefeed
        end_setup = "%%EndSetup" linefeed
        pages = page page_sections* dataline* document? dataline* page_trailer?
        page = "%%Page:" dataline
        page_sections =  (page_bounding_box / page_orientation / page_setup)
        page_trailer = "%%PageTrailer" dataline*
        page_bounding_box = "%%PageBoundingBox:" dataline
        page_orientation = "%%PageOrientation:" dataline
        document = begin_document dataline* end_document
        begin_document = "%%BeginDocument:" dataline
        end_document = "%%EndDocument" linefeed
        feature = begin_feature dataline* end_feature
        begin_feature = "%%BeginFeature:" dataline
        end_feature = "%%EndFeature" linefeed
        page_setup = begin_page_setup dataline* feature* dataline* end_page_setup
        begin_page_setup = "%%BeginPageSetup"
        end_page_setup= "%%EndPageSetup" linefeed
        document_trailer = trailer comments*
        trailer = "%%Trailer" linefeed
        dataline = (!postscript_prefix anything linefeed) postscript_continue*
        anything = ~r".*"
        _       = " "*
        eof = "%%EOF" linefeed
        linefeed      = ~r"[\x0a]"
        formfeed    = ~r"[\x0c]"
        whitespace = ~r"\s"
        postscript_prefix = ~r"%%[A-Z]"
        postscript_continue = "%%+" anything linefeed
        empty = ""
        ws          = ~"\s*"
        emptyline   = ws+
        """
    )
    with open(filename, encoding="utf8", errors="ignore") as f:
        content = f.read()

    tree = grammar_postscript_commands.parse(content)
    pv = PostscriptVisitor()
    pv.visit(tree)

    properties = []
    properties.append({'property':'header', 'value':pv.header})
    properties.append({'property':'pjl_commands', 'value':pv.pjl_commands})
    properties.append({'property':'pages', 'value':pv.pages})
    properties.append({'property':'trailer', 'value':pv.trailer})


    postscript_printjob = PrintJob('postscript', properties)
    return postscript_printjob



        
@keyword
def compare_print_jobs(type, reference_file, test_file):
    """Compares several print job metadata/properties of ``reference_file`` and ``test_file``.
        
        ``type`` can either be ``pcl`` or ``ps`` (postscript).

        The compared properties depend on the ``type``.

            PCL
                
                - Paper Source
                - Copies
                - Duplex
                - Page Orientation
                - Paper Format
            
            Postscript
                
                - Comments
                - Pages
                - Features
                - Setup
                - Header
                - Trailer
                - PJL Commands
        
        Result is passed if all properties are equal. 
        
        ``reference_file`` and ``test_file`` may be .ps or .pcl files


        Examples:
        | = Keyword =    |  = reference_file =  | = test_file =       |  = comment = |
        | Compare Print Jobs | reference.pcl | candidate.pcl | #Performs a property comparison of both files |
        | Compare Print Jobs | reference.ps | candidate.ps | #Performs a property comparison of both files |
        
        
        """
    if type=='ps':
        reference_print_job = get_postscript_print_job(reference_file)
        test_print_job = get_postscript_print_job(test_file)
    elif type=='pcl':
        reference_print_job = get_pcl_print_job(reference_file)
        test_print_job = get_pcl_print_job(test_file)
    elif type=='afp':
        pass
    compare_properties(reference_print_job, test_print_job)

def compare_properties(reference_print_job, test_print_job):
    properties_are_equal = True
    list_difference = []
    print("Reference File:")
    pprint(reference_print_job.properties)
    
    print("\n")
    print("Candidate File:")
    pprint(test_print_job.properties)
    
    print("\n")
    pprint(DeepDiff(reference_print_job.properties, test_print_job.properties, verbose_level=2))
    for reference_property_item in reference_print_job.properties:
        test_property_item = next((item for item in test_print_job.properties if item["property"] == reference_property_item["property"]), None)
        if reference_property_item['value']!=test_property_item['value']:
            properties_are_equal = False

            for x in reference_property_item['value']:
                if x not in test_property_item['value']:
                    list_difference.append({'file':'reference', 'property':reference_property_item['property'], 'value':x})
            
            for x in test_property_item['value']:
                if x not in reference_property_item['value']:
                    list_difference.append({'file':'test', 'property':test_property_item['property'], 'value':x})

            
            
            #print("Reference Property ", reference_property_item['property'], ":", reference_property_item['value'], " is not equal to ","Test Property ", test_property_item['property'], ":", test_property_item['value'])
    if properties_are_equal==False:
        
        pprint(list_difference)
        print("\n")
#        print(*list_difference, sep = "\n") 
        raise AssertionError('The compared print jobs are different.')

@keyword
def check_print_job_property(print_job, property, value):
    """Checks if one specific ``property`` of a ``print_job`` file has an expected ``value``.
        
        ``print_job`` is a printer file, with filetype ``pcl`` or ``ps`` (postscript).

        The possible compared ``property`` depends on the ``type``.

            PCL
                
                - Paper Source
                - Copies
                - Duplex
                - Page Orientation
                - Paper Format
            
            Postscript
                
                - Comments
                - Pages
                - Features
                - Setup
                - Header
                - Trailer
                - PJL Commands
        
        Result is passed if the value of ``property`` is equal to ``value``. 
        
        Examples:

        | ${value1} =  | Create Dictionary  | page=1  | property=copies  | value=1 |
        | ${value2} =  | Create Dictionary  | page=1  | property=paper_source  | value=2 |
        | Check Print Job Property  | pcl_file.pcl  | pcl_commands  | ${value1} |
        | Check Print Job Property  | pcl_file.pcl  | pcl_commands  | ${value2} |

        | ${value1} =  | Create Dictionary  | property=Copyright  | value=ExampleCompany Inc. |
        | ${value2} =  | Create Dictionary  | property=LanguageLevel  | value=2 |
        | Check Print Job Property  | postscript_file.ps  | header  | ${value1} |
        | Check Print Job Property  | postscript_file.ps  | header  | ${value2} |
        
        """

    test_property_item = next((item for item in print_job.properties if item["property"] == property), None)
    if not test_property_item:
        raise AssertionError('The property does not exist:', property, value)
    if value not in test_property_item['value']:
        print("Expected property: ", property, "\n\nExpected value:\n")
        pprint(value)
        print(" \nActual value:\n")
        pprint(test_property_item['value'])
        raise AssertionError('The print job property check failed.')

def printTable(myDict, colList=None):
   """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
   If column names (colList) aren't specified, they will show in random order.
   Author: Thierry Husson - Use it as you want but don't blame me.
   """
   if type(myDict) is not list:
       myDict = [myDict]
   if not colList: colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict: myList.append([str(item[col] if item[col] is not None else '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   for item in myList: print(formatStr.format(*item))