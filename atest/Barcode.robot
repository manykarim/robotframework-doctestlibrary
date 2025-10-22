*** Settings ***
Library    DocTest.VisualTest    show_diff=true    take_screenshots=true    screenshot_format=png
Library    Collections

*** Test Cases ***
Read Datamatrices in Image
    ${data}    Get Barcodes    testdata/datamatrix.png
    Length Should Be    ${data}    2
    Should Be True    $data[0] == {'x': 5, 'y': 7, 'height': 95, 'width': 96, 'value': 'Stegosaurus'} 
    Should Be True    $data[1] == {'x': 298, 'y': 7, 'height': 95, 'width': 95, 'value': 'Plesiosaurus'}
    Should Be True    $data[0]['value'] == 'Stegosaurus'    
    Should Be True    $data[1]['value'] == 'Plesiosaurus'    

Read Barcodes in PDF
    ${data}    Get Barcodes    testdata/sample_barcodes.pdf
    Length Should Be    ${data}    12
    ${expected_values}    Create List    This is a QR Code by TEC-IT    This is a QR Code by TEC-IT for mobile applications    1234567890    ABC-1234    ABC-1234-/+    ABC-abc-1234    0012345000065    90311017    0725272730706    9780201379624    This is a Data Matrix by TEC-IT    This is a Data Matrix by TEC-IT
    ${expected}    Evaluate    [{'x': 757, 'y': 1620, 'height': 207, 'width': 207}, {'x': 1198, 'y': 1598, 'height': 244, 'width': 244}, {'x': 160, 'y': 1651, 'height': 122, 'width': 413}, {'x': 467, 'y': 1309, 'height': 159, 'width': 663}, {'x': 509, 'y': 1021, 'height': 159, 'width': 564}, {'x': 485, 'y': 725, 'height': 159, 'width': 629}, {'x': 312, 'y': 399, 'height': 159, 'width': 204}, {'x': 1039, 'y': 399, 'height': 159, 'width': 278}, {'x': 984, 'y': 93, 'height': 158, 'width': 396}, {'x': 236, 'y': 90, 'height': 158, 'width': 396}, {'x': 480, 'y': 2025, 'height': 183, 'width': 184}, {'x': 979, 'y': 1971, 'height': 271, 'width': 272}]
    Should Be True    all(actual['value'] == expected for actual, expected in zip(${data}, ${expected_values}))
    Should Be True    all(abs(actual[key]-exp[key]) <= 1 for actual, exp in zip(${data}, ${expected}) for key in ('x','y','width','height'))


Read Datamatrices with Assertion Engine
    Get Barcodes    testdata/datamatrix.png    contains    Stegosaurus
    Get Barcodes    testdata/datamatrix.png    contains    Plesiosaurus
    Get Barcodes    testdata/datamatrix.png    not contains    Brontosaurus

*** Keywords ***
Assert Barcode Approximately
    [Arguments]    ${actual}    ${expected}
    ${actual}=    Convert To Dictionary If Needed    ${actual}
    ${expected}=    Convert To Dictionary If Needed    ${expected}
    ${actual_value}=    Get From Dictionary    ${actual}    value
    ${expected_value}=    Get From Dictionary    ${expected}    value
    Should Be Equal    ${actual_value}    ${expected_value}
    FOR    ${key}    IN    x    y    width    height
        ${actual_coord}=    Get From Dictionary    ${actual}    ${key}
        ${expected_coord}=    Get From Dictionary    ${expected}    ${key}
        ${diff}=    Evaluate    abs(${actual_coord} - ${expected_coord})
        Should Be True    ${diff} <= 1
    END

Convert To Dictionary If Needed
    [Arguments]    ${item}
    ${status}    ${_}=    Run Keyword And Ignore Error    Get From Dictionary    ${item}    value
    Run Keyword If    '${status}' == 'PASS'    Return From Keyword    ${item}
    ${literal}=    Evaluate    __import__('ast').literal_eval(${item})
    Return From Keyword    ${literal}
