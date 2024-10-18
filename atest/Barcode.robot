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
    Should Be True    $data[0] == {'x': 757, 'y': 1620, 'height': 207, 'width': 207, 'value': 'This is a QR Code by TEC-IT'}
    Should Be True    $data[1] == {'x': 1198, 'y': 1598, 'height': 244, 'width': 244, 'value': 'This is a QR Code by TEC-IT for mobile applications'}
    Should Be True    $data[2] == {'x': 160, 'y': 1651, 'height': 122, 'width': 413, 'value': '1234567890'}
    Should Be True    $data[3] == {'x': 467, 'y': 1309, 'height': 159, 'width': 663, 'value': 'ABC-1234'}
    Should Be True    $data[4] == {'x': 509, 'y': 1021, 'height': 159, 'width': 564, 'value': 'ABC-1234-/+'}
    Should Be True    $data[5] == {'x': 485, 'y': 725, 'height': 159, 'width': 629, 'value': 'ABC-abc-1234'}
    Should Be True    $data[6] == {'x': 312, 'y': 399, 'height': 159, 'width': 204, 'value': '0012345000065'}
    Should Be True    $data[7] == {'x': 1039, 'y': 399, 'height': 159, 'width': 278, 'value': '90311017'}
    Should Be True    $data[8] == {'x': 984, 'y': 93, 'height': 158, 'width': 396, 'value': '0725272730706'}
    Should Be True    $data[9] == {'x': 236, 'y': 90, 'height': 158, 'width': 396, 'value': '9780201379624'}
    Should Be True    $data[10] == {'x': 480, 'y': 2025, 'height': 183, 'width': 184, 'value': 'This is a Data Matrix by TEC-IT'}
    Should Be True    $data[11] == {'x': 979, 'y': 1971, 'height': 271, 'width': 272, 'value': 'This is a Data Matrix by TEC-IT'}

Read Datamatrices with Assertion Engine
    Get Barcodes    testdata/datamatrix.png    contains    Stegosaurus
    Get Barcodes    testdata/datamatrix.png    contains    Plesiosaurus
    Get Barcodes    testdata/datamatrix.png    not contains    Brontosaurus
