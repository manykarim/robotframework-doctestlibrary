*** Settings ***
Library     Browser    timeout=20s
Library     DocTest.VisualTest
Library     OperatingSystem

*** Test Cases ***
Open Browser
    New Browser    browser=chromium    headless=True
    New Page    url=https://the-internet.herokuapp.com/login
    Compare All Elements    Login
    Fill Text    //input[@name='username']    tomsmith
    Fill Text    //input[@name='password']    SuperSecretPassword!
    Click    button >> text=Login
    Get Text    .flash    *=    You logged into a secure area!
    ${pagename}    Set Variable    Login_Successful
    Compare All Elements    LoginSuccessful

*** Keywords ***
Compare All Elements
    [Arguments]    ${pagename}
    ${elements}    Get Elements    input, button, h1, h2, h3, h4
    FOR    ${element}    IN    @{elements}
        Log    ${element}
        ${nodeType}    Execute JavaScript    (elem) => elem.getAttribute("type")    ${element}
        ${nodeName}    Execute JavaScript    (elem) => elem.getAttribute("name")    ${element}
        ${className}    Execute JavaScript    (elem) => elem.getAttribute("class")    ${element}
        ${id}    Get Property    ${element}    id
        ${reference_screenshot_exists}    Run Keyword And Return Status    File Should Exist
        ...    ${CURDIR}/reference/${pagename}_${nodeType}_${nodeName}_${id}_${className}.png
        IF    ${reference_screenshot_exists}
            Take Screenshot    filename=${CURDIR}/candidate/${pagename}_${nodeType}_${nodeName}_${id}_${className}
            ...    selector=${element}
            Run Keyword And Ignore Error    Compare Images
            ...    ${CURDIR}/reference/${pagename}_${nodeType}_${nodeName}_${id}_${className}.png
            ...    ${CURDIR}/candidate/${pagename}_${nodeType}_${nodeName}_${id}_${className}.png    move_tolerance=1
        ELSE
            Take Screenshot    filename=${CURDIR}/reference/${pagename}_${nodeType}_${nodeName}_${id}_${className}
            ...    selector=${element}
        END
    END
