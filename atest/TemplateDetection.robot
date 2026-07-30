*** Settings ***
Documentation     Library-level template detection (issue #127).
...               These tests exist at acceptance level because only a real Robot
...               import exercises Robot Framework's argument type conversion — the
...               mechanism that previously rejected documented aliases such as
...               ``movement_detection=classic`` before library code could run.
Library           DocTest.VisualTest    template_detection=sift    movement_detection=classic
Library           Collections

*** Test Cases ***
Library Level Template Detection Is Applied Without Per-Call Argument
    [Documentation]    template_detection=sift at import is used when the call omits detection.
    ${coordinates}=    Image Should Contain Template    testdata/Beach_left.jpg    testdata/Beach_cropped.jpg
    Should Not Be Empty    ${coordinates}

Explicit Detection Argument Overrides The Library Level Setting
    ${coordinates}=    Image Should Contain Template    testdata/Beach_left.jpg    testdata/Beach_cropped.jpg
    ...    detection=template    threshold=0.999
    Should Not Be Empty    ${coordinates}
    Dictionary Should Contain Key    ${coordinates}    confidence

Documented Classic Alias Is Accepted At Import
    [Documentation]    movement_detection=classic is documented in the README and must
    ...    resolve to template rather than being refused during import.
    ${method}=    Get Library Instance    DocTest.VisualTest
    Should Be Equal    ${method.movement_detection}    template

Set Template Detection Keyword Changes The Default
    [Documentation]    Switching to template matching makes the result carry a
    ...    confidence value, which the sift/orb branches do not return.
    ...    No teardown is needed: the library uses Robot's default TEST scope,
    ...    so the next test gets a fresh instance with the import value.
    Set Template Detection    template
    ${coordinates}=    Image Should Contain Template    testdata/Beach_left.jpg    testdata/Beach_cropped.jpg
    ...    threshold=0.999
    Dictionary Should Contain Key    ${coordinates}    confidence

Setter Does Not Leak Into The Next Test
    [Documentation]    Pins the TEST-scope behavior the docs now describe.
    ${lib}=    Get Library Instance    DocTest.VisualTest
    Should Be Equal    ${lib.template_detection}    sift

Unsupported Template Detection Value Is Rejected
    Run Keyword And Expect Error    *Unsupported template detection method 'text'*
    ...    Set Template Detection    text
