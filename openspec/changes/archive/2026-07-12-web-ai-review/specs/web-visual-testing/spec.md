# Spec delta: web-visual-testing (web-ai-review)

## ADDED Requirements

### Requirement: Optional AI-assisted review
Web comparisons SHALL support the library's optional LLM assist (`llm`,
`llm_override`, `llm_prompt` options) and SHALL provide the model with the
capture context — browser, viewport, URL and the DOM-analysis verdict when
available — alongside the visual evidence. Absence of the [ai] extra SHALL
degrade gracefully (warning, normal comparison result).

#### Scenario: AI approves a non-important web change
- **WHEN** `Compare Page To Baseline    home    llm=True    llm_override=True` fails visually and the model approves
- **THEN** the keyword passes and the LLM decision is recorded in the sidecar

#### Scenario: Model sees the web context
- **WHEN** the LLM assist runs for a web comparison
- **THEN** its input includes the capture context and DOM verdict
