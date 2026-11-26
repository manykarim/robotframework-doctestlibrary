# Spec Delta

## Purpose

Defines how the LLM-assisted comparison and extraction features interact with the pydantic-ai runtime installed through the `[ai]` extra, so they keep working across the supported runtime versions and model endpoints.

## ADDED Requirements

### Requirement: Supported pydantic-ai runtime range
The `[ai]` extra SHALL declare a bounded pydantic-ai range covering every major version the LLM client is verified against (currently 1.40 up to, but excluding, 3). The LLM keywords SHALL behave identically on any version within that range.

#### Scenario: Works on the oldest supported runtime
- **WHEN** the LLM client runs an agent request with pydantic-ai at the declared lower bound
- **THEN** the request is constructed without error and a structured LLM decision is returned

#### Scenario: Works on the newest supported major
- **WHEN** the LLM client runs the same request with the newest allowed pydantic-ai 2.x
- **THEN** the request is constructed without error and returns the same structured decision

#### Scenario: Unverified major is not installed
- **WHEN** a user installs `robotframework-doctestlibrary[ai]` after a pydantic-ai 3.0 release exists
- **THEN** the resolver selects a version within the declared range instead of 3.x

### Requirement: Output retry budget is honored
The configured output retry budget (`llm_output_retries` / `DOCTEST_LLM_OUTPUT_RETRIES`, default 3) SHALL bound how many times a malformed model response is retried for output validation, on every supported runtime version.

#### Scenario: Malformed responses are retried up to the budget
- **WHEN** the model returns unparsable output on every attempt and the budget is 3
- **THEN** the model is called 4 times (initial attempt plus 3 retries) before the run fails

### Requirement: OpenAI provider uses the Chat Completions API
When the provider is `openai` (the default), including a custom `base_url` pointing at an OpenAI-compatible endpoint, the LLM client SHALL send requests through the Chat Completions API on every supported runtime version.

#### Scenario: OpenAI-compatible endpoint keeps working after a runtime upgrade
- **WHEN** the provider is `openai`, `DOCTEST_LLM_BASE_URL` points to an OpenAI-compatible endpoint, and pydantic-ai 2.x is installed
- **THEN** the request targets the Chat Completions endpoint of that base URL, as it did with pydantic-ai 1.x

#### Scenario: Non-OpenAI providers are passed through
- **WHEN** the provider is set to another value such as `anthropic` or `azure`
- **THEN** the model identifier uses that provider's prefix unchanged

### Requirement: No runtime console banner
Using the LLM keywords SHALL NOT print promotional or observability banners from the LLM runtime to the console, unless the user explicitly configured the runtime to show them.

#### Scenario: Robot console stays clean
- **WHEN** a Robot suite runs an LLM-assisted comparison with pydantic-ai 2.x and no banner-related environment variable is set
- **THEN** the console output contains no pydantic-ai/Logfire banner

#### Scenario: User preference wins
- **WHEN** the user has explicitly set the runtime's banner environment variable before the library is imported
- **THEN** the library does not override that value
