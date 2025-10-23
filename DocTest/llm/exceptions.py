class LLMDependencyError(RuntimeError):
    """Raised when optional LLM dependencies are not available."""

    def __init__(self, message: str = ""):
        default = (
            "LLM support requires the optional dependencies. Install with "
            "'pip install robotframework-doctestlibrary[ai]' and provide the necessary "
            "configuration values."
        )
        super().__init__(message or default)
