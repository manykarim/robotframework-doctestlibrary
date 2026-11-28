# Spec Delta

## MODIFIED Requirements

### Requirement: Version-conditional dependencies preserved as markers
Python-version-dependent constraints SHALL be expressed as PEP 508 environment markers preserving the selection intent for Python 3.11–3.15:
- numpy `>=2.1` on 3.13+ and `>=2.5.2` on 3.15+
- scipy `>=1.11` on 3.12+
- scikit-image `>=0.22` on 3.12, `>=0.25` on 3.13+, and `>=0.27.0rc1` on 3.15+ (an explicitly allowed pre-release, the first with 3.15 wheels)
- deepdiff `>=6.0` on 3.12+ and `>=9.2.1` on 3.15+
- setuptools present on 3.12+

Dependencies whose next major release breaks the library SHALL carry an upper bound: PyMuPDF `>=1.26,<2`, opencv-python-headless `>=4.8,<5`, pydantic-ai-slim `>=1.107.7,<3` (the `[ai]` extra). `requires-python` SHALL be `>=3.11`: Python 3.10 is past end of life and unsupported by the current scientific stack. Python 3.16 SHALL NOT be declared supported until every compiled dependency publishes wheels for it.

#### Scenario: Resolution matches the expectation table per interpreter
- **WHEN** the environment is synced for each of Python 3.11, 3.12, 3.13, 3.14, and 3.15
- **THEN** the resolved versions of numpy, scipy, scikit-image, deepdiff, PyMuPDF, opencv-python-headless, and pydantic-ai-slim satisfy the documented expectation table for that interpreter

#### Scenario: Binary-only install on Python 3.15
- **WHEN** the built wheel is installed with all extras into a fresh Python 3.15 environment on Windows or Linux using pip with source builds disabled
- **THEN** installation succeeds, with scikit-image resolved to a 0.27 release and no package built from source

#### Scenario: Unsupported interpreter is rejected
- **WHEN** a user attempts to install the package on Python 3.10
- **THEN** the installer refuses the package because of `requires-python`
