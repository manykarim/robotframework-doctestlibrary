# Spec: comparison-correctness

## Purpose
Correctness guarantees of the comparison keywords: boolean option coercion for LLM flags, active barcode masking, block-SSIM sensitivity to inverted content, explicit print-job errors, page-scoped masks for all document types, deterministic document-handle release, and hardened reference downloading.
## Requirements
### Requirement: LLM option flags accept Robot string booleans
`Compare Pdf Documents` SHALL interpret the `llm`, `llm_enabled`, `use_llm`, and `llm_override` keyword options with the same boolean coercion as every other boolean option (`_as_bool`), so the Robot Framework string values `False`/`false`/`no`/`0` disable the feature and `True`/`true`/`yes`/`1` enable it. The LLM MUST NOT be invoked, and an LLM verdict MUST NOT override a comparison result, when the corresponding flag coerces to false.

#### Scenario: String False disables the LLM
- **WHEN** `Compare Pdf Documents` is called with `llm=False` and `llm_override=False` passed as Robot string values on two differing PDFs
- **THEN** no LLM call is made and the keyword fails on the real difference

#### Scenario: String True enables the LLM
- **WHEN** `Compare Pdf Documents` is called with `llm=True` (string) and an LLM client is configured
- **THEN** the LLM path is invoked exactly as with `${True}`

### Requirement: contains_barcodes excludes barcode areas and checks their content
When `Compare Images` is called with `contains_barcodes=${True}`, the library SHALL detect barcodes in both documents, exclude the detected barcode areas from the visual comparison, and compare the decoded barcode values between reference and candidate. With the default `contains_barcodes=${False}` the comparison behavior SHALL be unchanged.

#### Scenario: Barcode render jitter is ignored but content is compared
- **WHEN** `Compare Images` runs with `contains_barcodes=${True}` on two documents whose only pixel difference lies inside a barcode that decodes to the same value
- **THEN** the comparison passes

#### Scenario: Differing barcode content fails
- **WHEN** `Compare Images` runs with `contains_barcodes=${True}` on two documents whose barcodes decode to different values
- **THEN** the comparison fails and the differing barcode values are reported

### Requirement: Block-based SSIM detects anti-correlated blocks
Block-based SSIM comparison SHALL treat a block's raw SSIM score (range −1…1) without absolute-value transformation, so a strongly negative block score (e.g. photometrically inverted content) fails the block similarity check. Reported block-SSIM scores SHALL remain within a sane range for negative raw scores.

#### Scenario: Inverted block fails block-based SSIM
- **WHEN** a comparison runs with `block_based_ssim=${True}` and one block of the candidate is a photometric inversion of the reference (raw block SSIM ≈ −1)
- **THEN** the block check reports the page as different

### Requirement: Print-job comparison errors are explicit
`Compare Print Jobs` SHALL raise a clear, typed error naming the unsupported value when called with a print-job type it cannot handle, and SHALL treat a property present in the reference but missing from the candidate (or vice versa) as a reported difference rather than an internal error.

#### Scenario: Unknown type raises ValueError
- **WHEN** `Compare Print Jobs` is called with type `afp` or a misspelled type
- **THEN** a `ValueError` naming the unsupported type is raised (not `UnboundLocalError`)

#### Scenario: Missing property is a difference
- **WHEN** the reference print job contains a property section the candidate lacks
- **THEN** the keyword fails with the print-jobs-differ assertion listing the missing property (not `TypeError`)

### Requirement: EAST text detection honors non-square dimensions
The EAST text extractor SHALL apply the caller-requested detector width and height without transposition, producing detection geometry consistent with the requested dimensions.

#### Scenario: Non-square EAST size
- **WHEN** text extraction runs with EAST width 320 and height 640
- **THEN** the resized detector input is 320 wide and 640 high and returned box ratios correspond to those axes

### Requirement: Sidecar mask metadata reflects the caller's mask
The JSON result sidecar's mask section SHALL record the mask definition the caller supplied, unaffected by internal processing state such as watermark handling.

#### Scenario: Watermark comparison preserves mask metadata
- **WHEN** `Compare Images` runs with `result_json=${True}`, a `mask` argument, and a `watermark_file` on documents with visual differences
- **THEN** the sidecar's `masks.mask` equals the supplied mask definition (not watermark image data)

### Requirement: Page-scoped ignore areas apply to all document types
Page-scoped ignore areas (`{"page": N, ...}`) SHALL be applied uniformly for every supported input format, including PCL and PostScript documents.

#### Scenario: Page-scoped mask on PCL
- **WHEN** a `.pcl` document is compared with an area mask scoped to `page: 1`
- **THEN** the mask is applied to page 1 exactly as it would be for a PDF input

### Requirement: PDF document handles are released deterministically
Every code path that opens a PDF document handle SHALL close it deterministically (context manager or try/finally), including error paths, rather than relying on garbage collection.

#### Scenario: Text keywords close their documents
- **WHEN** `PDF Should Contain Strings` runs against a PDF (passing or failing)
- **THEN** the underlying document handle is closed before the keyword returns

### Requirement: Hardened reference downloading
URL reference fetching SHALL validate the scheme of every redirect hop against the allowlist (with `ftp://` no longer allowed by default), SHALL abort a transfer as soon as the configured `max_size` is exceeded rather than after completion, and SHALL NOT accumulate temporary files from failed downloads.

#### Scenario: Redirect to disallowed scheme is rejected
- **WHEN** an allowed `https://` URL responds with a redirect to a `file://` or other non-allowlisted scheme
- **THEN** the download fails with a clear error and nothing is fetched from the redirect target

#### Scenario: Oversized download aborts early
- **WHEN** a download exceeds `max_size` mid-transfer
- **THEN** the transfer stops at the limit and the partial temp file is removed
