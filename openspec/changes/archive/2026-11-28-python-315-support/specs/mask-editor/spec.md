# Spec Delta

## ADDED Requirements

### Requirement: Upload-relative mask target is platform-correct
After a document is uploaded in the mask editor, the suggested `masks.json` target SHALL be in the same folder as the uploaded file, whichever path separator the server's platform uses.

#### Scenario: Windows server path
- **WHEN** the upload endpoint returns `C:\data\uploads\9ae81ac8\ref.png`
- **THEN** the editor suggests `C:\data\uploads\9ae81ac8\masks.json` as the mask target, and saving to it succeeds

#### Scenario: POSIX server path
- **WHEN** the upload endpoint returns `/srv/data/uploads/9ae81ac8/ref.png`
- **THEN** the editor suggests `/srv/data/uploads/9ae81ac8/masks.json`

### Requirement: Browser-picked mask target keeps the native separator
When a user enters a new file name in the file browser's save mode, the selected target SHALL be that name inside the browsed directory, joined with the directory path's own separator.

#### Scenario: New file in a Windows directory
- **WHEN** the browser shows `C:\data\artifacts` and the user enters `browser_masks.json`
- **THEN** the selected mask target is `C:\data\artifacts\browser_masks.json`

#### Scenario: New file in a POSIX directory
- **WHEN** the browser shows `/srv/data/artifacts` and the user enters `browser_masks.json`
- **THEN** the selected mask target is `/srv/data/artifacts/browser_masks.json`
