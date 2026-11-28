# Design

## Context

See proposal.md (Why). The byte-level readers from `fix-print-job-false-passes` (`DocTest/PrintJobParsing.py`) already provide:
- `split_jobs()`, which yields PJL lines plus the language and data of each section;
- `iter_pcl5()`, which yields `PclCommand(parameter, group, terminator, value, payload)` and page-eject tokens;
- `read_pcl5_pages()`, which applies the GhostPCL-verified page-eject rules;
- `iter_pclxl()`, which yields `(operator, {attribute_id: value})`;
- the DSC scanner `read_postscript()`, with features per page;
- `find_ghostscript()` and an EndPage-hook page counter.

The exploration verified these behaviors (Ghostscript 10.08.0):
- A `setpagedevice` redefinition in `userdict`, run under `-dSAFER` before the job, captures every requested page-device dictionary, including `Duplex`/`Tumble` that the output device ignores.
- A DSC header can claim Letter while the code requests A4.

The current page model only records the commands issued *on* a page. A ticket needs the state *in effect* when each page is ejected.

## Goals / Non-Goals

**Goals:**
- A single ticket model and comparison path for PCL 5, PCL XL and PostScript, built on the existing readers without changing their output (the `print-job-comparison` behavior stays as it is).
- Deterministic normalization tables, each traceable to its specification.

**Non-Goals:**
- Device-model databases (PPD or printer-specific tray names). The user mapping covers that.
- Finishing semantics such as staple or punch positions. They stay raw PJL or DSC values.
- Text, colour or ink checks.
- Splitting print jobs.

## Decisions

### D1: A new module `DocTest/PrintJobTicket.py`; the keywords live in `PrintJobTests.py`
The module holds:
- dataclasses `Setting(value, source, raw)`, `PageTicket` and `JobTicket`;
- the normalization tables;
- the per-language state machines;
- comparison and assertion helpers.

`PrintJobTests.py` exposes the three keywords, so the existing library import is enough.
- **Alternative rejected:** a new Robot library. It would need a separate import and documentation entry, for three keywords that belong with the print-job keywords.

### D2: Return type is a plain dictionary
`Get Print Job Ticket` returns `{"languages": [...], "page_count": n, "settings_source": ..., "job": {...}, "pjl": {...}, "pages": [{"page": 1, "paper_size": "A4", ..., "sources": {...}, "raw": {...}}]}`.
- Robot users can use `${ticket}[pages][0][duplex]` without custom classes.
- The comparison helpers accept such a dictionary or a file path.

### D3: Effective-value state machines per language
- **PJL** (PJL Technical Reference, ed. 10, ch. 6):
  - `@PJL DEFAULT` writes the user defaults. `@PJL SET` writes the current environment.
  - The current environment is reloaded from the defaults at every PJL reset: a UEL outside `JOB`…`EOJ`, `@PJL JOB`, `@PJL EOJ`, `@PJL RESET`, and `@PJL INITIALIZE`, which also clears the defaults.
  - So a `DEFAULT` value only reaches the pages after the next reset, while a `SET` value lasts until that reset.
  - The variables used are `PAPER`, `ORIENTATION`, `DUPLEX`/`BINDING`, `COPIES`/`QTY`, `MEDIASOURCE`, `MANUALFEED`, `MEDIATYPE`, `OUTBIN`, `RESOLUTION` and `RENDERMODE`. All variables are also kept raw.
- **PCL 5:**
  - The state is initialized from the PJL current environment plus the language defaults (copies 1, simplex, portrait).
  - Each page-setup command updates the state.
  - At every page eject (the same rules as `read_pcl5_pages`, reused rather than reimplemented) a `PageTicket` snapshot is taken.
  - `ESC E` reloads the PJL current environment, which is how the PCL reset is defined.
  - Handled commands: `&lA` size, `&lH` source (0 = eject, not a tray), `&lO`, `&lS`, `&lX`, `&lG` bin, `&lM` media number, and `&n#Wd<name>` media name. The last needs the payload bytes: `PclCommand` gains an additive `payload_offset` field.
- **PCL XL:**
  - Each `BeginPage` sets MediaSize, MediaSource, MediaType, MediaDestination (attribute 0x24, newly read), Orientation and Simplex/DuplexPageMode.
  - Attributes a page does not set fall back to the PJL state.
  - `PageCopies` on `EndPage` sets copies.
- **PostScript:**
  - The state starts from PJL, then takes document DSC features (`*PageSize`, `*Duplex`, `*InputSlot`, `*OutputBin`, `*MediaType`, `*ManualFeed`), `%%Orientation`, and per-page features and comments.
  - With Ghostscript, the captured `setpagedevice` requests are applied in order with the page they occur on (keys `PageSize`, `Duplex`, `Tumble`, `NumCopies`, `MediaPosition`, `MediaType`, `ManualFeed`, `OutputType`). They take precedence over DSC (D6).
  - The page-device `Orientation` key is *not* used: PLRM 3 Table 6.3 defines it as a rotation for roll-fed media, not portrait/landscape.
  - Orientation comes from DSC orientation comments. Without them it comes from the requested `PageSize` (width > height → landscape).

### D4: Normalization tables (checked against the primary documents)
Sources:
- HP PCL 5 Technical Reference Manual (1992) = TRM;
- PCL 5 Comparison Guide (1999) = CG;
- HP PCL Implementor's Guide 6.0 = IG;
- PCL XL Feature Reference Class 2.0 r1.8 App. G, Class 2.1 and 3.0 Supplements;
- PJL Technical Reference ed. 10 (1997), Table A-2;
- PLRM 3 §6.2; PPD 4.3; DSC 3.0.

- **Paper sizes:**
  - PCL 5 `&l#A` (TRM 5-2, CG 1-1A/1-1C, IG 7.2): 1 Executive, 2 Letter, 3 Legal, 6/11 Ledger, 10 JIS Exec, 24 A6 (IG only), 25 A5, 26 A4, 27 A3, 44 JIS B6, 45 JIS B5, 46 JIS B4, 65 B5 paper, 71 Hagaki, 72 Oufuku-Hagaki, 80 Monarch, 81 Com10, 90 DL, 91 C5, 100 B5 envelope.
  - PCL XL MediaSize: 0 Letter, 1 Legal, 2 A4, 3 Executive, 4 Ledger, 5 A3, 6 Com10, 7 Monarch, 8 C5, 9 DL, 10 JIS B4, 11 JIS B5, 12 B5 envelope, 13 B5 paper (2.1), 14 Hagaki, 15 Oufuku-Hagaki, 16 A5, 17 A6, 18 JIS B6, 19 JIS 8K, 20 JIS 16K, 21 JIS Exec. 96 (eDefaultPaperSize) → `unspecified`. MediaSize *names* are device-dependent (no list in the spec), so they are matched against the common names and otherwise reported raw.
  - PJL `PAPER`: LETTER, LEGAL, A5, A4, EXECUTIVE, LEDGER, A3, COM10, C5, DL, MONARCH, B5 (envelope), JISB4, JISB5, JPOST (Hagaki), JPOSTD (Oufuku-Hagaki).
  - The normalized name describes the sheet size, so B5 paper and B5 envelope (both 176×250 mm) are both `B5`.
  - PostScript dimensions match a name within ±2 pt in either orientation. Unknown sizes are reported as `<w>x<h>mm` rounded to whole millimetres.
- **Duplex**, always reported for the physical sheet:
  - PCL 5 `&l0S`/`1S`/`2S` → simplex/long_edge/short_edge. TRM 4-5 defines these on the physical page, independent of orientation.
  - PJL `DUPLEX=OFF` → simplex; `ON` with `BINDING=LONGEDGE` (default) or `SHORTEDGE` → long_edge or short_edge.
  - PCL XL SimplexPageMode → simplex. DuplexPageMode is relative to the page's orientation (2.1 Supplement p.10): vertical binding = long edge in portrait and short edge in landscape; horizontal is the opposite.
  - PostScript `Duplex false` → simplex. `Tumble` is defined in default user space (PLRM 3 Table 6.4): `Tumble false` = long edge for a portrait page size, short edge for a landscape page size, and the opposite for `Tumble true`.
  - DSC/PPD `*Duplex`: None → simplex, DuplexNoTumble → long_edge, DuplexTumble → short_edge.
- **Orientation:**
  - PCL 5 and PCL XL 0–3 → portrait, landscape, reverse_portrait, reverse_landscape. PCL XL 4 (eDefaultOrientation) → `unspecified`.
  - PJL and DSC know only Portrait/Landscape.
- **Generic tray, bin and media names**, used only where the specs name them:
  - **Trays:**
    - PCL 5 (CG 1-8): 2 → `manual`, 3 → `manual_envelope`, 6 → `envelope`, 7 → `auto`. The numbers 1, 4, 5, 8 and 20+ are printer-specific (e.g. 4 is the lower tray on old models and the MP tray on the LJ4000), so they stay raw.
    - PCL XL: 0 → `unspecified`, 1 → `auto`, 2 → `manual`, 3 → `multipurpose`, 4 → `upper`, 5 → `lower`, 6 → `envelope`, 7 → `third`; 8+ external trays stay raw.
    - PJL: `MANUALFEED=ON` → `manual`; `MEDIASOURCE` TRAY1–3 stay raw.
    - PostScript: `ManualFeed true` and `*ManualFeed True` → `manual`.
  - **Bins:**
    - PCL 5 `&l#G`: 1 → `face_down`, 2 → `face_up` (TRM 4-12 upper/lower, CG face-down/face-up); 0 → `unspecified`; 3+ printer-specific.
    - PCL XL MediaDestination: 0 → `unspecified`, 1 → `face_down`, 2 → `face_up`, 3 → `job_offset`; 5+ external.
    - PJL `OUTBIN`: UPPER → `face_down`, LOWER → `face_up`.
  - **Media:**
    - PCL 5 `&l#M` 0–4 (DeskJet): plain, bond, special, glossy, transparency.
    - PCL 5 `&n#Wd<name>` (name length + 1 = `#`), PCL XL MediaType strings (empty = plain), PJL `MEDIATYPE` (PAPER = plain) and PPD `*MediaType` names are lower-cased with spaces collapsed.

  Everything else is reported as `<language>:<raw>`, with language one of `pcl5`, `pclxl`, `pjl`, `ps`.

### D5: `unspecified` versus defaults
- A setting with no value from the job, PJL or a language-defined default is `unspecified`. It compares unequal to every concrete value, so a missing setting is never silently equal to a guessed printer default.
- Language defaults are copies 1, simplex and portrait for PCL 5 and PCL XL, and copies 1 for PostScript. They are reported with `source: default`.

### D6: PostScript capture through Ghostscript
- Ghostscript runs with `-q -dSAFER -dNOPAUSE -dBATCH -sDEVICE=nullpage -c <spy> -f <job>`.
- The spy prints one line per `setpagedevice` call and one per page end:
  - `DOCTEST-SPD <page> <key>=<value> …`, using `==only` for values;
  - `DOCTEST-PAGE`.
- The output is parsed with a small tokenizer for PostScript arrays, booleans, numbers and names.
- The run reuses `find_ghostscript()` and the error handling from `count_postscript_pages`. A failure falls back to DSC only and records the reason in `settings_source`; it does not fail the keyword, because the DSC data is still valid.

### D7: Comparison output
- Differences are computed as `(scope, setting, reference, candidate)`, where scope is `job`, `pjl` or `page N`.
- Pages beyond the shorter job are reported as `page N: missing in candidate/reference`.
- The `AssertionError` message starts with `The compared print job tickets are different:` and lists up to 20 differences (`… and N more`). The full list and both tickets are logged.
- The ignore handling reuses `_ignored_names()` from `PrintJobTests.py`, so defaults and name matching are the same as in `Compare Print Jobs`.

### D8: Mapping input
- `mapping` accepts a dictionary or a path to a JSON file of the form `{"input_tray": {"pcl5:4": "drawer2", "ps:Tray2": "drawer2"}, "output_bin": {...}, "media_type": {...}}`.
- Keys are the language-qualified raw values exactly as the ticket shows them, so users can copy them from a ticket into the mapping.
- Unknown sections in the mapping are an error, so typos don't silently map nothing.

## Risks / Trade-offs

- **[Normalization table errors]** Each table entry gets a unit test, and a task verifies the tables against the primary specs before release.
- **[Page-relative bindings]** PCL XL DuplexPageMode and PostScript `Tumble` are relative to the page orientation, while PCL 5 and PJL bindings are physical. Converting them to the sheet edge depends on the page's orientation being known; when a PostScript page size is unknown, `Tumble` is interpreted for a portrait page. Unit tests cover portrait and landscape for both.
- **[Jobs that bypass the `setpagedevice` spy]** (for example `systemdict /setpagedevice get exec`) DSC values remain in the ticket, and the `settings_source` field shows which sources were used.
- **[Ghostscript run cost on large PostScript jobs]** The spy uses the `nullpage` device, so the job is interpreted but nothing is rasterized. A benchmark task records the cost on a large job.
- **[Cross-format comparisons fail on job-level PJL]** This is expected: drivers emit different PJL. The documented pattern `ignore_properties=pjl,resolution,color_mode` is shown in the keyword docs and README.

## Migration Plan

This change is purely additive: three new keywords, no behavior change to existing keywords. It ships in a minor release after `fix-print-job-false-passes`. Rollback means not using the new keywords.
