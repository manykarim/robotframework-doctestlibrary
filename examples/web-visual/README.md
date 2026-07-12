# Web Visual Testing Examples

A runnable tour of `DocTest.WebVisualTest` against a small gallery of realistic
pages (`pages/`): an app dashboard with dynamic widgets, a checkout form, a
responsive product grid, SVG + canvas charts, and a long typographic article.

## Run it

```bash
pip install robotframework-doctestlibrary[browser]
rfbrowser init
robot web_visual_demo.robot     # 1st run: creates baselines (passes with warnings)
robot web_visual_demo.robot     # 2nd run: real comparisons against the baselines
```

Baselines land in `./visual_baselines/` as plain PNGs (plus `.dom.json` semantic
snapshots — the suite enables `dom_analysis`). Commit baselines to version
control in real projects.

## Things to try

- **Break something**: edit `pages/dashboard.html` (change a metric), re-run,
  and watch the comparison fail with diff regions.
- **Update baselines**: `robot --variable REFERENCE_RUN:True web_visual_demo.robot`
  overwrites all baselines with fresh captures.
- **Review visually**: run with sidecars and load the run into the dashboard:

  ```bash
  pip install robotframework-doctestlibrary[dashboard]
  robot --variable REFERENCE_RUN:False web_visual_demo.robot
  doctest-dashboard serve &
  doctest-dashboard ingest output.xml
  ```

  Failures appear with side-by-side/overlay/highlight viewers, diff-region
  navigation and one-click baseline promotion.
- **Selenium instead of Playwright**: import `SeleniumLibrary` in the suite and
  use `full_page=False` — the same keywords work unchanged.

Each test in `web_visual_demo.robot` is commented with the feature it shows;
the full guide lives in
[docs/web-visual-testing.md](../../docs/web-visual-testing.md).
