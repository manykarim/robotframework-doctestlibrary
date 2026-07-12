import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, assetUrl, ComparisonDetail, Page, Region } from "./api";
import { useViewTransform, ZoomPane } from "./ZoomPane";

type Mode = "side-by-side" | "overlay" | "blink" | "swipe" | "highlight";
const MODES: Mode[] = ["side-by-side", "overlay", "blink", "swipe", "highlight"];

export function ComparisonView({ comparisonId }: { comparisonId: number }) {
  const [detail, setDetail] = useState<ComparisonDetail | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [mode, setMode] = useState<Mode>("overlay");
  const [regionIndex, setRegionIndex] = useState(-1);
  const [message, setMessage] = useState<{ kind: string; text: string } | null>(null);
  const [reason, setReason] = useState("");
  const [regionText, setRegionText] = useState<
    { same: boolean; reference_text: string; candidate_text: string } | null | "loading"
  >(null);
  const [history, setHistory] = useState<any[]>([]);

  const load = () =>
    api.comparison(comparisonId).then(setDetail).catch((e) =>
      setMessage({ kind: "error", text: e.message }),
    );
  useEffect(() => {
    load();
    api.history(comparisonId).then((body) => setHistory(body.history)).catch(() => setHistory([]));
  }, [comparisonId]);

  const page: Page | undefined = detail?.pages[pageIndex];
  const regions: Region[] = page?.regions ?? [];
  const highlightImage = page?.images["candidate_with_diff"];
  // "highlight" is only offered when the sidecar shipped the rendering (v1.1+)
  const availableModes: Mode[] = highlightImage ? MODES : MODES.slice(0, 4) as Mode[];

  useEffect(() => {
    if (mode === "highlight" && !highlightImage) setMode("overlay");
  }, [mode, highlightImage]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((event.target as HTMLElement).tagName)) return;
      const num = parseInt(event.key, 10);
      if (num >= 1 && num <= availableModes.length) setMode(availableModes[num - 1]);
      if (event.key === "n") nextRegion(1);
      if (event.key === "p") nextRegion(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const nextRegion = (step: number) => {
    if (!regions.length) return;
    setRegionText(null);
    setRegionIndex((current) => (current + step + regions.length) % regions.length);
  };

  const explainRegion = async () => {
    if (!detail || !page || regionIndex < 0) return;
    setRegionText("loading");
    try {
      const result = await api.regionText(detail.id, page.page_no, regions[regionIndex]);
      setRegionText(result);
    } catch (e: any) {
      setRegionText(null);
      setMessage({ kind: "error", text: e.message });
    }
  };

  const act = async (action: () => Promise<any>, successText: string) => {
    setMessage(null);
    try {
      await action();
      setMessage({ kind: "success", text: successText });
      await load();
    } catch (e: any) {
      const alternatives = e.detail?.alternatives;
      setMessage({
        kind: "error",
        text: alternatives ? `${e.message} (alternatives: ${alternatives.join(", ")})` : e.message,
      });
    }
  };

  if (!detail) return <p className="note">Loading…{message && ` ${message.text}`}</p>;

  const isDegraded = !!detail.degraded;
  const dpi = detail.sidecar_json?.reference?.dpi;
  const title = detail.label || detail.test_name || detail.keyword;
  const hasPages = detail.pages.length > 0;

  return (
    <div>
      <div className="toolbar">
        <a href="javascript:history.back()">← Back</a>
        <strong data-testid="comparison-title">{title}</strong>
        <span className="chip" data-testid="comparison-keyword">{detail.keyword}</span>
        {detail.suite && <span className="note" data-testid="comparison-suite">{detail.suite}</span>}
        <span className={`badge ${detail.status}`} data-testid="comparison-status">
          {detail.status}
        </span>
        <span className={`badge ${detail.review_state}`} data-testid="review-state">
          {detail.review_state}
        </span>
        {isDegraded && <span className="badge degraded">degraded</span>}
        {dpi && <span className="dpi-banner" data-testid="dpi-banner">DPI {dpi}</span>}
        <ContextChips context={detail.sidecar_json?.context} />
      </div>

      {message && (
        <p className={message.kind} data-testid="action-message">
          {message.text}
        </p>
      )}

      {isDegraded ? (
        <DegradedView detail={detail} />
      ) : hasPages ? (
        <>
          <div className="toolbar">
            {detail.pages.map((p, index) => (
              <button
                key={p.id}
                className={index === pageIndex ? "active" : ""}
                data-testid={`page-tab-${p.page_no}`}
                onClick={() => {
                  setPageIndex(index);
                  setRegionIndex(-1);
                }}
              >
                Page {p.page_no} {p.status === "FAIL" ? "✗" : "✓"}
              </button>
            ))}
          </div>
          <div className="toolbar">
            {availableModes.map((m, index) => (
              <button
                key={m}
                className={mode === m ? "active" : ""}
                data-testid={`mode-${m}`}
                onClick={() => setMode(m)}
              >
                {index + 1} {m}
              </button>
            ))}
            <button data-testid="next-region" onClick={() => nextRegion(1)} disabled={!regions.length}>
              next diff (n)
            </button>
            <button data-testid="prev-region" onClick={() => nextRegion(-1)} disabled={!regions.length}>
              prev diff (p)
            </button>
            {regionIndex >= 0 && regions[regionIndex] && (
              <>
                <span className="note" data-testid="region-indicator">
                  region {regionIndex + 1}/{regions.length}
                </span>
                <a
                  data-testid="add-mask-from-region"
                  href={editorLink(detail, regions[regionIndex], dpi)}
                >
                  <button>Add ignore mask</button>
                </a>
                <button data-testid="explain-region" onClick={explainRegion}>
                  Explain region (text)
                </button>
              </>
            )}
            <span className="note">wheel = zoom · drag = pan · double-click = fit</span>
          </div>
          {page && (
            <PageViewer
              page={page}
              mode={mode}
              highlight={regionIndex >= 0 ? regions[regionIndex] : null}
            />
          )}
        </>
      ) : (
        <p className="note" data-testid="no-pages-note">
          This comparison has no page renderings — review the facet and history data below.
        </p>
      )}

      {regionText === "loading" && <p className="note">Extracting region text…</p>}
      {regionText && regionText !== "loading" && (
        <div className="pane" data-testid="region-text-panel">
          <div className="toolbar">
            <strong>Text in region</strong>
            <span
              className={`badge ${regionText.same ? "PASS" : "FAIL"}`}
              data-testid="region-text-verdict"
            >
              {regionText.same ? "same text" : "text differs"}
            </span>
          </div>
          <div className="region-text-columns">
            <div>
              <p className="note">Reference</p>
              <pre data-testid="region-text-reference">{regionText.reference_text || "(no text found)"}</pre>
            </div>
            <div>
              <p className="note">Candidate</p>
              <pre data-testid="region-text-candidate">{regionText.candidate_text || "(no text found)"}</pre>
            </div>
          </div>
        </div>
      )}

      {(detail.sidecar_json?.facets?.length ?? 0) > 0 && (
        <div className="pane" data-testid="facets-panel">
          <strong>Differences by facet</strong>
          {detail.sidecar_json.facets.map((facet: any, index: number) => (
            <details key={index} className="facet" data-testid={`facet-${facet.facet}`}>
              <summary>
                <span className="badge FAIL">{facet.facet}</span> {facet.description}
              </summary>
              <pre className="facet-details">{facet.details}</pre>
            </details>
          ))}
        </div>
      )}

      {history.length > 1 && (
        <div className="pane" data-testid="history-panel">
          <strong>History (same test across runs)</strong>
          <table className="grid history-table">
            <thead>
              <tr><th>Run</th><th>Imported</th><th>Status</th><th>Review</th><th>Score</th></tr>
            </thead>
            <tbody>
              {history.map((entry) => (
                <tr key={entry.comparison_id}
                    className={entry.comparison_id === comparisonId ? "history-current" : ""}
                    data-testid={`history-row-${entry.comparison_id}`}>
                  <td>
                    <a href={`#/comparisons/${entry.comparison_id}`}>
                      {entry.run_name} #{entry.run_id}
                    </a>
                  </td>
                  <td>{entry.imported_at}</td>
                  <td><span className={`badge ${entry.status}`}>{entry.status}</span></td>
                  <td><span className={`badge ${entry.review_state}`}>{entry.review_state}</span></td>
                  <td>{entry.score != null ? Number(entry.score).toFixed(4) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="pane decision-bar">
        <div className="toolbar">
          <input
            data-testid="decision-reason"
            placeholder="reason (optional for accept, recommended for reject)"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            style={{ width: 380 }}
          />
          {page && detail.pages.length >= 1 && (
            <button
              className="primary"
              data-testid="accept-page"
              disabled={isDegraded}
              onClick={() =>
                act(() => api.acceptPage(page.id, reason || undefined), "Page accepted — baseline promoted")
              }
            >
              Accept page
            </button>
          )}
          <button
            className="primary"
            data-testid="accept-comparison"
            disabled={isDegraded}
            onClick={() =>
              act(() => api.acceptComparison(detail.id, reason || undefined), "Accepted — baseline promoted")
            }
          >
            Accept document
          </button>
          <button
            className="danger"
            data-testid="reject-comparison"
            onClick={() => act(() => api.rejectComparison(detail.id, reason || undefined), "Rejected")}
          >
            Reject
          </button>
          <a data-testid="bugdata-link" href={`/api/comparisons/${detail.id}/bugdata`}>
            <button>Download bug data</button>
          </a>
        </div>
      </div>
    </div>
  );
}

/** Capture configuration of web comparisons (sidecar `context`, additive). */
function ContextChips({ context }: { context?: Record<string, any> | null }) {
  if (!context) return null;
  const dprValue = Number(context.device_pixel_ratio);
  const chips: [string, string][] = [];
  if (context.browser) chips.push(["browser", String(context.browser)]);
  if (context.viewport) chips.push(["viewport", String(context.viewport)]);
  if (dprValue > 1) chips.push(["dpr", `@${dprValue}x`]);
  if (!chips.length && !context.url) return null;
  return (
    <span className="context-chips" data-testid="context-chips">
      {chips.map(([key, value]) => (
        <span key={key} className="chip" data-testid={`context-${key}`}>
          {value}
        </span>
      ))}{" "}
      {context.url && (
        <span className="note" data-testid="context-url" title={String(context.url)}>
          {String(context.url).replace(/^https?:\/\//, "").slice(0, 60)}
        </span>
      )}
    </span>
  );
}

function editorLink(detail: ComparisonDetail, region: Region, dpi?: number): string {
  const params = new URLSearchParams({
    file: detail.reference_path || "",
    comparison: String(detail.id),
    x: String(Math.max(0, region.x - 5)),
    y: String(Math.max(0, region.y - 5)),
    width: String(region.width + 10),
    height: String(region.height + 10),
  });
  if (dpi) params.set("dpi", String(dpi));
  return `#/editor?${params.toString()}`;
}

function PageViewer({ page, mode, highlight }: { page: Page; mode: Mode; highlight: Region | null }) {
  const ref = page.images["reference"];
  const cand = page.images["candidate"];
  const diff = page.images["diff"];
  const highlighted = page.images["candidate_with_diff"];
  const [blinkOn, setBlinkOn] = useState(false);
  const [swipe, setSwipe] = useState(50);
  const [overlayOpacity, setOverlayOpacity] = useState(50);
  const [showRegions, setShowRegions] = useState(true);
  const view = useViewTransform();
  const paneAreaRef = useRef<HTMLDivElement>(null);
  const fittedFor = useRef<number | null>(null);

  // Fit the page into the pane once per page — on first image load (or
  // immediately when the browser already has it cached).
  const fitToImage = (img: HTMLImageElement | null) => {
    if (!img || !img.complete || !img.naturalWidth) return;
    if (fittedFor.current === page.id) return;
    const pane = paneAreaRef.current?.querySelector(".zoom-pane");
    if (!pane) return;
    const rect = pane.getBoundingClientRect();
    view.fitTo(img.naturalWidth, img.naturalHeight, rect.width, rect.height);
    fittedFor.current = page.id;
  };
  const fitProps = {
    ref: fitToImage,
    onLoad: (e: React.SyntheticEvent<HTMLImageElement>) => fitToImage(e.currentTarget),
  };

  useEffect(() => {
    if (mode !== "blink") return;
    const interval = setInterval(() => setBlinkOn((value) => !value), 600);
    return () => clearInterval(interval);
  }, [mode]);

  // Region navigation centers the viewport at the current zoom level
  useEffect(() => {
    if (!highlight || !paneAreaRef.current) return;
    const pane = paneAreaRef.current.querySelector(".zoom-pane");
    if (!pane) return;
    const rect = pane.getBoundingClientRect();
    view.centerOn(
      highlight.x + highlight.width / 2,
      highlight.y + highlight.height / 2,
      rect.width,
      rect.height,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlight]);

  const overlays = useMemo(() => {
    const scale = view.transform.scale;
    return (
      <>
        {showRegions &&
          page.regions.map((region, index) => (
            <div
              key={index}
              data-testid={`region-outline-${index}`}
              style={{
                position: "absolute",
                left: region.x,
                top: region.y,
                width: region.width,
                height: region.height,
                outline: `${1.5 / scale}px dashed #ff5722`,
                pointerEvents: "none",
              }}
            />
          ))}
        {highlight && (
          <div
            data-testid="region-highlight"
            style={{
              position: "absolute",
              left: highlight.x,
              top: highlight.y,
              width: highlight.width,
              height: highlight.height,
              outline: `${3 / scale}px solid #ff9800`,
              pointerEvents: "none",
            }}
          />
        )}
      </>
    );
  }, [highlight, showRegions, page.regions, view.transform.scale]);

  if (!ref || !cand) {
    return (
      <div className="viewer">
        {Object.entries(page.images).map(([kind, token]) => (
          <div key={kind}>
            <p className="note">{kind}</p>
            <img src={assetUrl(token)} />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="viewer" data-testid={`viewer-${mode}`} ref={paneAreaRef}>
      <div className="toolbar viewer-controls">
        <button data-testid="fit-button" onClick={() => view.reset()}>Fit</button>
        <button data-testid="actual-size-button" onClick={() => view.actualSize()}>100%</button>
        <label className="note">
          <input
            type="checkbox"
            data-testid="toggle-regions"
            checked={showRegions}
            onChange={(e) => setShowRegions(e.target.checked)}
          />{" "}
          diff regions ({page.regions.length})
        </label>
      </div>
      {mode === "side-by-side" && (
        <div className="images">
          <div className="viewer-col">
            <p className="note">Reference</p>
            <ZoomPane view={view} testid="pane-reference">
              <div className="overlay-wrap">
                <img src={assetUrl(ref)} {...fitProps} />
                {overlays}
              </div>
            </ZoomPane>
          </div>
          <div className="viewer-col">
            <p className="note">Candidate</p>
            <ZoomPane view={view} testid="pane-candidate">
              <div className="overlay-wrap">
                <img src={assetUrl(cand)} />
                {overlays}
              </div>
            </ZoomPane>
          </div>
          {diff && (
            <div className="viewer-col">
              <p className="note">Diff</p>
              <ZoomPane view={view} testid="pane-diff">
                <img src={assetUrl(diff)} />
              </ZoomPane>
            </div>
          )}
        </div>
      )}
      {mode === "overlay" && (
        <div>
          <div className="toolbar">
            <label className="note">candidate opacity</label>
            <input
              type="range"
              min={0}
              max={100}
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(parseInt(e.target.value, 10))}
            />
          </div>
          <ZoomPane view={view} testid="pane-overlay">
            <div className="overlay-wrap">
              <img src={assetUrl(ref)} {...fitProps} />
              <img className="cand" src={assetUrl(cand)} style={{ opacity: overlayOpacity / 100 }} />
              {overlays}
            </div>
          </ZoomPane>
        </div>
      )}
      {mode === "blink" && (
        <ZoomPane view={view} testid="pane-blink">
          <div className="overlay-wrap">
            <img src={assetUrl(ref)} {...fitProps} style={{ visibility: blinkOn ? "hidden" : "visible" }} />
            <img className="cand" src={assetUrl(cand)} style={{ visibility: blinkOn ? "visible" : "hidden" }} />
            {overlays}
          </div>
        </ZoomPane>
      )}
      {mode === "swipe" && (
        <div>
          <div className="toolbar">
            <label className="note">swipe</label>
            <input
              type="range"
              min={0}
              max={100}
              value={swipe}
              onChange={(e) => setSwipe(parseInt(e.target.value, 10))}
            />
          </div>
          <ZoomPane view={view} testid="pane-swipe">
            <div className="swipe-wrap">
              <img src={assetUrl(ref)} {...fitProps} />
              <div className="clip" style={{ clipPath: `inset(0 ${100 - swipe}% 0 0)` }}>
                <img src={assetUrl(cand)} />
              </div>
              {overlays}
            </div>
          </ZoomPane>
        </div>
      )}
      {mode === "highlight" && highlighted && (
        <ZoomPane view={view} testid="pane-highlight">
          <div className="overlay-wrap">
            <img src={assetUrl(highlighted)} {...fitProps} />
            {overlays}
          </div>
        </ZoomPane>
      )}
      <p className="note">
        SSIM: {page.score?.toFixed(6) ?? "n/a"} | threshold: {page.threshold ?? "n/a"} | regions:{" "}
        {page.regions.length} | zoom:{" "}
        <span data-testid="zoom-level">{view.transform.scale.toFixed(2)}×</span>
      </p>
    </div>
  );
}

function DegradedView({ detail }: { detail: ComparisonDetail }) {
  return (
    <div className="pane" data-testid="degraded-view">
      <p className="error">
        Limited data: this run was ingested without <code>result_json</code> sidecars. Only combined
        screenshots are available — per-page review, diff-region navigation and accept are disabled.
        Enable <code>result_json=true</code> on the DocTest library to unlock full review.
      </p>
      <div className="images">
        {detail.images.map((token) => (
          <img key={token} src={assetUrl(token)} />
        ))}
      </div>
    </div>
  );
}
