import React, { useEffect, useMemo, useState } from "react";
import { api, assetUrl, ComparisonDetail, Page, Region } from "./api";

type Mode = "side-by-side" | "overlay" | "blink" | "swipe";
const MODES: Mode[] = ["side-by-side", "overlay", "blink", "swipe"];

export function ComparisonView({ comparisonId }: { comparisonId: number }) {
  const [detail, setDetail] = useState<ComparisonDetail | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [mode, setMode] = useState<Mode>("overlay");
  const [regionIndex, setRegionIndex] = useState(-1);
  const [message, setMessage] = useState<{ kind: string; text: string } | null>(null);
  const [reason, setReason] = useState("");

  const load = () =>
    api.comparison(comparisonId).then(setDetail).catch((e) =>
      setMessage({ kind: "error", text: e.message }),
    );
  useEffect(() => {
    load();
  }, [comparisonId]);

  const page: Page | undefined = detail?.pages[pageIndex];
  const regions: Region[] = page?.regions ?? [];

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((event.target as HTMLElement).tagName)) return;
      const num = parseInt(event.key, 10);
      if (num >= 1 && num <= MODES.length) setMode(MODES[num - 1]);
      if (event.key === "n") nextRegion(1);
      if (event.key === "p") nextRegion(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const nextRegion = (step: number) => {
    if (!regions.length) return;
    setRegionIndex((current) => (current + step + regions.length) % regions.length);
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

  return (
    <div>
      <div className="toolbar">
        <a href="javascript:history.back()">← Back</a>
        <strong data-testid="comparison-keyword">{detail.keyword}</strong>
        <span className={`badge ${detail.status}`} data-testid="comparison-status">
          {detail.status}
        </span>
        <span className={`badge ${detail.review_state}`} data-testid="review-state">
          {detail.review_state}
        </span>
        {isDegraded && <span className="badge degraded">degraded</span>}
        {dpi && <span className="dpi-banner" data-testid="dpi-banner">DPI {dpi}</span>}
      </div>

      {message && (
        <p className={message.kind} data-testid="action-message">
          {message.text}
        </p>
      )}

      {isDegraded ? (
        <DegradedView detail={detail} />
      ) : (
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
            {MODES.map((m, index) => (
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
              </>
            )}
          </div>
          {page && <PageViewer page={page} mode={mode} highlight={regionIndex >= 0 ? regions[regionIndex] : null} />}
        </>
      )}

      <div className="pane">
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
              onClick={() => act(() => api.acceptPage(page.id, reason || undefined), "Page accepted — baseline promoted")}
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
  const [blinkOn, setBlinkOn] = useState(false);
  const [swipe, setSwipe] = useState(50);
  const [overlayOpacity, setOverlayOpacity] = useState(50);

  useEffect(() => {
    if (mode !== "blink") return;
    const interval = setInterval(() => setBlinkOn((value) => !value), 600);
    return () => clearInterval(interval);
  }, [mode]);

  const highlightStyle = useMemo(() => {
    if (!highlight) return null;
    return {
      position: "absolute" as const,
      left: highlight.x,
      top: highlight.y,
      width: highlight.width,
      height: highlight.height,
      outline: "3px solid #ff9800",
      pointerEvents: "none" as const,
    };
  }, [highlight]);

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
    <div className="viewer" data-testid={`viewer-${mode}`}>
      {mode === "side-by-side" && (
        <div className="images">
          <div>
            <p className="note">Reference</p>
            <div className="overlay-wrap">
              <img src={assetUrl(ref)} />
              {highlightStyle && <div style={highlightStyle} data-testid="region-highlight" />}
            </div>
          </div>
          <div>
            <p className="note">Candidate</p>
            <div className="overlay-wrap">
              <img src={assetUrl(cand)} />
              {highlightStyle && <div style={highlightStyle} />}
            </div>
          </div>
          {diff && (
            <div>
              <p className="note">Diff</p>
              <img src={assetUrl(diff)} />
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
          <div className="overlay-wrap">
            <img src={assetUrl(ref)} />
            <img className="cand" src={assetUrl(cand)} style={{ opacity: overlayOpacity / 100 }} />
            {highlightStyle && <div style={highlightStyle} data-testid="region-highlight" />}
          </div>
        </div>
      )}
      {mode === "blink" && (
        <div className="overlay-wrap">
          <img src={assetUrl(ref)} style={{ visibility: blinkOn ? "hidden" : "visible" }} />
          <img className="cand" src={assetUrl(cand)} style={{ visibility: blinkOn ? "visible" : "hidden" }} />
          {highlightStyle && <div style={highlightStyle} />}
        </div>
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
          <div className="swipe-wrap">
            <img src={assetUrl(ref)} />
            <div className="clip" style={{ clipPath: `inset(0 ${100 - swipe}% 0 0)` }}>
              <img src={assetUrl(cand)} />
            </div>
            {highlightStyle && <div style={highlightStyle} />}
          </div>
        </div>
      )}
      <p className="note">
        SSIM: {page.score?.toFixed(6) ?? "n/a"} | threshold: {page.threshold ?? "n/a"} | regions:{" "}
        {page.regions.length}
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
