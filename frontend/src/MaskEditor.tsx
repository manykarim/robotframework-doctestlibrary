import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Image as KonvaImage, Layer, Rect, Stage, Transformer } from "react-konva";
import Konva from "konva";
import { api, assetUrl, MaskEntry, Region } from "./api";
import { FileBrowser } from "./FileBrowser";

const DOCUMENT_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf", ".ps", ".pcl", ".tif", ".tiff", ".bmp", ".gif"];
const isDocumentFile = (name: string) =>
  DOCUMENT_EXTENSIONS.some((extension) => name.toLowerCase().endsWith(extension));
const isJsonFile = (name: string) => name.toLowerCase().endsWith(".json");

type Unit = "px" | "mm" | "cm" | "pt";
const UNIT_FACTORS: Record<Unit, (dpi: number) => number> = {
  px: () => 1,
  mm: (dpi) => dpi / 25.4,
  cm: (dpi) => dpi / 2.54,
  pt: (dpi) => dpi / 72.0,
};

const toPx = (value: number, unit: Unit, dpi: number) => value * UNIT_FACTORS[unit](dpi);
const fromPx = (px: number, unit: Unit, dpi: number) => px / UNIT_FACTORS[unit](dpi);
const round2 = (value: number) => Math.round(value * 100) / 100;

/** Client-side mirror of IgnoreAreaManager's shorthand rules ("top:10;bottom:5"). */
function parseShorthand(input: string): MaskEntry[] | null {
  const entries: MaskEntry[] = [];
  for (const part of input.split(";")) {
    if (!part.trim()) continue;
    const [location, percent] = part.split(":");
    if (!["top", "bottom", "left", "right"].includes(location)) return null;
    if (!/^\d+$/.test(percent ?? "")) return null;
    entries.push({ page: "all", type: "area", location, percent });
  }
  return entries.length ? entries : null;
}

export function MaskEditor({ routeQuery }: { routeQuery: string }) {
  const query = useMemo(() => new URLSearchParams(routeQuery), [routeQuery]);
  const [docFile, setDocFile] = useState(query.get("file") || "");
  const [maskFile, setMaskFile] = useState(query.get("maskfile") || "");
  const [pageNo, setPageNo] = useState(1);
  const [pageCount, setPageCount] = useState(1);
  const [dpi, setDpi] = useState<number>(parseInt(query.get("dpi") || "0", 10) || 0);
  const [imageToken, setImageToken] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState({ width: 800, height: 600 });
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [masks, setMasks] = useState<MaskEntry[]>(() => {
    if (query.get("x") !== null) {
      return [{
        page: "all",
        name: "From diff region",
        type: "coordinates",
        x: parseInt(query.get("x")!, 10),
        y: parseInt(query.get("y")!, 10),
        width: parseInt(query.get("width")!, 10),
        height: parseInt(query.get("height")!, 10),
        unit: "px",
      }];
    }
    return [];
  });
  const [selected, setSelected] = useState<number>(-1);
  const [previewBoxes, setPreviewBoxes] = useState<Region[]>([]);
  const [diffRegions, setDiffRegions] = useState<Region[]>([]);
  const [showDiffRegions, setShowDiffRegions] = useState(true);
  const [message, setMessage] = useState<{ kind: string; text: string } | null>(null);
  const [recompareResult, setRecompareResult] = useState<string | null>(null);
  const [importText, setImportText] = useState("");
  const [browser, setBrowser] = useState<"doc" | "mask" | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const comparisonId = query.get("comparison") ? parseInt(query.get("comparison")!, 10) : null;
  const drawState = useRef<{ startX: number; startY: number; index: number } | null>(null);

  // Load page rendering
  useEffect(() => {
    if (!docFile) return;
    fetch(
      `/api/page-image?file=${encodeURIComponent(docFile)}&page=${pageNo}` +
        (dpi ? `&dpi=${dpi}` : ""),
    )
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
        return response.json();
      })
      .then((info) => {
        setImageToken(info.image);
        setImageSize(info.image_size);
        setPageCount(info.page_count);
        setDpi(info.dpi);
        setMessage(null);
      })
      .catch((e) => setMessage({ kind: "error", text: String(e.message || e) }));
  }, [docFile, pageNo]);

  useEffect(() => {
    if (!imageToken) return;
    const element = new window.Image();
    element.src = assetUrl(imageToken);
    element.onload = () => setImage(element);
  }, [imageToken]);

  // Diff-region overlay when arriving from a comparison
  useEffect(() => {
    if (!comparisonId) return;
    api.comparison(comparisonId).then((detail) => {
      const failing = detail.pages.find((page) => page.status === "FAIL");
      if (failing) setDiffRegions(failing.regions);
    });
  }, [comparisonId]);

  // Debounced live pattern preview through the real extraction path.
  // Patterns are validated locally first: while the user is mid-keystroke
  // ("[", "[0-9]{2}[") the regex does not compile — show a hint instead of
  // sending requests that can only fail.
  const isCompilable = (pattern: string) => {
    try {
      new RegExp(pattern);
      return true;
    } catch {
      return false;
    }
  };
  const patternMasks = masks.filter(
    (mask) => ["pattern", "line_pattern", "word_pattern"].includes(mask.type) && mask.pattern,
  );
  const validPatternMasks = patternMasks.filter((mask) => isCompilable(String(mask.pattern)));
  const hasIncompletePattern = validPatternMasks.length < patternMasks.length;
  const patternKey = JSON.stringify(validPatternMasks);
  useEffect(() => {
    if (!docFile || validPatternMasks.length === 0) {
      setPreviewBoxes([]);
      return;
    }
    const timer = setTimeout(() => {
      api
        .maskPreview({
          file: docFile, page: pageNo, masks: validPatternMasks, dpi: dpi || undefined,
        })
        .then((result) => setPreviewBoxes(result.resolved_areas))
        .catch((e) => {
          setPreviewBoxes([]);
          // 422 = pattern the backend's regex dialect rejects — a typing-state
          // condition, not an application error
          if (e.status !== 422) setMessage({ kind: "error", text: e.message });
        });
    }, 400);
    return () => clearTimeout(timer);
  }, [patternKey, docFile, pageNo]);

  const scale = Math.min(1, 900 / imageSize.width);

  const updateMask = (index: number, patch: Partial<MaskEntry>) => {
    setMasks((current) => current.map((mask, i) => (i === index ? { ...mask, ...patch } : mask)));
  };

  const loadMasks = async () => {
    try {
      const result = await api.getMasks(maskFile);
      setMasks(result.masks);
      setMessage({ kind: "success", text: `Loaded ${result.masks.length} masks` });
    } catch (e: any) {
      setMessage({ kind: "error", text: e.message });
    }
  };

  const saveMasks = async () => {
    try {
      const result = await api.putMasks(maskFile, masks);
      setMessage({ kind: "success", text: `Saved ${result.masks.length} masks to ${result.file}` });
    } catch (e: any) {
      setMessage({ kind: "error", text: e.message });
    }
  };

  const importMasks = () => {
    const text = importText.trim();
    if (!text) return;
    try {
      const parsed = JSON.parse(text);
      setMasks((current) => [...current, ...(Array.isArray(parsed) ? parsed : [parsed])]);
      setMessage({ kind: "success", text: "Imported JSON masks" });
      return;
    } catch {
      /* not JSON — try shorthand */
    }
    const shorthand = parseShorthand(text);
    if (shorthand) {
      setMasks((current) => [...current, ...shorthand]);
      setMessage({ kind: "success", text: `Imported ${shorthand.length} shorthand masks` });
    } else {
      setMessage({ kind: "error", text: "Input is neither JSON nor top:10;bottom:5 shorthand" });
    }
  };

  const recompare = async () => {
    if (!comparisonId) return;
    setRecompareResult("running…");
    try {
      const result = await api.recompare(comparisonId, masks);
      setRecompareResult(result.status);
    } catch (e: any) {
      setRecompareResult(null);
      setMessage({ kind: "error", text: e.message });
    }
  };

  // Drawing new coordinate masks by drag on empty canvas
  const onStageMouseDown = (event: Konva.KonvaEventObject<MouseEvent>) => {
    if (event.target !== event.target.getStage() && event.target.className !== "Image") return;
    const stage = event.target.getStage()!;
    const pointer = stage.getPointerPosition()!;
    const x = pointer.x / scale;
    const y = pointer.y / scale;
    const entry: MaskEntry = {
      page: "all", name: `Mask ${masks.length + 1}`, type: "coordinates",
      x: Math.round(x), y: Math.round(y), width: 1, height: 1, unit: "px",
    };
    drawState.current = { startX: x, startY: y, index: masks.length };
    setMasks((current) => [...current, entry]);
    setSelected(masks.length);
  };

  const onStageMouseMove = (event: Konva.KonvaEventObject<MouseEvent>) => {
    if (!drawState.current) return;
    const stage = event.target.getStage()!;
    const pointer = stage.getPointerPosition()!;
    const { startX, startY, index } = drawState.current;
    const x = pointer.x / scale;
    const y = pointer.y / scale;
    updateMask(index, {
      x: Math.round(Math.min(startX, x)),
      y: Math.round(Math.min(startY, y)),
      width: Math.max(1, Math.round(Math.abs(x - startX))),
      height: Math.max(1, Math.round(Math.abs(y - startY))),
    });
  };

  const onStageMouseUp = () => {
    if (drawState.current) {
      const { index } = drawState.current;
      // Discard accidental click-draws
      setMasks((current) =>
        current.filter(
          (mask, i) => i !== index || (Number(mask.width) > 3 && Number(mask.height) > 3),
        ),
      );
      drawState.current = null;
    }
  };

  const effectiveDpi = dpi || 72;

  const uploadFile = async (file: File) => {
    setMessage(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/upload", { method: "POST", body: form });
      if (!response.ok) {
        if (response.status === 405 || response.status === 404) {
          // POST fell through to the static-file mount: the backend predates uploads
          throw new Error(
            "The running dashboard server does not support uploads yet. " +
              "Restart it (stop and start `doctest-dashboard serve`) to pick up the update.",
          );
        }
        let detail: any;
        try {
          detail = (await response.json()).detail;
        } catch {
          detail = response.statusText;
        }
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const result = await response.json();
      setDocFile(result.path);
      setPageNo(1);
      if (!maskFile) {
        // suggest saving masks next to the uploaded file
        const folder = result.path.slice(0, result.path.lastIndexOf("/"));
        setMaskFile(`${folder}/masks.json`);
      }
      setMessage({ kind: "success", text: `Uploaded ${result.name}` });
    } catch (e: any) {
      setMessage({ kind: "error", text: String(e.message || e) });
    }
  };

  const onBrowseSelect = (path: string) => {
    if (browser === "doc") {
      setDocFile(path);
      setPageNo(1);
    } else if (browser === "mask") {
      setMaskFile(path);
      // Picking an existing file loads it right away; new files just set the target
      api
        .getMasks(path)
        .then((result) => {
          setMasks(result.masks);
          setMessage({ kind: "success", text: `Loaded ${result.masks.length} masks` });
        })
        .catch(() => undefined);
    }
    setBrowser(null);
  };

  return (
    <div>
      {browser && (
        <FileBrowser
          title={browser === "doc" ? "Open document or image" : "Select masks.json (pick a file or name a new one)"}
          mode={browser === "doc" ? "open" : "save"}
          fileFilter={browser === "doc" ? isDocumentFile : isJsonFile}
          defaultFilename={browser === "mask" ? "masks.json" : undefined}
          onSelect={onBrowseSelect}
          onClose={() => setBrowser(null)}
        />
      )}
      <div className="toolbar">
        <input
          data-testid="doc-file"
          placeholder="document/image file (under a configured root)"
          value={docFile}
          onChange={(e) => setDocFile(e.target.value)}
          style={{ width: 420 }}
        />
        <button data-testid="browse-doc" onClick={() => setBrowser("doc")}>
          Browse…
        </button>
        <button data-testid="upload-doc" onClick={() => uploadInputRef.current?.click()}>
          Upload…
        </button>
        <input
          ref={uploadInputRef}
          data-testid="upload-input"
          type="file"
          accept={DOCUMENT_EXTENSIONS.join(",")}
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) uploadFile(file);
            e.target.value = "";
          }}
        />
        {pageCount > 1 && (
          <>
            <button disabled={pageNo <= 1} onClick={() => setPageNo(pageNo - 1)}>◀</button>
            <span className="note">page {pageNo}/{pageCount}</span>
            <button disabled={pageNo >= pageCount} onClick={() => setPageNo(pageNo + 1)}>▶</button>
          </>
        )}
        <span className="dpi-banner" data-testid="editor-dpi">DPI {effectiveDpi}</span>
        {diffRegions.length > 0 && (
          <label className="note">
            <input
              type="checkbox"
              checked={showDiffRegions}
              onChange={(e) => setShowDiffRegions(e.target.checked)}
            />
            show diff regions
          </label>
        )}
      </div>

      <div className="toolbar">
        <input
          data-testid="mask-file"
          placeholder="masks.json path"
          value={maskFile}
          onChange={(e) => setMaskFile(e.target.value)}
          style={{ width: 420 }}
        />
        <button data-testid="browse-mask" onClick={() => setBrowser("mask")}>
          Browse…
        </button>
        <button data-testid="load-masks" onClick={loadMasks} disabled={!maskFile}>Load</button>
        <button className="primary" data-testid="save-masks" onClick={saveMasks} disabled={!maskFile}>
          Save
        </button>
        {comparisonId && (
          <>
            <button data-testid="recompare" onClick={recompare}>
              Recompare with these masks
            </button>
            {recompareResult && (
              <span className={`badge ${recompareResult}`} data-testid="recompare-result">
                {recompareResult}
              </span>
            )}
          </>
        )}
      </div>

      {message && (
        <p className={message.kind} data-testid="editor-message">{message.text}</p>
      )}
      {patternMasks.length > 0 && (
        <p className="note" data-testid="pattern-status">
          {hasIncompletePattern
            ? "⚠ Incomplete or invalid regular expression — live preview is paused while you type"
            : `${previewBoxes.length} pattern match${previewBoxes.length === 1 ? "" : "es"} highlighted on this page`}
        </p>
      )}

      <div className="editor-layout">
        <Stage
          width={imageSize.width * scale}
          height={imageSize.height * scale}
          scaleX={scale}
          scaleY={scale}
          className="editor-canvas"
          onMouseDown={onStageMouseDown}
          onMouseMove={onStageMouseMove}
          onMouseUp={onStageMouseUp}
          data-testid="editor-stage"
        >
          <Layer listening={false}>
            {image && <KonvaImage image={image} width={imageSize.width} height={imageSize.height} />}
          </Layer>
          <Layer listening={false}>
            {showDiffRegions &&
              diffRegions.map((region, index) => (
                <Rect
                  key={`diff-${index}`}
                  x={region.x} y={region.y} width={region.width} height={region.height}
                  stroke="#ff9800" strokeWidth={2 / scale} dash={[6, 4]}
                />
              ))}
            {previewBoxes.map((box, index) => (
              <Rect
                key={`preview-${index}`}
                x={box.x} y={box.y} width={box.width} height={box.height}
                fill="rgba(37,99,235,0.35)" stroke="#2563eb" strokeWidth={1.5 / scale}
              />
            ))}
            {masks.map((mask, index) =>
              mask.type === "area" ? (
                <AreaBand key={`area-${index}`} mask={mask} size={imageSize} />
              ) : null,
            )}
          </Layer>
          <Layer>
            {masks.map((mask, index) =>
              mask.type === "coordinates" ? (
                <CoordRect
                  key={`coord-${index}`}
                  mask={mask}
                  dpi={effectiveDpi}
                  selected={selected === index}
                  onSelect={() => setSelected(index)}
                  onChange={(patch) => updateMask(index, patch)}
                />
              ) : null,
            )}
          </Layer>
        </Stage>

        <div className="editor-side">
          <div className="pane">
            <strong>Masks</strong>
            <div className="mask-list" data-testid="mask-list">
              {masks.map((mask, index) => (
                <div
                  key={index}
                  className={`mask-row ${selected === index ? "selected" : ""}`}
                  data-testid={`mask-row-${index}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelected(index)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(index);
                    }
                  }}
                >
                  <span className="badge degraded">{mask.type}</span>
                  <span>{mask.name || `#${index + 1}`}</span>
                  <span className="note">p:{String(mask.page ?? "all")}</span>
                  <button
                    style={{ marginLeft: "auto" }}
                    data-testid={`delete-mask-${index}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setMasks((current) => current.filter((_, i) => i !== index));
                      setSelected(-1);
                    }}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
            <div className="toolbar">
              <button
                data-testid="add-coordinates-mask"
                onClick={() => {
                  setMasks((current) => [...current, {
                    page: "all", name: `Mask ${masks.length + 1}`, type: "coordinates",
                    x: 10, y: 10, width: 100, height: 50, unit: "px",
                  }]);
                  setSelected(masks.length);
                }}
              >
                + coordinates
              </button>
              <button
                data-testid="add-area-mask"
                onClick={() => {
                  setMasks((current) => [...current, { page: "all", type: "area", location: "top", percent: 10 }]);
                  setSelected(masks.length);
                }}
              >
                + area
              </button>
              <button
                data-testid="add-pattern-mask"
                onClick={() => {
                  setMasks((current) => [...current, { page: "all", type: "pattern", pattern: "", xoffset: 0, yoffset: 0 }]);
                  setSelected(masks.length);
                }}
              >
                + pattern
              </button>
            </div>
          </div>

          {selected >= 0 && masks[selected] && (
            <MaskProperties
              mask={masks[selected]}
              dpi={effectiveDpi}
              onChange={(patch) => updateMask(selected, patch)}
            />
          )}

          <div className="pane">
            <strong>Import</strong>
            <p className="note">JSON list/object or shorthand like top:10;bottom:5</p>
            <textarea
              data-testid="import-text"
              rows={3}
              style={{ width: "100%" }}
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
            />
            <button data-testid="import-button" onClick={importMasks}>Import</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function AreaBand({ mask, size }: { mask: MaskEntry; size: { width: number; height: number } }) {
  const percent = Number(mask.percent ?? 10);
  let x = 0, y = 0, width = size.width, height = size.height;
  if (mask.location === "top") height = (size.height * percent) / 100;
  if (mask.location === "bottom") {
    height = (size.height * percent) / 100;
    y = size.height - height;
  }
  if (mask.location === "left") width = (size.width * percent) / 100;
  if (mask.location === "right") {
    width = (size.width * percent) / 100;
    x = size.width - width;
  }
  return <Rect x={x} y={y} width={width} height={height} fill="rgba(124,58,237,0.3)" />;
}

function CoordRect({
  mask, dpi, selected, onSelect, onChange,
}: {
  mask: MaskEntry;
  dpi: number;
  selected: boolean;
  onSelect: () => void;
  onChange: (patch: Partial<MaskEntry>) => void;
}) {
  const shapeRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const unit = (mask.unit ?? "px") as Unit;
  useEffect(() => {
    if (selected && transformerRef.current && shapeRef.current) {
      transformerRef.current.nodes([shapeRef.current]);
      transformerRef.current.getLayer()?.batchDraw();
    }
  }, [selected]);
  const px = {
    x: toPx(Number(mask.x), unit, dpi),
    y: toPx(Number(mask.y), unit, dpi),
    width: toPx(Number(mask.width), unit, dpi),
    height: toPx(Number(mask.height), unit, dpi),
  };
  const commit = (next: { x: number; y: number; width: number; height: number }) => {
    onChange({
      x: round2(fromPx(next.x, unit, dpi)),
      y: round2(fromPx(next.y, unit, dpi)),
      width: round2(fromPx(next.width, unit, dpi)),
      height: round2(fromPx(next.height, unit, dpi)),
    });
  };
  return (
    <>
      <Rect
        ref={shapeRef}
        {...px}
        draggable
        fill="rgba(220,38,38,0.25)"
        stroke={selected ? "#2563eb" : "#dc2626"}
        strokeWidth={2}
        onClick={onSelect}
        onTap={onSelect}
        onDragEnd={(e) => commit({ ...px, x: e.target.x(), y: e.target.y() })}
        onTransformEnd={() => {
          const node = shapeRef.current!;
          const next = {
            x: node.x(),
            y: node.y(),
            width: Math.max(2, node.width() * node.scaleX()),
            height: Math.max(2, node.height() * node.scaleY()),
          };
          node.scaleX(1);
          node.scaleY(1);
          commit(next);
        }}
      />
      {selected && <Transformer ref={transformerRef} rotateEnabled={false} />}
    </>
  );
}

function MaskProperties({
  mask, dpi, onChange,
}: {
  mask: MaskEntry;
  dpi: number;
  onChange: (patch: Partial<MaskEntry>) => void;
}) {
  return (
    <div className="pane" data-testid="mask-properties">
      <strong>Properties</strong>
      <div className="field-grid">
        <label>name</label>
        <input
          data-testid="prop-name"
          value={mask.name ?? ""}
          onChange={(e) => onChange({ name: e.target.value })}
        />
        <label>page</label>
        <input
          data-testid="prop-page"
          value={String(mask.page ?? "all")}
          onChange={(e) => {
            const value = e.target.value;
            onChange({ page: /^\d+$/.test(value) ? parseInt(value, 10) : value });
          }}
        />
        {mask.type === "coordinates" && (
          <>
            {(["x", "y", "width", "height"] as const).map((field) => (
              <React.Fragment key={field}>
                <label>{field}</label>
                <input
                  type="number"
                  step="any"
                  data-testid={`prop-${field}`}
                  value={Number(mask[field])}
                  onChange={(e) => onChange({ [field]: parseFloat(e.target.value) || 0 })}
                />
              </React.Fragment>
            ))}
            <label>unit</label>
            <select
              data-testid="prop-unit"
              value={mask.unit ?? "px"}
              onChange={(e) => {
                // Convert stored values so the rectangle stays put on screen
                const oldUnit = (mask.unit ?? "px") as Unit;
                const newUnit = e.target.value as Unit;
                const convert = (value: number) =>
                  round2(fromPx(toPx(value, oldUnit, dpi), newUnit, dpi));
                onChange({
                  unit: newUnit,
                  x: convert(Number(mask.x)),
                  y: convert(Number(mask.y)),
                  width: convert(Number(mask.width)),
                  height: convert(Number(mask.height)),
                });
              }}
            >
              {["px", "mm", "cm", "pt"].map((unit) => (
                <option key={unit} value={unit}>{unit}</option>
              ))}
            </select>
          </>
        )}
        {mask.type === "area" && (
          <>
            <label>location</label>
            <select
              data-testid="prop-location"
              value={mask.location ?? "top"}
              onChange={(e) => onChange({ location: e.target.value })}
            >
              {["top", "bottom", "left", "right"].map((location) => (
                <option key={location} value={location}>{location}</option>
              ))}
            </select>
            <label>percent</label>
            <input
              type="range" min={1} max={100}
              data-testid="prop-percent"
              value={Number(mask.percent ?? 10)}
              onChange={(e) => onChange({ percent: parseInt(e.target.value, 10) })}
            />
          </>
        )}
        {["pattern", "line_pattern", "word_pattern"].includes(mask.type) && (
          <>
            <label>type</label>
            <select
              data-testid="prop-pattern-type"
              value={mask.type}
              onChange={(e) => onChange({ type: e.target.value })}
            >
              {["pattern", "line_pattern", "word_pattern"].map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
            <label>regex</label>
            <input
              data-testid="prop-pattern"
              value={mask.pattern ?? ""}
              onChange={(e) => onChange({ pattern: e.target.value })}
            />
            <label>xoffset</label>
            <input
              type="number"
              data-testid="prop-xoffset"
              value={Number(mask.xoffset ?? 0)}
              onChange={(e) => onChange({ xoffset: parseInt(e.target.value, 10) || 0 })}
            />
            <label>yoffset</label>
            <input
              type="number"
              data-testid="prop-yoffset"
              value={Number(mask.yoffset ?? 0)}
              onChange={(e) => onChange({ yoffset: parseInt(e.target.value, 10) || 0 })}
            />
          </>
        )}
      </div>
    </div>
  );
}
