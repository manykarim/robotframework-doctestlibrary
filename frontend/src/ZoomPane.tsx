import React, { useEffect, useRef, useState } from "react";

export interface ViewTransform {
  scale: number;
  tx: number;
  ty: number;
}

const MIN_SCALE = 0.25;
const MAX_SCALE = 8;

/** Shared zoom/pan state. Pass the same instance to several ZoomPanes to
 *  keep their viewports synchronized (side-by-side review). */
export function useViewTransform() {
  const [transform, setTransform] = useState<ViewTransform>({ scale: 1, tx: 0, ty: 0 });

  const zoomAt = (cx: number, cy: number, factor: number) =>
    setTransform((prev) => {
      const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, prev.scale * factor));
      const k = scale / prev.scale;
      return { scale, tx: cx - (cx - prev.tx) * k, ty: cy - (cy - prev.ty) * k };
    });

  const panBy = (dx: number, dy: number) =>
    setTransform((prev) => ({ ...prev, tx: prev.tx + dx, ty: prev.ty + dy }));

  const reset = () => setTransform({ scale: 1, tx: 0, ty: 0 });

  /** Center the viewport on a point given in image coordinates. */
  const centerOn = (x: number, y: number, viewportWidth: number, viewportHeight: number) =>
    setTransform((prev) => ({
      ...prev,
      tx: viewportWidth / 2 - x * prev.scale,
      ty: viewportHeight / 2 - y * prev.scale,
    }));

  return { transform, zoomAt, panBy, reset, centerOn };
}

export type ViewTransformApi = ReturnType<typeof useViewTransform>;

export function ZoomPane({
  view,
  children,
  testid,
  height = "62vh",
}: {
  view: ViewTransformApi;
  children: React.ReactNode;
  testid?: string;
  height?: string;
}) {
  const paneRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  // React registers wheel listeners passively; zooming must preventDefault
  // to stop the page from scrolling, so attach natively.
  useEffect(() => {
    const pane = paneRef.current;
    if (!pane) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = pane.getBoundingClientRect();
      const factor = event.deltaY < 0 ? 1.2 : 1 / 1.2;
      view.zoomAt(event.clientX - rect.left, event.clientY - rect.top, factor);
    };
    pane.addEventListener("wheel", onWheel, { passive: false });
    return () => pane.removeEventListener("wheel", onWheel);
  }, [view]);

  return (
    <div
      ref={paneRef}
      className="zoom-pane"
      data-testid={testid}
      data-zoom={view.transform.scale.toFixed(2)}
      style={{ height }}
      onDoubleClick={() => view.reset()}
      onMouseDown={(event) => {
        drag.current = { x: event.clientX, y: event.clientY };
      }}
      onMouseMove={(event) => {
        if (!drag.current || event.buttons !== 1) return;
        view.panBy(event.clientX - drag.current.x, event.clientY - drag.current.y);
        drag.current = { x: event.clientX, y: event.clientY };
      }}
      onMouseUp={() => {
        drag.current = null;
      }}
      onMouseLeave={() => {
        drag.current = null;
      }}
    >
      <div
        className="zoom-content"
        style={{
          transform: `translate(${view.transform.tx}px, ${view.transform.ty}px) scale(${view.transform.scale})`,
          transformOrigin: "0 0",
        }}
      >
        {children}
      </div>
    </div>
  );
}
