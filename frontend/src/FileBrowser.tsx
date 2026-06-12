import React, { useEffect, useState } from "react";

interface Entry {
  name: string;
  path: string;
  type: "dir" | "file";
  size: number | null;
}

interface BrowseResponse {
  roots?: { name: string; path: string }[];
  path?: string;
  parent?: string | null;
  entries?: Entry[];
}

export interface FileBrowserProps {
  title: string;
  /** "open": pick an existing file. "save": pick a folder (or existing file) and a filename. */
  mode: "open" | "save";
  /** Show only files whose name matches (directories always shown). */
  fileFilter?: (name: string) => boolean;
  defaultFilename?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

async function browse(path?: string): Promise<BrowseResponse> {
  const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
  const response = await fetch(url);
  if (!response.ok) {
    let detail: string;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    if (response.status === 404 && !path) {
      // The route itself is missing: UI is newer than the running backend
      detail =
        "The running dashboard server does not provide the file browser yet. " +
        "Restart it (stop and start `doctest-dashboard serve`) to pick up the update.";
    }
    throw new Error(detail);
  }
  return response.json();
}

function formatSize(size: number | null): string {
  if (size === null) return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

export function FileBrowser({
  title, mode, fileFilter, defaultFilename, onSelect, onClose,
}: FileBrowserProps) {
  const [listing, setListing] = useState<BrowseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState(defaultFilename ?? "");

  const navigate = (path?: string) =>
    browse(path)
      .then((result) => {
        setListing(result);
        setError(null);
        // Jump straight in when there is exactly one root
        if (!path && result.roots && result.roots.length === 1) {
          navigate(result.roots[0].path);
        }
      })
      .catch((e) => setError(String(e.message || e)));

  useEffect(() => {
    navigate();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const entries = (listing?.entries ?? []).filter(
    (entry) => entry.type === "dir" || !fileFilter || fileFilter(entry.name),
  );

  const pickFile = (entry: Entry) => {
    if (mode === "open") {
      onSelect(entry.path);
    } else {
      setFilename(entry.name);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="file-browser">
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <strong>{title}</strong>
          <button data-testid="fb-close" onClick={onClose}>✕</button>
        </div>

        {error && <p className="error">{error}</p>}
        {listing?.roots && listing.roots.length === 0 && (
          <p className="note">
            No browsable locations yet. Use the <strong>Upload…</strong> button to bring a
            file from your machine, start the server with <code>--root /your/testdata</code>,
            or ingest a run — ingested run folders become browsable automatically.
          </p>
        )}

        {listing?.path && (
          <div className="fb-pathbar">
            <button
              data-testid="fb-up"
              disabled={!listing.parent && !listing.roots}
              onClick={() => (listing.parent ? navigate(listing.parent) : navigate())}
            >
              ↑ Up
            </button>
            <span className="note fb-path" title={listing.path}>{listing.path}</span>
          </div>
        )}

        <div className="fb-list" data-testid="fb-list">
          {listing?.roots?.map((root) => (
            <div
              key={root.path}
              className="fb-entry"
              data-testid={`fb-entry-${root.name}`}
              onClick={() => navigate(root.path)}
            >
              <span className="fb-icon">🗄</span>
              <span>{root.path}</span>
            </div>
          ))}
          {listing?.path &&
            entries.map((entry) => (
              <div
                key={entry.path}
                className="fb-entry"
                data-testid={`fb-entry-${entry.name}`}
                onClick={() => (entry.type === "dir" ? navigate(entry.path) : pickFile(entry))}
                onDoubleClick={() => entry.type === "file" && mode === "save" && onSelect(entry.path)}
              >
                <span className="fb-icon">{entry.type === "dir" ? "📁" : "📄"}</span>
                <span>{entry.name}</span>
                <span className="note fb-size">{formatSize(entry.size)}</span>
              </div>
            ))}
          {listing?.path && entries.length === 0 && (
            <p className="note">No matching files in this folder.</p>
          )}
        </div>

        {mode === "save" && listing?.path && (
          <div className="modal-footer">
            <label className="note">File name</label>
            <input
              data-testid="fb-filename"
              value={filename}
              onChange={(event) => setFilename(event.target.value)}
              placeholder="masks.json"
            />
            <button
              className="primary"
              data-testid="fb-select"
              disabled={!filename.trim()}
              onClick={() => onSelect(`${listing.path}/${filename.trim()}`)}
            >
              Select
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
