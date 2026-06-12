import React, { useEffect, useState } from "react";
import { api, Run, TestRow, assetUrl } from "./api";
import { ComparisonView } from "./ComparisonView";
import { MaskEditor } from "./MaskEditor";

/** Backend features this UI build depends on (advertised by /api/health).
 *  When any are missing, the running server is older than the served UI. */
const REQUIRED_FEATURES = ["browse", "upload", "recompare", "upload-results"];

const RESULT_FILE_EXTENSIONS = [".xml", ".png", ".jpg", ".jpeg", ".json", ".pdf"];

/** Tiny hash router: #/ | #/runs/:id | #/comparisons/:id | #/editor?... */
function useHashRoute(): [string, (hash: string) => void] {
  const [route, setRoute] = useState(window.location.hash || "#/");
  useEffect(() => {
    const onChange = () => setRoute(window.location.hash || "#/");
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return [route, (hash) => (window.location.hash = hash)];
}

function useBackendCheck(): string | null {
  const [warning, setWarning] = useState<string | null>(null);
  useEffect(() => {
    fetch("/api/health")
      .then(async (response) => {
        if (!response.ok) throw new Error(`health check failed (${response.status})`);
        const health = await response.json();
        const features: string[] = health.features ?? [];
        const missing = REQUIRED_FEATURES.filter((f) => !features.includes(f));
        if (missing.length) {
          setWarning(
            `The running dashboard server (v${health.version ?? "?"}) is older than this ` +
              `user interface — it lacks: ${missing.join(", ")}. ` +
              "Restart it (stop and start `doctest-dashboard serve`) to update.",
          );
        }
      })
      .catch((e) => setWarning(`Cannot reach the dashboard server: ${e.message}`));
  }, []);
  return warning;
}

export function App() {
  const [route] = useHashRoute();
  const backendWarning = useBackendCheck();
  let view: React.ReactNode;
  if (route.startsWith("#/runs/")) {
    view = <RunDetail runId={parseInt(route.split("/")[2], 10)} />;
  } else if (route.startsWith("#/comparisons/")) {
    view = <ComparisonView comparisonId={parseInt(route.split("/")[2], 10)} />;
  } else if (route.startsWith("#/editor")) {
    view = <MaskEditor routeQuery={route.split("?")[1] || ""} />;
  } else {
    view = <RunList />;
  }
  return (
    <>
      <header className="topbar">
        <h1>
          <a href="#/">doctest-dashboard</a>
        </h1>
        <a href="#/" data-testid="nav-runs">Runs</a>
        <a href="#/editor" data-testid="nav-editor">Mask Editor</a>
      </header>
      {backendWarning && (
        <div className="server-warning" data-testid="server-warning">
          ⚠ {backendWarning}
        </div>
      )}
      <main>{view}</main>
    </>
  );
}

function RunList() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [ingestPath, setIngestPath] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const folderInputRef = React.useRef<HTMLInputElement>(null);
  const load = () => api.runs().then(setRuns).catch((e) => setMessage(e.message));
  useEffect(() => {
    load();
  }, []);
  const ingest = async () => {
    setMessage(null);
    try {
      const summary = await api.ingest(ingestPath);
      setMessage(`Ingested run ${summary.run_id}: ${summary.comparisons} comparisons`);
      load();
    } catch (e: any) {
      setMessage(e.message);
    }
  };
  const uploadResultsFolder = async (files: FileList) => {
    setMessage(null);
    const form = new FormData();
    let count = 0;
    for (const file of Array.from(files)) {
      const relative = (file as any).webkitRelativePath || file.name;
      if (!RESULT_FILE_EXTENSIONS.some((ext) => relative.toLowerCase().endsWith(ext))) continue;
      form.append("files", file, relative);
      count++;
    }
    if (!count) {
      setMessage("The selected folder contains no result files (output.xml, images, sidecars)");
      return;
    }
    setMessage(`Uploading ${count} files…`);
    try {
      const response = await fetch("/api/upload-results", { method: "POST", body: form });
      if (!response.ok) {
        const detail = (await response.json()).detail;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const result = await response.json();
      const summary = result.runs
        .map((run: any) => `run ${run.run_id}: ${run.comparisons} comparisons`)
        .join(", ");
      setMessage(`Uploaded ${result.stored} files — ingested ${summary}`);
      load();
    } catch (e: any) {
      setMessage(String(e.message || e));
    }
  };
  return (
    <div>
      <div className="toolbar">
        <input
          data-testid="ingest-path"
          placeholder="/path/to/output.xml"
          value={ingestPath}
          onChange={(e) => setIngestPath(e.target.value)}
          style={{ width: 420 }}
        />
        <button className="primary" data-testid="ingest-button" onClick={ingest}>
          Ingest output.xml
        </button>
        <button
          data-testid="upload-results"
          title="Pick a Robot Framework output folder; output.xml, screenshots and sidecars are uploaded and ingested"
          onClick={() => folderInputRef.current?.click()}
        >
          Upload results folder…
        </button>
        <input
          ref={folderInputRef}
          data-testid="upload-results-input"
          type="file"
          multiple
          style={{ display: "none" }}
          {...({ webkitdirectory: "" } as any)}
          onChange={(e) => {
            if (e.target.files?.length) uploadResultsFolder(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {message && <p className="note" data-testid="ingest-message">{message}</p>}
      <table className="grid" data-testid="run-list">
        <thead>
          <tr>
            <th>Run</th>
            <th>Imported</th>
            <th>Comparisons</th>
            <th>Failed</th>
            <th>Unresolved</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.id}
              className="clickable"
              data-testid={`run-row-${run.id}`}
              onClick={() => (window.location.hash = `#/runs/${run.id}`)}
            >
              <td>{run.name}</td>
              <td>{run.imported_at}</td>
              <td>{run.comparisons}</td>
              <td>{run.failed}</td>
              <td>
                <span className={`badge ${run.unresolved ? "unresolved" : "passed"}`}>
                  {run.unresolved}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {runs.length === 0 && <p className="note">No runs ingested yet.</p>}
    </div>
  );
}

function RunDetail({ runId }: { runId: number }) {
  const [rows, setRows] = useState<TestRow[]>([]);
  const [statusFilter, setStatusFilter] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  useEffect(() => {
    const params = new URLSearchParams();
    if (statusFilter) params.set("status", statusFilter);
    if (stateFilter) params.set("review_state", stateFilter);
    const query = params.toString() ? `?${params.toString()}` : "";
    api.tests(runId, query).then(setRows).catch(() => setRows([]));
  }, [runId, statusFilter, stateFilter]);
  return (
    <div>
      <div className="toolbar">
        <a href="#/">← Runs</a>
        <select
          data-testid="status-filter"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All statuses</option>
          <option value="fail">FAIL</option>
          <option value="pass">PASS</option>
        </select>
        <select
          data-testid="state-filter"
          value={stateFilter}
          onChange={(e) => setStateFilter(e.target.value)}
        >
          <option value="">All review states</option>
          <option value="unresolved">unresolved</option>
          <option value="accepted">accepted</option>
          <option value="rejected">rejected</option>
          <option value="passed">passed</option>
        </select>
      </div>
      <table className="grid" data-testid="test-grid">
        <thead>
          <tr>
            <th>Diff</th>
            <th>Test</th>
            <th>Keyword</th>
            <th>Comparison</th>
            <th>Review</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.comparison_id}
              className="clickable"
              data-testid={`comparison-row-${row.comparison_id}`}
              onClick={() => (window.location.hash = `#/comparisons/${row.comparison_id}`)}
            >
              <td>{row.thumbnail ? <img className="thumb" src={assetUrl(row.thumbnail)} /> : "—"}</td>
              <td>{row.name}</td>
              <td>{row.keyword}</td>
              <td>
                <span className={`badge ${row.status}`}>{row.status}</span>{" "}
                {row.degraded ? <span className="badge degraded">degraded</span> : null}
              </td>
              <td>
                <span className={`badge ${row.review_state}`}>{row.review_state}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
