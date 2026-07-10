import React, { Suspense, useEffect, useState } from "react";
import { api, BatchResult, DiffGroup, Run, TestRow, assetUrl } from "./api";
import { ComparisonView } from "./ComparisonView";

// konva stays out of the initial bundle: the editor loads on demand
const MaskEditor = React.lazy(() =>
  import("./MaskEditor").then((m) => ({ default: m.MaskEditor })),
);

/** Backend features this UI build depends on (advertised by /api/health).
 *  When any are missing, the running server is older than the served UI. */
const REQUIRED_FEATURES = ["browse", "upload", "recompare", "upload-results", "batch-accept", "lifecycle", "diff-groups", "root-cause", "history"];

const RESULT_FILE_EXTENSIONS = [".xml", ".png", ".jpg", ".jpeg", ".json", ".pdf"];
const PAGE_SIZE = 50;

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
    view = (
      <Suspense fallback={<p className="note">Loading editor…</p>}>
        <MaskEditor routeQuery={route.split("?")[1] || ""} />
      </Suspense>
    );
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
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [flaky, setFlaky] = useState<any[]>([]);
  const folderInputRef = React.useRef<HTMLInputElement>(null);
  const load = () => {
    api.runs().then((page) => setRuns(page.runs)).catch((e) => setMessage(e.message));
    api.flaky().then((body) => setFlaky(body.flaky)).catch(() => setFlaky([]));
  };
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
  const deleteRun = async (runId: number) => {
    setConfirmDelete(null);
    try {
      await api.deleteRun(runId);
      setMessage(`Deleted run ${runId}`);
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
            <th></th>
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
              <td onClick={(e) => e.stopPropagation()}>
                {confirmDelete === run.id ? (
                  <span className="confirm-inline" data-testid={`confirm-delete-${run.id}`}>
                    delete run?
                    <button
                      className="danger"
                      data-testid={`confirm-delete-yes-${run.id}`}
                      onClick={() => deleteRun(run.id)}
                    >
                      Yes
                    </button>
                    <button onClick={() => setConfirmDelete(null)}>No</button>
                  </span>
                ) : (
                  <button
                    data-testid={`delete-run-${run.id}`}
                    title="Delete this run from the dashboard (files on disk are not touched)"
                    onClick={() => setConfirmDelete(run.id)}
                  >
                    🗑
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {runs.length === 0 && <p className="note">No runs ingested yet.</p>}
      {flaky.length > 0 && (
        <details className="pane" data-testid="flaky-panel">
          <summary>
            <strong>Flaky comparisons ({flaky.length})</strong> — status flipped across recent runs
          </summary>
          <table className="grid">
            <thead>
              <tr><th>Test</th><th>Flips</th><th>Runs seen</th><th>Last status</th></tr>
            </thead>
            <tbody>
              {flaky.map((item) => (
                <tr key={item.identity} data-testid="flaky-row">
                  <td title={item.identity}>{item.name}</td>
                  <td>{item.flips}</td>
                  <td>{item.occurrences}</td>
                  <td><span className={`badge ${item.last_status}`}>{item.last_status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}

function RunDetail({ runId }: { runId: number }) {
  const [rows, setRows] = useState<TestRow[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [statusFilter, setStatusFilter] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [confirm, setConfirm] = useState<{ ids: number[] | "run"; count: number } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"flat" | "groups">("flat");
  const [groups, setGroups] = useState<DiffGroup[]>([]);
  const [ungrouped, setUngrouped] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = () => {
    const params = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(offset) });
    if (statusFilter) params.set("status", statusFilter);
    if (stateFilter) params.set("review_state", stateFilter);
    api
      .tests(runId, `?${params.toString()}`)
      .then((page) => {
        setRows(page.rows);
        setTotal(page.total);
      })
      .catch(() => {
        setRows([]);
        setTotal(0);
      });
  };
  useEffect(load, [runId, statusFilter, stateFilter, offset]);

  const loadGroups = () =>
    api
      .groups(runId)
      .then((response) => {
        setGroups(response.groups);
        setUngrouped(response.ungrouped);
      })
      .catch(() => setGroups([]));
  useEffect(() => {
    if (viewMode === "groups") loadGroups();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, runId]);

  const eligible = rows.filter(
    (r) => r.status === "FAIL" && r.review_state === "unresolved" && !r.degraded,
  );
  const toggle = (id: number) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const runBatch = async (target: number[] | "run") => {
    setConfirm(null);
    setMessage(null);
    try {
      const result: BatchResult =
        target === "run" ? await api.acceptRun(runId) : await api.acceptBatch(target);
      const skippedNote = result.skipped.length
        ? ` — skipped ${result.skipped.length}: ${result.skipped
            .map((s) => `#${s.comparison_id} (${s.reason})`)
            .slice(0, 3)
            .join("; ")}`
        : "";
      setMessage(`Accepted ${result.accepted.length} comparison(s)${skippedNote}`);
      setSelected(new Set());
      load();
      if (viewMode === "groups") loadGroups();
    } catch (e: any) {
      setMessage(e.message);
    }
  };

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const pageIndex = Math.floor(offset / PAGE_SIZE);

  return (
    <div>
      <div className="toolbar">
        <a href="#/">← Runs</a>
        <button
          className={viewMode === "flat" ? "active" : ""}
          data-testid="view-flat"
          onClick={() => setViewMode("flat")}
        >
          All
        </button>
        <button
          className={viewMode === "groups" ? "active" : ""}
          data-testid="view-groups"
          onClick={() => setViewMode("groups")}
        >
          By similarity
        </button>
        <select
          data-testid="status-filter"
          value={statusFilter}
          onChange={(e) => {
            setOffset(0);
            setStatusFilter(e.target.value);
          }}
        >
          <option value="">All statuses</option>
          <option value="fail">FAIL</option>
          <option value="pass">PASS</option>
        </select>
        <select
          data-testid="state-filter"
          value={stateFilter}
          onChange={(e) => {
            setOffset(0);
            setStateFilter(e.target.value);
          }}
        >
          <option value="">All review states</option>
          <option value="unresolved">unresolved</option>
          <option value="accepted">accepted</option>
          <option value="rejected">rejected</option>
          <option value="passed">passed</option>
        </select>
        <button
          data-testid="accept-run"
          disabled={!rows.some((r) => r.review_state === "unresolved")}
          onClick={() => setConfirm({ ids: "run", count: -1 })}
        >
          Accept all unresolved in run
        </button>
        {selected.size > 0 && (
          <button
            className="primary"
            data-testid="accept-selected"
            onClick={() => setConfirm({ ids: Array.from(selected), count: selected.size })}
          >
            Accept selected ({selected.size})
          </button>
        )}
      </div>

      {confirm && (
        <div className="pane confirm-bar" data-testid="confirm-batch">
          <span>
            {confirm.ids === "run"
              ? "Promote the candidates of ALL unresolved failed comparisons in this run to new baselines?"
              : `Promote ${confirm.count} candidate file(s) to new baselines?`}{" "}
            Multi-page PDFs are promoted as whole documents; every promotion is audited.
          </span>
          <button className="primary" data-testid="confirm-batch-yes" onClick={() => runBatch(confirm.ids)}>
            Confirm
          </button>
          <button onClick={() => setConfirm(null)}>Cancel</button>
        </div>
      )}
      {message && <p className="note" data-testid="batch-message">{message}</p>}

      {viewMode === "groups" && (
        <div data-testid="groups-view">
          {groups.length === 0 && (
            <p className="note" data-testid="groups-empty">
              No similarity groups — {ungrouped} unresolved failure(s) are each unique.
            </p>
          )}
          {groups.map((group) => (
            <div className="pane group-card" key={group.group_key}
                 data-testid={`group-${group.group_key}`}>
              <div className="toolbar">
                {group.thumbnail && <img className="thumb" src={assetUrl(group.thumbnail)} />}
                <strong data-testid={`group-count-${group.group_key}`}>
                  {group.count} comparisons with the same difference
                </strong>
                <button
                  className="primary"
                  data-testid={`accept-group-${group.group_key}`}
                  onClick={() =>
                    setConfirm({
                      ids: group.members.map((member) => member.comparison_id),
                      count: group.count,
                    })
                  }
                >
                  Accept group ({group.count})
                </button>
                <button
                  data-testid={`expand-group-${group.group_key}`}
                  onClick={() =>
                    setExpanded(expanded === group.group_key ? null : group.group_key)
                  }
                >
                  {expanded === group.group_key ? "Hide members" : "Show members"}
                </button>
              </div>
              {expanded === group.group_key && (
                <ul className="group-members">
                  {group.members.map((member) => (
                    <li key={member.comparison_id}>
                      <a href={`#/comparisons/${member.comparison_id}`}>
                        #{member.comparison_id} {member.name}
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          {groups.length > 0 && (
            <p className="note">{ungrouped} unresolved failure(s) not in any group.</p>
          )}
        </div>
      )}

      {viewMode === "flat" && (
      <>
      <table className="grid" data-testid="test-grid">
        <thead>
          <tr>
            <th>
              <input
                type="checkbox"
                data-testid="select-page"
                checked={eligible.length > 0 && eligible.every((r) => selected.has(r.comparison_id))}
                onChange={(e) =>
                  setSelected(
                    e.target.checked ? new Set(eligible.map((r) => r.comparison_id)) : new Set(),
                  )
                }
              />
            </th>
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
              <td onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  data-testid={`select-${row.comparison_id}`}
                  disabled={row.status !== "FAIL" || row.review_state !== "unresolved" || !!row.degraded}
                  checked={selected.has(row.comparison_id)}
                  onChange={() => toggle(row.comparison_id)}
                />
              </td>
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

      <div className="toolbar" data-testid="grid-pager">
        <button
          disabled={pageIndex === 0}
          data-testid="pager-prev"
          onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
        >
          ◀ Prev
        </button>
        <span className="note" data-testid="pager-info">
          page {pageIndex + 1}/{pageCount} — {total} comparison(s)
        </span>
        <button
          disabled={offset + PAGE_SIZE >= total}
          data-testid="pager-next"
          onClick={() => setOffset(offset + PAGE_SIZE)}
        >
          Next ▶
        </button>
      </div>
      </>
      )}
    </div>
  );
}
