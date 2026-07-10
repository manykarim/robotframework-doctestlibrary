export interface Run {
  id: number;
  name: string;
  output_xml_path: string;
  imported_at: string;
  comparisons: number;
  unresolved: number;
  failed: number;
}

export interface TestRow {
  test_id: number;
  suite: string;
  name: string;
  test_status: string;
  comparison_id: number;
  keyword: string;
  status: string;
  degraded: number;
  review_state: string;
  thumbnail: string | null;
}

export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Page {
  id: number;
  page_no: number;
  status: string;
  score: number | null;
  threshold: number | null;
  review_state: string;
  regions: Region[];
  images: Record<string, string>;
}

export interface ComparisonDetail {
  id: number;
  keyword: string;
  status: string;
  degraded: number;
  review_state: string;
  reference_path: string | null;
  candidate_path: string | null;
  sidecar_json: any;
  images: string[];
  pages: Page[];
}

export interface MaskEntry {
  page?: string | number;
  name?: string;
  type: string;
  [key: string]: any;
}

async function request(method: string, url: string, body?: any) {
  const response = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    let detail: any;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    const error: any = new Error(
      typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail),
    );
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return response;
}

export interface RunsPage {
  runs: Run[];
  total: number;
}

export interface GridPage {
  rows: TestRow[];
  total: number;
}

export interface GroupMember {
  comparison_id: number;
  name: string;
  thumbnail: string | null;
}

export interface DiffGroup {
  group_key: string;
  count: number;
  thumbnail: string | null;
  members: GroupMember[];
}

export interface GroupsResponse {
  groups: DiffGroup[];
  ungrouped: number;
}

export interface BatchResult {
  accepted: { comparison_id: number }[];
  skipped: { comparison_id: number; reason: string }[];
}

export const api = {
  runs: (limit = 100, offset = 0): Promise<RunsPage> =>
    request("GET", `/api/runs?limit=${limit}&offset=${offset}`).then((r) => r.json()),
  deleteRun: (runId: number) =>
    request("DELETE", `/api/runs/${runId}`).then((r) => r.json()),
  tests: (runId: number, params = ""): Promise<GridPage> =>
    request("GET", `/api/runs/${runId}/tests${params}`).then((r) => r.json()),
  groups: (runId: number): Promise<GroupsResponse> =>
    request("GET", `/api/runs/${runId}/groups`).then((r) => r.json()),
  acceptRun: (runId: number, reason?: string): Promise<BatchResult> =>
    request("POST", `/api/runs/${runId}/accept`, { reason }).then((r) => r.json()),
  acceptBatch: (ids: number[], reason?: string): Promise<BatchResult> =>
    request("POST", "/api/comparisons/accept-batch", { ids, reason }).then((r) => r.json()),
  comparison: (id: number): Promise<ComparisonDetail> =>
    request("GET", `/api/comparisons/${id}`).then((r) => r.json()),
  ingest: (outputXml: string) =>
    request("POST", "/api/ingest", { output_xml: outputXml }).then((r) => r.json()),
  acceptComparison: (id: number, reason?: string) =>
    request("POST", `/api/comparisons/${id}/accept`, { reason }).then((r) => r.json()),
  acceptPage: (id: number, reason?: string) =>
    request("POST", `/api/pages/${id}/accept`, { reason }).then((r) => r.json()),
  rejectComparison: (id: number, reason?: string) =>
    request("POST", `/api/comparisons/${id}/reject`, { reason }).then((r) => r.json()),
  getMasks: (file: string): Promise<{ file: string; masks: MaskEntry[] }> =>
    request("GET", `/api/masks?file=${encodeURIComponent(file)}`).then((r) => r.json()),
  putMasks: (file: string, masks: MaskEntry[] | string) =>
    request("PUT", "/api/masks", { file, masks }).then((r) => r.json()),
  maskPreview: (body: any) =>
    request("POST", "/api/mask-preview", body).then((r) => r.json()),
  recompare: (comparisonId: number, masks: any, settings?: any) =>
    request("POST", "/api/recompare", { comparison_id: comparisonId, masks, settings }).then((r) =>
      r.json(),
    ),
  regionText: (comparisonId: number, pageNo: number, region: Region) =>
    request("POST", `/api/comparisons/${comparisonId}/region-text`, {
      page_no: pageNo,
      region,
    }).then((r) => r.json()),
  history: (comparisonId: number) =>
    request("GET", `/api/comparisons/${comparisonId}/history`).then((r) => r.json()),
  flaky: () => request("GET", "/api/flaky").then((r) => r.json()),
  capabilities: () => request("GET", "/api/capabilities").then((r) => r.json()),
};

export const assetUrl = (token: string) => `/api/assets/${encodeURIComponent(token)}`;
