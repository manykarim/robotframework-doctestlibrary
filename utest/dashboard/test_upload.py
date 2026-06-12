"""Upload endpoint tests: workspace storage, type/size limits, integration."""

from helpers import REF_IMAGE


def _upload(client, name, content):
    return client.post("/api/upload", files={"file": (name, content)})


def test_upload_stores_in_workspace_root(client, config):
    response = _upload(client, "my photo.png", REF_IMAGE.read_bytes())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "my_photo.png"  # sanitized
    assert body["size"] == REF_IMAGE.stat().st_size
    path = body["path"]
    assert str(config.data_dir / "uploads") in path
    assert config.is_within_roots(__import__("pathlib").Path(path))


def test_uploaded_image_usable_by_engine(client):
    path = _upload(client, "ref.png", REF_IMAGE.read_bytes()).json()["path"]
    response = client.get("/api/page-image", params={"file": path})
    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 1
    assert client.get(f"/api/assets/{body['image']}").status_code == 200

    preview = client.post("/api/mask-preview", json={
        "file": path, "page": 1,
        "masks": {"type": "coordinates", "x": 0, "y": 0, "width": 10, "height": 10}})
    assert preview.status_code == 200


def test_uploads_workspace_visible_in_browser(client, config):
    _upload(client, "ref.png", REF_IMAGE.read_bytes())
    roots = client.get("/api/browse").json()["roots"]
    assert any(root["path"] == str(config.data_dir / "uploads") for root in roots)
    # the internal scratch dir stays hidden
    assert all(root["path"] != str(config.data_dir / "scratch") for root in roots)


def test_unsupported_extension_415(client):
    response = _upload(client, "evil.exe", b"MZ")
    assert response.status_code == 415


def test_oversized_upload_413(client, monkeypatch):
    import doctest_dashboard.server.app  # noqa: F401  (limit lives in the closure)

    # 100 MB real upload would be slow; verify the limit logic via a small file
    # against a temporarily reduced limit is not possible (closure constant),
    # so send slightly over the real limit boundary cheaply: skip if too slow.
    content = b"x" * (100 * 1024 * 1024 + 1)
    response = _upload(client, "big.png", content)
    assert response.status_code == 413


def _folder_as_multipart(base):
    """Simulate a browser webkitdirectory upload: every file with its
    folder-relative path as the multipart filename."""
    files = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            files.append(("files", (path.relative_to(base).as_posix(), path.read_bytes())))
    return files


def test_upload_results_folder_ingests_run(client, config, sidecar_output_xml):
    response = client.post(
        "/api/upload-results", files=_folder_as_multipart(sidecar_output_xml.parent))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stored"] > 0
    assert len(body["runs"]) == 1
    assert body["runs"][0]["comparisons"] == 3
    assert body["runs"][0]["sidecar_comparisons"] == 3

    # the uploaded copies are fully reviewable: page images serve as assets
    run_id = body["runs"][0]["run_id"]
    rows = client.get(f"/api/runs/{run_id}/tests", params={"status": "fail"}).json()
    detail = client.get(f"/api/comparisons/{rows[0]['comparison_id']}").json()
    diff_token = detail["pages"][0]["images"]["diff"]
    assert client.get(f"/api/assets/{diff_token}").status_code == 200


def test_upload_results_without_output_xml_422(client):
    response = client.post(
        "/api/upload-results", files=[("files", ("screenshots/a.png", b"png"))])
    assert response.status_code == 422
    assert "output.xml" in response.json()["detail"]


def test_upload_results_traversal_rejected(client, config):
    response = client.post(
        "/api/upload-results",
        files=[
            ("files", ("../evil.xml", b"<robot/>")),
            ("files", ("/abs/evil.xml", b"<robot/>")),
        ])
    # both skipped -> nothing ingestable, and nothing written outside uploads
    assert response.status_code == 422
    assert not (config.data_dir / "evil.xml").exists()
    assert response.json() is not None


def test_upload_results_skips_irrelevant_files(client, sidecar_output_xml):
    files = _folder_as_multipart(sidecar_output_xml.parent)
    files.append(("files", ("video.mp4", b"x")))
    response = client.post("/api/upload-results", files=files)
    assert response.status_code == 200
    assert response.json()["skipped"] >= 1


def test_mask_save_next_to_upload(client):
    """The editor's suggested masks.json target (upload folder) is writable."""
    path = _upload(client, "ref.png", REF_IMAGE.read_bytes()).json()["path"]
    folder = path.rsplit("/", 1)[0]
    response = client.put("/api/masks", json={
        "file": f"{folder}/masks.json", "masks": "top:10"})
    assert response.status_code == 200
