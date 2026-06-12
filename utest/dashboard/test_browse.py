"""File-browse endpoint tests: listing, navigation, root confinement."""


def test_browse_lists_roots(client, config, tmp_path):
    config.add_root(tmp_path)
    body = client.get("/api/browse").json()
    assert any(root["path"] == str(tmp_path) for root in body["roots"])


def test_browse_without_configured_roots_shows_uploads_workspace(client, config):
    roots = client.get("/api/browse").json()["roots"]
    assert [root["path"] for root in roots] == [str(config.data_dir / "uploads")]


def test_browse_lists_directory_entries(client, config, tmp_path):
    base = tmp_path / "browseme"
    base.mkdir()
    config.add_root(base)
    (base / "sub").mkdir()
    (base / "image.png").write_bytes(b"x" * 10)
    (base / "masks.json").write_text("[]")
    (base / ".hidden").write_text("secret")

    body = client.get("/api/browse", params={"path": str(base)}).json()
    names = [entry["name"] for entry in body["entries"]]
    assert names == ["sub", "image.png", "masks.json"]  # dirs first, hidden excluded
    types = {entry["name"]: entry["type"] for entry in body["entries"]}
    assert types["sub"] == "dir"
    assert types["image.png"] == "file"
    sizes = {entry["name"]: entry["size"] for entry in body["entries"]}
    assert sizes["image.png"] == 10
    assert sizes["sub"] is None


def test_browse_parent_stops_at_root(client, config, tmp_path):
    config.add_root(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    body = client.get("/api/browse", params={"path": str(sub)}).json()
    assert body["parent"] == str(tmp_path)
    body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    assert body["parent"] is None  # cannot climb out of the root


def test_browse_outside_roots_403(client, tmp_path):
    assert client.get("/api/browse", params={"path": str(tmp_path)}).status_code == 403
    assert client.get("/api/browse", params={"path": "/etc"}).status_code == 403


def test_browse_file_path_404(client, config, tmp_path):
    config.add_root(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("x")
    assert client.get("/api/browse", params={"path": str(target)}).status_code == 404
