import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/health").json()["ready"] is True
    s = client.get("/api/v1/status").json()
    assert s["ready"] and s["data_as_of"]


@pytest.mark.parametrize("path", [
    "/api/v1/market/catalog", "/api/v1/market/snapshot", "/api/v1/market/quality", "/api/v1/forecast",
    "/api/v1/forecast/tce?vessel_class=Capesize", "/api/v1/forecast/explain?h=21", "/api/v1/models",
    "/api/v1/backtest?h=21", "/api/v1/reference", "/api/v1/risk", "/api/v1/market/series?keys=bdry,bdi",
])
def test_get_endpoints(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text


def test_forecast_shape(client):
    f = client.get("/api/v1/forecast").json()
    assert len(f["bdry"]["path"]) == 63
    p = f["bdry"]["path"][20]
    assert p["lo95"] <= p["lo80"] <= p["point"] <= p["hi80"] <= p["hi95"]
    assert f["bdi"]["nowcast"]["p10"] <= f["bdi"]["nowcast"]["level"] <= f["bdi"]["nowcast"]["p90"]


@pytest.mark.parametrize("path", ["/api/v1/optimize/allocation", "/api/v1/optimize/timing", "/api/v1/optimize/speed",
                                  "/api/v1/landed-cost", "/api/v1/stress"])
def test_post_endpoints_with_defaults(client, path):
    r = client.post(path, json={})
    assert r.status_code == 200, r.text


def test_validation_errors(client):
    assert client.post("/api/v1/optimize/timing", json={"vessel_class": "Yacht"}).status_code == 422
    assert client.get("/api/v1/forecast/tce?vessel_class=Yacht").status_code == 422
    bad = {"cargoes": [{"id": "X", "load_port": "samarinda", "discharge_port": "paradip", "quantity_t": 1000,
                        "laycan_start": 5, "laycan_end": 2}]}
    assert client.post("/api/v1/optimize/allocation", json=bad).status_code == 422


def test_admin_refresh_disabled_without_token(client):
    assert client.post("/api/v1/admin/refresh").status_code == 404


def test_empty_cargo_list_is_rejected(client):
    assert client.post("/api/v1/optimize/allocation", json={"cargoes": []}).status_code == 422


def test_bad_start_date_is_422(client):
    assert client.get("/api/v1/market/series?keys=bdry&start=notadate").status_code == 422


@pytest.mark.parametrize("path", ["/api/v1/optimize/timing", "/api/v1/optimize/speed", "/api/v1/landed-cost", "/api/v1/stress"])
def test_infeasible_voyage_is_rejected(client, path):
    body = {"vessel_class": "Capesize", "load_port": "richards_bay", "discharge_port": "gopalpur", "cargo_t": 150000}
    r = client.post(path, json=body)
    assert r.status_code == 422 and "Infeasible" in r.json()["detail"]


def test_bands_contain_point(client):
    for p in client.get("/api/v1/forecast").json()["bdry"]["horizons"]:
        assert p["lo95"] <= p["lo80"] <= p["point"] <= p["hi80"] <= p["hi95"]
