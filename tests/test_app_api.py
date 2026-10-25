import asyncio
from io import BytesIO

import cv2
import httpx
from PIL import Image

from app import MAX_UPLOAD_BYTES, app, products_by_code
from src.garment_catalog import load_catalog


def request(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8501",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


class LocalApiClient:
    def get(self, path, **kwargs):
        return request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return request("POST", path, **kwargs)

    def put(self, path, **kwargs):
        return request("PUT", path, **kwargs)


client = LocalApiClient()


def test_dashboard_and_health_are_local_prototype():
    response = client.get("/")

    assert response.status_code == 200
    assert "DepthX" in response.text
    assert "Local pose API ready" in response.text
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["permissions-policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["real_depth_camera_connected"] is False
    assert health.json()["camera_accessed"] is False


def test_api_rejects_cross_origin_and_non_loopback_hosts():
    cross_origin = client.get(
        "/api/health",
        headers={"Origin": "http://untrusted.example"},
    )
    invalid_host = client.get(
        "/api/health",
        headers={"Host": "untrusted.example"},
    )

    assert cross_origin.status_code == 403
    assert invalid_host.status_code == 400


def test_catalogue_lookup_and_unknown_code():
    response = client.post("/api/tag/lookup", json={"code": " dx-tee-001 "})

    assert response.status_code == 200
    assert response.json()["product"]["code"] == "DX-TEE-001"
    assert client.post(
        "/api/tag/lookup", json={"code": "missing"}
    ).status_code == 404


def test_catalogue_update_is_validated_and_in_memory():
    original = next(product for product in load_catalog() if product["code"] == "DX-TEE-001")
    response = client.put(
        "/api/catalog/DX-TEE-001",
        json={"name": "Temporary sample name", "chest_cm": 105, "waist_cm": 99},
    )

    assert response.status_code == 200
    assert products_by_code["dx-tee-001"]["name"] == "Temporary sample name"
    invalid = client.put(
        "/api/catalog/DX-TEE-001",
        json={"name": "Bad values", "chest_cm": -1, "waist_cm": 99},
    )
    assert invalid.status_code == 422

    products_by_code["dx-tee-001"].update(original)


def test_tag_scanner_endpoint_decodes_qr_product_code():
    qr_code = cv2.QRCodeEncoder_create().encode("DX-TEE-001")
    qr_code = cv2.resize(qr_code, (300, 300), interpolation=cv2.INTER_NEAREST)
    qr_code = cv2.copyMakeBorder(
        qr_code, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
    )
    success, encoded = cv2.imencode(".png", qr_code)
    assert success

    response = client.post(
        "/api/tag/scan",
        files={"image": ("tag.png", encoded.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["code"] == "DX-TEE-001"
    assert response.json()["catalogue_match"] is True


def test_pose_endpoint_runs_model_and_never_claims_measurements():
    with Image.open("data/sample/person.jpg") as sample:
        output = BytesIO()
        sample.save(output, format="JPEG")

    response = client.post(
        "/api/pose",
        files={"image": ("person.jpg", output.getvalue(), "image/jpeg")},
        data={"garment_code": "DX-TEE-001"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")
    report = payload["report"]
    assert report["depth"]["real_depth_camera_connected"] is False
    assert report["body_measurements"]["chest_circumference"]["value_cm"] is None
    assert report["body_measurements"]["waist_circumference"]["value_cm"] is None
    assert report["fit_analysis_status"] == "unavailable"
    assert report["privacy"]["image_saved_to_application_storage"] is False


def test_upload_endpoint_rejects_invalid_and_oversized_files():
    invalid = client.post(
        "/api/tag/scan",
        files={"image": ("tag.jpg", b"not an image", "image/jpeg")},
    )
    oversized = client.post(
        "/api/tag/scan",
        files={
            "image": (
                "large.jpg",
                b"x" * (MAX_UPLOAD_BYTES + 1),
                "image/jpeg",
            )
        },
    )

    assert invalid.status_code == 400
    assert oversized.status_code == 413
