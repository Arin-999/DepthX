"""Loopback-only FastAPI server for the DepthX HTML dashboard."""

import base64
import logging
import time
import urllib.parse
import warnings
from io import BytesIO
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from src.garment_catalog import load_catalog, scan_garment_tag
from src.pose_service import analyze_uploaded_image, build_pose_report


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "depthx_dashboard.html"
SAMPLE_IMAGE_PATH = ROOT / "data" / "sample" / "person.jpg"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
LOGGER = logging.getLogger("depthx")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS * 2

app = FastAPI(
    title="DepthX local prototype API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
products = load_catalog()
products_by_code = {product["code"].casefold(): product for product in products}


class ProductUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    chest_cm: float = Field(gt=0, le=300)
    waist_cm: float = Field(gt=0, le=300)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Product name cannot be blank.")
        return normalized_name


class ProductLookup(BaseModel):
    code: str = Field(min_length=1, max_length=80)


@app.middleware("http")
async def add_local_security_headers(request, call_next):
    request_host = request.headers.get("host", "").split(":")[0].strip("[]")
    if request_host.casefold() not in {"127.0.0.1", "localhost"}:
        return JSONResponse(
            {"detail": "This local API accepts loopback hostnames only."},
            status_code=400,
        )

    origin = request.headers.get("origin")
    if origin:
        try:
            parsed_origin = urllib.parse.urlsplit(origin)
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid request origin."},
                status_code=403,
            )
        if (
            parsed_origin.scheme != "http"
            or parsed_origin.hostname != request_host.casefold()
            or parsed_origin.port not in (None, 8501)
        ):
            return JSONResponse(
                {"detail": "Cross-origin requests are not accepted."},
                status_code=403,
            )

    content_length = request.headers.get("content-length")
    if request.method in {"POST", "PUT"} and content_length:
        try:
            request_size = int(content_length)
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid Content-Length header."},
                status_code=400,
            )
        if request_size > MAX_UPLOAD_BYTES + 64 * 1024:
            return JSONResponse(
                {"detail": "Request exceeds the local upload limit."},
                status_code=413,
            )

    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; font-src 'self' data:; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )
    return response


def decode_image(content: bytes) -> Image.Image:
    """Decode a bounded raster image; never persist uploaded bytes."""
    try:
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(
                        status_code=415,
                        detail="Upload a JPEG, PNG, or WebP image.",
                    )
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image exceeds the supported pixel limit.",
                    )
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image.",
        ) from exc


async def read_image(upload: UploadFile) -> Image.Image:
    """Read and decode a bounded image in memory without blocking the event loop."""
    try:
        content = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Image exceeds the "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                ),
            )
        return await run_in_threadpool(decode_image, content)
    finally:
        await upload.close()


def product_by_code(code: str) -> dict | None:
    return products_by_code.get(code.strip().casefold())


def encode_preview(image_rgb: np.ndarray) -> str:
    preview = image_rgb
    height, width = image_rgb.shape[:2]
    longest_edge = max(height, width)
    if longest_edge > 1600:
        scale = 1600 / longest_edge
        preview = cv2.resize(
            image_rgb,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    if not success:
        raise RuntimeError("Could not encode the pose preview.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


@app.get("/")
async def dashboard():
    if not HTML_PATH.is_file():
        raise HTTPException(status_code=500, detail="Dashboard HTML is missing.")
    return FileResponse(
        HTML_PATH,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "application": "DepthX",
        "pose_model_available": (ROOT / "models" / "pose_landmarker_lite.task").is_file(),
        "real_depth_camera_connected": False,
        "camera_accessed": False,
    }


@app.get("/api/catalog")
async def catalog():
    return {"units": "cm", "products": products}


@app.get("/api/sample-image")
async def sample_image():
    if not SAMPLE_IMAGE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Sample image is unavailable.")
    return FileResponse(
        SAMPLE_IMAGE_PATH,
        media_type="image/jpeg",
        filename="depthx-sample.jpg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/pose")
async def pose_analysis(
    image: Annotated[UploadFile, File()],
    garment_code: Annotated[str, Form()] = "",
    garment_name: Annotated[str, Form()] = "Sample garment",
    garment_chest_cm: Annotated[float, Form(gt=0, le=300)] = 100,
    garment_waist_cm: Annotated[float, Form(gt=0, le=300)] = 90,
):
    decoded = await read_image(image)
    started_at = time.perf_counter()
    try:
        annotated_image, landmarks = await run_in_threadpool(
            analyze_uploaded_image, decoded
        )
    except FileNotFoundError as exc:
        LOGGER.error("Pose model is unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Pose model is unavailable. Check the local model file.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Pose inference failed")
        raise HTTPException(
            status_code=500,
            detail="Pose inference failed. Check the server log for details.",
        ) from exc

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    visibility = (
        float(np.mean([landmark.visibility for landmark in landmarks]))
        if landmarks
        else None
    )
    product = product_by_code(garment_code) if garment_code else None
    report = build_pose_report(
        source_name="uploaded image",
        image=decoded,
        landmarks=landmarks,
        garment={
            "code": product["code"] if product else garment_code or None,
            "name": garment_name[:120],
            "units": "cm",
            "chest_circumference": garment_chest_cm,
            "waist_circumference": garment_waist_cm,
        },
        elapsed_ms=elapsed_ms,
        mean_visibility=visibility,
    )
    report["performance"]["note"] = (
        "Single-request pose processing time; not a benchmark or accuracy measure."
    )
    try:
        preview_data_url = await run_in_threadpool(
            encode_preview, annotated_image
        )
    except (cv2.error, RuntimeError) as exc:
        LOGGER.exception("Could not encode pose result preview")
        raise HTTPException(
            status_code=500,
            detail="Could not encode the pose preview.",
        ) from exc
    return JSONResponse(
        {
            "report": report,
            "annotated_image": preview_data_url,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/tag/lookup")
async def lookup_tag(payload: ProductLookup):
    product = product_by_code(payload.code)
    if product is None:
        raise HTTPException(
            status_code=404,
            detail=f"No sample catalogue item matches code {payload.code.strip()}.",
        )
    return {"code": product["code"], "product": product}


@app.post("/api/tag/scan")
async def scan_tag(image: Annotated[UploadFile, File()]):
    decoded = await read_image(image)
    try:
        code = await run_in_threadpool(scan_garment_tag, np.asarray(decoded))
    except (ValueError, cv2.error) as exc:
        LOGGER.exception("Tag scan failed")
        raise HTTPException(
            status_code=422,
            detail="Could not decode a QR code or supported barcode.",
        ) from exc
    if not code:
        raise HTTPException(
            status_code=422,
            detail="No QR code or supported barcode was decoded.",
        )
    product = product_by_code(code)
    return {
        "code": code,
        "product": product,
        "catalogue_match": product is not None,
    }


@app.put("/api/catalog/{code}")
async def update_catalogue_record(code: str, update: ProductUpdate):
    """Update an example record only for the lifetime of this server process."""
    product = product_by_code(code)
    if product is None:
        raise HTTPException(status_code=404, detail="Sample product was not found.")
    product["name"] = update.name.strip()
    product["measurements"]["chest_cm"] = update.chest_cm
    product["measurements"]["waist_cm"] = update.waist_cm
    return {"code": product["code"], "product": product}


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8501,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
