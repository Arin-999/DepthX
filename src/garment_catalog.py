"""Sample garment catalogue and image-based tag identification."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "product_catalog.json"
REQUIRED_MEASUREMENTS = ("chest_cm", "waist_cm")


def load_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    """Load and validate the sample catalogue; measurements are centimetres."""
    with path.open("r", encoding="utf-8") as catalog_file:
        data = json.load(catalog_file)

    if not isinstance(data, dict) or data.get("units") != "cm":
        raise ValueError("Garment catalogue must declare centimetre units.")

    products = data.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("Garment catalogue must contain at least one product.")

    seen_codes: set[str] = set()
    for product in products:
        if not isinstance(product, dict):
            raise ValueError("Each catalogue product must be a JSON object.")
        for field in ("code", "name", "category", "size", "color"):
            if not isinstance(product.get(field), str) or not product[field].strip():
                raise ValueError(f"Catalogue product requires a non-empty '{field}'.")

        normalized_code = product["code"].strip().casefold()
        if normalized_code in seen_codes:
            raise ValueError(f"Duplicate product code: {product['code']}")
        seen_codes.add(normalized_code)

        measurements = product.get("measurements")
        if not isinstance(measurements, dict):
            raise ValueError(
                f"Product {product['code']} requires a measurements object."
            )
        for measurement in REQUIRED_MEASUREMENTS:
            value = measurements.get(measurement)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Product {product['code']} has an invalid {measurement}."
                )

    return products


def find_product_by_code(
    products: list[dict[str, Any]], code: str
) -> dict[str, Any] | None:
    """Find an exact product code, ignoring surrounding whitespace and case."""
    normalized_code = code.strip().casefold()
    return next(
        (
            product
            for product in products
            if product["code"].strip().casefold() == normalized_code
        ),
        None,
    )


def scan_garment_tag(image_rgb: np.ndarray) -> str | None:
    """Decode the first QR or supported 1D barcode in an RGB image."""
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.size == 0
    ):
        raise ValueError("Tag image must be a non-empty RGB image.")

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    qr_code, _, _ = cv2.QRCodeDetector().detectAndDecode(image_bgr)
    if qr_code:
        return qr_code.strip() or None

    barcode_detector = cv2.barcode_BarcodeDetector()
    found, codes, _, _ = barcode_detector.detectAndDecodeWithType(image_bgr)
    if found:
        return next((code.strip() for code in codes if code.strip()), None)
    return None
