import json

import cv2
import numpy as np
import pytest

from src.garment_catalog import (
    find_product_by_code,
    load_catalog,
    scan_garment_tag,
)


def test_sample_catalog_contains_products_with_cm_measurements():
    products = load_catalog()

    assert len(products) >= 3
    assert all(
        product["measurements"]["chest_cm"] > 0
        and product["measurements"]["waist_cm"] > 0
        for product in products
    )


def test_find_product_by_code_ignores_case_and_whitespace():
    products = load_catalog()

    assert find_product_by_code(products, "  dx-tee-001 ") is products[0]
    assert find_product_by_code(products, "not-in-catalogue") is None


def test_catalog_rejects_non_centimetre_units(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"units": "inches", "products": []}))

    with pytest.raises(ValueError, match="centimetre"):
        load_catalog(path)


def test_tag_scanner_rejects_invalid_image_shape():
    with pytest.raises(ValueError, match="RGB image"):
        scan_garment_tag(np.zeros((12, 12), dtype=np.uint8))


def test_tag_scanner_returns_none_for_blank_image():
    assert scan_garment_tag(np.zeros((120, 120, 3), dtype=np.uint8)) is None


def test_tag_scanner_decodes_qr_product_code():
    qr_code = cv2.QRCodeEncoder_create().encode("DX-TEE-001")
    qr_code = cv2.resize(qr_code, (300, 300), interpolation=cv2.INTER_NEAREST)
    qr_code = cv2.copyMakeBorder(
        qr_code,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    image_rgb = cv2.cvtColor(qr_code, cv2.COLOR_GRAY2RGB)

    assert scan_garment_tag(image_rgb) == "DX-TEE-001"
