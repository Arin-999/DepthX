# DepthX | SahaAI Fit

DepthX is a local-first Windows prototype that serves the supplied HTML
dashboard from a FastAPI backend. It reuses the existing MediaPipe pose model
and sample garment catalogue. It does not claim validated body measurements
or clothing-size accuracy.

## Windows 11 quick start

Use Python 3.11 (64-bit) and PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8501**. The server binds to loopback only; the API does
not enable interactive API documentation or access a camera. If activation is
blocked by PowerShell policy, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Stop the local server with **Ctrl+C**.

## Try the prototype

1. Select **Use sample** to load the included image or upload a JPG/PNG.
2. Choose **Analyze pose** to run MediaPipe and view the 2D landmark overlay.
3. Download the JSON report or CSV summary from the results panel.
4. Open **Garment tag** and enter a code, choose a sample product, or upload a
   QR/supported 1D barcode image.
5. The garment fields can be edited and updated for the current server run;
   those example catalogue edits are kept in memory and are not written back to
   the catalogue file.

Run automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Local API

- `GET /api/health` — application, model, and camera status.
- `GET /api/catalog` — sample products and centimetre units.
- `GET /api/sample-image` — included sample image.
- `POST /api/pose` — multipart image and optional garment fields; returns an
  annotated preview and downloadable-report data.
- `POST /api/tag/lookup` — look up a product code.
- `POST /api/tag/scan` — decode a QR or supported 1D barcode from an image.
- `PUT /api/catalog/{code}` — update a sample record in server memory.

Image uploads are limited to 10 MB and 20 megapixels. They are decoded for
inference, are not persisted to application output files, and are closed after
processing. Starlette may temporarily spool larger multipart uploads to the
operating system's temporary directory; that temporary file is closed and
removed after the request. The server adds no-cache, frame, content-type,
referrer, and permissions-policy headers. This is a single-user local
prototype, not a production deployment recipe for public networks.

## Measurement and depth limitations

- Pose landmarks are 2D estimates from a still RGB image.
- The application does not connect to a real depth camera. Synthetic depth
  created by the legacy pipeline is explicitly test data, not sensor data.
- Chest and waist body circumferences are unavailable. Pose or synthetic depth
  does not establish validated body measurements, so fit recommendations are
  withheld.
- Displayed processing duration is a single-run diagnostic, not a benchmark.
  Landmark visibility is not a body-measurement accuracy score.
- The privacy switch is a UI simulation and never controls or opens a camera.

## Existing command-line pipeline

The original research pipeline remains available:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

It generates synthetic RGB-D test artifacts under `outputs/`; those runtime
artifacts are ignored by Git.
