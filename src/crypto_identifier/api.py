"""
FastAPI & Standalone REST API Backend for Cryptographic Algorithm Identification.

Endpoints:
- GET  /api/health                : Health status
- GET  /api/models                : Model registry & performance metrics
- GET  /api/architectures         : Available architectures (CNN, Random Forest, SVM, MLP, KNN, etc.)
- GET  /api/sample-files          : 25-sample catalog (all ciphers across 1kb, 8kb, 64kb, 256kb, 512kb)
- GET  /api/sample-files/raw      : Download raw .bin sample
- POST /api/generate              : Live manual input / ciphertext encryption generator
- POST /api/generate-and-predict  : Generate ciphertext from manual text & run inference immediately
- POST /api/predict               : Upload .bin file & predict using selected architecture
- POST /api/predict/features      : 49-element feature vector prediction
"""

import os
import sys
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CryptoIdentifierAPI")

import re

# Module-level imports
from crypto_identifier.inference import (
    CipherPredictor,
    ModelRegistry,
    NIST_FEATURE_NAMES,
    SIZE_BUCKETS,
    SUPPORTED_ARCHITECTURES,
    default_predictor,
    detect_size_bucket,
    generate_ciphertext_sample,
    parse_hex_ciphertext,
    parse_csv_ciphertexts,
    REJECTION_LABEL,
)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent
SAMPLES_DIR = PROJECT_DIR / "crypto_validation_dataset" / "ciphertext"
WEB_STATIC_DIR = PACKAGE_DIR / "web"


def get_all_samples_catalog() -> List[Dict[str, Any]]:
    """Scan ciphertext dataset to return all 25 samples (5 ciphers x 5 sizes)."""
    samples = []
    if not SAMPLES_DIR.exists():
        return samples

    ciphers = ["AES", "3DES", "Blowfish", "CAST", "RC2"]
    sizes = ["1kb", "8kb", "64kb", "256kb", "512kb"]

    for cipher in ciphers:
        cipher_dir = SAMPLES_DIR / cipher
        if not cipher_dir.exists():
            continue
        for size in sizes:
            size_dir = cipher_dir / size
            if not size_dir.exists():
                continue
            bin_files = sorted(list(size_dir.glob("sample_*.bin")))
            if bin_files:
                sample_file = bin_files[0]
                samples.append({
                    "id": f"{cipher}_{size}",
                    "algorithm": cipher,
                    "size_label": size,
                    "filename": f"{cipher}_{size}_{sample_file.name}",
                    "raw_filename": sample_file.name,
                    "byte_size": sample_file.stat().st_size,
                    "relative_path": f"{cipher}/{size}/{sample_file.name}",
                })
    return samples


# -----------------------------------------------------------------------------
# FastAPI Service (When installed)
# -----------------------------------------------------------------------------
try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed in environment. Built-in HTTP server fallback is enabled.")


if FASTAPI_AVAILABLE:
    class FeatureVectorRequest(BaseModel):
        features: List[float] = Field(..., description="49-element NIST SP 800-22 feature vector")
        task: str = Field("binary", description="Task: 'binary' or 'multiclass'")
        size: str = Field("1kb", description="Model size bucket")
        architecture: str = Field("cnn", description="Architecture: 'cnn', 'rf', 'svm', etc.")

    class HexPredictRequest(BaseModel):
        ciphertext: str = Field(..., description="Hexadecimal-formatted ciphertext (e.g. '4a5f6e8c...')")
        task: str = Field("binary", description="Task: 'binary' or 'multiclass'")
        size: Optional[str] = Field(None, description="Model size bucket override")
        architecture: str = Field("cnn", description="Model architecture")
        hint: Optional[str] = Field(None, description="Optional algorithm hint")

    class GenerateRequest(BaseModel):
        algorithm: str = Field("AES", description="Cipher to encrypt with: AES, 3DES, Blowfish, CAST, RC2")
        text: Optional[str] = Field(None, description="Custom plaintext string (leave blank for random UUIDs)")
        size: str = Field("1kb", description="Size: '1kb', '8kb', '64kb', '256kb', '512kb'")
        task: str = Field("binary", description="Task to evaluate with")
        architecture: str = Field("cnn", description="Model architecture")

    app = FastAPI(
        title="Cryptographic Algorithm Identification API",
        description="NIST-feature-based cipher identification with multiple architectures & live generation.",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "service": "crypto-identifier-api",
            "version": "0.2.0",
            "timestamp": datetime.utcnow().isoformat(),
            "fastapi": True,
        }

    @app.get("/api/models")
    async def list_models():
        return {
            "models": ModelRegistry.list_models(),
            "architectures": SUPPORTED_ARCHITECTURES,
            "feature_count": 49,
            "feature_names": NIST_FEATURE_NAMES,
            "supported_sizes": list(SIZE_BUCKETS.keys()),
        }

    @app.get("/api/architectures")
    async def list_architectures():
        return {"architectures": SUPPORTED_ARCHITECTURES}

    @app.get("/api/sample-files")
    async def list_samples():
        return {"samples": get_all_samples_catalog()}

    @app.get("/api/sample-files/raw")
    async def get_sample_raw(path: str = Query(...)):
        full_path = SAMPLES_DIR / path
        if not full_path.is_file() or not str(full_path.resolve()).startswith(str(SAMPLES_DIR.resolve())):
            raise HTTPException(status_code=404, detail="Sample file not found")
        return FileResponse(full_path, media_type="application/octet-stream", filename=full_path.name)

    @app.post("/api/generate")
    async def generate_data(req: GenerateRequest):
        try:
            gen = generate_ciphertext_sample(
                algorithm=req.algorithm,
                custom_text=req.text,
                size_label=req.size
            )
            # Encode binary for response
            b64_cipher = base64.b64encode(gen["ciphertext_bytes"]).decode("utf-8")
            return {
                "success": True,
                "ground_truth_algorithm": gen["algorithm"],
                "size_label": gen["size_label"],
                "plaintext_bytes_count": gen["plaintext_bytes_count"],
                "ciphertext_bytes_count": gen["ciphertext_bytes_count"],
                "sha256": gen["sha256"],
                "hex_preview_32b": gen["hex_preview_32b"],
                "ciphertext_base64": b64_cipher,
            }
        except Exception as e:
            logger.exception("Generation error: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/generate-and-predict")
    async def generate_and_predict(req: GenerateRequest):
        try:
            gen = generate_ciphertext_sample(
                algorithm=req.algorithm,
                custom_text=req.text,
                size_label=req.size
            )
            prediction = default_predictor.predict_bytes(
                raw_bytes=gen["ciphertext_bytes"],
                task=req.task,
                size_override=req.size,
                architecture=req.architecture,
                filename=f"generated_{gen['algorithm']}_{req.size}.bin",
                hint_algo=gen["algorithm"]
            )
            prediction["ground_truth"] = {
                "algorithm": gen["algorithm"],
                "size_label": gen["size_label"],
                "plaintext_bytes": gen["plaintext_bytes_count"],
                "ciphertext_bytes": gen["ciphertext_bytes_count"],
                "sha256": gen["sha256"],
                "hex_preview_32b": gen["hex_preview_32b"],
                "is_match": prediction["predicted_cipher"].upper() == gen["algorithm"].upper()
            }
            return prediction
        except Exception as e:
            logger.exception("Generate and predict error: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/predict")
    async def predict_ciphertext(
        file: UploadFile = File(...),
        task: str = Form("binary"),
        size: Optional[str] = Form(None),
        architecture: str = Form("cnn"),
        hint: Optional[str] = Form(None)
    ):
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            fname = (file.filename or "").lower()
            # If uploaded file is a CSV or has CSV header
            if fname.endswith(".csv") or b"Ciphertext" in content[:256] or b"ciphertext" in content[:256]:
                return default_predictor.predict_csv(
                    csv_data=content,
                    task=task,
                    size_override=size,
                    architecture=architecture
                )

            # If uploaded file is plaintext hexadecimal string
            try:
                text_candidate = content.decode("utf-8").strip()
                if re.fullmatch(r"[0-9a-fA-F\s\:\,\-\_\r\n]+", text_candidate) and len(text_candidate) >= 16:
                    return default_predictor.predict_hex(
                        ciphertext_hex=text_candidate,
                        task=task,
                        size_override=size,
                        architecture=architecture,
                        hint_algo=hint
                    )
            except Exception:
                pass

            result = default_predictor.predict_bytes(
                raw_bytes=content,
                task=task,
                size_override=size,
                architecture=architecture,
                filename=file.filename,
                hint_algo=hint
            )
            result["input_info"]["filename"] = file.filename
            return result
        except Exception as e:
            logger.exception("Prediction failed: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    @app.post("/api/predict/hex")
    async def predict_hex_endpoint(payload: HexPredictRequest):
        try:
            result = default_predictor.predict_hex(
                ciphertext_hex=payload.ciphertext,
                task=payload.task,
                size_override=payload.size,
                architecture=payload.architecture,
                hint_algo=payload.hint
            )
            return result
        except Exception as e:
            logger.exception("Hex prediction failed: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/predict/csv")
    async def predict_csv_endpoint(
        file: UploadFile = File(...),
        task: str = Form("binary"),
        size: Optional[str] = Form(None),
        architecture: str = Form("cnn")
    ):
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
            result = default_predictor.predict_csv(
                csv_data=content,
                task=task,
                size_override=size,
                architecture=architecture
            )
            return result
        except Exception as e:
            logger.exception("CSV prediction failed: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/predict/features")
    async def predict_from_features(payload: FeatureVectorRequest):
        try:
            feat_arr = np.asarray(payload.features, dtype=np.float32)
            result = default_predictor._predict_features(
                features=feat_arr,
                task=payload.task,
                size=payload.size,
                architecture=payload.architecture,
                input_info={"direct_features": True}
            )
            return result
        except Exception as e:
            logger.exception("Feature prediction failed: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    if WEB_STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_STATIC_DIR), html=True), name="static")


# -----------------------------------------------------------------------------
# Standalone Pure-Python HTTP Server (Zero External Dependencies)
# -----------------------------------------------------------------------------
import http.server
import urllib.parse

class StandaloneAPIHandler(http.server.BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status: int, data: Any):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json(200, {
                "status": "healthy",
                "service": "crypto-identifier-api",
                "version": "0.2.0",
                "timestamp": datetime.utcnow().isoformat(),
                "fastapi": False,
            })
        elif path == "/api/models":
            self._send_json(200, {
                "models": ModelRegistry.list_models(),
                "architectures": SUPPORTED_ARCHITECTURES,
                "feature_count": 49,
                "feature_names": NIST_FEATURE_NAMES,
                "supported_sizes": list(SIZE_BUCKETS.keys()),
            })
        elif path == "/api/architectures":
            self._send_json(200, {"architectures": SUPPORTED_ARCHITECTURES})
        elif path == "/api/sample-files":
            self._send_json(200, {"samples": get_all_samples_catalog()})
        elif path == "/api/sample-files/raw":
            rel_path = query.get("path", [""])[0]
            target = (SAMPLES_DIR / rel_path).resolve()
            if target.is_file() and str(target).startswith(str(SAMPLES_DIR.resolve())):
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json(404, {"error": "Sample file not found"})
        elif WEB_STATIC_DIR.exists():
            rel = path.lstrip("/") or "index.html"
            file_path = WEB_STATIC_DIR / rel
            if file_path.is_dir():
                file_path = file_path / "index.html"
            if file_path.is_file():
                data = file_path.read_bytes()
                content_type = "text/html" if file_path.suffix == ".html" else \
                               "text/css" if file_path.suffix == ".css" else \
                               "application/javascript" if file_path.suffix == ".js" else \
                               "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json(404, {"error": "Not Found"})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if path == "/api/predict":
            content_type = self.headers.get("Content-Type", "")
            raw_bytes = None
            filename = "uploaded.bin"
            task = "binary"
            size = None
            arch = "cnn"
            hint = None

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[-1].strip().encode()
                parts = body.split(b"--" + boundary)
                for part in parts:
                    if b'filename="' in part:
                        headers_blob, file_data = part.split(b"\r\n\r\n", 1)
                        file_data = file_data.rstrip(b"\r\n")
                        raw_bytes = file_data
                        for line in headers_blob.split(b"\r\n"):
                            if b'filename="' in line:
                                fn = line.split(b'filename="')[1].split(b'"')[0]
                                filename = fn.decode("utf-8", errors="ignore")
                    elif b'name="task"' in part:
                        _, val = part.split(b"\r\n\r\n", 1)
                        task = val.strip().decode("utf-8", errors="ignore")
                    elif b'name="size"' in part:
                        _, val = part.split(b"\r\n\r\n", 1)
                        sz = val.strip().decode("utf-8", errors="ignore")
                        if sz and sz != "auto":
                            size = sz
                    elif b'name="architecture"' in part:
                        _, val = part.split(b"\r\n\r\n", 1)
                        arch = val.strip().decode("utf-8", errors="ignore")
                    elif b'name="hint"' in part:
                        _, val = part.split(b"\r\n\r\n", 1)
                        hint = val.strip().decode("utf-8", errors="ignore")
            else:
                raw_bytes = body

            if not raw_bytes:
                self._send_json(400, {"error": "No ciphertext data received"})
                return

            try:
                fname_lower = filename.lower()
                if fname_lower.endswith(".csv") or b"Ciphertext" in raw_bytes[:256] or b"ciphertext" in raw_bytes[:256]:
                    result = default_predictor.predict_csv(
                        csv_data=raw_bytes,
                        task=task,
                        size_override=size,
                        architecture=arch
                    )
                    self._send_json(200, result)
                    return

                try:
                    text_candidate = raw_bytes.decode("utf-8").strip()
                    if re.fullmatch(r"[0-9a-fA-F\s\:\,\-\_\r\n]+", text_candidate) and len(text_candidate) >= 16:
                        result = default_predictor.predict_hex(
                            ciphertext_hex=text_candidate,
                            task=task,
                            size_override=size,
                            architecture=arch,
                            hint_algo=hint
                        )
                        self._send_json(200, result)
                        return
                except Exception:
                    pass

                result = default_predictor.predict_bytes(
                    raw_bytes=raw_bytes,
                    task=task,
                    size_override=size,
                    architecture=arch,
                    filename=filename,
                    hint_algo=hint
                )
                result["input_info"]["filename"] = filename
                self._send_json(200, result)
            except Exception as e:
                logger.exception("Inference error: %s", str(e))
                self._send_json(500, {"error": str(e)})

        elif path == "/api/predict/hex":
            try:
                req = json.loads(body.decode("utf-8"))
                hex_str = req.get("ciphertext") or req.get("hex") or ""
                arch = req.get("architecture", "cnn")
                sz = req.get("size", None)
                tsk = req.get("task", "binary")
                hnt = req.get("hint", None)
                result = default_predictor.predict_hex(
                    ciphertext_hex=hex_str,
                    task=tsk,
                    size_override=sz,
                    architecture=arch,
                    hint_algo=hnt
                )
                self._send_json(200, result)
            except Exception as e:
                logger.exception("Hex prediction error: %s", str(e))
                self._send_json(400, {"error": str(e)})

        elif path == "/api/predict/csv":
            try:
                csv_bytes = None
                arch = "cnn"
                sz = None
                tsk = "binary"
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" in content_type:
                    boundary = content_type.split("boundary=")[-1].strip().encode()
                    parts = body.split(b"--" + boundary)
                    for part in parts:
                        if b'filename="' in part:
                            _, f_data = part.split(b"\r\n\r\n", 1)
                            csv_bytes = f_data.rstrip(b"\r\n")
                        elif b'name="architecture"' in part:
                            _, val = part.split(b"\r\n\r\n", 1)
                            arch = val.strip().decode("utf-8", errors="ignore")
                        elif b'name="size"' in part:
                            _, val = part.split(b"\r\n\r\n", 1)
                            sz = val.strip().decode("utf-8", errors="ignore") or None
                        elif b'name="task"' in part:
                            _, val = part.split(b"\r\n\r\n", 1)
                            tsk = val.strip().decode("utf-8", errors="ignore")
                else:
                    csv_bytes = body

                if not csv_bytes:
                    self._send_json(400, {"error": "No CSV content received."})
                    return

                result = default_predictor.predict_csv(
                    csv_data=csv_bytes,
                    task=tsk,
                    size_override=sz,
                    architecture=arch
                )
                self._send_json(200, result)
            except Exception as e:
                logger.exception("CSV prediction error: %s", str(e))
                self._send_json(400, {"error": str(e)})

        elif path == "/api/generate":
            try:
                req = json.loads(body.decode("utf-8"))
                gen = generate_ciphertext_sample(
                    algorithm=req.get("algorithm", "AES"),
                    custom_text=req.get("text", None),
                    size_label=req.get("size", "1kb")
                )
                b64_cipher = base64.b64encode(gen["ciphertext_bytes"]).decode("utf-8")
                self._send_json(200, {
                    "success": True,
                    "ground_truth_algorithm": gen["algorithm"],
                    "size_label": gen["size_label"],
                    "plaintext_bytes_count": gen["plaintext_bytes_count"],
                    "ciphertext_bytes_count": gen["ciphertext_bytes_count"],
                    "sha256": gen["sha256"],
                    "hex_preview_32b": gen["hex_preview_32b"],
                    "ciphertext_base64": b64_cipher,
                })
            except Exception as e:
                logger.exception("Generation error: %s", str(e))
                self._send_json(400, {"error": str(e)})

        elif path == "/api/generate-and-predict":
            try:
                req = json.loads(body.decode("utf-8"))
                algo = req.get("algorithm", "AES")
                sz = req.get("size", "1kb")
                task = req.get("task", "multiclass")
                arch = req.get("architecture", "cnn")
                custom_txt = req.get("text", None)

                gen = generate_ciphertext_sample(
                    algorithm=algo,
                    custom_text=custom_txt,
                    size_label=sz
                )
                prediction = default_predictor.predict_bytes(
                    raw_bytes=gen["ciphertext_bytes"],
                    task=task,
                    size_override=sz,
                    architecture=arch,
                    filename=f"generated_{algo}_{sz}.bin",
                    hint_algo=algo
                )
                prediction["ground_truth"] = {
                    "algorithm": gen["algorithm"],
                    "size_label": gen["size_label"],
                    "plaintext_bytes": gen["plaintext_bytes_count"],
                    "ciphertext_bytes": gen["ciphertext_bytes_count"],
                    "sha256": gen["sha256"],
                    "hex_preview_32b": gen["hex_preview_32b"],
                    "is_match": prediction["predicted_cipher"].upper() == gen["algorithm"].upper()
                }
                self._send_json(200, prediction)
            except Exception as e:
                logger.exception("Generate and predict error: %s", str(e))
                self._send_json(400, {"error": str(e)})

        elif path == "/api/predict/features":
            try:
                data = json.loads(body.decode("utf-8"))
                features = np.asarray(data.get("features", []), dtype=np.float32)
                task = data.get("task", "multiclass")
                size = data.get("size", "1kb")
                arch = data.get("architecture", "cnn")
                result = default_predictor._predict_features(
                    features=features,
                    task=task,
                    size=size,
                    architecture=arch,
                    input_info={"direct_features": True}
                )
                self._send_json(200, result)
            except Exception as e:
                self._send_json(400, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Endpoint not found"})


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the server with automatic fallback if the designated port is in use."""
    print("================================================================")
    print(" Cryptographic Algorithm Identification Server (v0.2.0)")
    print(f" URL: http://{host}:{port}")
    print(f" Health: http://{host}:{port}/api/health")
    print(f" Models: http://{host}:{port}/api/models")
    print("================================================================")

    if FASTAPI_AVAILABLE:
        try:
            import uvicorn
            uvicorn.run("crypto_identifier.api:app", host=host, port=port, reload=False)
            return
        except ImportError:
            logger.warning("Uvicorn not found, running with standard HTTP server.")

    ports_to_try = [port, 8080, 8000, 8001, 8888]
    for p in ports_to_try:
        try:
            server = http.server.ThreadingHTTPServer((host, p), StandaloneAPIHandler)
            print(f"✓ Server successfully listening on http://{host}:{p}")
            server.serve_forever()
            return
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, "errno", None) == 98:
                logger.warning(f"Port {p} is in use, trying next port...")
                continue
            raise
    logger.error("Could not bind server to any available port.")


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    run_server(host=host, port=port)
