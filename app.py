"""Flask web application and HTTP API service for ARACHNE 3D reconstruction.

Exposes the standalone Python reconstruction engine over HTTP with CORS support,
asynchronous job queuing, structured JSON responses, and secure sandboxed file delivery:
    GET  /api/health
    POST /api/reconstruct
    GET  /api/jobs/<job_id>
    GET  /api/jobs/<job_id>/result
    GET  /api/jobs/<job_id>/files/<filename>
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Optional, Set
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Import ARACHNE single-view reconstruction pipeline
from arachne.single_view.pipeline import (
    ALLOWED_ARTIFACT_NAMES,
    MODEL_MAP,
    SUPPORTED_IMAGE_EXTENSIONS,
    SceneReconstructionConfig,
    SceneReconstructionResult,
    reconstruct_scene,
    validate_image,
)

# Application Configuration
BASE_DIR = Path(__file__).resolve().parent
WEB_OUTPUT_DIR = BASE_DIR / "web_outputs"
JOBS_DIR = WEB_OUTPUT_DIR / "jobs"
UPLOADS_DIR = WEB_OUTPUT_DIR / "uploads"

# Ensure output directories exist
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
ALLOWED_DOWNLOAD_FILES = ALLOWED_ARTIFACT_NAMES
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB max upload

# CORS Configuration
ALLOWED_ORIGINS_ENV = os.environ.get(
    "ARACHNE_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
)
ALLOWED_ORIGINS: Set[str] = {
    orig.strip() for orig in ALLOWED_ORIGINS_ENV.split(",") if orig.strip()
}
ALLOW_ALL_ORIGINS = "*" in ALLOWED_ORIGINS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory registry for jobs
jobs_lock = threading.Lock()
jobs_registry: Dict[str, Dict[str, Any]] = {}


# --- CORS Middleware ---

@app.after_request
def add_cors_headers(response):
    """Applies cross-origin resource sharing headers for Next.js and frontend integration."""
    origin = request.headers.get("Origin")
    if ALLOW_ALL_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    elif not origin and not ALLOW_ALL_ORIGINS:
        # Same-origin / direct browser navigation
        pass

    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


@app.route("/api/<path:subpath>", methods=["OPTIONS"])
def api_options_preflight(subpath):
    """Handles HTTP OPTIONS preflight requests for all /api endpoints."""
    return "", 204


# --- Statistics Calculation & Reconciliation Helper ---

def calculate_missing_statistics(
    job_dir: Path,
    existing_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensures all reconstruction statistics (resolution, vertices, triangles, points, depth)
    are present and accurate. If missing or incomplete, inspects the actual output files
    (scene_mesh.obj, scene_mesh.glb, scene_pointcloud.ply, uploaded.*, depth_summary.json)
    to calculate genuine geometry measurements.
    """
    stats: Dict[str, Any] = dict(existing_stats or {})

    # 1. Resolve Input Resolution (input_width, input_height, image_width, image_height)
    width = stats.get("image_width") or stats.get("input_width")
    height = stats.get("image_height") or stats.get("input_height")

    if not width or not height or width <= 0 or height <= 0:
        image_candidates = [
            list(job_dir.glob("uploaded.*")),
            [job_dir / "input_image.png"],
            [job_dir / "depth_input_rgb.png"],
        ]
        for candidate_list in image_candidates:
            for img_file in candidate_list:
                if img_file.exists() and img_file.is_file():
                    try:
                        with Image.open(img_file) as im:
                            width, height = im.size
                            break
                    except Exception:
                        pass
            if width and height:
                break

        if not width or not height:
            summary_path = job_dir / "depth_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as sf:
                        sdata = json.load(sf)
                        width = sdata.get("image_width")
                        height = sdata.get("image_height")
                except Exception:
                    pass

        if not width or not height:
            npy_path = job_dir / "depth_metric.npy"
            if npy_path.exists():
                try:
                    depth_arr = np.load(str(npy_path), mmap_mode="r")
                    height, width = int(depth_arr.shape[0]), int(depth_arr.shape[1])
                except Exception:
                    pass

    # 2. Resolve Mesh Vertices & Triangles (mesh_vertices, num_vertices, mesh_triangles, num_faces)
    num_vertices = stats.get("num_vertices") or stats.get("mesh_vertices") or stats.get("vertex_count")
    num_faces = (
        stats.get("num_faces")
        or stats.get("mesh_triangles")
        or stats.get("face_count")
        or stats.get("triangle_count")
        or stats.get("num_triangles")
    )

    if num_vertices is None or num_faces is None:
        obj_path = job_dir / "scene_mesh.obj"
        if obj_path.exists() and obj_path.is_file():
            try:
                v_count = 0
                f_count = 0
                with open(obj_path, "rb") as of:
                    for line in of:
                        if line.startswith(b"v "):
                            v_count += 1
                        elif line.startswith(b"f "):
                            f_count += 1
                if num_vertices is None and v_count > 0:
                    num_vertices = v_count
                if num_faces is None and f_count > 0:
                    num_faces = f_count
            except Exception:
                pass

        if num_vertices is None or num_faces is None:
            glb_path = job_dir / "scene_mesh.glb"
            if glb_path.exists() and glb_path.is_file():
                try:
                    import trimesh
                    loaded = trimesh.load(str(glb_path), file_type="glb")
                    if isinstance(loaded, trimesh.Trimesh):
                        if num_vertices is None:
                            num_vertices = len(loaded.vertices)
                        if num_faces is None:
                            num_faces = len(loaded.faces)
                    elif isinstance(loaded, trimesh.Scene):
                        v_total = sum(len(g.vertices) for g in loaded.geometry.values() if hasattr(g, "vertices"))
                        f_total = sum(len(g.faces) for g in loaded.geometry.values() if hasattr(g, "faces"))
                        if num_vertices is None:
                            num_vertices = v_total
                        if num_faces is None:
                            num_faces = f_total
                except Exception:
                    pass

    # 3. Resolve Point Count (point_count, num_points)
    point_count = stats.get("num_points") or stats.get("point_count")
    if point_count is None:
        ply_path = job_dir / "scene_pointcloud.ply"
        if ply_path.exists() and ply_path.is_file():
            try:
                with open(ply_path, "rb") as pf:
                    for line in pf:
                        line_str = line.decode("ascii", errors="ignore").strip()
                        if line_str.startswith("element vertex "):
                            parts = line_str.split()
                            if len(parts) >= 3 and parts[2].isdigit():
                                point_count = int(parts[2])
                                break
                        elif line_str == "end_header":
                            break
            except Exception:
                pass

        if point_count is None:
            npy_path = job_dir / "depth_metric.npy"
            if npy_path.exists():
                try:
                    depth_arr = np.load(str(npy_path), mmap_mode="r")
                    point_count = int(np.count_nonzero(np.isfinite(depth_arr) & (depth_arr > 0)))
                except Exception:
                    pass

    # 4. Resolve Depth Range (min_depth_m, max_depth_m, median_depth_m)
    min_d = stats.get("min_depth_m")
    max_d = stats.get("max_depth_m")
    med_d = stats.get("median_depth_m")

    if min_d is None or max_d is None:
        summary_path = job_dir / "depth_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as sf:
                    sdata = json.load(sf)
                    d_stats = sdata.get("depth_statistics", {})
                    if min_d is None:
                        min_d = d_stats.get("min_depth_m")
                    if max_d is None:
                        max_d = d_stats.get("max_depth_m")
                    if med_d is None:
                        med_d = d_stats.get("median_depth_m")
            except Exception:
                pass

        if min_d is None or max_d is None:
            npy_path = job_dir / "depth_metric.npy"
            if npy_path.exists():
                try:
                    depth_arr = np.load(str(npy_path))
                    valid = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0)]
                    if len(valid) > 0:
                        if min_d is None:
                            min_d = round(float(np.min(valid)), 2)
                        if max_d is None:
                            max_d = round(float(np.max(valid)), 2)
                        if med_d is None:
                            med_d = round(float(np.median(valid)), 2)
                except Exception:
                    pass

    # 5. Populate File Sizes on Disk
    file_sizes = dict(stats.get("file_sizes") or {})
    for fname in ["scene_mesh.glb", "scene_mesh.obj", "scene_pointcloud.ply", "depth_vis.png"]:
        fpath = job_dir / fname
        if fpath.exists() and fpath.is_file():
            file_sizes[fname] = fpath.stat().st_size
    stats["file_sizes"] = file_sizes

    # 6. Set Synchronized Values across all property variants
    if width is not None and height is not None:
        stats["input_width"] = int(width)
        stats["image_width"] = int(width)
        stats["input_height"] = int(height)
        stats["image_height"] = int(height)

    if num_vertices is not None:
        stats["mesh_vertices"] = int(num_vertices)
        stats["num_vertices"] = int(num_vertices)
        stats["vertex_count"] = int(num_vertices)

    if num_faces is not None:
        stats["mesh_triangles"] = int(num_faces)
        stats["num_faces"] = int(num_faces)
        stats["face_count"] = int(num_faces)
        stats["triangle_count"] = int(num_faces)
        stats["num_triangles"] = int(num_faces)

    if point_count is not None:
        stats["point_count"] = int(point_count)
        stats["num_points"] = int(point_count)

    if min_d is not None:
        stats["min_depth_m"] = round(float(min_d), 2)
        stats["depth_min_m"] = round(float(min_d), 2)
    if max_d is not None:
        stats["max_depth_m"] = round(float(max_d), 2)
        stats["depth_max_m"] = round(float(max_d), 2)
    if med_d is not None:
        stats["median_depth_m"] = round(float(med_d), 2)
        stats["depth_median_m"] = round(float(med_d), 2)

    return stats


# --- Background Worker ---

def run_reconstruction_worker(
    job_id: str,
    image_path: Path,
    scene_type: str = "indoor",
    fov: float = 60.0,
    depth_jump_ratio: float = 0.08,
    mesh_stride: int = 1,
) -> None:
    """Background worker thread executing the reconstruction engine in an isolated job folder."""
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def update_job_status(status: str, stage: str, progress: int, error: Optional[str] = None):
        with jobs_lock:
            if job_id in jobs_registry:
                jobs_registry[job_id]["status"] = status
                jobs_registry[job_id]["stage"] = stage
                jobs_registry[job_id]["progress"] = progress
                if error:
                    jobs_registry[job_id]["error"] = error

    try:
        update_job_status("processing", "Running metric depth estimation and 3D surface meshing...", 30)

        recon_cfg = SceneReconstructionConfig(
            scene_type=scene_type,
            fov_deg=fov,
            depth_jump_ratio=depth_jump_ratio,
            mesh_stride=mesh_stride,
        )

        # Execute the standalone Python reconstruction engine
        result: SceneReconstructionResult = reconstruct_scene(
            image_path=image_path,
            config=recon_cfg,
            output_dir=job_dir,
            job_id=job_id,
            base_url_prefix="/api/jobs",
        )

        final_metrics = calculate_missing_statistics(
            job_dir,
            result.statistics if result.success else None,
        )

        with jobs_lock:
            if result.success:
                jobs_registry[job_id]["status"] = "completed"
                jobs_registry[job_id]["stage"] = "Reconstruction complete"
                jobs_registry[job_id]["progress"] = 100
                jobs_registry[job_id]["metrics"] = final_metrics
                jobs_registry[job_id]["files"] = {
                    "glb": f"/api/jobs/{job_id}/files/scene_mesh.glb",
                    "obj": f"/api/jobs/{job_id}/files/scene_mesh.obj",
                    "ply": f"/api/jobs/{job_id}/files/scene_pointcloud.ply",
                    "depth_vis": f"/api/jobs/{job_id}/files/depth_vis.png",
                    "manifest": f"/api/jobs/{job_id}/files/scene_manifest.json",
                }
                res_dict = result.to_dict()
                res_dict["statistics"] = final_metrics
                jobs_registry[job_id]["result_data"] = res_dict
            else:
                jobs_registry[job_id]["status"] = "failed"
                jobs_registry[job_id]["stage"] = result.stage
                jobs_registry[job_id]["progress"] = 0
                jobs_registry[job_id]["error"] = result.error
                jobs_registry[job_id]["result_data"] = result.to_dict()

    except Exception as e:
        update_job_status("failed", "Reconstruction failed", 0, error=str(e))


# --- HTTP Routes ---

@app.route("/")
def index():
    """Renders the standalone Fieldwork web interface (optional development UI)."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Reports API service status, CUDA GPU availability, and model device."""
    cuda_available = False
    device_name = "CPU"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "service": "ARACHNE 3D Reconstruction Service",
        "cuda_available": cuda_available,
        "device_name": device_name,
        "timestamp": time.time(),
    })


@app.route("/api/reconstruct", methods=["POST"])
def create_reconstruction_job():
    """Accepts an uploaded image and initiates an asynchronous 3D reconstruction job."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    raw_filename = secure_filename(file.filename)
    ext = Path(raw_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }), 400

    # Parse calibration options with defensive range clamping
    scene_type = request.form.get("scene_type", "indoor").strip().lower()
    if scene_type not in MODEL_MAP:
        scene_type = "indoor"

    try:
        fov = float(request.form.get("fov", 60.0))
        fov = max(20.0, min(140.0, fov))
    except (ValueError, TypeError):
        fov = 60.0

    try:
        depth_jump = float(request.form.get("depth_jump_ratio", 0.08))
        depth_jump = max(0.01, min(0.5, depth_jump))
    except (ValueError, TypeError):
        depth_jump = 0.08

    try:
        stride = int(request.form.get("mesh_stride", 1))
        stride = max(1, min(4, stride))
    except (ValueError, TypeError):
        stride = 1

    # Generate isolated job ID and directory
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    upload_save_path = job_dir / f"uploaded{ext}"
    try:
        file.save(str(upload_save_path))
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save upload: {e}"}), 500

    # Validate image file integrity
    is_valid, val_err = validate_image(upload_save_path)
    if not is_valid:
        try:
            upload_save_path.unlink()
        except Exception:
            pass
        return jsonify({"success": False, "error": val_err or "Corrupt or invalid image."}), 400

    # Register initial job state
    with jobs_lock:
        jobs_registry[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Job queued for processing...",
            "progress": 5,
            "created_at": time.time(),
            "error": None,
            "metrics": None,
            "files": {},
            "result_data": None,
        }

    # Dispatch worker thread
    worker = threading.Thread(
        target=run_reconstruction_worker,
        kwargs={
            "job_id": job_id,
            "image_path": upload_save_path,
            "scene_type": scene_type,
            "fov": fov,
            "depth_jump_ratio": depth_jump,
            "mesh_stride": stride,
        },
        daemon=True,
    )
    worker.start()

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "message": "Reconstruction job started successfully.",
    }), 200


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Returns current status, progress stage, metrics, and artifact URLs for an active job."""
    if not re.match(r"^[a-f0-9\-]{36}$", job_id):
        return jsonify({"success": False, "error": "Invalid job ID format."}), 404

    with jobs_lock:
        job = jobs_registry.get(job_id)

    if not job:
        # Check if job exists on disk from a prior server session
        job_dir = JOBS_DIR / job_id
        manifest_path = job_dir / "scene_manifest.json"
        glb_path = job_dir / "scene_mesh.glb"
        if manifest_path.exists() and glb_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                raw_stats = manifest_data.get("statistics", manifest_data.get("bounding_box_meters", {}))
                complete_metrics = calculate_missing_statistics(job_dir, raw_stats)
                return jsonify({
                    "success": True,
                    "job_id": job_id,
                    "status": "completed",
                    "stage": "Reconstruction complete (loaded from disk)",
                    "progress": 100,
                    "metrics": complete_metrics,
                    "files": {
                        "glb": f"/api/jobs/{job_id}/files/scene_mesh.glb",
                        "obj": f"/api/jobs/{job_id}/files/scene_mesh.obj",
                        "ply": f"/api/jobs/{job_id}/files/scene_pointcloud.ply",
                        "depth_vis": f"/api/jobs/{job_id}/files/depth_vis.png",
                    },
                })
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Job '{job_id}' not found."}), 404

    # Ensure in-memory metrics also have all aliases and calculated values
    metrics_data = job.get("metrics")
    if metrics_data and job.get("status") == "completed":
        job_dir = JOBS_DIR / job_id
        metrics_data = calculate_missing_statistics(job_dir, metrics_data)

    return jsonify({
        "success": True,
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job.get("error"),
        "metrics": metrics_data,
        "files": job.get("files", {}),
    })


@app.route("/api/jobs/<job_id>/result", methods=["GET"])
def get_job_result(job_id: str):
    """Returns the structured final reconstruction result matching the integration contract."""
    if not re.match(r"^[a-f0-9\-]{36}$", job_id):
        return jsonify({"success": False, "error": "Invalid job ID format."}), 404

    with jobs_lock:
        job = jobs_registry.get(job_id)

    if not job:
        # Check disk manifest
        job_dir = JOBS_DIR / job_id
        manifest_path = job_dir / "scene_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                complete_stats = calculate_missing_statistics(job_dir, manifest_data.get("statistics", {}))
                return jsonify({
                    "success": True,
                    "job_id": job_id,
                    "status": "completed",
                    "stage": "Reconstruction complete (loaded from disk)",
                    "statistics": complete_stats,
                    "artifacts": manifest_data.get("artifacts", {}),
                }), 200
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Job '{job_id}' not found."}), 404

    if job["status"] == "completed":
        job_dir = JOBS_DIR / job_id
        result_data = job.get("result_data")
        if result_data:
            result_data["statistics"] = calculate_missing_statistics(job_dir, result_data.get("statistics", {}))
            return jsonify(result_data), 200

        # Build fallback from job registry
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "completed",
            "stage": job.get("stage", "Reconstruction complete"),
            "statistics": calculate_missing_statistics(job_dir, job.get("metrics", {})),
            "artifacts": {
                "glb": {
                    "available": bool(job.get("files", {}).get("glb")),
                    "download_url": job.get("files", {}).get("glb"),
                },
                "obj": {
                    "available": bool(job.get("files", {}).get("obj")),
                    "download_url": job.get("files", {}).get("obj"),
                },
                "ply": {
                    "available": bool(job.get("files", {}).get("ply")),
                    "download_url": job.get("files", {}).get("ply"),
                },
                "depth_map": {
                    "available": bool(job.get("files", {}).get("depth_vis")),
                    "download_url": job.get("files", {}).get("depth_vis"),
                },
            },
        }), 200

    return jsonify({
        "success": job["status"] != "failed",
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "progress": job.get("progress", 0),
        "error": job.get("error"),
    }), 200


@app.route("/api/jobs/<job_id>/files/<filename>", methods=["GET"])
def download_job_file(job_id: str, filename: str):
    """Safely serves generated 3D models and diagnostics without path traversal vulnerabilities."""
    if not re.match(r"^[a-f0-9\-]{36}$", job_id):
        abort(404)

    if filename not in ALLOWED_DOWNLOAD_FILES:
        abort(404)

    job_dir = (JOBS_DIR / job_id).resolve()
    target_file = (job_dir / filename).resolve()

    # Sandboxing: Ensure target path is strictly inside the designated job directory
    if not str(target_file).startswith(str(job_dir)):
        abort(403)

    if not target_file.exists() or not target_file.is_file():
        abort(404)

    as_attachment = request.args.get("download", "0") in {"1", "true", "yes"}

    mimetype = None
    if filename.endswith(".glb"):
        mimetype = "model/gltf-binary"
    elif filename.endswith(".obj"):
        mimetype = "model/obj"
    elif filename.endswith(".ply"):
        mimetype = "application/x-ply"
    elif filename.endswith(".png"):
        mimetype = "image/png"
    elif filename.endswith(".json"):
        mimetype = "application/json"

    return send_from_directory(
        directory=str(job_dir),
        path=filename,
        as_attachment=as_attachment,
        mimetype=mimetype,
        download_name=filename,
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[INFO] Starting ARACHNE 3D Reconstruction Service...")
    print(f"[INFO] Server URL: http://{host}:{port}")
    print(f"[INFO] CORS Allowed Origins: {', '.join(sorted(ALLOWED_ORIGINS)) if not ALLOW_ALL_ORIGINS else '*'}\n")
    app.run(host=host, port=port, debug=False)
