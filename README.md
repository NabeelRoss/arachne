# ARACHNE: AI-Powered Single-Image 3D Scene Reconstruction

ARACHNE reconstructs complete, navigable 3D environments (rooms, buildings, streets, landscapes) from a single 2D photograph. It features an interactive local web interface built with **Flask** and **Three.js**, with GPU-accelerated metric depth estimation and 3D surface mesh generation.

---

## Architecture Overview

```
[ Web Browser ]
      │  (Drag & Drop Upload: JPG, PNG, WEBP)
      ▼
[ Flask Server (app.py) ]
      │  (Validates image, assigns unique UUID job_id, runs worker thread)
      ├─────────────────────────────────────────────┐
      ▼                                             ▼
[ Metric Depth Estimation ]                 [ 3D Scene Reconstruction ]
• Depth Anything V2 Metric Base             • Pinhole camera ray unprojection
• FP16 CUDA (~1.5 GB VRAM on RTX 3050)      • Vectorized occlusion jump disconnection
• Indoor (0–20m) & Outdoor (0–80m) modes    • Standards-compliant binary GLB export
      │                                             │
      └─────────────────────┬───────────────────────┘
                            ▼
          [ web_outputs/jobs/<job_id>/ ]
          • scene_mesh.glb (Three.js browser preview & download)
          • scene_mesh.obj (Wavefront OBJ for Blender & CAD)
          • scene_pointcloud.ply (Metric RGB point cloud)
          • depth_vis.png (Colormapped depth visualization)
          • scene_manifest.json (Bounding box & camera intrinsics)
```

### Scientific / Geometric Limitation
A single 2D photograph inherently captures line-of-sight visible surfaces (2.5D metric surface geometry). Surfaces completely hidden behind foreground occlusions (e.g., backs of furniture, surfaces behind walls) are unobserved and disconnected rather than hallucinated into inaccurate closed solids.

---

## Hardware & Environment Requirements

* **Operating System**: Windows 10 / 11
* **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) or equivalent with CUDA support
* **Conda Environment**: `3d_recon`

### Manual Package Installation

Activate your existing environment and install any missing requirements:

```bash
conda activate 3d_recon

# 1. Install Flask web framework
pip install flask werkzeug

# 2. Ensure compatible transformers & PyTorch (Option A: PyTorch 2.6 CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Or Option B (if keeping PyTorch 2.5.1):
# pip install "transformers==4.48.3"

# 3. Ensure OpenCV and 3D dependencies are up to date
pip install --upgrade opencv-python Pillow trimesh open3d
```

---

## Launching the Web Application

1. Activate your Conda environment:
   ```bash
   conda activate 3d_recon
   ```

2. Start the local Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## Web Application Workflow

1. **Upload Source Image**: Drag and drop a photograph of an entire scene (or click *browse file*). Supports `.jpg`, `.jpeg`, `.png`, and `.webp` up to 25 MB.
2. **Select Scene Calibration**:
   * **Indoor Room**: Optimized for metric depth up to 20 meters (rooms, furniture, walls, interiors).
   * **Outdoor Scene**: Optimized for metric depth up to 80 meters (buildings, streets, landscapes).
3. **Generate 3D Model**: Click **Generate 3D Model**. The application runs the pipeline in a background worker thread, reporting genuine progress stages.
4. **Interactive 3D Inspection**:
   * **Left-Click + Drag**: Orbit / rotate around the scene.
   * **Right-Click + Drag**: Pan through the scene.
   * **Scroll Wheel**: Zoom in and out.
   * **Toolbar**: Reset camera viewpoint, toggle wireframe, toggle ground grid, or enable auto-orbit.
5. **Download Outputs**: Download the exported models directly:
   * **Download GLB**: Binary glTF 2.0 with embedded RGB vertex colors (ready for Three.js, Windows 3D Viewer, and web applications).
   * **Download OBJ**: Wavefront OBJ mesh with vertex colors (ready for Blender and CAD).
   * **Download PLY**: Dense metric 3D point cloud.
   * **Download Depth Map**: Metric colormapped depth image (`.png`).

---

## Command-Line Alternative (CLI)

The CLI pipeline in `main.py` remains fully supported:

```bash
# Reconstruct a single image into output/my_scene/
python main.py --single-image images/my_room.jpg -o output/my_scene

# Automatically discover and reconstruct an image placed in the images/ directory:
python main.py --single-image images -o output/my_scene
```

---

## Running the Test Suite

Run the web application test suite (routes, validation, security, and mocked pipeline flow):

```bash
conda activate 3d_recon
pytest tests/test_web_app.py -v
```

Run the complete ARACHNE test suite:

```bash
pytest tests/ -v
```

