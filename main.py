#!/usr/bin/env python3
"""CLI entry point for Arachne 3D Reconstruction:
Step 1: Image Loading & Validation
Step 2: Feature Extraction and Matching
Step 3: Camera Pose Estimation & Sparse Triangulation
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from arachne.core.types import DatasetSummary, MatchingSummary, SparseReconstruction, ValidationSeverity
from arachne.features.extractor import FeatureExtractorConfig, SIFTExtractor
from arachne.features.matcher import FeatureMatcher, MatcherConfig
from arachne.features.visualization import export_matching_results
from arachne.geometry.pose import PoseEstimationConfig
from arachne.geometry.reconstruction import (
    ReconstructionConfig,
    SparseReconstructor,
    export_reconstruction_outputs,
)
from arachne.geometry.triangulation import TriangulationConfig
from arachne.validation.validator import ImageValidator, ValidationConfig


def format_validation_dict(summary: DatasetSummary) -> Dict[str, Any]:
    """Serializes dataset validation summary to a dictionary."""
    return {
        "dataset_path": str(summary.dataset_path),
        "total_images_found": summary.total_images_found,
        "valid_images_count": summary.valid_images_count,
        "warning_images_count": summary.warning_images_count,
        "rejected_images_count": summary.rejected_images_count,
        "is_reconstruction_ready": summary.is_reconstruction_ready,
        "dataset_issues": [
            {"severity": issue.severity.value, "message": issue.message}
            for issue in summary.dataset_issues
        ],
        "images": [
            {
                "filename": img.filename,
                "path": str(img.path),
                "is_valid": img.is_valid,
                "width": img.width,
                "height": img.height,
                "channels": img.channels,
                "blur_score": round(img.blur_score, 2),
                "mean_brightness": round(img.mean_brightness, 2),
                "focal_length_pixels": round(img.exif.focal_length_pixels, 1)
                if img.exif and img.exif.focal_length_pixels
                else None,
                "camera_model": f"{img.exif.make or ''} {img.exif.model or ''}".strip()
                if img.exif
                else None,
                "issues": [
                    {"severity": iss.severity.value, "message": iss.message}
                    for iss in img.issues
                ],
            }
            for img in summary.images
        ],
    }


def print_step1_summary(summary: DatasetSummary, verbose: bool = False) -> None:
    """Prints a structured summary of image validation results to stdout."""
    print("=" * 80)
    print("  ARACHNE 3D RECONSTRUCTION - STEP 1: IMAGE VALIDATION")
    print("=" * 80)
    print(f"Directory:         {summary.dataset_path}")
    print(f"Total Discovered:  {summary.total_images_found}")
    print(f"Valid Images:       {summary.valid_images_count}")
    print(f"Images w/ Warnings: {summary.warning_images_count}")
    print(f"Rejected Images:    {summary.rejected_images_count}")
    print("-" * 80)

    if summary.dataset_issues:
        print("Dataset-Level Findings:")
        for issue in summary.dataset_issues:
            print(f"  [{issue.severity.value}] {issue.message}")
        print("-" * 80)

    print(f"{'Filename':<28} | {'Resolution':<12} | {'Blur Score':<10} | {'Status':<10}")
    print("-" * 80)

    for img in summary.images:
        status = "REJECTED" if not img.is_valid else ("WARNING" if img.has_warnings else "VALID")
        res_str = f"{img.width}x{img.height}" if img.width > 0 else "N/A"
        blur_str = f"{img.blur_score:.1f}" if img.blur_score > 0 else "N/A"
        print(f"{img.filename[:27]:<28} | {res_str:<12} | {blur_str:<10} | {status:<10}")

        if verbose or not img.is_valid or img.has_warnings:
            for iss in img.issues:
                print(f"    └─ [{iss.severity.value}] {iss.message}")

    print("=" * 80)
    if summary.is_reconstruction_ready:
        print("[SUCCESS] Validation passed: Ready for Step 2 (Feature Extraction & Matching).")
    else:
        print("[FAILED] Dataset does not meet minimum requirements for 3D reconstruction.")
    print("=" * 80)


def print_step2_summary(summary: MatchingSummary) -> None:
    """Prints a structured summary of feature extraction and matching to stdout."""
    print("\n" + "=" * 80)
    print("  ARACHNE 3D RECONSTRUCTION - STEP 2: FEATURE EXTRACTION & MATCHING")
    print("=" * 80)
    print(f"Images Processed:  {summary.total_images_processed}")
    print(f"Pairs Tested:      {summary.total_pairs_tested}")
    print(f"Overlapping Pairs: {summary.overlapping_pairs_count}")
    if summary.output_dir:
        print(f"Output Directory:  {summary.output_dir}")
    print("-" * 80)

    # Feature counts per image
    print(f"{'Filename':<35} | {'SIFT Features':<15} | {'Status':<12}")
    print("-" * 80)
    for name, feat in summary.features.items():
        if feat.error_message:
            status = "FAILED"
        elif feat.has_insufficient_features:
            status = "LOW_FEATURES"
        else:
            status = "OK"
        print(f"{name[:34]:<35} | {feat.num_features:<15} | {status:<12}")
        if feat.error_message:
            print(f"    └─ [ERROR] {feat.error_message}")

    print("-" * 80)
    print("Pairwise Overlap Summary (Top Pairs by Match Count):")
    print(f"{'Image Pair':<45} | {'Good Matches':<14} | {'Overlap':<10}")
    print("-" * 80)

    sorted_pairs = sorted(
        summary.pair_matches.values(),
        key=lambda pm: pm.num_matches,
        reverse=True,
    )

    display_limit = min(10, len(sorted_pairs))
    for pm in sorted_pairs[:display_limit]:
        pair_label = f"{pm.image_path_1.name[:20]} <-> {pm.image_path_2.name[:20]}"
        overlap_str = "YES" if pm.is_overlapping else "NO"
        print(f"{pair_label:<45} | {pm.num_matches:<14} | {overlap_str:<10}")

    if len(sorted_pairs) > display_limit:
        print(f"... and {len(sorted_pairs) - display_limit} additional pairs.")

    print("=" * 80)


def print_step3_summary(reconstruction: SparseReconstruction, sparse_dir: Path) -> None:
    """Prints a structured summary of camera pose estimation and sparse triangulation."""
    print("\n" + "=" * 80)
    print("  ARACHNE 3D RECONSTRUCTION - STEP 3: CAMERA POSES & SPARSE CLOUD")
    print("=" * 80)

    registered_cams = [c for c in reconstruction.cameras.values() if c.is_registered]
    print(f"Cameras Registered:     {len(registered_cams)} / {len(reconstruction.cameras)}")
    print(f"Triangulated 3D Points: {reconstruction.total_points}")
    print(f"Mean Reprojection Error: {reconstruction.mean_reprojection_error:.3f} px")
    print(f"Sparse Cloud (PLY):     {sparse_dir / 'points3D.ply'}")
    print(f"Camera Poses (JSON):    {sparse_dir / 'cameras.json'}")
    print("-" * 80)

    print(f"{'Camera':<25} | {'Optical Center (X, Y, Z)':<35} | {'Status':<12}")
    print("-" * 80)
    for cam in reconstruction.cameras.values():
        if cam.is_registered:
            c = cam.camera_center
            c_str = f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]"
            status = "REGISTERED"
        else:
            c_str = "N/A"
            status = "FAILED"
        print(f"{cam.image_name[:24]:<25} | {c_str:<35} | {status:<12}")

    # Compute and display geometry diagnostics
    if reconstruction.total_points > 0:
        from arachne.geometry.diagnostics import compute_diagnostics
        diag = compute_diagnostics(reconstruction)
        print("-" * 80)
        print("GEOMETRIC DIAGNOSTICS & DISTRIBUTIONS:")
        print(f"  Point Cloud Extents (X, Y, Z):  [{diag.point_cloud_bbox.span_x:.2f}, {diag.point_cloud_bbox.span_y:.2f}, {diag.point_cloud_bbox.span_z:.2f}]")
        print(f"  Robust Extents (p5 - p95):       [{diag.robust_bbox_p5_p95.span_x:.2f}, {diag.robust_bbox_p5_p95.span_y:.2f}, {diag.robust_bbox_p5_p95.span_z:.2f}]")
        print(f"  Valid Depth Distribution:        Median: {diag.depth_distribution.median:.2f} | Range [p10-p90]: [{diag.depth_distribution.p10:.2f}, {diag.depth_distribution.p90:.2f}]")
        print(f"  Triangulation Angles:            Median: {diag.angle_distribution.median:.2f}° | Mean: {diag.angle_distribution.mean:.2f}° | Max: {diag.angle_distribution.max:.2f}°")
        print(f"  Reprojection Error Distribution: Median: {diag.error_distribution.median:.3f} px | p75: {diag.error_distribution.p75:.3f} px | p90: {diag.error_distribution.p90:.3f} px")
        print(f"  Camera Baseline Span:            Max: {diag.baseline_metrics.max:.2f} | Mean: {diag.baseline_metrics.mean:.2f}")
        print(f"  Interactive WebGL Viewer:        {sparse_dir / 'diagnostics_viewer.html'}")
        print(f"  Camera Frustums Wireframe (PLY): {sparse_dir / 'cameras_frustums.ply'}")

    print("=" * 80)
    if reconstruction.total_points > 0:
        ply_file_str = str((sparse_dir / "points3D.ply").resolve()).replace("\\", "/")
        viewer_file_str = str((sparse_dir / "diagnostics_viewer.html").resolve()).replace("\\", "/")
        viewer_url = f"file://{viewer_file_str}"

        print("[SUCCESS] Sparse 3D reconstruction complete.")
        print(f"[TIP] Visualize in Open3D: o3d.io.read_point_cloud('{ply_file_str}')")
        print(f"[TIP] Open interactive 3D viewer: {viewer_url}")
    else:
        print("[WARNING] No 3D points were successfully triangulated. Check image overlap and parallax.")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Arachne 3D Reconstruction: Validation, Feature Matching & Sparse Pose Estimation."
    )
    # Multi-Image Photogrammetry Arguments
    parser.add_argument(
        "--input", "-i",
        default=None,
        type=Path,
        help="Path to directory containing input images (or a single image file).",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively scan subdirectories for images.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=3,
        help="Minimum required valid images (default: 3).",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=60.0,
        help="Laplacian variance threshold for blur detection (default: 60.0).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to output validation report JSON.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose details for all images.",
    )

    # Step 2 Arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./output"),
        help="Directory to save match visualizations and reconstruction outputs (default: ./output).",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Only run Step 1 (image validation) and skip feature extraction/matching.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4096,
        help="Maximum SIFT features to extract per image (0 = all, default: 4096).",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold (default: 0.75).",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=15,
        help="Minimum good matches to classify pair as overlapping (default: 15).",
    )
    parser.add_argument(
        "--max-viz",
        type=int,
        default=20,
        help="Maximum match visualizations to render and save (default: 20).",
    )

    # Step 3 Arguments
    parser.add_argument(
        "--skip-pose",
        action="store_true",
        help="Skip camera pose estimation and sparse point cloud triangulation.",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=1.5,
        help="RANSAC reprojection error threshold in pixels for Essential Matrix (default: 1.5).",
    )
    parser.add_argument(
        "--reproj-thresh",
        type=float,
        default=4.0,
        help="Maximum allowable reprojection error in pixels for triangulated 3D points (default: 4.0).",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=15,
        help="Minimum RANSAC inliers to verify relative pose (default: 15).",
    )
    parser.add_argument(
        "--min-triangulation-angle",
        type=float,
        default=1.0,
        help="Minimum allowable triangulation parallax angle in degrees (default: 1.0).",
    )
    parser.add_argument(
        "--min-initial-angle",
        type=float,
        default=1.5,
        help="Minimum median triangulation angle for initial pair selection in degrees (default: 1.5).",
    )

    # Single-Image AI Reconstruction Arguments
    parser.add_argument(
        "--single-image",
        type=Path,
        default=None,
        help="Path to a single image for AI-based 3D scene reconstruction.",
    )
    parser.add_argument(
        "--from-depth-dir",
        type=Path,
        default=None,
        help="Reconstruct 3D scene from existing depth outputs without rerunning depth inference.",
    )
    parser.add_argument(
        "--depth-model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        help="Hugging Face model ID for metric depth estimation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Inference device: 'cuda' or 'cpu' (default: auto-detect).",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 half-precision on GPU.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Horizontal camera field of view in degrees (default: 60.0).",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=None,
        help="Explicit focal length in pixels (overrides --fov).",
    )
    parser.add_argument(
        "--depth-jump-ratio",
        type=float,
        default=0.08,
        help="Maximum relative depth jump between adjacent pixels to disconnect occlusion edges (default: 0.08).",
    )
    parser.add_argument(
        "--max-edge-length",
        type=float,
        default=0.35,
        help="Maximum allowable 3D triangle edge length in meters (default: 0.35).",
    )
    parser.add_argument(
        "--mesh-stride",
        type=int,
        default=1,
        help="Grid subsampling stride for mesh generation (1 = full resolution, 2 = half resolution).",
    )

    args = parser.parse_args()

    # --- Mode Dispatch: Single-Image AI Reconstruction vs. Multi-Image Photogrammetry ---
    # Case A: Direct 3D reconstruction from existing depth outputs without rerunning inference
    if args.from_depth_dir is not None:
        from arachne.single_view.reconstruct import SingleView3DConfig, reconstruct_scene_from_files

        depth_dir = args.from_depth_dir.resolve()
        depth_npy = depth_dir / "depth_metric.npy"
        rgb_img = depth_dir / "depth_input_rgb.png"

        if not depth_npy.exists():
            # Check if any .npy file exists in directory
            npy_candidates = list(depth_dir.glob("*.npy"))
            if npy_candidates:
                depth_npy = npy_candidates[0]
            else:
                print(f"[ERROR] Could not find depth .npy file in: {depth_dir}")
                return 1

        if not rgb_img.exists():
            # Check for other image files in directory
            img_candidates = [
                f for f in depth_dir.glob("*")
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"} and "vis" not in f.stem.lower()
            ]
            if img_candidates:
                rgb_img = img_candidates[0]
            else:
                print(f"[ERROR] Could not find RGB reference image in: {depth_dir}")
                return 1

        print("=" * 80)
        print("ARACHNE SINGLE-IMAGE 3D PIPELINE: STEP 2 3D SCENE RECONSTRUCTION")
        print("=" * 80)
        print(f"[INFO] Using Existing Depth: {depth_npy.resolve()}")
        print(f"[INFO] Using RGB Reference:  {rgb_img.resolve()}")

        recon_cfg = SingleView3DConfig(
            fov_deg=args.fov,
            focal_length_px=args.focal_length_px,
            depth_discontinuity_ratio=args.depth_jump_ratio,
            max_edge_length_m=args.max_edge_length,
            mesh_stride=args.mesh_stride,
        )

        scene = reconstruct_scene_from_files(
            depth_npy_path=depth_npy,
            rgb_image_path=rgb_img,
            output_dir=depth_dir,
            config=recon_cfg,
            prefix="scene",
            focal_length_px=args.focal_length_px,
            fov_deg=args.fov,
        )

        print("\n--- 3D Scene Reconstruction Summary ---")
        print(f"Point Cloud Points:  {scene.num_points:,}")
        print(f"Mesh Vertices:       {scene.num_vertices:,}")
        print(f"Mesh Triangles:      {scene.num_faces:,}")
        print(f"Scene Bounding Box:  Extents [X: {scene.bounding_box['extents'][0]:.2f}m, Y: {scene.bounding_box['extents'][1]:.2f}m, Z: {scene.bounding_box['extents'][2]:.2f}m]")
        print(f"Exported PLY Cloud:  {scene.output_files.get('ply')}")
        print(f"Exported GLB Mesh:   {scene.output_files.get('glb')}")
        print(f"Exported OBJ Mesh:   {scene.output_files.get('obj')}")
        print(f"Scene Manifest:      {scene.output_files.get('manifest')}")
        print("[NOTICE] 2.5D Visibility: Surfaces behind foreground occlusions are unobserved in a single view.")
        print("=" * 80)
        return 0

    # Case B: Single-image input (runs depth estimation + full 3D reconstruction)
    single_image_target = None
    if args.single_image is not None:
        single_image_target = args.single_image
    elif args.input is not None:
        if args.input.is_file():
            single_image_target = args.input
        elif args.input.is_dir():
            from arachne.io.loader import discover_image_files
            discovered_candidates = discover_image_files(args.input)
            if len(discovered_candidates) == 1:
                single_image_target = discovered_candidates[0]

    if single_image_target is not None:
        from arachne.io.loader import SUPPORTED_EXTENSIONS, discover_image_files
        from arachne.single_view.depth import DepthEstimationConfig, MetricDepthEstimator, save_depth_outputs
        from arachne.single_view.reconstruct import SingleView3DConfig, reconstruct_3d_scene, export_3d_scene

        # If a directory was provided (e.g. --single-image images), discover images within it
        if single_image_target.is_dir():
            found_images = discover_image_files(single_image_target)
            if not found_images:
                print(f"[ERROR] No supported images found in directory: {single_image_target.resolve()}")
                print(f"[INFO] Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
                return 1
            if len(found_images) == 1:
                single_image_target = found_images[0]
                print(f"[INFO] Discovered 1 image in '{single_image_target.parent.name}/': {single_image_target.name} ({single_image_target.suffix.upper()})")
            else:
                print(f"[INFO] Discovered {len(found_images)} images in '{single_image_target.name}/':")
                for i, img_path in enumerate(found_images, 1):
                    print(f"  [{i}] {img_path.name} ({img_path.suffix.upper()})")
                single_image_target = found_images[0]
                print(f"[INFO] Selected: {single_image_target.name} (pass --single-image <path> to select a different image)")
        elif not single_image_target.exists():
            print(f"[ERROR] Image file does not exist: {single_image_target.resolve()}")
            return 1
        else:
            if single_image_target.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"[ERROR] Unsupported image format: '{single_image_target.suffix}' for file {single_image_target.name}")
                print(f"[INFO] Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
                return 1

        print("=" * 80)
        print("ARACHNE SINGLE-IMAGE 3D PIPELINE: STEP 1 METRIC DEPTH ESTIMATION")
        print("=" * 80)
        depth_cfg = DepthEstimationConfig(
            model_id=args.depth_model_id,
            device=args.device,
            use_fp16=not args.no_fp16,
        )
        estimator = MetricDepthEstimator(config=depth_cfg)
        print(f"[INFO] Source Image:   {single_image_target.resolve()}")
        print(f"[INFO] Image Format:   {single_image_target.suffix.upper()} (Supported)")
        print(f"[INFO] Model:          {depth_cfg.model_id}")
        print(f"[INFO] Target Device:  {estimator.device} (FP16: {depth_cfg.use_fp16})")

        result = estimator.estimate_depth(single_image_target)
        single_out = args.output_dir if args.output_dir != Path("./output") else (args.output_dir / "single_view")
        saved = save_depth_outputs(result, output_dir=single_out)

        print("\n--- Metric Depth Estimation Results ---")
        print(f"Resolution:        {result.width} x {result.height}")
        print(f"Device Used:       {result.device_used}")
        print(f"Depth Statistics:  Min: {result.min_depth:.2f} m | Max: {result.max_depth:.2f} m | Median: {result.median_depth:.2f} m")
        print(f"Raw Depth (.npy):  {saved['npy']}")
        print(f"Visual Map (.png): {saved['vis']}")
        print(f"Summary (.json):   {saved['summary']}")

        # Immediately proceed to Step 2: 3D Scene Reconstruction
        print("\n" + "=" * 80)
        print("ARACHNE SINGLE-IMAGE 3D PIPELINE: STEP 2 3D SCENE RECONSTRUCTION")
        print("=" * 80)

        recon_cfg = SingleView3DConfig(
            fov_deg=args.fov,
            focal_length_px=args.focal_length_px,
            depth_discontinuity_ratio=args.depth_jump_ratio,
            max_edge_length_m=args.max_edge_length,
            mesh_stride=args.mesh_stride,
        )

        rgb_image_input = result.color_image if result.color_image is not None else single_image_target
        if isinstance(rgb_image_input, (str, Path)):
            from arachne.io.loader import load_image
            rgb_image_input = load_image(rgb_image_input, color_mode="RGB")

        scene = reconstruct_3d_scene(
            depth_map=result.depth_map,
            rgb_image=rgb_image_input,
            config=recon_cfg,
            focal_length_px=args.focal_length_px,
            fov_deg=args.fov,
        )
        scene_files = export_3d_scene(scene, output_dir=single_out, prefix="scene")

        print("--- 3D Scene Reconstruction Summary ---")
        print(f"Point Cloud Points:  {scene.num_points:,}")
        print(f"Mesh Vertices:       {scene.num_vertices:,}")
        print(f"Mesh Triangles:      {scene.num_faces:,}")
        print(f"Scene Bounding Box:  Extents [X: {scene.bounding_box['extents'][0]:.2f}m, Y: {scene.bounding_box['extents'][1]:.2f}m, Z: {scene.bounding_box['extents'][2]:.2f}m]")
        print(f"Exported PLY Cloud:  {scene_files.get('ply')}")
        print(f"Exported GLB Mesh:   {scene_files.get('glb')}")
        print(f"Exported OBJ Mesh:   {scene_files.get('obj')}")
        print(f"Scene Manifest:      {scene_files.get('manifest')}")
        print("[NOTICE] 2.5D Visibility: Surfaces behind foreground occlusions are unobserved in a single view.")
        print("=" * 80)
        return 0

    if args.input is None:
        print("[ERROR] Please provide --input <directory_of_images> or --single-image <image_file>.")
        return 1

    # --- Multi-Image Step 1: Image Loading & Validation ---
    val_config = ValidationConfig(
        min_dataset_images=args.min_images,
        blur_threshold=args.blur_threshold,
    )
    validator = ImageValidator(config=val_config)
    dataset_summary = validator.validate_dataset(args.input, recursive=args.recursive)
    print_step1_summary(dataset_summary, verbose=args.verbose)

    if args.output_report:
        report_data = format_validation_dict(dataset_summary)
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Validation report written to: {args.output_report.resolve()}")

    if not dataset_summary.is_reconstruction_ready or args.skip_matching:
        return 0 if dataset_summary.is_reconstruction_ready else 1

    # --- Step 2: Feature Extraction and Matching ---
    valid_images = [img for img in dataset_summary.images if img.is_valid]
    if len(valid_images) < 2:
        print("[ERROR] Fewer than 2 valid images available for pairwise feature matching.")
        return 1

    extractor_config = FeatureExtractorConfig(n_features=args.max_features)
    extractor = SIFTExtractor(config=extractor_config)
    features_dict = extractor.extract_dataset(valid_images)

    matcher_config = MatcherConfig(
        ratio_threshold=args.ratio_threshold,
        min_matches=args.min_matches,
    )
    matcher = FeatureMatcher(config=matcher_config)
    matching_summary = matcher.match_all_pairs(features_dict)

    export_matching_results(
        matching_summary,
        output_dir=args.output_dir,
        max_visualizations=args.max_viz,
    )
    print_step2_summary(matching_summary)

    if args.skip_pose:
        return 0 if matching_summary.overlapping_pairs_count > 0 else 1

    # --- Step 3: Camera Pose Estimation & Sparse Triangulation ---
    reconstruction_config = ReconstructionConfig(
        pose_config=PoseEstimationConfig(
            ransac_threshold_px=args.ransac_thresh,
            min_inliers=args.min_inliers,
        ),
        triangulation_config=TriangulationConfig(
            max_reprojection_error_px=args.reproj_thresh,
            min_triangulation_angle_deg=args.min_triangulation_angle,
        ),
        min_initial_angle_deg=args.min_initial_angle,
    )
    reconstructor = SparseReconstructor(config=reconstruction_config)
    reconstruction = reconstructor.reconstruct(
        dataset_summary=dataset_summary,
        matching_summary=matching_summary,
        load_rgb_colors=True,
    )

    sparse_dir = args.output_dir / "sparse"
    export_reconstruction_outputs(reconstruction, output_dir=args.output_dir)
    print_step3_summary(reconstruction, sparse_dir)

    return 0 if reconstruction.total_points > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
