"""Script to run HaMeR on MP4/COLMAP data and save outputs to a pickle file."""

import pickle
import shutil
import json
import pandas as pd
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import tyro
from egoallo.hand_detection_structs import (
    SavedHamerOutputs,
    SingleHandHamerOutputWrtCamera,
)


from hamer_helper import HamerHelper
from tqdm.auto import tqdm

# Import your MP4/COLMAP path finder
from modify_aria_inference import MP4ColmapTrajectoryPaths
import sys
import os
from pathlib import Path
repo_root = Path(__file__).resolve().parent
print(" repo root is : ", repo_root)
# exit(0)
sys.path.append(os.path.join(str(repo_root), "hamer"))
def main(traj_root: Path, overwrite: bool = False) -> None:
    """Run HaMeR on MP4/COLMAP trajectory. We'll save outputs to
    `traj_root/hamer_outputs.pkl` and `traj_root/hamer_outputs_render`.

    Arguments:
        traj_root: The root directory of the trajectory containing MP4/COLMAP data.
        overwrite: If True, overwrite any existing HaMeR outputs.
    """

    paths = MP4ColmapTrajectoryPaths.find(traj_root)
    
    # Check required files exist
    if not paths.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {paths.images_dir}")
    if not paths.video_metadata_path.exists():
        raise FileNotFoundError(f"Video metadata not found: {paths.video_metadata_path}")
    if not paths.slam_csv_path.exists():
        raise FileNotFoundError(f"SLAM trajectory not found: {paths.slam_csv_path}")

    pickle_out = traj_root / "hamer_outputs.pkl"
    hamer_render_out = traj_root / "hamer_outputs_render"
    
    run_hamer_and_save_mp4(paths, pickle_out, hamer_render_out, overwrite)


def run_hamer_and_save_mp4(
    paths: MP4ColmapTrajectoryPaths, 
    pickle_out: Path, 
    hamer_render_out: Path, 
    overwrite: bool
) -> None:
    """Run HaMeR on MP4 frames and save results"""
    
    # if not overwrite:
    #     # assert not pickle_out.exists(), f"Output file already exists: {pickle_out}"
    #     # assert not hamer_render_out.exists(), f"Render directory already exists: {hamer_render_out}"
    #     # NEW
    #     if hamer_render_out.exists():
    #         import shutil
    #         shutil.rmtree(hamer_render_out)
        
    # else:
    #     pickle_out.unlink(missing_ok=True)
    #     shutil.rmtree( hamer_render_out, ignore_errors=True)

    hamer_render_out.mkdir(exist_ok=True)
    hamer_helper = HamerHelper()

    # Load video metadata
    with open(paths.video_metadata_path, 'r') as f:
        video_metadata = json.load(f)
    
    fps = video_metadata.get('fps', 30.0)
    frame_count = video_metadata.get('frame_count', 0)
    
    # Load SLAM trajectory for timestamps
    print("Printing current paths:\n {} \n {} \n {} ".format(paths.video_metadata_path, paths.slam_csv_path, paths.images_dir))
    slam_df = pd.read_csv(paths.slam_csv_path)
    slam_timestamps_us = slam_df['tracking_timestamp_us'].values
    slam_timestamps_ns = slam_timestamps_us * 1000  # Convert to nanoseconds
    
    # Find all image files
    image_files = sorted(list(paths.images_dir.glob("*.jpg")) + 
                        list(paths.images_dir.glob("*.jpeg")) + 
                        list(paths.images_dir.glob("*.png")))
    
    print(f"Found {len(image_files)} image files")
    print(f"SLAM trajectory has {len(slam_df)} poses")
    
    # For MP4/COLMAP data, we need to estimate camera parameters
    # These are approximate values - you might need to adjust based on your camera
    focal_length = 450  # Approximate focal length in pixels
    image_width = video_metadata.get('width', 1408)
    image_height = video_metadata.get('height', 1408)
    
    # Since we don't have Aria device calibration, we'll use identity transforms
    # In a real scenario, you'd want to calibrate these properly
    T_device_cam = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # [qx, qy, qz, qw, tx, ty, tz]
    T_cpf_cam = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])     # Assume CPF == device for MP4

    # Dict from timestamp in nanoseconds to hand detections
    detections_left_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None] = {}
    detections_right_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None] = {}

    pbar = tqdm(enumerate(image_files), total=len(image_files))
    
    for i, image_path in pbar:
        # pbar.set_description(f"Processing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        # Convert BGR to RGB for HaMeR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate timestamp for this frame
        # Map frame index to SLAM timestamp
        if i < len(slam_timestamps_ns):
            timestamp_ns = int(slam_timestamps_ns[i])
        else:
            # If we have more images than SLAM poses, estimate timestamp
            timestamp_ns = int(slam_timestamps_ns[0] + (i / fps) * 1e9)
        
        # Run HaMeR hand detection
        try:
            hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
                image_rgb,
                focal_length=focal_length,
            )
        except Exception as e:
            print(f"Error running HaMeR on frame {i}: {e}")
            hamer_out_left, hamer_out_right = None, None

        # Store left hand detection
        if hamer_out_left is None:
            detections_left_wrt_cam[timestamp_ns] = None
        else:
            detections_left_wrt_cam[timestamp_ns] = {
                "verts": hamer_out_left["verts"],
                "keypoints_3d": hamer_out_left["keypoints_3d"],
                "mano_hand_pose": hamer_out_left["mano_hand_pose"],
                "mano_hand_betas": hamer_out_left["mano_hand_betas"],
                "mano_hand_global_orient": hamer_out_left["mano_hand_global_orient"],
            }

        # Store right hand detection
        if hamer_out_right is None:
            detections_right_wrt_cam[timestamp_ns] = None
        else:
            detections_right_wrt_cam[timestamp_ns] = {
                "verts": hamer_out_right["verts"],
                "keypoints_3d": hamer_out_right["keypoints_3d"],
                "mano_hand_pose": hamer_out_right["mano_hand_pose"],
                "mano_hand_betas": hamer_out_right["mano_hand_betas"],
                "mano_hand_global_orient": hamer_out_right["mano_hand_global_orient"],
            }

        # Create visualization
        composited = image_rgb.copy()
        composited = hamer_helper.composite_detections(
            composited,
            hamer_out_left,
            border_color=(255, 100, 100),
            focal_length=focal_length,
        )
        composited = hamer_helper.composite_detections(
            composited,
            hamer_out_right,
            border_color=(100, 100, 255),
            focal_length=focal_length,
        )
        
        # Add text annotations
        font_scale = 10.0 / 2880.0 * image.shape[0]
        composited = put_text(
            composited,
            "L detections: "
            + (
                "0" if hamer_out_left is None else str(hamer_out_left["verts"].shape[0])
            ),
            0,
            color=(255, 100, 100),
            font_scale=font_scale,
        )
        composited = put_text(
            composited,
            "R detections: "
            + (
                "0"
                if hamer_out_right is None
                else str(hamer_out_right["verts"].shape[0])
            ),
            1,
            color=(100, 100, 255),
            font_scale=font_scale,
        )
        composited = put_text(
            composited,
            f"frame={i:06d}, ns={timestamp_ns}",
            2,
            color=(255, 255, 255),
            font_scale=font_scale,
        )

        # Save visualization
        output_image = np.concatenate(
            [
                # Darken original image for contrast
                (image_rgb * 0.6).astype(np.uint8),
                composited,
            ],
            axis=1,
        )
        
        output_path = hamer_render_out / f"{i:06d}.jpeg"
        iio.imwrite(str(output_path), output_image, quality=90)

    # Create output structure compatible with EgoAllo
    outputs = SavedHamerOutputs(
        mano_faces_right=hamer_helper.get_mano_faces("right"),
        mano_faces_left=hamer_helper.get_mano_faces("left"),
        detections_right_wrt_cam=detections_right_wrt_cam,
        detections_left_wrt_cam=detections_left_wrt_cam,
        T_device_cam=T_device_cam,
        T_cpf_cam=T_cpf_cam,
    )
    
    # Save to pickle file
    with open(pickle_out, "wb") as f:
        pickle.dump(outputs, f)
    
    print(f"\nHaMeR processing complete!")
    print(f"Results saved to: {pickle_out}")
    print(f"Visualizations saved to: {hamer_render_out}")
    
    # Print detection statistics
    left_detections = sum(1 for d in detections_left_wrt_cam.values() if d is not None)
    right_detections = sum(1 for d in detections_right_wrt_cam.values() if d is not None)
    total_frames = len(detections_left_wrt_cam)
    
    print(f"\nDetection Statistics:")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Left hand detections: {left_detections} ({100*left_detections/total_frames:.1f}%)")
    print(f"  Right hand detections: {right_detections} ({100*right_detections/total_frames:.1f}%)")
    
    if left_detections == 0 and right_detections == 0:
        print("\n⚠️  WARNING: No hand detections found!")
        print("   This will significantly reduce pose estimation quality.")
        print("   Consider:")
        print("   - Checking if hands are visible in the video")
        print("   - Adjusting lighting/contrast")
        print("   - Manually reviewing some frames in hamer_outputs_render/")


def put_text(
    image: np.ndarray,
    text: str,
    line_number: int,
    color: tuple[int, int, int],
    font_scale: float,
) -> np.ndarray:
    """Put some text on the top-left corner of an image."""
    image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    
    # Black outline for better visibility
    cv2.putText(
        image,
        text=text,
        org=(2, 1 + int(15 * font_scale * (line_number + 1))),
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=max(int(font_scale), 1) + 1,
        lineType=cv2.LINE_AA,
    )
    
    # Colored text
    cv2.putText(
        image,
        text=text,
        org=(2, 1 + int(15 * font_scale * (line_number + 1))),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=max(int(font_scale), 1),
        lineType=cv2.LINE_AA,
    )
    return image


def estimate_camera_parameters_from_colmap(traj_root: Path) -> dict:
    """Estimate camera parameters from COLMAP data if available"""
    
    # Look for COLMAP camera file
    cameras_file = None
    for possible_path in [
        traj_root / "cameras.txt",
        traj_root / "colmap" / "cameras.txt",
        traj_root / "sparse_text" / "cameras.txt",
    ]:
        if possible_path.exists():
            cameras_file = possible_path
            break
    print(" camera file is : ", cameras_file)
    exit(0)
    if cameras_file is None:
        print("No COLMAP cameras.txt found, using default parameters")
        return {
            'focal_length': 450,
            'cx': 704,  # image_width / 2
            'cy': 704,  # image_height / 2
            'width': 1408,
            'height': 1408
        }
    
    try:
        # Parse COLMAP cameras.txt
        with open(cameras_file, 'r') as f:
            lines = f.readlines()
        
        # Skip comments
        lines = [line for line in lines if not line.startswith('#')]
        
        if len(lines) > 0:
            # Parse first camera (format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...)
            parts = lines[0].strip().split()
            if len(parts) >= 4:
                width = int(parts[2])
                height = int(parts[3])
                
                # Extract focal length (depends on camera model)
                if len(parts) >= 5:
                    focal_length = float(parts[4])
                else:
                    focal_length = min(width, height) * 0.8  # Rough estimate
                
                return {
                    'focal_length': focal_length,
                    'cx': width / 2,
                    'cy': height / 2,
                    'width': width,
                    'height': height
                }
    except Exception as e:
        print(f"Error parsing COLMAP cameras: {e}")
    
    # Fallback to defaults
    return {
        'focal_length': 450,
        'cx': 704,
        'cy': 704,
        'width': 1408,
        'height': 1408
    }


if __name__ == "__main__":
    tyro.cli(main)