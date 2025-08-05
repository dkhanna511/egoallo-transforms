from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import tyro
import viser
from tqdm import tqdm

from egoallo import fncsmpl
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.network import EgoDenoiseTraj
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from scipy.spatial.transform import Rotation as R


# Import the classes from your modified inference script
from modify_aria_inference import MP4ColmapTrajectoryPaths, load_semidense_points_and_find_ground


def quaternion_inverse(q):
    """Compute inverse of quaternion [qx, qy, qz, qw]"""
    qx, qy, qz, qw = q
    norm_sq = qx*qx + qy*qy + qz*qz + qw*qw
    return np.array([-qx/norm_sq, -qy/norm_sq, -qz/norm_sq, qw/norm_sq])

def transform_inverse(T):
    """Compute inverse of transformation [qx, qy, qz, qw, tx, ty, tz]"""
    q, t = T[:4], T[4:]
    
    # Inverse rotation
    q_inv = quaternion_inverse(q)
    
    # Inverse translation: t_inv = -R_inv * t
    r_inv = R.from_quat(q_inv)
    t_inv = -r_inv.apply(t)
    
    return np.array([q_inv[0], q_inv[1], q_inv[2], q_inv[3], 
                     t_inv[0], t_inv[1], t_inv[2]])

def transform_compose(T1, T2):
    """Compose two transformations T1 @ T2
    Each transform is [qx, qy, qz, qw, tx, ty, tz]
    """
    # Extract rotation and translation
    q1, t1 = T1[:4], T1[4:]
    q2, t2 = T2[:4], T2[4:]
    
    # Convert quaternions to rotation objects
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    
    # Compose rotations
    r_composed = r1 * r2
    q_composed = r_composed.as_quat()
    
    # Compose translations: t_composed = t1 + R1 * t2
    t_composed = t1 + r1.apply(t2)
    
    return np.array([q_composed[0], q_composed[1], q_composed[2], q_composed[3], 
                     t_composed[0], t_composed[1], t_composed[2]])


# Function to convert CPF-to-World to Device-to-World (following egoallo pattern)
def cpf_to_device_world_transform(Ts_world_cpf, T_device_cpf_param):
    """
    Convert CPF-to-World transformation(s) to Device-to-World transformation(s)
    Following the egoallo pattern: T_world_device = T_world_cpf @ T_device_cpf.inverse()
    
    Args:
        Ts_world_cpf: numpy array of shape (N, 7) or (7,) representing World to CPF transforms
        T_device_cpf_param: numpy array [qx, qy, qz, qw, tx, ty, tz] representing CPF to Device transform
    
    Returns:
        Ts_world_device: numpy array same shape as input representing World to Device transforms
    """
    # Handle both single transform and batch of transforms
    if Ts_world_cpf.ndim == 1:
        # Single transform case
        T_device_cpf_inv = transform_inverse(T_device_cpf_param)
        T_world_device = transform_compose(Ts_world_cpf, T_device_cpf_inv)
        return T_world_device
    else:
        # Batch of transforms case
        T_device_cpf_inv = transform_inverse(T_device_cpf_param)
        Ts_world_device = []
        
        for i in range(Ts_world_cpf.shape[0]):
            T_world_cpf = Ts_world_cpf[i]
            T_world_device = transform_compose(T_world_cpf, T_device_cpf_inv)
            Ts_world_device.append(T_world_device)
        
        return np.array(Ts_world_device)


def main(
    search_root_dir: Path,
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
) -> None:
    """Visualization script for outputs from EgoAllo with MP4/COLMAP data.

    Arguments:
        search_root_dir: Root directory where inputs/outputs are stored. All
            NPZ files in this directory will be assumed to be outputs from EgoAllo.
        smplh_npz_path: Path to the SMPLH model NPZ file.
    """
    device = torch.device("cuda")

    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    def get_file_list():
        return ["None"] + sorted(
            str(p.relative_to(search_root_dir))
            for p in search_root_dir.glob("**/egoallo_outputs/*.npz")
        )

    options = get_file_list()
    file_dropdown = server.gui.add_dropdown("File", options=options)

    refresh_file_list = server.gui.add_button("Refresh File List")

    @refresh_file_list.on_click
    def _(_) -> None:
        file_dropdown.options = get_file_list()

    trajectory_folder = server.gui.add_folder("Trajectory")

    current_file = "None"
    loop_cb = lambda: None

    while True:
        loop_cb()
        if current_file != file_dropdown.value:
            current_file = file_dropdown.value

            # Clear the scene.
            server.scene.reset()

            if current_file != "None":
                # Clear the folder by removing then re-adding it.
                trajectory_folder.remove()
                trajectory_folder = server.gui.add_folder("Trajectory")

                with trajectory_folder:
                    npz_path = Path(search_root_dir / current_file).resolve()
                    loop_cb = load_and_visualize(
                        server,
                        npz_path,
                        body_model,
                        device=device,
                    )
                    args = npz_path.parent / (npz_path.stem + "_args.yaml")
                    if args.exists():
                        with server.gui.add_folder("Args"):
                            server.gui.add_markdown(
                                "```\n" + args.read_text() + "\n```"
                            )


def load_and_visualize(
    server: viser.ViserServer,
    npz_path: Path,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
) -> Callable[[], int]:
    """Load and visualize trajectory from NPZ file with MP4/COLMAP data"""
    
    # Load the NPZ outputs
    outputs = np.load(npz_path)
    expected_keys = [
        "Ts_world_cpf",
        "Ts_world_root",
        "body_quats",
        "left_hand_quats",
        "right_hand_quats",
        "betas",
        "frame_nums",
        "timestamps_ns",
    ]
    assert all(key in outputs for key in expected_keys), (
        f"Missing keys in NPZ file. Expected: {expected_keys}, Found: {list(outputs.keys())}"
    )
    (num_samples, timesteps, _, _) = outputs["body_quats"].shape

    # We assume the directory structure is:
    # - some trajectory root
    #     - egoallo_outputs/
    #         - the npz file
    traj_dir = npz_path.resolve().parent.parent
    paths = MP4ColmapTrajectoryPaths.find(traj_dir)

    pose_timestamps_sec = outputs["timestamps_ns"] / 1e9

    Ts_world_cpf = torch.from_numpy(outputs["Ts_world_cpf"])
    print(" Ts_World _CPF : ", Ts_world_cpf)
    
    rotation_matrix_cpf_to_device = np.array([
        [ 0, -1,  0],  # Device_x = -CPF_y (up becomes down)
        [ 1,  0,  0],  # Device_y = CPF_x (left stays left)
        [ 0,  0,  1]   # Device_z = CPF_z (forward stays forward)
    ])
    
    rot_cpf_to_device = R.from_matrix(rotation_matrix_cpf_to_device)
    quat_cpf_to_device = rot_cpf_to_device.as_quat()  # [qx, qy, qz, qw]
    T_device_cpf = np.array([quat_cpf_to_device[0], quat_cpf_to_device[1], 
                         quat_cpf_to_device[2], quat_cpf_to_device[3], 
                         0.0, 0.0, 0.0])
    # Ts_world_device = Ts_world_cpf  
    Ts_world_device = cpf_to_device_world_transform(Ts_world_cpf, T_device_cpf)
    
    # Get temporally corresponded HaMeR detections
    if paths.hamer_outputs is not None:
        print(" am i coming here?")
        try:
            hamer_detections = CorrespondedHamerDetections.load(
                paths.hamer_outputs,
                pose_timestamps_sec,
            )
            print("Loaded HaMeR hand detections")
        except Exception as e:
            print(f"Warning: Could not load HaMeR detections: {e}")
            hamer_detections = None
    else:
        print("No hand detections found.")
        # exit(0)
        hamer_detections = None

    # Get temporally corresponded Aria wrist and palm estimates
    if paths.wrist_and_palm_poses_csv is not None:
        try:
            aria_detections = CorrespondedAriaHandWristPoseDetections.load(
                paths.wrist_and_palm_poses_csv,
                pose_timestamps_sec,
                Ts_world_device=Ts_world_device.numpy(),
            )
            print("Loaded Aria hand detections")
        except Exception as e:
            print(f"Warning: Could not load Aria hand detections: {e}")
            aria_detections = None
    else:
        aria_detections = None

    if paths.splat_path is not None:
        print("Found splat at", paths.splat_path)
    else:
        print("No scene splat found.")

    # Get point cloud + floor
    points_data, floor_z = load_semidense_points_and_find_ground(paths.points_csv_path)

    traj = EgoDenoiseTraj(
        betas=torch.from_numpy(outputs["betas"]).to(device),
        body_rotmats=SO3(
            torch.from_numpy(outputs["body_quats"]),
        )
        .as_matrix()
        .to(device),
        # We weren't saving contacts originally. We added it September 28th.
        contacts=torch.zeros((num_samples, timesteps, 21), device=device)
        if "contacts" not in outputs
        else torch.from_numpy(outputs["contacts"]).to(device),
        hand_rotmats=SO3(
            torch.from_numpy(
                np.concatenate(
                    [
                        outputs["left_hand_quats"],
                        outputs["right_hand_quats"],
                    ],
                    axis=-2,
                )
            ).to(device)
        ).as_matrix(),
    )
    Ts_world_cpf = torch.from_numpy(outputs["Ts_world_cpf"]).to(device)

    def get_ego_video(
        start_index: int,
        end_index: int,
        total_duration: float,
    ) -> bytes:
        """Helper function that returns the egocentric video corresponding to
        some start/end pose index using MP4 frames."""
        
        # Load video metadata to get frame rate and timing info
        try:
            with open(paths.video_metadata_path, 'r') as f:
                video_metadata = json.load(f)
            
            original_fps = video_metadata.get('fps', 30.0)
            total_frames = video_metadata.get('frame_count', 0)
            
            print(f"Video metadata: fps={original_fps}, total_frames={total_frames}")
        except Exception as e:
            print(f"Warning: Could not load video metadata: {e}")
            original_fps = 30.0
            total_frames = len(list(paths.images_dir.glob("*.jpg")))

        # Calculate which image frames correspond to the pose timestamps
        start_ns = int(outputs["timestamps_ns"][start_index])
        end_ns = int(outputs["timestamps_ns"][min(end_index, len(outputs["timestamps_ns"]) - 1)])
        
        # Get the first timestamp from the trajectory
        first_ns = int(outputs["timestamps_ns"][0])
        
        # Calculate frame indices
        duration_from_start_sec = (start_ns - first_ns) / 1e9
        duration_total_sec = (end_ns - start_ns) / 1e9
        
        image_start_index = max(0, int(duration_from_start_sec * original_fps))
        num_frames_needed = max(1, int(duration_total_sec * original_fps))
        image_end_index = min(image_start_index + num_frames_needed, total_frames)
        
        print(f"Loading frames {image_start_index} to {image_end_index}")
        
        frames = []
        # Load frames from the images directory
        for i in tqdm(range(image_start_index, image_end_index)):
            frame_path = paths.images_dir / f"frame_{i:06d}.jpg"
            print(" frame_path is : ", frame_path)
            # exit(0)
            if not frame_path.exists():
                # Try different naming conventions
                frame_path = paths.images_dir / f"{i:06d}.jpg"
                if not frame_path.exists():
                    frame_path = paths.images_dir / f"frame_{i:06d}.png"
                    if not frame_path.exists():
                        print(f"Warning: Frame not found: {frame_path}")
                        continue
            
            try:
                image_array = cv2.imread(str(frame_path))
                if image_array is None:
                    print(f"Warning: Could not load image: {frame_path}")
                    continue
                    
                # Resize and rotate similar to original code
                image_array = cv2.resize(
                    image_array, (1920, 1080), interpolation=cv2.INTER_AREA
                )
                image_array = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
                # Convert BGR to RGB for imageio
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                frames.append(image_array)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                continue

        if not frames:
            print("Warning: No frames loaded, creating dummy frame")
            # Create a dummy black frame
            dummy_frame = np.zeros((800, 800, 3), dtype=np.uint8)
            frames = [dummy_frame]

        fps = len(frames) / max(total_duration, 0.1)  # Avoid division by zero
        output = io.BytesIO()
        
        try:
            iio.imwrite(
                output,
                frames,
                fps=fps,
                extension=".mp4",
                codec="libx264",
                pixelformat="yuv420p",
                quality=None,
                ffmpeg_params=["-crf", "23"],
            )
        except Exception as e:
            print(f"Error creating video: {e}")
            # Return empty bytes if video creation fails
            return b""
            
        return output.getvalue()

    return visualize_traj_and_hand_detections(
        server,
        Ts_world_cpf,
        traj,
        body_model,
        hamer_detections,
        aria_detections,
        points_data,
        paths.splat_path,
        floor_z=floor_z,
        get_ego_video=get_ego_video,
    )


if __name__ == "__main__":
    tyro.cli(main)