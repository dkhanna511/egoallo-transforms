from __future__ import annotations

import dataclasses
import time
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import viser
import yaml
import cv2

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
    load_denoiser,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections


@dataclasses.dataclass
class MP4ColmapTrajectoryPaths:
    """Paths for MP4/COLMAP converted data"""
    traj_root: Path
    slam_csv_path: Path
    points_csv_path: Path
    images_dir: Path
    hamer_outputs: Optional[Path] = None
    wrist_and_palm_poses_csv: Optional[Path] = None
    splat_path: Optional[Path] = None

    @classmethod
    def find(cls, traj_root: Path) -> "MP4ColmapTrajectoryPaths":
        """Find all required files in the trajectory directory"""
        traj_root = Path(traj_root)
        
        slam_csv_path = traj_root / "closed_loop_trajectory.csv"
        points_csv_path = traj_root / "semidense_points.csv.gz"
        # video_metadata_path = traj_root / "video_metadata.json"
        images_dir = traj_root / "images"
        
        # Optional files
        hamer_outputs = traj_root / "hamer_outputs.pkl"
        if not hamer_outputs.exists():
            hamer_outputs = None
            
        wrist_and_palm_poses_csv = traj_root / "wrist_and_palm_poses.csv"
        if not wrist_and_palm_poses_csv.exists():
            wrist_and_palm_poses_csv = None
            
        splat_path = traj_root / "scene.splat"
        if not splat_path.exists():
            splat_path = None
        
        return cls(
            traj_root=traj_root,
            slam_csv_path=slam_csv_path,
            points_csv_path=points_csv_path,
            images_dir=images_dir,
            hamer_outputs=hamer_outputs,
            wrist_and_palm_poses_csv=wrist_and_palm_poses_csv,
            splat_path=splat_path
        )


class MP4ColmapInputTransforms:
    """
    Input transforms for MP4/COLMAP data with proper coordinate system handling.
    
    Coordinate Systems:
    - Your camera system: X(forward), Y(left), Z(up)
    - CPF (EgoAllo expects): X(left), Y(up), Z(forward)
    """
    
    def __init__(self, Ts_world_cpf: torch.Tensor, Ts_world_device: torch.Tensor, 
                 pose_timesteps: torch.Tensor, device: torch.device):
        self.Ts_world_cpf = Ts_world_cpf.to(device)
        self.Ts_world_device = Ts_world_device.to(device) 
        self.pose_timesteps = pose_timesteps.to(device)
    
    @classmethod
    def load(cls, traj_paths: MP4ColmapTrajectoryPaths, fps: int = 30, 
             device: torch.device = torch.device("cpu")) -> "MP4ColmapInputTransforms":
        """
        Load transforms from CSV data and convert to CPF coordinate system.
        
        CSV contains world-to-device transforms in your camera coordinate system.
        We need to:
        1. Invert world-to-device to get device-to-world (camera poses in world)
        2. Transform from your coordinate system to CPF coordinate system
        """
        
        # Load SLAM poses
        df = pd.read_csv(traj_paths.slam_csv_path)
        
        # Validate required columns
        required_cols = ['tracking_timestamp_us', 'tx_world_device', 'ty_world_device', 'tz_world_device',
                        'qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Convert timestamps
        timestamps_us = df['tracking_timestamp_us'].values
        timestamps_sec = timestamps_us / 1e6
        
        # Extract world-to-device transforms in your camera coordinate system
        translations_device_to_world = np.stack([
            df['tx_world_device'].values,  # X = forward
            df['ty_world_device'].values,  # Y = left
            df['tz_world_device'].values   # Z = up
        ], axis=1)
        
        quaternions_device_to_world = np.stack([
            df['qx_world_device'].values,
            df['qy_world_device'].values, 
            df['qz_world_device'].values,
            df['qw_world_device'].values
        ], axis=1)
        
        print(f"=== PROCESSING WORLD-TO-DEVICE TRANSFORMS ===")
        print(f"Input coordinate system: X(forward), Y(left), Z(up)")
        print(f"Loaded {len(translations_device_to_world)} world-to-device transforms")
        print(f"World-to-device translation ranges:")
        print(f"  X(forward): [{translations_device_to_world[:,0].min():.3f}, {translations_device_to_world[:,0].max():.3f}]")
        print(f"  Y(left): [{translations_device_to_world[:,1].min():.3f}, {translations_device_to_world[:,1].max():.3f}]")
        print(f"  Z(up): [{translations_device_to_world[:,2].min():.3f}, {translations_device_to_world[:,2].max():.3f}]")
        
        # Step 1: Invert world-to-device transforms to get device-to-world (camera poses in world)
        # translations_device_to_world = []
        # rotations_device_to_world = []
        T_device_to_world = []
        R_device_to_world = []
        
        for i in range(len(translations_device_to_world)):
            # Get world-to-device transform
            qx, qy, qz, qw = quaternions_device_to_world[i]
            r_device_to_world = cls._quaternion_to_rotation_matrix(qx, qy, qz, qw)
            t_device_to_world = translations_device_to_world[i]
            
            # Invert to get device-to-world transform
            # For SE(3): [R t; 0 1]^(-1) = [R^T -R^T*t; 0 1]
            # R_device_to_world = R_world_to_device.T
            # t_device_to_world = -R_device_to_world @ t_world_to_device
            
            T_device_to_world.append(t_device_to_world)
            R_device_to_world.append(r_device_to_world)
        
        T_device_to_world = np.array(T_device_to_world)
        R_device_to_world = np.array(R_device_to_world)
        
        print(f"Device-to-world translation ranges (camera positions in world):")
        print(f"  X(forward): [{T_device_to_world[:,0].min():.3f}, {T_device_to_world[:,0].max():.3f}]")
        print(f"  Y(left): [{T_device_to_world[:,1].min():.3f}, {T_device_to_world[:,1].max():.3f}]")
        print(f"  Z(up): [{T_device_to_world[:,2].min():.3f}, {T_device_to_world[:,2].max():.3f}]")
        
        # Step 2: Transform from your coordinate system to CPF coordinate system
        # Your system: X(forward), Y(left), Z(up) → CPF: X(left), Y(up), Z(forward)
        translations_cpf = np.stack([
            T_device_to_world[:, 0],  # CPF X(left) = Your Y(left)
            T_device_to_world[:, 1],  # CPF Y(up) = Your Z(up)  
            T_device_to_world[:, 2]   # CPF Z(forward) = Your X(forward)
        ], axis=1)
        
        # Transform rotations with the same coordinate transformation
        rotations_cpf = []
        for i in range(len(R_device_to_world)):
            R_your_system = R_device_to_world[i]
            
            # Coordinate transformation matrix: Your system → CPF
            # [X,Y,Z] → [Y,Z,X] (permute axes)
            T_your_to_cpf = np.array([
                [0, 1, 0],  # CPF X(left) = Your Y(left)
                [0, 0, 1],  # CPF Y(up) = Your Z(up)
                [1, 0, 0]   # CPF Z(forward) = Your X(forward)
            ])
            # 
            # Transform rotation: R_cpf = T * R_your * T^T
            
            # R_cpf = T_your_to_cpf @ R_your_system @ T_your_to_cpf.T
            # rotations_cpf.append(R_cpf)


            R_cpf = R_your_system.copy()
            rotations_cpf.append(R_cpf)

        
        rotations_cpf = np.array(rotations_cpf)

        print(f"=== FINAL CPF COORDINATE SYSTEM ===")
        print(f"Output coordinate system: X(left), Y(up), Z(forward)")
        print(f"CPF translation ranges:")
        print(f"  X(left): [{translations_cpf[:,0].min():.3f}, {translations_cpf[:,0].max():.3f}]")
        print(f"  Y(up): [{translations_cpf[:,1].min():.3f}, {translations_cpf[:,1].max():.3f}]")
        print(f"  Z(forward): [{translations_cpf[:,2].min():.3f}, {translations_cpf[:,2].max():.3f}]")
        
        # Create SE3 transforms in CPF coordinate system
        Ts_world_device = SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(torch.from_numpy(rotations_cpf).float()),
            translation=torch.from_numpy(translations_cpf).float()
        ).parameters()
        
        # Assume CPF is same as device (common approximation for regular cameras)
        Ts_world_cpf = Ts_world_device.clone()
        
        pose_timesteps = torch.from_numpy(timestamps_sec).float()
        
        return cls(Ts_world_cpf, Ts_world_device, pose_timesteps, device)
    
    @staticmethod
    def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert quaternion (qx, qy, qz, qw) to rotation matrix"""
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm == 0:
            return np.eye(3)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        return R
    
    def to(self, device: torch.device) -> "MP4ColmapInputTransforms":
        """Move transforms to specified device"""
        return MP4ColmapInputTransforms(
            self.Ts_world_cpf.to(device),
            self.Ts_world_device.to(device),
            self.pose_timesteps.to(device),
            device
        )

def load_semidense_points_and_find_ground(points_csv_path: Path) -> Tuple[np.ndarray, float]:
    """
    Load point cloud from semidense_points.csv.gz and estimate ground plane.
    
    Uses the points as they are in the CSV without coordinate system assumptions.
    The CSV contains points in the MPS coordinate system:
    - px_world, py_world, pz_world: 3D coordinates in world frame
    
    Ground plane estimation assumes the ground is the lowest major plane.
    """
    if not points_csv_path.exists():
        print(f"Warning: Points file not found: {points_csv_path}")
        return np.array([]), 0.0
    
    print(f"Loading point cloud from: {points_csv_path}")
    df = pd.read_csv(points_csv_path, compression='gzip')
    
    # Validate required columns
    required_cols = ['px_world', 'py_world', 'pz_world']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in point cloud CSV: {missing_cols}")
    
    # Extract points as they are in the CSV (MPS world coordinate system)
    points = np.stack([
        df['px_world'].values,
        df['py_world'].values,
        df['pz_world'].values
    ], axis=1)
    
    print(f"Loaded {len(points)} points")
    print(f"Point cloud ranges:")
    print(f"  px_world: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"  py_world: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"  pz_world: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    # Based on observation: camera looking down at table, ground is further forward (higher Z)
    # and sparser than the dense table points
    
    # Analyze point distribution in forward direction (assuming pz_world = forward/depth)
    z_values = points[:, 2]  # pz_world
    z_sorted = np.sort(z_values)
    
    print(f"Forward direction (pz_world) analysis:")
    print(f"  Min Z: {z_sorted[0]:.3f}")
    print(f"  Max Z: {z_sorted[-1]:.3f}")
    
    # Create depth bins to analyze point density
    n_bins = 20
    hist, bin_edges = np.histogram(z_values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"Point density analysis across depth bins:")
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        print(f"  Bin {i:2d}: Z={center:6.3f}, Points={count:4d}")
    
    # Ground is likely in the forward region with lower point density
    # Look for the furthest region that still has reasonable number of points
    min_points_threshold = max(10, len(points) * 0.01)  # At least 1% of points or 10 points
    
    # Find bins with sufficient points, starting from the furthest
    valid_bins = [(i, center, count) for i, (center, count) in enumerate(zip(bin_centers, hist)) 
                  if count >= min_points_threshold]
    
    if valid_bins:
        # Take the furthest valid bin as potential ground region
        furthest_bin_idx, furthest_center, furthest_count = valid_bins[-1]
        
        # Ground level is somewhere in this furthest region
        # Use points in this bin to estimate the ground plane
        bin_start = bin_edges[furthest_bin_idx]
        bin_end = bin_edges[furthest_bin_idx + 1]
        
        ground_region_mask = (z_values >= bin_start) & (z_values <= bin_end)
        ground_region_points = points[ground_region_mask]
        
        if len(ground_region_points) > 0:
            # The "ground level" in the context of the vertical axis
            # We need to find which axis represents vertical (up/down)
            # Look at Y-axis as it's most likely to be vertical
            y_values_in_ground_region = ground_region_points[:, 1]  # py_world
            ground_level = np.percentile(y_values_in_ground_region, 10)  # Bottom 10% in Y
            
            print(f"Ground region found at Z=[{bin_start:.3f}, {bin_end:.3f}] with {len(ground_region_points)} points")
            print(f"Ground level (py_world): {ground_level:.3f}")
        else:
            ground_level = np.percentile(points[:, 1], 10)  # Fallback
            print(f"Using fallback ground level: {ground_level:.3f}")
    else:
        ground_level = np.percentile(points[:, 1], 10)  # Fallback
        print(f"No valid ground region found, using fallback: {ground_level:.3f}")
    
    return points, ground_level

@dataclasses.dataclass
class Args:
    traj_root: Path
    """Directory containing converted data with proper coordinate system alignment:
    
    traj_root/
        closed_loop_trajectory.csv     # Poses in your camera coordinate system
        semidense_points.csv.gz        # Points in your camera coordinate system (converted from COLMAP)
        video_metadata.json
        images/
            frame_000000.jpg
            ...
        hamer_outputs.pkl (optional)
        egoallo_outputs/ (will be created)
    """
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    glasses_x_angle_offset: float = 0.0
    """Rotate the CPF poses by some X angle."""
    start_index: int = 0
    """Index within the downsampled trajectory to start inference at."""
    traj_length: int = 128
    """How many timesteps to estimate body motion for."""
    num_samples: int = 1
    """Number of samples to take."""
    guidance_mode: GuidanceMode = "aria_hamer"
    """Which guidance mode to use."""
    guidance_inner: bool = True
    """Whether to apply guidance optimizer between denoising steps."""
    guidance_post: bool = True
    """Whether to apply guidance optimizer after diffusion sampling."""
    save_traj: bool = True
    """Whether to save the output trajectory."""
    visualize_traj: bool = False
    """Whether to visualize the trajectory after sampling."""


def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"=== EgoAllo Inference with Proper Coordinate Transformations ===")

    # Find trajectory paths
    traj_paths = MP4ColmapTrajectoryPaths.find(args.traj_root)
    print(f"Loading data from: {args.traj_root}")
    
    # Check required files
    required_files = [
        traj_paths.slam_csv_path,
        traj_paths.images_dir
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    if traj_paths.splat_path is not None:
        print("Found splat at", traj_paths.splat_path)
    else:
        print("No scene splat found.")
    
    # Load point cloud with coordinate transformation
    points_data, floor_z = load_semidense_points_and_find_ground(traj_paths.points_csv_path)
    
    # Load transforms with coordinate system transformation
    transforms = MP4ColmapInputTransforms.load(traj_paths, fps=30).to(device=device)
    
    # Validate trajectory length
    max_length = len(transforms.pose_timesteps) - args.start_index - 1
    if args.traj_length > max_length:
        print(f"Warning: Requested trajectory length {args.traj_length} > available length {max_length}")
        args.traj_length = max_length
        print(f"Adjusted trajectory length to {args.traj_length}")
    
    # Extract trajectory segment (note off-by-one for relative transforms)
    Ts_world_cpf = (
        SE3(
            transforms.Ts_world_cpf[
                args.start_index : args.start_index + args.traj_length + 1
            ]
        )
        @ SE3.from_rotation(
            SO3.from_x_radians(
                transforms.Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
            )
        )
    ).parameters()
    
    pose_timestamps_sec = transforms.pose_timesteps[
        args.start_index + 1 : args.start_index + args.traj_length + 1
    ]
    
    Ts_world_device = transforms.Ts_world_device[
        args.start_index + 1 : args.start_index + args.traj_length + 1
    ]

    # print("translations device to world : ", Ts_world_device)
    # print("translations CPF to world", Ts_world_cpf)
    # exit(0)
    # Print final trajectory info
    cpf_translations = SE3(Ts_world_cpf).translation()
    print(f"=== FINAL TRAJECTORY INFO ===")
    print(f"CPF trajectory translation range:")
    print(f"  X(left): [{cpf_translations[:, 0].min():.3f}, {cpf_translations[:, 0].max():.3f}]")
    print(f"  Y(up): [{cpf_translations[:, 1].min():.3f}, {cpf_translations[:, 1].max():.3f}]")
    print(f"  Z(forward): [{cpf_translations[:, 2].min():.3f}, {cpf_translations[:, 2].max():.3f}]")
    print(f"Ground plane at Y={floor_z:.3f}")
    print(f"Processing {args.traj_length} timesteps")
    
    del transforms

    # Load hand detections (optional)
    hamer_detections = None
    if traj_paths.hamer_outputs is not None:
        try:
            hamer_detections = CorrespondedHamerDetections.load(
                traj_paths.hamer_outputs,
                pose_timestamps_sec,
            ).to(device)
            print("Loaded HaMeR hand detections")
        except Exception as e:
            print(f"Warning: Could not load HaMeR detections: {e}")
            hamer_detections = None
    else:
        print("No HaMeR hand detections found.")

    # Load Aria wrist and palm estimates (optional)
    aria_detections = None
    if traj_paths.wrist_and_palm_poses_csv is not None:
        try:
            aria_detections = CorrespondedAriaHandWristPoseDetections.load(
                traj_paths.wrist_and_palm_poses_csv,
                pose_timestamps_sec,
                Ts_world_device=Ts_world_device.cpu().numpy(),
            ).to(device)
            print("Loaded Aria hand detections")
        except Exception as e:
            print(f"Warning: Could not load Aria hand detections: {e}")
            aria_detections = None
    else:
        print("No Aria hand detections found.")

    # Setup visualization server if requested
    server = None
    if args.visualize_traj:
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)

    # Load models
    print("=== LOADING MODELS ===")
    print("Loading denoiser network...")
    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    
    print("Loading SMPL-H body model...")
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    # Run inference
    print("=== RUNNING INFERENCE ===")
    print("Running sampling with stitching...")
    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=hamer_detections,
        aria_detections=aria_detections,
        num_samples=args.num_samples,
        device=device,
        floor_z=floor_z,
    )

    # Save outputs
    if args.save_traj:
        print("=== SAVING RESULTS ===")
        save_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"_{args.start_index}-{args.start_index + args.traj_length}"
            + "_corrected_coords"  # Indicate proper coordinate handling
        )
        out_path = args.traj_root / "egoallo_outputs" / (save_name + ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't overwrite existing files
        counter = 1
        while out_path.exists():
            out_path = args.traj_root / "egoallo_outputs" / (save_name + f"_{counter}.npz")
            counter += 1
        
        # Save args for reproducibility
        args_path = out_path.with_suffix("") / "_args.yaml"
        args_path.parent.mkdir(parents=True, exist_ok=True)
        args_path.write_text(yaml.dump(dataclasses.asdict(args)))

        # Apply poses to body model and save
        posed = traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, Ts_world_cpf[..., 1:, :]
        )
        
        print(f"Saving results to {out_path}...")
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].cpu().numpy(),
            Ts_world_root=Ts_world_root.cpu().numpy(),
            body_quats=posed.local_quats[..., :21, :].cpu().numpy(),
            left_hand_quats=posed.local_quats[..., 21:36, :].cpu().numpy(),
            right_hand_quats=posed.local_quats[..., 36:51, :].cpu().numpy(),
            contacts=traj.contacts.cpu().numpy(),
            betas=traj.betas.cpu().numpy(),
            frame_nums=np.arange(args.start_index, args.start_index + args.traj_length),
            timestamps_ns=(pose_timestamps_sec.cpu().numpy() * 1e9).astype(np.int64),
        )
        print("Results saved successfully!")

    # Visualize results
    if args.visualize_traj:
        print("=== STARTING VISUALIZATION ===")
        assert server is not None
        print("Starting visualization server...")
        print("The body should now be properly positioned relative to the ground!")
        
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf[1:],
            traj,
            body_model,
            hamer_detections,
            aria_detections,
            points_data=points_data,
            splat_path=traj_paths.splat_path,
            floor_z=floor_z,
        )
        
        print("Visualization server running. Press Ctrl+C to stop.")
        print("Expected coordinate system in visualization:")
        print("- X: left direction")
        print("- Y: up direction (ground should be around Y=floor_z)")
        print("- Z: forward direction")
        
        try:
            while True:
                loop_cb()
        except KeyboardInterrupt:
            print("Stopping visualization server...")

    print("=== INFERENCE COMPLETE ===")
    print("Summary:")
    print(f"- Processed {args.traj_length} timesteps")  
    print(f"- Coordinate transformation: Camera → CPF applied")
    print(f"- Ground plane detected at Y={floor_z:.3f}")
    if args.save_traj:
        print(f"- Results saved to: {out_path}")
    print("The body should now be correctly positioned!")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))