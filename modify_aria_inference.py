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
from scipy.spatial.transform import Rotation as R

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    
    
    def make_transform(quat_xyzw, t):
        """
        quat_xyzw: quaternion [x,y,z,w]
        t: translation [tx,ty,tz]
        Returns: 4x4 homogeneous transform
        """
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = t
        return T

    
    @classmethod
    def load(cls, traj_paths: MP4ColmapTrajectoryPaths, matrix:np.ndarray, fps: int = 30, 
             device: torch.device = torch.device("cpu")) -> "MP4ColmapInputTransforms":
   
        
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
        print(f"Device to world translation ranges:")
        print(f"  X(forward): [{translations_device_to_world[:,0].min():.3f}, {translations_device_to_world[:,0].max():.3f}]")
        print(f"  Y(left): [{translations_device_to_world[:,1].min():.3f}, {translations_device_to_world[:,1].max():.3f}]")
        print(f"  Z(up): [{translations_device_to_world[:,2].min():.3f}, {translations_device_to_world[:,2].max():.3f}]")
        
        # Step 1: Invert world-to-device transforms to get device-to-world (camera poses in world)
        # translations_device_to_world = []
        # rotations_device_to_world = []
        T_device_to_world = []
        R_device_to_world = []
        
        T_cpf_to_world = []
        R_cpf_to_world = []
        
        ####  THIS IS MODIFIED TO CORRECT TRANSFORMATIONS #########
        # R_cpf_device = np.array([[0, 1, 0],   # CPF x (left) <- Camera y (left)
        #     [0, 0, 1],   # CPF y (up) <- Camera z (up)  
        #     [1, 0, 0] ])
        R_cpf_device = matrix.copy()
        R_cpf_device = np.array([
            [0, -1, 0],  # CPF x (left) <- -Camera x (right)
            [0, 0, -1],  # CPF y (up) <- -Camera y (down)  
            [1, 0, 0]    # CPF z (forward) <- Camera z (forward)
        ]).T
#         T_device_cpf = np.array([
#             [-3.18601373e-02, -9.98629532e-01, -4.15209507e-02,  4.99302002e-03],
#             [ 7.93353401e-01,  1.74815473e-09, -6.08761350e-01, -5.14600901e-02],
#             [ 6.07927062e-01, -5.23360076e-02,  7.92266136e-01, -4.99498807e-02],
#             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
# ])
        Ts_world_cpf = []
        Ts_world_device = []
        
        T_device_cpf = np.eye(4)
        T_device_cpf[:3, :3] = R_cpf_device
        
        for i in range(len(translations_device_to_world)):
            # Get world-to-device transform
            # qx, qy, qz, qw = quaternions_device_to_world[i]
            # t_device_to_world = translations_device_to_world[i]
            # r_device_to_world = cls._quaternion_to_rotation_matrix(qx, qy, qz, qw)
            T_world_device = cls.make_transform(quaternions_device_to_world[i], translations_device_to_world[i] )
            
            T_world_cpf = T_world_device @ T_device_cpf
            # r_cpf_to_world = r_device_to_world @ Transformation_device_to_cpf  
            # t_cpf_to_world = t_device_to_world ## For physical transformation in location
            
            
            Ts_world_device.append(T_world_device)
            Ts_world_cpf.append(T_world_cpf)
            # T_cpf_to_world.append(t_cpf_to_world)
            # R_cpf_to_world.append(r_cpf_to_world)

        
        Ts_world_device = np.array(Ts_world_device)
        Ts_world_cpf = np.array(Ts_world_cpf)
        
        # T_device_to_world = np.array(T_device_to_world) ## These are 1 x 3 vectors for translation
        # R_device_to_world = np.array(R_device_to_world) ## These are 3 x 3 Rotation matrices derived from the quaternions
        
        # T_cpf_to_world = np.array(T_cpf_to_world) ## These are 1 x 3 vectors for translation
        # R_cpf_to_world = np.array(R_cpf_to_world) ## These are 3 x 3 Rotation matrices derived from the quaternions
        
        Ts_world_cpf = SE3.from_matrix(torch.from_numpy(np.array(Ts_world_cpf))).parameters().to(torch.float32)
        Ts_world_device=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_device))).parameters().to(torch.float32)
        
        # print(R_device_to_world.shape)
        # exit(0)
        # print(f"Device-to-world translation ranges (camera positions in world):")
        # print(f"  X(forward): [{T_device_to_world[:,0].min():.3f}, {T_device_to_world[:,0].max():.3f}]")
        # print(f"  Y(left): [{T_device_to_world[:,1].min():.3f}, {T_device_to_world[:,1].max():.3f}]")
        # print(f"  Z(up): [{T_device_to_world[:,2].min():.3f}, {T_device_to_world[:,2].max():.3f}]")
        
        # Step 2: Transform from your coordinate system to CPF coordinate system
        # Your system: X(forward), Y(left), Z(up) → CPF: X(left), Y(up), Z(forward)
        
        # Transform rotations with the same coordinate transformation
        
        # translations_world_device = [] 
        # rotations_device_world = []
        # for i in range(len(R_device_to_world)):
        #     r_world_device = R_device_to_world[i]
        #     t_world_device = T_device_to_world[i]
            
        #     rotations_device_world.append(r_world_device)
        #     rotations_device_world.append(t_world_device)

        
        # rotations_device_world = np.array(rotations_device_world)

        # print(f"=== FINAL CPF COORDINATE SYSTEM ===")
        # print(f"Output coordinate system: X(left), Y(up), Z(forward)")
        # print(f"CPF translation ranges:")
        # print(f"  X(left): [{translations_cpf[:,0].min():.3f}, {translations_cpf[:,0].max():.3f}]")
        # print(f"  Y(up): [{translations_cpf[:,1].min():.3f}, {translations_cpf[:,1].max():.3f}]")
        # print(f"  Z(forward): [{translations_cpf[:,2].min():.3f}, {translations_cpf[:,2].max():.3f}]")
        
        # Create SE3 transforms in CPF coordinate system
        # Ts_world_device = SE3.from_rotation_and_translation(
        #     rotation=SO3.from_matrix(torch.from_numpy(R_device_to_world).float()),
        #     translation=torch.from_numpy(T_device_to_world).float()
        # ).parameters()
        
        # Assume CPF is same as device (common approximation for regular cameras)
        # Ts_world_cpf = SE3.from_rotation_and_translation(
        #     rotation=SO3.from_matrix(torch.from_numpy(R_cpf_to_world).float()),
        #     translation=torch.from_numpy(T_cpf_to_world).float()
        # ).parameters()
        
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, RANSACRegressor
from pathlib import Path
from typing import Tuple, Optional
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def fit_plane_ransac(points: np.ndarray, max_trials: int = 1000, residual_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to 3D points using RANSAC.
    
    Returns:
        plane_normal: Normal vector of the plane (a, b, c) where ax + by + cz + d = 0
        plane_point: A point on the plane
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    # RANSAC for plane fitting
    # We'll fit z = ax + by + c, then convert to normal form
    X = points[:, :2]  # x, y coordinates
    y = points[:, 2]   # z coordinates
    
    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=max_trials,
        residual_threshold=residual_threshold,
        random_state=42
    )
    
    ransac.fit(X, y)
    
    # Get plane parameters: z = ax + by + c
    # Convert to normal form: ax + by - z + c = 0
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    plane_normal = np.array([a, b, -1])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize
    
    # Get a point on the plane
    plane_point = np.array([0, 0, c])
    
    return plane_normal, plane_point, ransac.inlier_mask_

def calculate_plane_area(points: np.ndarray, plane_normal: np.ndarray) -> float:
    """
    Calculate the area of a plane based on the convex hull of projected points.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        # Project points onto the plane using PCA
        pca = PCA(n_components=2)
        # Center the points
        centered_points = points - np.mean(points, axis=0)
        
        # Create a coordinate system on the plane
        # Use the two principal components as the plane basis
        points_2d = pca.fit_transform(centered_points)
        
        # Calculate area using convex hull
        if len(points_2d) >= 3:
            hull = ConvexHull(points_2d)
            return hull.volume  # In 2D, volume is actually area
        else:
            return 0.0
    except:
        # Fallback: approximate area as bounding box
        x_range = np.ptp(points[:, 0])  # peak-to-peak
        y_range = np.ptp(points[:, 1])
        return x_range * y_range

def get_plane_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of plane points."""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return min_coords, max_coords

def create_plane_rectangle(plane_points: np.ndarray, plane_normal: np.ndarray, 
                         margin: float = 0.1) -> np.ndarray:
    """
    Create a rectangle mesh on the detected plane for visualization.
    
    Args:
        plane_points: Points belonging to the plane
        plane_normal: Normal vector of the plane
        margin: Extra margin around the points (as fraction of range)
    
    Returns:
        rectangle_corners: 4 corners of the rectangle on the plane
    """
    # Get bounds of the plane points
    min_coords, max_coords = get_plane_bounds(plane_points)
    
    # Add margin
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    
    margin_x = range_x * margin
    margin_y = range_y * margin
    
    # Create rectangle corners
    x_min, x_max = min_coords[0] - margin_x, max_coords[0] + margin_x
    y_min, y_max = min_coords[1] - margin_y, max_coords[1] + margin_y
    
    # Calculate Z values for each corner using the plane equation
    # plane_normal · (point - plane_point) = 0
    # Solve for z: z = (plane_normal[0]*(plane_point[0] - x) + 
    #                   plane_normal[1]*(plane_point[1] - y) + 
    #                   plane_normal[2]*plane_point[2]) / plane_normal[2]
    
    plane_point = np.mean(plane_points, axis=0)  # Use centroid as reference point
    
    def get_z_on_plane(x, y):
        if abs(plane_normal[2]) < 1e-6:  # Nearly vertical plane
            return plane_point[2]
        return (np.dot(plane_normal, plane_point) - plane_normal[0]*x - plane_normal[1]*y) / plane_normal[2]
    
    corners = np.array([
        [x_min, y_min, get_z_on_plane(x_min, y_min)],
        [x_max, y_min, get_z_on_plane(x_max, y_min)],
        [x_max, y_max, get_z_on_plane(x_max, y_max)],
        [x_min, y_max, get_z_on_plane(x_min, y_max)]
    ])
    
    return corners

def detect_multiple_planes(points: np.ndarray, max_planes: int = 3, min_points_per_plane: int = 100) -> list:
    """
    Detect multiple planes in the point cloud using iterative RANSAC.
    
    Returns list of (plane_normal, plane_point, inlier_points, inlier_mask)
    """
    planes = []
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))
    
    for i in range(max_planes):
        if len(remaining_points) < min_points_per_plane:
            break
            
        try:
            # Fit plane to remaining points
            plane_normal, plane_point, inlier_mask = fit_plane_ransac(
                remaining_points, 
                residual_threshold=0.05  # 2cm threshold
            )
            
            inlier_points = remaining_points[inlier_mask]
            
            if len(inlier_points) < min_points_per_plane:
                break
            
            # Calculate plane area
            plane_area = calculate_plane_area(inlier_points, plane_normal)
            
            # Store plane info with original indices
            original_inlier_indices = remaining_indices[inlier_mask]
            planes.append({
                'normal': plane_normal,
                'point': plane_point,
                'inlier_points': inlier_points,
                'original_indices': original_inlier_indices,
                'num_points': len(inlier_points),
                'area': plane_area
            })
            
            print(f"Plane {i+1}: {len(inlier_points)} points, area: {plane_area:.2f}, normal: {plane_normal}")
            
            # Remove inlier points for next iteration
            remaining_points = remaining_points[~inlier_mask]
            remaining_indices = remaining_indices[~inlier_mask]
            
        except Exception as e:
            print(f"Could not fit plane {i+1}: {e}")
            break
    
    return planes

def identify_ground_plane(planes: list, points: np.ndarray) -> Tuple[Optional[dict], float]:
    """
    Identify which plane is most likely the ground based on:
    1. Most horizontal plane (normal vector close to vertical)
    2. LARGEST AREA (not point density)
    3. Reasonable height (not too high)
    """
    if not planes:
        return None, 0.0
    
    ground_candidates = []
    
    for i, plane in enumerate(planes):
        normal = plane['normal']
        plane_point = plane['point']
        
        # Measure how horizontal the plane is
        # A horizontal plane should have normal close to [0, 0, ±1]
        verticality = abs(normal[2])  # How close to vertical the normal is
        
        # Calculate average height of plane points
        plane_heights = plane['inlier_points'][:, 2]  # Z is up
        avg_height = np.mean(plane_heights)
        
        # Get plane area (this is the key change!)
        plane_area = plane['area']
        
        ground_candidates.append({
            'plane_idx': i,
            'plane': plane,
            'verticality': verticality,
            'avg_height': avg_height,
            'area': plane_area,
            'num_points': plane['num_points']
        })
        
        print(f"Plane {i+1} analysis:")
        print(f"  Verticality (normal Z component): {verticality:.3f}")
        print(f"  Average height: {avg_height:.3f}")
        print(f"  Area: {plane_area:.3f}")
        print(f"  Number of points: {plane['num_points']}")
    
    # NEW SCORING: Prioritize AREA over point count
    def ground_score(candidate):
        # Higher score = more likely to be ground
        verticality_score = candidate['verticality'] * 10  # Weight horizontality heavily
        area_score = candidate['area'] / 100.0  # Area is the key factor now!
        height_penalty = max(0, candidate['avg_height']) * -0.5  # Penalize high planes
        
        # Don't use point count in scoring to avoid dense table bias
        total_score = verticality_score + area_score + height_penalty
        
        print(f"  Ground score: {total_score:.3f} (vert: {verticality_score:.3f}, "
              f"area: {area_score:.3f}, height_penalty: {height_penalty:.3f})")
        
        return total_score
    
    ground_candidates.sort(key=ground_score, reverse=True)
    
    best_candidate = ground_candidates[0]
    ground_plane = best_candidate['plane']
    
    # Calculate ground level as the average Z coordinate of ground plane points
    ground_level = np.mean(ground_plane['inlier_points'][:, 2])
    
    print(f"\nSelected ground plane: Plane {best_candidate['plane_idx'] + 1}")
    print(f"Ground level (Z): {ground_level:.3f}")
    print(f"Ground plane area: {ground_plane['area']:.2f}")
    
    return ground_plane, ground_level

def visualize_planes(points: np.ndarray, planes: list, ground_plane: Optional[dict] = None):
    """Visualize the point cloud and detected planes with rectangle overlay on ground."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points in gray
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='gray', alpha=0.2, s=1, label='All points')
    
    # Plot each plane with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, plane in enumerate(planes):
        color = colors[i % len(colors)]
        inlier_points = plane['inlier_points']
        
        label = f"Plane {i+1} (area: {plane['area']:.1f})"
        if ground_plane and plane is ground_plane:
            label += " - GROUND"
            
        ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2],
                  c=color, alpha=0.8, s=3, label=label)
    
    # Draw rectangle on ground plane
    if ground_plane is not None:
        try:
            rectangle_corners = create_plane_rectangle(
                ground_plane['inlier_points'], 
                ground_plane['normal'],
                margin=0.05
            )
            
            # Create rectangle by connecting corners
            rect_x = [rectangle_corners[0, 0], rectangle_corners[1, 0], 
                     rectangle_corners[2, 0], rectangle_corners[3, 0], rectangle_corners[0, 0]]
            rect_y = [rectangle_corners[0, 1], rectangle_corners[1, 1], 
                     rectangle_corners[2, 1], rectangle_corners[3, 1], rectangle_corners[0, 1]]
            rect_z = [rectangle_corners[0, 2], rectangle_corners[1, 2], 
                     rectangle_corners[2, 2], rectangle_corners[3, 2], rectangle_corners[0, 2]]
            
            ax.plot(rect_x, rect_y, rect_z, 'k-', linewidth=3, label='Ground Rectangle')
            
            # Fill the rectangle (optional)
            ax.plot_trisurf(rectangle_corners[:, 0], rectangle_corners[:, 1], rectangle_corners[:, 2], 
                           alpha=0.3, color='yellow')
            
        except Exception as e:
            print(f"Could not draw ground rectangle: {e}")
    
    ax.set_xlabel('X (px_world)')
    ax.set_ylabel('Y (py_world)')  
    ax.set_zlabel('Z (pz_world)')
    ax.legend()
    ax.set_title('Point Cloud with Detected Planes (Area-Based Ground Detection)')
    
    # Set equal aspect ratio
    max_range = np.array([points[:,0].max()-points[:,0].min(),
                         points[:,1].max()-points[:,1].min(),
                         points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def load_semidense_points_and_find_ground(points_csv_path: Path, visualize: bool = False) -> Tuple[np.ndarray, float]:
    """
    Load point cloud from semidense_points.csv.gz and estimate ground plane using area-based detection.
    
    Args:
        points_csv_path: Path to the semidense points CSV file
        visualize: Whether to show a 3D visualization of the detected planes
    
    Returns:
        points: The loaded point cloud
        ground_level: Estimated ground level (Z coordinate)
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
    
    # Extract points
    points = np.stack([df['px_world'].values, df['py_world'].values, df['pz_world'].values], axis=1)
    
    print(f"Loaded {len(points)} points")
    print(f"Point cloud ranges:")
    print(f"  px_world: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"  py_world: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"  pz_world: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    # Detect multiple planes in the point cloud
    print("\nDetecting planes...")
    planes = detect_multiple_planes(points, max_planes=11, min_points_per_plane=200)
    
    if not planes:
        print("No planes detected! Using fallback method.")
        # Fallback: assume ground is at the 10th percentile of the lowest coordinate
        coord_ranges = [
            (points[:, 0].max() - points[:, 0].min(), 0),  # X range
            (points[:, 1].max() - points[:, 1].min(), 1),  # Y range  
            (points[:, 2].max() - points[:, 2].min(), 2)   # Z range
        ]
        # The axis with largest range is likely horizontal, smallest might be vertical
        coord_ranges.sort(reverse=True)
        vertical_axis = coord_ranges[-1][1]  # Axis with smallest range
        ground_level = np.percentile(points[:, vertical_axis], 10)
        return points, ground_level
    
    # Identify which plane is the ground
    print("\nIdentifying ground plane...")
    ground_plane, ground_level = identify_ground_plane(planes, points)
    
    if ground_plane is None:
        print("Could not identify ground plane!")
        ground_level = np.percentile(points[:, 2], 10)  # Fallback
    
    # Visualization
    # if visualize:
    points_up = points - np.array([0,  0, ground_level])
    
    planes_up = detect_multiple_planes(points_up, max_planes=11, min_points_per_plane=200)

    ground_plane_up, ground_level_up = identify_ground_plane(planes_up, points_up)
    # print(" New ground plane is : ",ground_level_up)
    # visualize_planes(points_up, planes_up, ground_plane_up)
    # exit(0)
    return points_up, ground_level

# Additional utility functions for filtering points based on ground level
def filter_points_above_ground(points: np.ndarray, ground_level: float, 
                             height_threshold: float = 0.05) -> np.ndarray:
    """Filter points that are above the ground by at least height_threshold."""
    return points[points[:, 2] > ground_level + height_threshold]

def filter_points_near_ground(points: np.ndarray, ground_level: float,
                            tolerance: float = 0.02) -> np.ndarray:
    """Filter points that are close to the ground level."""
    return points[np.abs(points[:, 2] - ground_level) <= tolerance]
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
    traj_length: int = 256
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
    rotation_matrices = {
        # Your original assumption
        "forward_left_up": np.array([
            [0, 0, 1],  # CPF x (left) <- Camera y (left)  
            [1, 0, 0],  # CPF y (up) <- Camera z (up)
            [0, 1, 0]   # CPF z (forward) <- Camera x (forward)
        ]).T,
        
        # Common computer vision convention (OpenCV, etc.)
        "right_down_forward": np.array([
            [0, -1, 0],  # CPF x (left) <- -Camera x (right)
            [0, 0, -1],  # CPF y (up) <- -Camera y (down)  
            [1, 0, 0]    # CPF z (forward) <- Camera z (forward)
        ]).T,
        
        # Another common CV convention
        "right_forward_down": np.array([
            [0, 1, 0],   # CPF x (left) <- Camera y (forward)
            [0, 0, -1],  # CPF y (up) <- -Camera z (down)
            [-1, 0, 0]   # CPF z (forward) <- -Camera x (right)
        ]).T,
        
        # Graphics/OpenGL convention  
        "right_up_back": np.array([
            [0, 1, 0],   # CPF x (left) <- Camera y (up)
            [-1, 0, 0],  # CPF y (up) <- -Camera x (right)
            [0, 0, -1]   # CPF z (forward) <- -Camera z (back)
        ]).T,
        
        # Robotics convention (ROS)
        "forward_left_up_ros": np.array([
            [0, 1, 0],   # CPF x (left) <- Camera y (left)
            [0, 0, 1],   # CPF y (up) <- Camera z (up)  
            [1, 0, 0]    # CPF z (forward) <- Camera x (forward)
        ]).T,
        
        # Unity/Unreal convention
        "right_up_forward": np.array([
            [0, 0, 1],   # CPF x (left) <- Camera z (forward)
            [-1, 0, 0],  # CPF y (up) <- -Camera x (right)
            [0, 1, 0]    # CPF z (forward) <- Camera y (up)
        ]).T,
        
        # Another possible variant
        "forward_right_up": np.array([
            [0, -1, 0],  # CPF x (left) <- -Camera y (right)
            [0, 0, 1],   # CPF y (up) <- Camera z (up)
            [1, 0, 0]    # CPF z (forward) <- Camera x (forward)  
        ]).T,
        
        # Camera pointing down
        "down_left_forward": np.array([
            [0, 1, 0],   # CPF x (left) <- Camera y (left)
            [0, 0, 1],   # CPF y (up) <- Camera z (forward) 
            [-1, 0, 0]   # CPF z (forward) <- -Camera x (down)
        ]).T,
    }
    
    
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
    # points_data +=np.array([0, abs(floor_z), 0])
    
    # for keys in rotation_matrices.keys():
    matrix = rotation_matrices["right_down_forward"]
    #     print(" matrix is : ", matrix)
        # exit(0)
        # Load transforms with coordinate system transformation
    transforms = MP4ColmapInputTransforms.load(traj_paths, matrix = matrix, fps=30).to(device=device)
    
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
                args.start_index : args.start_index + max_length + 1
            ]
        )
        @ SE3.from_rotation(
            SO3.from_x_radians(
                transforms.Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
            )
        )
    ).parameters()
    
    pose_timestamps_sec = transforms.pose_timesteps[
        args.start_index + 1 : args.start_index + max_length + 1
    ]
    
    Ts_world_device = transforms.Ts_world_device[
        args.start_index + 1 : args.start_index + max_length + 1
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
    print(" transformation from CPF to world is ; ", Ts_world_cpf)
#     Ts_world_cpf = torch.tensor([
#     [ 0.6315,  0.6440, -0.3249, -0.2844,  0.4612,  0.3702,  0.3164],
#     [ 0.6314,  0.6444, -0.3249, -0.2837,  0.4623,  0.3677,  0.3164],
#     [ 0.6315,  0.6447, -0.3247, -0.2830,  0.4637,  0.3648,  0.3164],
#     [ 0.6317,  0.6450, -0.3245, -0.2823,  0.4652,  0.3616,  0.3162],
#     [ 0.6316,  0.6455, -0.3242, -0.2816,  0.4670,  0.3581,  0.3158],
#     [ 0.6311,  0.6463, -0.3241, -0.2808,  0.4690,  0.3543,  0.3153],
#     [ 0.6303,  0.6473, -0.3244, -0.2799,  0.4712,  0.3504,  0.3147],
#     [ 0.6293,  0.6484, -0.3253, -0.2790,  0.4734,  0.3464,  0.3141],
#     [ 0.6282,  0.6492, -0.3264, -0.2781,  0.4757,  0.3424,  0.3135],
#     [ 0.6276,  0.6495, -0.3275, -0.2776,  0.4779,  0.3383,  0.3128],
#     [ 0.6272,  0.6493, -0.3289, -0.2773,  0.4798,  0.3342,  0.3119],
#     [ 0.6270,  0.6486, -0.3304, -0.2776,  0.4812,  0.3300,  0.3109],
#     [ 0.6265,  0.6479, -0.3320, -0.2785,  0.4819,  0.3258,  0.3098],
#     [ 0.6250,  0.6480, -0.3333, -0.2799,  0.4819,  0.3215,  0.3086],
#     [ 0.6229,  0.6487, -0.3348, -0.2813,  0.4811,  0.3170,  0.3076],
#     [ 0.6208,  0.6494, -0.3362, -0.2824,  0.4796,  0.3123,  0.3069],
#     [ 0.6194,  0.6498, -0.3377, -0.2831,  0.4776,  0.3072,  0.3064],
#     [ 0.6187,  0.6497, -0.3387, -0.2837,  0.4749,  0.3018,  0.3060],
#     [ 0.6190,  0.6493, -0.3386, -0.2841,  0.4714,  0.2958,  0.3056],
#     [ 0.6199,  0.6485, -0.3384, -0.2841,  0.4670,  0.2894,  0.3048],
#     [ 0.6209,  0.6474, -0.3389, -0.2837,  0.4617,  0.2824,  0.3035],
#     [ 0.6214,  0.6466, -0.3399, -0.2833,  0.4554,  0.2749,  0.3019],
#     [ 0.6213,  0.6463, -0.3411, -0.2829,  0.4481,  0.2669,  0.2999],
#     [ 0.6207,  0.6465, -0.3422, -0.2823,  0.4399,  0.2583,  0.2977],
#     [ 0.6196,  0.6476, -0.3428, -0.2814,  0.4309,  0.2490,  0.2954],
#     [ 0.6182,  0.6498, -0.3421, -0.2801,  0.4211,  0.2390,  0.2931],
#     [ 0.6167,  0.6531, -0.3401, -0.2785,  0.4106,  0.2283,  0.2907],
#     [ 0.6155,  0.6564, -0.3372, -0.2766,  0.3992,  0.2169,  0.2883],
#     [ 0.6149,  0.6595, -0.3338, -0.2747,  0.3869,  0.2052,  0.2857],
#     [ 0.6148,  0.6624, -0.3303, -0.2724,  0.3738,  0.1931,  0.2833],
#     [ 0.6150,  0.6650, -0.3272, -0.2693,  0.3598,  0.1806,  0.2817],
#     [ 0.6145,  0.6683, -0.3250, -0.2650,  0.3450,  0.1678,  0.2816],
#     [ 0.6137,  0.6719, -0.3230, -0.2601,  0.3296,  0.1545,  0.2830],
#     [ 0.6120,  0.6764, -0.3212, -0.2544,  0.3142,  0.1412,  0.2844],
#     [ 0.6096,  0.6815, -0.3200, -0.2481,  0.2992,  0.1279,  0.2850],
#     [ 0.6067,  0.6869, -0.3184, -0.2422,  0.2849,  0.1148,  0.2845],
#     [ 0.6043,  0.6918, -0.3153, -0.2383,  0.2710,  0.1018,  0.2834],
#     [ 0.6029,  0.6952, -0.3124, -0.2361,  0.2573,  0.0890,  0.2821],
#     [ 0.6014,  0.6980, -0.3107, -0.2338,  0.2441,  0.0763,  0.2810],
#     [ 0.5991,  0.7011, -0.3104, -0.2308,  0.2315,  0.0635,  0.2805],
#     [ 0.5965,  0.7040, -0.3108, -0.2280,  0.2194,  0.0506,  0.2813],
#     [ 0.5943,  0.7062, -0.3110, -0.2267,  0.2077,  0.0376,  0.2829],
#     [ 0.5928,  0.7074, -0.3105, -0.2275,  0.1963,  0.0245,  0.2848],
#     [ 0.5925,  0.7074, -0.3095, -0.2298,  0.1850,  0.0112,  0.2864],
#     [ 0.5926,  0.7069, -0.3083, -0.2326,  0.1738, -0.0026,  0.2873],
#     [ 0.5927,  0.7064, -0.3072, -0.2350,  0.1624, -0.0168,  0.2872],
#     [ 0.5925,  0.7062, -0.3065, -0.2371,  0.1508, -0.0316,  0.2864],
#     [ 0.5920,  0.7060, -0.3066, -0.2392,  0.1389, -0.0470,  0.2852],
#     [ 0.5914,  0.7054, -0.3071, -0.2415,  0.1268, -0.0628,  0.2836],
#     [ 0.5911,  0.7044, -0.3080, -0.2441,  0.1144, -0.0791,  0.2817],
#     [ 0.5910,  0.7028, -0.3092, -0.2472,  0.1018, -0.0959,  0.2796],
#     [ 0.5909,  0.7012, -0.3105, -0.2506,  0.0891, -0.1132,  0.2776],
#     [ 0.5911,  0.6990, -0.3120, -0.2543,  0.0762, -0.1311,  0.2760],
#     [ 0.5913,  0.6968, -0.3139, -0.2576,  0.0632, -0.1496,  0.2746],
#     [ 0.5912,  0.6948, -0.3162, -0.2602,  0.0502, -0.1687,  0.2742],
#     [ 0.5911,  0.6931, -0.3185, -0.2623,  0.0372, -0.1883,  0.2746],
#     [ 0.5910,  0.6913, -0.3211, -0.2641,  0.0241, -0.2082,  0.2766],
#     [ 0.5905,  0.6897, -0.3238, -0.2659,  0.0109, -0.2282,  0.2798],
#     [ 0.5894,  0.6890, -0.3258, -0.2679, -0.0021, -0.2478,  0.2836],
#     [ 0.5878,  0.6890, -0.3268, -0.2701, -0.0148, -0.2665,  0.2866],
#     [ 0.5864,  0.6891, -0.3270, -0.2725, -0.0273, -0.2844,  0.2884],
#     [ 0.5851,  0.6891, -0.3277, -0.2748, -0.0399, -0.3013,  0.2894],
#     [ 0.5834,  0.6894, -0.3283, -0.2768, -0.0525, -0.3172,  0.2898],
#     [ 0.5813,  0.6903, -0.3289, -0.2782, -0.0652, -0.3321,  0.2903],
#     [ 0.5786,  0.6917, -0.3305, -0.2786, -0.0779, -0.3461,  0.2912],
#     [ 0.5759,  0.6928, -0.3332, -0.2780, -0.0903, -0.3590,  0.2923],
#     [ 0.5748,  0.6923, -0.3364, -0.2776, -0.1024, -0.3710,  0.2936],
#     [ 0.5759,  0.6900, -0.3392, -0.2778, -0.1143, -0.3823,  0.2951],
#     [ 0.5781,  0.6871, -0.3409, -0.2784, -0.1261, -0.3933,  0.2962],
#     [ 0.5804,  0.6846, -0.3415, -0.2789, -0.1381, -0.4041,  0.2967],
#     [ 0.5820,  0.6833, -0.3416, -0.2787, -0.1505, -0.4149,  0.2966],
#     [ 0.5823,  0.6838, -0.3408, -0.2779, -0.1630, -0.4255,  0.2958],
#     [ 0.5821,  0.6853, -0.3390, -0.2768, -0.1758, -0.4361,  0.2941],
#     [ 0.5819,  0.6872, -0.3363, -0.2759, -0.1890, -0.4464,  0.2920],
#     [ 0.5818,  0.6892, -0.3328, -0.2752, -0.2028, -0.4563,  0.2895],
#     [ 0.5817,  0.6911, -0.3294, -0.2747, -0.2172, -0.4657,  0.2871],
#     [ 0.5817,  0.6927, -0.3266, -0.2740, -0.2325, -0.4746,  0.2847],
#     [ 0.5816,  0.6942, -0.3247, -0.2727, -0.2484, -0.4828,  0.2826],
#     [ 0.5815,  0.6957, -0.3233, -0.2709, -0.2649, -0.4905,  0.2805],
#     [ 0.5818,  0.6970, -0.3217, -0.2689, -0.2822, -0.4979,  0.2792],
#     [ 0.5818,  0.6987, -0.3199, -0.2664, -0.3001, -0.5054,  0.2788],
#     [ 0.5811,  0.7014, -0.3179, -0.2631, -0.3186, -0.5131,  0.2800],
#     [ 0.5803,  0.7049, -0.3150, -0.2591, -0.3375, -0.5211,  0.2826],
#     [ 0.5797,  0.7089, -0.3112, -0.2539, -0.3562, -0.5294,  0.2851],
#     [ 0.5793,  0.7134, -0.3070, -0.2473, -0.3742, -0.5380,  0.2864],
#     [ 0.5793,  0.7179, -0.3025, -0.2401, -0.3912, -0.5466,  0.2863],
#     [ 0.5800,  0.7215, -0.2977, -0.2332, -0.4073, -0.5552,  0.2858],
#     [ 0.5810,  0.7243, -0.2936, -0.2272, -0.4225, -0.5640,  0.2854],
#     [ 0.5804,  0.7272, -0.2917, -0.2217, -0.4367, -0.5727,  0.2853],
#     [ 0.5782,  0.7306, -0.2909, -0.2173, -0.4499, -0.5812,  0.2861],
#     [ 0.5751,  0.7344, -0.2898, -0.2143, -0.4617, -0.5893,  0.2875],
#     [ 0.5722,  0.7381, -0.2873, -0.2129, -0.4722, -0.5970,  0.2890],
#     [ 0.5705,  0.7408, -0.2839, -0.2125, -0.4816, -0.6044,  0.2903],
#     [ 0.5692,  0.7431, -0.2809, -0.2121, -0.4902, -0.6113,  0.2910],
#     [ 0.5675,  0.7453, -0.2788, -0.2116, -0.4981, -0.6175,  0.2909],
#     [ 0.5660,  0.7469, -0.2779, -0.2112, -0.5053, -0.6232,  0.2901],
#     [ 0.5629,  0.7490, -0.2796, -0.2100, -0.5118, -0.6281,  0.2889],
#     [ 0.5586,  0.7512, -0.2828, -0.2091, -0.5175, -0.6323,  0.2882],
#     [ 0.5556,  0.7520, -0.2858, -0.2101, -0.5226, -0.6362,  0.2885],
#     [ 0.5547,  0.7508, -0.2887, -0.2125, -0.5273, -0.6399,  0.2893],
#     [ 0.5533,  0.7503, -0.2911, -0.2148, -0.5313, -0.6432,  0.2899],
#     [ 0.5519,  0.7499, -0.2931, -0.2173, -0.5347, -0.6462,  0.2904],
#     [ 0.5510,  0.7490, -0.2952, -0.2197, -0.5379, -0.6491,  0.2907],
#     [ 0.5496,  0.7486, -0.2975, -0.2214, -0.5408, -0.6518,  0.2909],
#     [ 0.5482,  0.7486, -0.2995, -0.2222, -0.5434, -0.6546,  0.2911],
#     [ 0.5470,  0.7489, -0.3011, -0.2221, -0.5456, -0.6573,  0.2911],
#     [ 0.5458,  0.7495, -0.3022, -0.2214, -0.5473, -0.6600,  0.2910],
#     [ 0.5451,  0.7501, -0.3028, -0.2203, -0.5487, -0.6626,  0.2908],
#     [ 0.5452,  0.7503, -0.3028, -0.2193, -0.5497, -0.6651,  0.2908],
#     [ 0.5463,  0.7499, -0.3023, -0.2186, -0.5503, -0.6674,  0.2910],
#     [ 0.5479,  0.7491, -0.3017, -0.2182, -0.5508, -0.6697,  0.2916],
#     [ 0.5497,  0.7482, -0.3008, -0.2180, -0.5511, -0.6720,  0.2922],
#     [ 0.5514,  0.7475, -0.2997, -0.2179, -0.5512, -0.6741,  0.2927],
#     [ 0.5532,  0.7466, -0.2985, -0.2179, -0.5512, -0.6762,  0.2932],
#     [ 0.5552,  0.7457, -0.2971, -0.2177, -0.5510, -0.6782,  0.2937],
#     [ 0.5573,  0.7447, -0.2957, -0.2175, -0.5507, -0.6801,  0.2943],
#     [ 0.5597,  0.7436, -0.2942, -0.2173, -0.5504, -0.6820,  0.2951],
#     [ 0.5620,  0.7425, -0.2927, -0.2171, -0.5500, -0.6839,  0.2959],
#     [ 0.5640,  0.7417, -0.2912, -0.2169, -0.5495, -0.6857,  0.2965],
#     [ 0.5655,  0.7411, -0.2899, -0.2166, -0.5491, -0.6874,  0.2970],
#     [ 0.5665,  0.7408, -0.2890, -0.2163, -0.5486, -0.6891,  0.2973],
#     [ 0.5668,  0.7409, -0.2883, -0.2161, -0.5482, -0.6906,  0.2973],
#     [ 0.5666,  0.7413, -0.2879, -0.2159, -0.5476, -0.6921,  0.2971],
#     [ 0.5661,  0.7418, -0.2877, -0.2157, -0.5470, -0.6936,  0.2969],
#     [ 0.5658,  0.7421, -0.2875, -0.2156, -0.5463, -0.6950,  0.2967],
#     [ 0.5658,  0.7422, -0.2874, -0.2157, -0.5456, -0.6965,  0.2966],
#     [ 0.5661,  0.7420, -0.2869, -0.2160, -0.5447, -0.6981,  0.2966],
#     [ 0.5664,  0.7418, -0.2864, -0.2166, -0.5439, -0.6996,  0.2966],
#     [ 0.5663,  0.7417, -0.2863, -0.2174, -0.5430, -0.7011,  0.2967]
# ], device='cuda:0')

    print(" shape is : ", Ts_world_cpf.shape)
    # exit(0)
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
            + f"_{args.start_index}-{args.start_index + max_length}"
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

        Ts_world_root_modified = Ts_world_root.clone()
        # Ts_world_root_modified[:, 2] += floor_z  # Lower by 1 meter in Z (up) direction

        

        print(f"Saving results to {out_path}...")
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].cpu().numpy(),
            Ts_world_root=Ts_world_root_modified.cpu().numpy(),
            body_quats=posed.local_quats[..., :21, :].cpu().numpy(),
            left_hand_quats=posed.local_quats[..., 21:36, :].cpu().numpy(),
            right_hand_quats=posed.local_quats[..., 36:51, :].cpu().numpy(),
            contacts=traj.contacts.cpu().numpy(),
            betas=traj.betas.cpu().numpy(),
            frame_nums=np.arange(args.start_index, args.start_index + max_length),
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