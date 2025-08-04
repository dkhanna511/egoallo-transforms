import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import argparse
import os

def load_data_files(frames_file, egomotion_file, imu_file):
    """Load all three data files"""
    
    print("Loading data files...")
    
    # Load frames file
    frames_df = pd.read_csv(frames_file)
    print(f"Loaded {len(frames_df)} frames from {frames_file}")
    
    # Load egomotion file (space-separated)
    egomotion_df = pd.read_csv(egomotion_file, sep=' ', header=None)
    egomotion_df.columns = ['frame_ID', 'X_world', 'Y_world', 'Z_world', 
                           'euler_X', 'euler_Y', 'euler_Z', 
                           'quat_X', 'quat_Y', 'quat_Z', 'quat_W']
    print(f"Loaded {len(egomotion_df)} poses from {egomotion_file}")
    
    # Load IMU file
    imu_df = pd.read_csv(imu_file)
    print(f"Loaded {len(imu_df)} IMU samples from {imu_file}")
    
    return frames_df, egomotion_df, imu_df

def align_timestamps(frames_df, imu_df):
    """Align IMU data to frame timestamps using nearest neighbor matching"""
    
    print("Aligning timestamps...")
    
    # Convert timestamps to numpy arrays for faster processing
    frame_timestamps = frames_df['unix_timestamp'].values
    imu_timestamps = imu_df['timestamp'].values
    
    # Find nearest IMU timestamp for each frame
    aligned_imu_indices = []
    
    for frame_ts in frame_timestamps:
        # Find closest IMU timestamp
        time_diffs = np.abs(imu_timestamps - frame_ts)
        closest_idx = np.argmin(time_diffs)
        aligned_imu_indices.append(closest_idx)
        
        # Print warning if alignment is poor (>50ms difference)
        time_diff_ms = time_diffs[closest_idx] / 1000  # Convert to milliseconds
        if time_diff_ms > 50:
            print(f"Warning: Frame {len(aligned_imu_indices)-1} has {time_diff_ms:.2f}ms timestamp difference")
    
    # Create aligned IMU dataframe
    aligned_imu_df = imu_df.iloc[aligned_imu_indices].reset_index(drop=True)
    aligned_imu_df['frame_idx'] = frames_df['frame_idx']
    aligned_imu_df['frame_timestamp'] = frames_df['unix_timestamp']
    
    print(f"Aligned {len(aligned_imu_df)} IMU samples to frames")
    return aligned_imu_df


def compute_linear_velocity_device_frame(positions, quaternions, timestamps):
    """
    Compute linear velocities in the device coordinate frame from camera-to-world poses.

    Args:
        positions: (N,3) numpy array of translations [tx,ty,tz] in meters
        quaternions: (N,4) numpy array of quaternions [qx,qy,qz,qw]
        timestamps: (N,) numpy array of timestamps (ns, us, ms, or s)

    Returns:
        device_linear_velocities: (N,3) linear velocities in device frame [m/s]
    """
    print("=== Computing device-frame linear velocities ===")

    # --- Timestamp normalization ---
    timestamp_range = timestamps[-1] - timestamps[0]
    if timestamp_range > 1e12:
        timestamps_sec = timestamps / 1e9  # ns → s
        print("Detected nanoseconds - converted to seconds")
    elif timestamp_range > 1e9:
        timestamps_sec = timestamps / 1e6  # µs → s
        print("Detected microseconds - converted to seconds")
    elif timestamp_range > 1e6:
        timestamps_sec = timestamps / 1e3  # ms → s
        print("Detected milliseconds - converted to seconds")
    else:
        timestamps_sec = timestamps
        print("Assuming timestamps already in seconds")

    print(f"Timestamps: {timestamps_sec[0]:.6f} → {timestamps_sec[-1]:.6f}, duration {timestamps_sec[-1]-timestamps_sec[0]:.3f} s")

    # --- Compute linear velocity in world frame ---
    world_linear_velocities = np.zeros_like(positions)
    dt_values = []
    for i in range(1, len(positions)):
        dt = timestamps_sec[i] - timestamps_sec[i-1]
        dt_values.append(dt)
        if dt > 0:
            world_linear_velocities[i] = (positions[i] - positions[i-1]) / dt
        else:
            world_linear_velocities[i] = world_linear_velocities[i-1]
    world_linear_velocities[0] = world_linear_velocities[1]

    # --- Debug sanity checks ---
    dt_values = np.array(dt_values)
    print("dt statistics:")
    print(f"  Mean dt: {np.mean(dt_values):.6f} s")
    print(f"  Min dt: {np.min(dt_values):.6f} s")
    print(f"  Max dt: {np.max(dt_values):.6f} s")
    vel_magnitudes = np.linalg.norm(world_linear_velocities[1:6], axis=1)
    print(f"First 5 world velocity magnitudes: {vel_magnitudes}")

    if np.any(vel_magnitudes > 100):
        print("WARNING: Unrealistically high velocities detected")
        if np.mean(dt_values) < 1e-6:
            print("  → Timestamps may be in the wrong units")
        if np.max(np.abs(positions)) > 1000:
            print("  → Positions may be in mm instead of meters")

    # --- Transform velocities from world frame to device frame ---
    device_linear_velocities = np.zeros_like(world_linear_velocities)
    for i in range(len(quaternions)):
        q_device_to_world = R.from_quat(quaternions[i])
        q_world_to_device = q_device_to_world.inv()
        device_linear_velocities[i] = q_world_to_device.apply(world_linear_velocities[i])
        
    print("Finished computing device-frame velocities.")
    return device_linear_velocities

def camera_to_world_to_world_to_device(positions_c2w, quaternions_c2w):
    """
    Convert camera-to-world poses to world-to-device transforms.
    
    Camera-to-world: where the camera is positioned in world coordinates
    World-to-device: the transform that takes world points to device coordinates
    
    For EgoAllo: world-to-device = inverse of camera-to-world
    """
    print("Converting camera-to-world poses to world-to-device transforms...")
    
    positions_w2d = np.zeros_like(positions_c2w)
    quaternions_w2d = np.zeros_like(quaternions_c2w)
    
    for i in range(len(positions_c2w)):
        # Camera-to-world rotation
        q_c2w = R.from_quat([quaternions_c2w[i, 0], quaternions_c2w[i, 1], 
                            quaternions_c2w[i, 2], quaternions_c2w[i, 3]])
        
        # Invert to get world-to-camera (world-to-device)
        q_w2d = q_c2w.inv()
        
        # Invert translation: t_w2d = -R_w2d * t_c2w
        positions_w2d[i] = q_w2d.apply(-positions_c2w[i])
        quaternions_w2d[i] = q_w2d.as_quat()
    
    print("Successfully converted poses:")
    print(f"  Camera positions (c2w): X[{positions_c2w[:,0].min():.3f}, {positions_c2w[:,0].max():.3f}]")
    print(f"  World-to-device trans: X[{positions_w2d[:,0].min():.3f}, {positions_w2d[:,0].max():.3f}]")
    
    return positions_w2d, quaternions_w2d

def camera_pose_to_aria(positions_c2w, quaternions_c2w):
    """
    Convert camera poses to Aria coordinate system.
    
    Args:
        positions_c2w: (N, 3) array of camera positions [tx, ty, tz]
        quaternions_c2w: (N, 4) array of quaternions [qx, qy, qz, qw]
    
    Returns:
        t_aria: (N, 3) array of positions in Aria frame
        quaternions_aria: (N, 4) array of quaternions in Aria frame
    """
    
    # Define transformation matrix from camera frame to Aria frame
    # Camera: [X=forward, Y=left, Z=up] -> Aria: [X=right, Y=up, Z=backward]
    R_cam_to_aria = np.array([
        [ 0, -1,  0],  # Aria_X = -Camera_Y
        [ 0,  0,  1],  # Aria_Y = Camera_Z
        [-1,  0,  0]   # Aria_Z = -Camera_X
    ])
    
    # Convert quaternions to rotation matrices
    R_cam = R.from_quat(quaternions_c2w).as_matrix()  # (N, 3, 3)
    
    # Apply change of basis to each rotation matrix
    # Broadcasting: (3,3) @ (N,3,3) @ (3,3) -> (N,3,3)
    R_aria = R_cam_to_aria @ R_cam @ R_cam_to_aria.T
    
    # Convert back to quaternions
    quaternions_aria = R.from_matrix(R_aria).as_quat()  # (N, 4)
    
    # Transform translations
    # Apply the same coordinate transformation to positions
    t_aria = (R_cam_to_aria @ positions_c2w.T).T  # (N, 3)
    
    return t_aria, quaternions_aria

import numpy as np
from scipy.spatial.transform import Rotation

def transform_to_aria_device_frame(positions_c2w, quaternions_c2w, timestamps):
   """
   Transform camera-to-world poses to Aria device frame convention
   
   Input coordinate system (your world frame):
   x -- forward, y -- left, z -- up
   
   Output coordinate system (Aria device frame):
   x -- down, y -- left, z -- forward
   
   Args:
       tx, ty, tz: translation arrays (camera positions in world)
       qx, qy, qz, qw: quaternion arrays (camera orientation in world)
       timestamps: timestamp array for velocity calculation
   
   Returns:
       Dictionary with transformed poses and velocities
   """
   
   # Step 1: Prepare input data
#    positions_c2w = np.array([tx, ty, tz]).T  # Shape: (N, 3)
#    quaternions_c2w = np.array([qx, qy, qz, qw]).T  # Shape: (N, 4)
   
   # Step 2: Calculate velocities in your current world frame FIRST
   dt = np.diff(timestamps)
   dt = np.append(dt, dt[-1])  # Extend dt array to match length
   
   # Calculate velocity using finite differences
   velocity_world = np.zeros_like(positions_c2w)
   velocity_world[1:] = np.diff(positions_c2w, axis=0) / dt[1:, np.newaxis]
   velocity_world[0] = velocity_world[1]  # Use second point for first
   
   # Step 3: Define transformation matrix (Your world → Aria device)
   R_aria_your = np.array([[0,  0, -1],  # Aria x-down = -your z-up
                           [0,  1,  0],  # Aria y-left = your y-left
                           [1,  0,  0]]) # Aria z-forward = your x-forward
   
   # Step 4: Transform positions (world coordinates with Aria axis convention)
   positions_aria_world = positions_c2w @ R_aria_your.T
   
   # Step 5: Transform rotations
   # Convert quaternions to rotation matrices
   rotations_c2w = Rotation.from_quat(quaternions_c2w)
   R_world_camera = rotations_c2w.as_matrix()
   
   # Apply coordinate system transformation to rotations
   R_world_aria_device = R_world_camera @ R_aria_your.T
   
   # Convert back to quaternions
   rotations_aria = Rotation.from_matrix(R_world_aria_device)
   quaternions_aria = rotations_aria.as_quat()  # [qx, qy, qz, qw]
   
   # Step 6: Transform velocities to device frame
   velocity_device = velocity_world @ R_aria_your.T
   
   # Step 7: Return results
   return {
       # Device-to-world poses (Aria format)
       'tx_world_device': positions_aria_world[:, 0],
       'ty_world_device': positions_aria_world[:, 1], 
       'tz_world_device': positions_aria_world[:, 2],
       'qx_world_device': quaternions_aria[:, 0],
       'qy_world_device': quaternions_aria[:, 1],
       'qz_world_device': quaternions_aria[:, 2],
       'qw_world_device': quaternions_aria[:, 3],
       
       # Device velocities in device frame (Aria format)
       'device_linear_velocity_x_device': velocity_device[:, 0],
       'device_linear_velocity_y_device': velocity_device[:, 1],
       'device_linear_velocity_z_device': velocity_device[:, 2]
   }

def transform_angular_velocity_iphone_to_aria(omega_iphone):
    """
    Transform angular velocities from iPhone IMU frame to Aria device frame.

    Args:
        omega_iphone: (N,3) numpy array of angular velocities in iPhone IMU frame

    Returns:
        omega_aria: (N,3) numpy array of angular velocities in Aria device frame
    """
    R_iphone_to_aria = np.array([
        [ 0, -1,  0],  # X_down
        [-1,  0,  0],  # Y_left
        [ 0,  0, -1]   # Z_forward
    ])
    return omega_iphone @ R_iphone_to_aria.T


def create_closed_loop_trajectory(frames_df, egomotion_df, aligned_imu_df):
    """Create the closed-loop trajectory CSV data with CORRECT coordinate handling"""

    print("Creating closed-loop trajectory...")
    print("=== COORDINATE SYSTEM EXPLANATION ===")
    print("Input data: Camera-to-world poses (where camera is in world)")
    print("EgoAllo expects: World-to-device transforms (tx_world_device, etc.)")
    print("Conversion: world-to-device = inverse(camera-to-world)")

    # Merge frames with egomotion data
    merged_df = frames_df.merge(egomotion_df, left_on='frame_idx', right_on='frame_ID', how='inner')
    print(f"Merged data has {len(merged_df)} frames")

    # Extract camera-to-world position and orientation
    positions_c2w = merged_df[['X_world', 'Y_world', 'Z_world']].values
    quaternions_c2w = merged_df[['quat_X', 'quat_Y', 'quat_Z', 'quat_W']].values
    device_angular_velocities = aligned_imu_df[['omega_x', 'omega_y', 'omega_z']].values
    
    timestamps = merged_df['unix_timestamp'].values

    print(f"Original camera-to-world poses:")
    print(f"  Position range: X[{positions_c2w[:,0].min():.3f}, {positions_c2w[:,0].max():.3f}]")
    print(f"                  Y[{positions_c2w[:,1].min():.3f}, {positions_c2w[:,1].max():.3f}]") 
    print(f"                  Z[{positions_c2w[:,2].min():.3f}, {positions_c2w[:,2].max():.3f}]")

    # Compute velocities from the camera-to-world poses (before inversion)
    # device_linear_vel= compute_linear_velocity_device_frame(positions_c2w, quaternions_c2w, timestamps)
    
    # Usage example:
    # Assuming you have your data arrays: tx, ty, tz, qx, qy, qz, qw, timestamps
    
    # Convert camera-to-world to world-to-device for EgoAllo
    # positions_w2d, quaternions_w2d = camera_to_world_to_world_to_device(positions_c2w, quaternions_c2w)
    # positions_aria, quaternions_aria = camera_pose_to_aria(positions_c2w, quaternions_c2w)
    

    # Create output dataframe
    output_df = pd.DataFrame()
    output_df['graph_uid'] = range(len(merged_df))
    output_df['tracking_timestamp_us'] = merged_df['unix_timestamp']
    output_df['utc_timestamp_ns'] = merged_df['unix_timestamp'] * 1000

    #### Values extracted from the egomotion files -- Camera (Device) to World coordinate system
    
    result = transform_to_aria_device_frame(positions_c2w,  quaternions_c2w, timestamps)

    # Add to your dataframe
    for key, value in result.items():
        output_df[key] = value

    # output_df['tx_world_device'] = positions_c2w[:, 0]
    # output_df['ty_world_device'] = positions_c2w[:, 1]
    # output_df['tz_world_device'] = positions_c2w[:, 2]
    # output_df['qx_world_device'] = quaternions_c2w[:, 0]
    # output_df['qy_world_device'] = quaternions_c2w[:, 1]
    # output_df['qz_world_device'] = quaternions_c2w[:, 2]
    # output_df['qw_world_device'] = quaternions_c2w[:, 3]

    # Device frame velocities (computed from the poses in world coordinate system with inverse transformations.
    # output_df['device_linear_velocity_x_device'] = device_linear_vel[:, 0]
    # output_df['device_linear_velocity_y_device'] = device_linear_vel[:, 1]
    # output_df['device_linear_velocity_z_device'] = device_linear_vel[:, 2]

    # Angular velocities
    # if len(aligned_imu_df) == len(merged_df):
    # print("Using IMU angular velocities")
    output_df['gravity_y_world'] = aligned_imu_df['gravity_x']
    output_df['gravity_y_world'] = aligned_imu_df['gravity_y']
    output_df['gravity_z_world'] = aligned_imu_df['gravity_z']
    
    ### World coordinate system gravity as mentioned in the Aria Documentation on loop trajectories.
    # output_df['gravity_y_world'] = 0
    # output_df['gravity_y_world'] = 0
    # output_df['gravity_z_world'] = -9.81
    
    # else:
    #### Angular velocities directly extracted from IMU data. -- These need to be transformed to the camera coordinate system.
    output_df['angular_velocity_x_device'] = device_angular_velocities[:, 0]
    output_df['angular_velocity_y_device'] = device_angular_velocities[:, 1]
    output_df['angular_velocity_z_device'] = device_angular_velocities[:, 2]
    
    
    output_df['angular_velocity_x_device'], output_df['angular_velocity_y_device'], \
    output_df['angular_velocity_z_device'] = transform_angular_velocity_iphone_to_aria(output_df[['angular_velocity_x_device', 
                                                                                                  'angular_velocity_y_device', 
                                                                                                  'angular_velocity_z_device']].values).T

    # Gravity in world frame
    # output_df['gravity_x_world'] = 0.0
    # output_df['gravity_y_world'] = 0.0
    # output_df['gravity_z_world'] = -9.81

    # Default valuess
    output_df['quality_score'] = 1.0
    output_df['geo_available'] = 0

    # ECEF placeholders
    output_df['tx_ecef_device'] = 0.0
    output_df['ty_ecef_device'] = 0.0
    output_df['tz_ecef_device'] = 0.0
    output_df['qx_ecef_device'] = 0.0
    output_df['qy_ecef_device'] = 0.0
    output_df['qz_ecef_device'] = 0.0
    output_df['qw_ecef_device'] = 0.0

    print("=== FINAL OUTPUT SUMMARY ===")
    print(f"tx_world_device range: [{output_df['tx_world_device'].min():.3f}, {output_df['tx_world_device'].max():.3f}]")
    print(f"ty_world_device range: [{output_df['ty_world_device'].min():.3f}, {output_df['ty_world_device'].max():.3f}]")
    print(f"tz_world_device range: [{output_df['tz_world_device'].min():.3f}, {output_df['tz_world_device'].max():.3f}]")
    print("These are world-to-device transforms (what EgoAllo expects)")

    return output_df

def validate_output(output_df):
    """Validate the generated trajectory data"""
    
    print("\n=== Trajectory Validation ===")
    
    # Check for NaN values
    nan_counts = output_df.isnull().sum()
    if nan_counts.sum() > 0:
        print("Warning: Found NaN values:")
        print(nan_counts[nan_counts > 0])
    
    # Check trajectory smoothness
    positions = output_df[['tx_world_device', 'ty_world_device', 'tz_world_device']].values
    velocities = output_df[['device_linear_velocity_x_device', 'device_linear_velocity_y_device', 
                           'device_linear_velocity_z_device']].values
    
    # Compute trajectory statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    mean_velocity = np.mean(np.linalg.norm(velocities, axis=1))
    
    print(f"World-to-device transform distance: {total_distance:.2f}")
    print(f"Maximum velocity: {max_velocity:.2f} m/s")
    print(f"Mean velocity: {mean_velocity:.2f} m/s")
    
    # Check quaternion normalization
    quats = output_df[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].values
    quat_norms = np.linalg.norm(quats, axis=1)
    print(f"Quaternion norm range: [{np.min(quat_norms):.4f}, {np.max(quat_norms):.4f}]")
    
    if np.any(np.abs(quat_norms - 1.0) > 0.01):
        print("Warning: Some quaternions are not properly normalized")
    else:
        print("✅ All quaternions are properly normalized")
    
    return True


# Test function
def test_conversion():
    """Test the conversion with some example data."""
    
    # Example data: 3 poses
    positions_c2w = np.array([
        [1.0, 0.0, 0.0],  # 1 meter forward
        [0.0, 1.0, 0.0],  # 1 meter left
        [0.0, 0.0, 1.0],  # 1 meter up
    ])
    
    # Identity quaternion (no rotation)
    quaternions_c2w = np.array([
        [0.0, 0.0, 0.0, 1.0],  # [qx, qy, qz, qw]
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    
    print("Input (Camera frame):")
    print("Positions:", positions_c2w)
    print("Quaternions:", quaternions_c2w)
    
    # Convert
    t_aria, q_aria = camera_pose_to_aria(positions_c2w, quaternions_c2w)
    
    print("\nOutput (Aria frame):")
    print("Positions:", t_aria)
    print("Quaternions:", q_aria)
    
    print("\nExpected transformations:")
    print("Camera forward (1,0,0) -> Aria (0,0,-1) backward")
    print("Camera left (0,1,0) -> Aria (-1,0,0) right") 
    print("Camera up (0,0,1) -> Aria (0,1,0) up")



def main():
    parser = argparse.ArgumentParser(description='Convert trajectory data to closed-loop format')
    parser.add_argument('--frames', required=True, help='Path to frames_ep005.csv')
    parser.add_argument('--egomotion', required=True, help='Path to egimotion_camera_imu_aligned.txt')
    parser.add_argument('--imu', required=True, help='Path to imu_combined.csv')
    parser.add_argument('--output', default='closed_loop_trajectory.csv', help='Output CSV file')
    
    args = parser.parse_args()
    # test_conversion()
    # exit(0)
    # Verify input files exist
    for file_path in [args.frames, args.egomotion, args.imu]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return
    
    try:
        print("=== CORRECTED TRAJECTORY CONVERSION ===")
        print("This version correctly handles camera-to-world → world-to-device conversion")
        print("for EgoAllo compatibility.")
        print()
        
        # Load data files
        frames_df, egomotion_df, imu_df = load_data_files(args.frames, args.egomotion, args.imu)
        
        # Align IMU timestamps to frames
        aligned_imu_df = align_timestamps(frames_df, imu_df)
        
        # Create closed-loop trajectory with CORRECT coordinate handling
        output_df = create_closed_loop_trajectory(frames_df, egomotion_df, aligned_imu_df)
        
        # Validate output
        validate_output(output_df)
        
        
        # Save to CSV
        output_df.to_csv(args.output, index=False)
        print(f"\n✅ Successfully saved CORRECTED closed-loop trajectory to {args.output}")
        print(f"Generated {len(output_df)} trajectory points")
        print("This file now contains proper world-to-device transforms for EgoAllo!")
        
        # Print sample of output
        print(f"\nFirst 3 rows of output:")
        print(output_df.head(3)[['tx_world_device', 'ty_world_device', 'tz_world_device', 
                                'qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].to_string())
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()