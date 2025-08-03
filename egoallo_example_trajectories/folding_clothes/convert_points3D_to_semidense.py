#!/usr/bin/env python3
"""
Convert COLMAP points3D.txt to semidense_points.csv.gz format compatible with EgoAllo.

COLMAP coordinate system: X(right), Y(down), Z(forward) 
Your camera coordinate system: X(forward), Y(left), Z(up)
Target format: semidense_points.csv.gz with columns [px_world, py_world, pz_world]
"""

import numpy as np
import pandas as pd
import gzip
from pathlib import Path
from typing import Tuple, Optional
import argparse


def read_colmap_points3d(points3d_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read COLMAP points3D.txt file and extract 3D points, colors, and errors.
    
    Format: POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
    
    Returns:
        points_3d: (N, 3) array of 3D points in COLMAP coordinate system
        colors: (N, 3) array of RGB colors [0-255]
        errors: (N,) array of reprojection errors
    """
    points_3d = []
    colors = []
    errors = []
    
    print(f"Reading COLMAP points3D.txt from: {points3d_path}")
    
    with open(points3d_path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            if len(parts) < 8:  # Minimum: ID + XYZ + RGB + ERROR
                print(f"Warning: Skipping malformed line {line_idx}: {line}")
                continue
            
            try:
                # Parse: POINT3D_ID X Y Z R G B ERROR
                point_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                error = float(parts[7])
                
                points_3d.append([x, y, z])
                colors.append([r, g, b])
                errors.append(error)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_idx}: {e}")
                continue
    
    points_3d = np.array(points_3d, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8)
    errors = np.array(errors, dtype=np.float64)
    
    print(f"Loaded {len(points_3d)} 3D points from COLMAP")
    print(f"COLMAP coordinate ranges:")
    print(f"  X(right): [{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}]")
    print(f"  Y(down): [{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}]")
    print(f"  Z(forward): [{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")
    
    return points_3d, colors, errors


def transform_colmap_to_camera_system(points_colmap: np.ndarray) -> np.ndarray:
    """
    Transform points from COLMAP coordinate system to your camera coordinate system.
    
    COLMAP: X(right), Y(down), Z(forward)
    Your camera: X(forward), Y(left), Z(up)
    
    Transformation:
    - COLMAP X(right) → Your -Y(left)  [right becomes -left]
    - COLMAP Y(down) → Your -Z(up)     [down becomes -up]
    - COLMAP Z(forward) → Your X(forward) [forward stays forward]
    """
    print("Transforming from COLMAP to your camera coordinate system...")
    print("COLMAP: X(right), Y(down), Z(forward) → Your: X(forward), Y(left), Z(up)")
    
    points_camera = np.zeros_like(points_colmap)
    
    # Apply transformation
    R = np.array([
        [1, 0,  0],
    [0,-1,  0],
    [0, 0, -1]
    ])
    points_colmap[:, 2] = -points_colmap[:, 2] +1 # Your Z(up) = -COLMAP Y(down)
    
    
    # points_camera[:, 0] = points_colmap[:, 2]   # Your X(forward) = COLMAP Z(forward)
    # points_camera[:, 1] = -points_colmap[:, 0]  # Your Y(left) = -COLMAP X(right)
    # points_camera[:, 2] = -points_colmap[:, 1]  # Your Z(up) = -COLMAP Y(down)
    # points_camera = points_colmap@ R.T


    
    print(f"Transformed coordinate ranges:")
    print(f"  X(forward): [{points_camera[:, 0].min():.3f}, {points_camera[:, 0].max():.3f}]")
    print(f"  Y(left): [{points_camera[:, 1].min():.3f}, {points_camera[:, 1].max():.3f}]")
    print(f"  Z(up): [{points_camera[:, 2].min():.3f}, {points_camera[:, 2].max():.3f}]")
    
    return points_colmap


def save_semidense_points(points_camera: np.ndarray, output_path: Path, 
                         colors: Optional[np.ndarray] = None, 
                         errors: Optional[np.ndarray] = None) -> None:
    """
    Save points in semidense_points.csv.gz format expected by EgoAllo.
    
    Args:
        points_camera: (N, 3) points in your camera coordinate system
        output_path: Path to save semidense_points.csv.gz
        colors: Optional (N, 3) RGB colors
        errors: Optional (N,) reprojection errors
    """
    print(f"Saving semidense_points.csv.gz to: {output_path}")
    
    # Create DataFrame with required columns
    df_data = {
        'px_world': points_camera[:, 0],  # X(forward) in your camera system
        'py_world': points_camera[:, 1],  # Y(left) in your camera system  
        'pz_world': points_camera[:, 2],  # Z(up) in your camera system
    }
    
    # Add optional columns if available
    if colors is not None:
        df_data['r'] = colors[:, 0]
        df_data['g'] = colors[:, 1] 
        df_data['b'] = colors[:, 2]
    
    if errors is not None:
        df_data['error'] = errors
    
    df = pd.DataFrame(df_data)
    
    # Save as compressed CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, compression='gzip', index=False)
    
    print(f"Saved {len(df)} points to {output_path}")
    print("Columns:", list(df.columns))


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP points3D.txt to semidense_points.csv.gz")
    parser.add_argument("--input_path", type=Path, help="Path to COLMAP points3D.txt file")
    parser.add_argument("--output_path", type=Path, help="Path to output semidense_points.csv.gz file")
    parser.add_argument("--include-colors", action="store_true", help="Include RGB color information")
    parser.add_argument("--include-errors", action="store_true", help="Include reprojection error information")
    parser.add_argument("--max-points", type=int, default=None, help="Limit number of points (for testing)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    # Read COLMAP points
    points_colmap, colors, errors = read_colmap_points3d(args.input_path)
    
    if len(points_colmap) == 0:
        raise ValueError("No valid 3D points found in COLMAP file")
    
    # Limit points if requested (useful for testing)
    if args.max_points is not None and len(points_colmap) > args.max_points:
        print(f"Limiting to {args.max_points} points for testing")
        indices = np.random.choice(len(points_colmap), args.max_points, replace=False)
        points_colmap = points_colmap[indices]
        colors = colors[indices] if colors is not None else None
        errors = errors[indices] if errors is not None else None
    
    # Transform coordinate system
    points_camera = transform_colmap_to_camera_system(points_colmap)
    
    # Save in semidense format
    save_semidense_points(
        points_camera, 
        args.output_path,
        colors if args.include_colors else None,
        errors if args.include_errors else None
    )
    
    print("Conversion complete!")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Points converted: {len(points_camera)}")


if __name__ == "__main__":
    # Example usage if run directly
    if len(__import__('sys').argv) == 1:
        print("Example usage:")
        print("python convert_colmap_points.py points3D.txt semidense_points.csv.gz --include-colors --include-errors")
        print()
        print("This script converts COLMAP points3D.txt to semidense_points.csv.gz format")
        print("with proper coordinate system transformation.")
    else:
        main()