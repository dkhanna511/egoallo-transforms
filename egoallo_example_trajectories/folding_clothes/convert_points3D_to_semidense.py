import numpy as np
import pandas as pd
import gzip
from pathlib import Path
from typing import Tuple, Optional
import argparse
import matplotlib.pyplot as plt

from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
from typing import Tuple, Optional
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

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

from mpl_toolkits.mplot3d import Axes3D



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
                residual_threshold=0.7  # 2cm threshold
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

    
def visualize_pointcloud(points: np.ndarray, title="Point Cloud (Aria World)"):
    """
    Visualize a 3D point cloud with matplotlib.
    
    Args:
        points: (N, 3) numpy array of points
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=2)
    
    # Axis labels (Aria world)
    ax.set_xlabel("X (down)")
    ax.set_ylabel("Y (left)")
    ax.set_zlabel("Z (forward)")
    ax.set_title(title)

    # Equal aspect ratio
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.show()


def transform_colmap_to_aria_world(points_colmap: np.ndarray) -> np.ndarray:
    """
    Transform COLMAP point cloud to Aria world coordinate system.
    
    COLMAP world: X(right), Y(down), Z(forward)
    Aria world: X(down), Y(left), Z(forward)
    
    Args:
        points_colmap: (N, 3) points in COLMAP world coordinates
    
    Returns:
        points_aria: (N, 3) points in Aria world coordinates
    """
    print("Transforming coordinate system: COLMAP world → Aria world")
    print("  COLMAP: X(right), Y(down), Z(forward)")
    print("  Aria:   X(down), Y(left), Z(forward)")
    
    # Transformation matrix: COLMAP world → Aria world
    # Aria x-down = +COLMAP y-down
    # Aria y-left = -COLMAP x-right  
    # Aria z-forward = +COLMAP z-forward
    R_aria_colmap = np.array([[0,  1,  0],  # Aria x-down = +COLMAP y-down
                              [1, 0,  0],  # Aria y-left = -COLMAP x-right  
                              [0,  0,  -1]]) # Aria z-forward = +COLMAP z-forward
    
    # Apply transformation
    points_aria = points_colmap @ R_aria_colmap.T
    # points_aria[:, 2] *= -1  # Flip X (down/up axis)
    points_aria = 0.15 * points_aria
     # Visualize
    # visualize_pointcloud(points_aria)
    # points_aria = R_aria_colmap @ points_colmap
    # exit(0)
    print(f"Transformed {len(points_aria)} points to Aria world coordinates")
    print(f"Aria world coordinate ranges:")
    print(f"  X(down): [{points_aria[:, 0].min():.3f}, {points_aria[:, 0].max():.3f}]")
    print(f"  Y(left): [{points_aria[:, 1].min():.3f}, {points_aria[:, 1].max():.3f}]")
    print(f"  Z(forward): [{points_aria[:, 2].min():.3f}, {points_aria[:, 2].max():.3f}]")
    planes = detect_multiple_planes(points_aria, max_planes=10, min_points_per_plane=200)
    
    if not planes:
        print("No planes detected! Using fallback method.")
        # Fallback: assume ground is at the 10th percentile of the lowest coordinate
        coord_ranges = [
            (points_aria[:, 0].max() - points_aria[:, 0].min(), 0),  # X range
            (points_aria[:, 1].max() - points_aria[:, 1].min(), 1),  # Y range  
            (points_aria[:, 2].max() - points_aria[:, 2].min(), 2)   # Z range
        ]
        # The axis with largest range is likely horizontal, smallest might be vertical
        coord_ranges.sort(reverse=True)
        vertical_axis = coord_ranges[-1][1]  # Axis with smallest range
        ground_level = np.percentile(points_aria[:, vertical_axis], 10)
        # return points_aria, ground_level
    
    # Identify which plane is the ground
    print("\nIdentifying ground plane...")
    ground_plane, ground_level = identify_ground_plane(planes, points_aria)
    
    if ground_plane is None:
        print("Could not identify ground plane!")
        ground_level = np.percentile(points_aria[:, 2], 10)  # Fallback
    
    # Visualization
    # if visualize:
    visualize_planes(points_aria, planes, ground_plane)
    
    return points_aria


def save_semidense_points(points_aria: np.ndarray, output_path: Path, 
                         colors: Optional[np.ndarray] = None, 
                         errors: Optional[np.ndarray] = None) -> None:
    """
    Save points in semidense_points.csv.gz format expected by EgoAllo/Aria.
    
    Args:
        points_aria: (N, 3) points in Aria world coordinate system
        output_path: Path to save semidense_points.csv.gz
        colors: Optional (N, 3) RGB colors
        errors: Optional (N,) reprojection errors
    """
    print(f"Saving semidense_points.csv.gz to: {output_path}")
    
    # Create DataFrame with required columns in Aria world coordinates
    df_data = {
        'px_world': points_aria[:, 0],  # X(down) in Aria world system
        'py_world': points_aria[:, 1],  # Y(left) in Aria world system  
        'pz_world': points_aria[:, 2],  # Z(forward) in Aria world system
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
    print("Coordinate system: Aria world (X-down, Y-left, Z-forward)")


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP points3D.txt to Aria-compatible semidense_points.csv.gz")
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
    
    # Transform coordinate system: COLMAP world → Aria world
    points_aria = transform_colmap_to_aria_world(points_colmap)
    
    # Save in semidense format
    save_semidense_points(
        points_aria, 
        args.output_path,
        colors if args.include_colors else None,
        errors if args.include_errors else None
    )
    
    print("\nConversion complete!")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Points converted: {len(points_colmap)}")
    print("Coordinate transformation: COLMAP world → Aria world")
    print("  COLMAP: X(right), Y(down), Z(forward)")
    print("  Aria:   X(down), Y(left), Z(forward)")


if __name__ == "__main__":
    # Example usage if run directly
    if len(__import__('sys').argv) == 1:
        print("Example usage:")
        print("python convert_colmap_points_aria.py --input_path points3D.txt --output_path semidense_points.csv.gz --include-colors --include-errors")
        print()
        print("This script converts COLMAP points3D.txt to Aria-compatible semidense_points.csv.gz format")
        print("with coordinate system transformation from COLMAP world to Aria world coordinates.")
        print()
        print("Coordinate Systems:")
        print("  COLMAP world: X(right), Y(down), Z(forward)")
        print("  Aria world:   X(down), Y(left), Z(forward)")
    else:
        main()