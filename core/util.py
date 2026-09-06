"""
Utility functions module

Helper functions used by SceneCtx, including:
- Geometry computation
- Wall processing
- Polygon computation
- Mesh loading and processing
"""

import numpy as np
import math
import trimesh
from shapely.geometry import Polygon
from typing import List, Dict, Tuple, Optional, Any, Set
from scipy.spatial import ConvexHull
import uuid
import os
import json
import hashlib
import colorsys
import time
from datetime import datetime


def log_timestamp() -> str:
    """Wall-clock timestamp for render progress logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_image_array(path: str) -> np.ndarray:
    """Read an image file into a numpy array.

    Uses Pillow directly so versions like ``9.5.0.post2`` do not break imageio's
    Pillow plugin version parser (``ValueError: invalid literal for int() ... 'post2'``).
    """
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img)


def write_image_array(path: str, arr: np.ndarray) -> None:
    """Write a numpy image array to disk via Pillow."""
    from PIL import Image

    Image.fromarray(np.asarray(arr)).save(path)


def allocate_millis_stamp(exclude=None) -> str:
    """Allocate a millisecond timestamp string; stamps in exclude are skipped (separates dir names from image names)."""
    blocked = set(exclude or ())
    while True:
        stamp = str(int(time.time() * 1000))
        if stamp not in blocked:
            return stamp
        time.sleep(0.001)


def allocate_view_output_dir(
    output_root: str,
    view_name: Optional[str] = None,
    *,
    sequence: bool = False,
) -> Tuple[str, str]:
    """Allocate view output directory: fixed name for topdown, millisecond timestamp otherwise. Returns (view_dir, dir_stamp).

    When sequence=True, directory name is ``{timestamp}_seq`` (camera pose sequence).
    dir_stamp identifies one render_view call; per-frame png names use allocate_millis_stamp(exclude={dir_stamp}).
    """
    os.makedirs(output_root, exist_ok=True)
    if view_name == "topdown":
        view_dir = os.path.join(output_root, "topdown")
        os.makedirs(view_dir, exist_ok=True)
        return view_dir, "topdown"
    while True:
        stamp = str(int(time.time() * 1000))
        dir_name = f"{stamp}_seq" if sequence else stamp
        view_dir = os.path.join(output_root, dir_name)
        if not os.path.exists(view_dir):
            os.makedirs(view_dir, exist_ok=True)
            return view_dir, dir_name
        time.sleep(0.001)


def resolve_topdown_image_path(output_root: str) -> str:
    """Top-down view: create topdown/ under output_root; single frame uses topdown.png."""
    view_dir, _ = allocate_view_output_dir(output_root, "topdown")
    return os.path.join(view_dir, "topdown.png")


def resolve_view_image_path(output_root: str, view_dir_name: Optional[str] = None) -> str:
    """Single view: allocate ``{view_dir_name or millisecond timestamp}/{image_stamp}.png`` path."""
    if view_dir_name:
        view_dir = os.path.join(output_root, view_dir_name)
        os.makedirs(view_dir, exist_ok=True)
        dir_stamp = view_dir_name
    else:
        view_dir, dir_stamp = allocate_view_output_dir(output_root)
    image_stamp = allocate_millis_stamp(exclude={dir_stamp})
    return os.path.join(view_dir, f"{image_stamp}.png")


def allocate_sequence_view_output_dir(
    output_root: str,
    view_dir_name: Optional[str] = None,
) -> Tuple[str, str]:
    """Sequence view directory: use view_dir_name if given, else ``{timestamp}_seq``."""
    os.makedirs(output_root, exist_ok=True)
    if view_dir_name:
        view_dir = os.path.join(output_root, view_dir_name)
        os.makedirs(view_dir, exist_ok=True)
        return view_dir, view_dir_name
    return allocate_view_output_dir(output_root, sequence=True)


SEMANTIC_BACKGROUND = (0, 0, 0)



def read_jsonl_line(file_path, line_number):
    """
    Read line n (0-based) from a JSONL file
    
    Args:
        file_path: Path to JSONL file
        line_number: Line number (0-based)
    
    Returns:
        dict: Parsed JSON for that line
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_number:
                return json.loads(line.strip())
    
    raise ValueError(f"Line number {line_number} is out of file range")



def generate_unique_id() -> str:
    """
    Generate a unique ID

    Purpose: unique identifier for walls, doors, windows, furniture, etc.
    Implementation: 4-character alphanumeric ID
    """
    import random
    import string
    chars = string.ascii_letters + string.digits  # upper/lower letters and digits
    return ''.join(random.choices(chars, k=4))


def point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float],
                          line_end: Tuple[float, float], clamp: bool = True) -> Tuple[float, Tuple[float, float]]:
    """
    Distance from point to segment or line, and closest point

    Purpose: snap door/window center to wall in add_door/add_window
    Implementation: vector projection onto the line

    Args:
        point: Target point (x, y)
        line_start: Start (x, y)
        line_end: End (x, y)
        clamp: Clamp projection to segment (True) or infinite line (False)

    Returns:
        (distance, closest point)
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Segment direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Vector from start to target point
    px = x0 - x1
    py = y0 - y1

    # Project onto segment
    if dx == 0 and dy == 0:
        return np.sqrt(px*px + py*py), (x1, y1)

    # Projection parameter t
    t = (px * dx + py * dy) / (dx * dx + dy * dy)
    
    if clamp:
        t = max(0, min(1, t))
        
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    distance = np.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)
    return distance, (closest_x, closest_y)


def calculate_wall_orientation(wall_start: Tuple[float, float], wall_end: Tuple[float, float],
                               vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Wall orientation ( inward-facing normal )

    Purpose: compute per-wall normal in add_walls for mesh generation
    Implementation:
        1. Compute two perpendicular directions (candidate normals)
        2. Offset slightly from wall center along each direction
        3. Test whether offset points lie inside floor polygon
        4. Pick the direction inside the polygon as inward normal

    Args:
        wall_start: Wall start
        wall_end: Wall end
        vertices: Room vertices (floor polygon, concave OK)

    Returns:
        Unit normal (nx, ny) pointing inward
    """
    dx = wall_end[0] - wall_start[0]
    dy = wall_end[1] - wall_start[1]
    length = np.sqrt(dx*dx + dy*dy)

    if length == 0:
        return (0, 0)

    # Two candidate normals (perpendicular to wall)
    normal1 = (-dy/length, dx/length)
    normal2 = (dy/length, -dx/length)

    # Wall center
    wall_center = ((wall_start[0] + wall_end[0])/2, (wall_start[1] + wall_end[1])/2)

    # Test offset from wall center along normal
    test_offset = 0.01  # 1 cm offset for testing

    # Two test points
    test_point1 = (
        wall_center[0] + normal1[0] * test_offset,
        wall_center[1] + normal1[1] * test_offset
    )
    test_point2 = (
        wall_center[0] + normal2[0] * test_offset,
        wall_center[1] + normal2[1] * test_offset
    )

    # Check which test point is inside polygon
    in_polygon1 = point_in_polygon(test_point1, vertices)
    in_polygon2 = point_in_polygon(test_point2, vertices)

    # Choose normal from test results
    if in_polygon1 and not in_polygon2:
        # Only normal1 points inward
        return normal1
    elif in_polygon2 and not in_polygon1:
        # Only normal2 points inward
        return normal2
    else:
        # Both inside or both outside (edge case)
        # Default to first normal
        # Other strategies possible
        return normal1


def try_find_closed_loop(walls: List[Dict]) -> Optional[List[Tuple[float, float]]]:
    """
    Test whether walls form a single closed simple loop.
    
    Algorithm:
    1. Count vertex degree.
    2. Degree 2 at every vertex implies one or more closed loops.
    3. Walk walls from a start vertex; check full coverage and return to start.
    
    Returns:
    Ordered vertex list if single closed loop; else None.
    """
    if not walls:
        return None
        
    # Build adjacency graph
    graph = {}
    for wall_idx, wall in enumerate(walls):
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        
        if s not in graph: graph[s] = []
        if e not in graph: graph[e] = []
        
        graph[s].append((wall_idx, e))
        graph[e].append((wall_idx, s))
        
    # 1. Every vertex degree must be 2 (simple loop, no branch, no dangling)
    for vertex, edges in graph.items():
        if len(edges) != 2:
            return None
            
    # 2. Walk the loop
    start_vertex = next(iter(graph.keys()))
    current_vertex = start_vertex
    visited_walls = set()
    ordered_vertices = []
    
    while True:
        ordered_vertices.append(current_vertex)
        edges = graph[current_vertex]
        
        # Find next unvisited wall
        next_edge = None
        for wall_idx, neighbor in edges:
            if wall_idx not in visited_walls:
                next_edge = (wall_idx, neighbor)
                break
        
        if not next_edge:
            break
            
        wall_idx, neighbor = next_edge
        visited_walls.add(wall_idx)
        current_vertex = neighbor
        
        if current_vertex == start_vertex:
            break
            
    # 3. Must visit all walls (single connected loop)
    if len(visited_walls) == len(walls):
        return ordered_vertices
        
    return None


def calculate_minimum_area_polygon_and_partitions(walls: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float, float]]]:
    """
    Minimum-area polygon enclosing all walls (concave OK) and interior partition walls
    
    Returns:
        (vertices, partitions)
        vertices: [(x1, y1), (x2, y2), ...] polygon vertices
        partitions: [(xs, ys, xe, ye, height), ...] interior partitions with height
    """
    # Try closed loop first
    closed_loop = try_find_closed_loop(walls)
    if closed_loop:
        # Closed loop => empty partitions
        print(f"✅ Input walls form a closed loop; using that order directly ({len(closed_loop)} vertices)")
        return closed_loop, []

    # Collect and dedupe endpoints
    points = []
    for wall in walls:
        points.extend([tuple(wall["s"]), tuple(wall["e"])])
    unique_points = list(dict.fromkeys(points))

    if len(unique_points) < 3:
        return unique_points, []

    # Compute concave polygon
    vertices = calculate_concave_polygon_from_walls(unique_points, walls)
    
    # Extract interior partitions
    partitions = []
    for wall in walls:
        # Check if wall lies on polygon boundary
        is_on_boundary = False
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            if is_wall_on_edge(wall, v1, v2):
                is_on_boundary = True
                break
        
        if not is_on_boundary:
            # Partition: [xs, ys, xe, ye, height]
            partitions.append((
                float(wall["s"][0]), float(wall["s"][1]), 
                float(wall["e"][0]), float(wall["e"][1]),
                float(wall["height"])
            ))
            
    return vertices, partitions


def calculate_concave_polygon_from_walls(points: List[Tuple[float, float]], walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    Compute concave polygon from walls

    Steps:
    1. Convex hull
    2. Check each hull edge against walls
    3. Replace non-collinear edges with wall paths for concavity

    Args:
        points: Unique wall endpoints
        walls: Wall list

    Returns:
        Polygon vertices (possibly concave)
    """
    # Step 1: convex hull
    points_array = np.array(points)
    hull = ConvexHull(points_array)
    convex_polygon = [points[i] for i in hull.vertices]
    
    if len(convex_polygon) < 3:
        return convex_polygon
    
    # Step 2: wall graph for path search
    wall_graph = build_wall_graph(walls)
    
    # Step 3: mark walls collinear with convex hull edges
    walls_on_convex = set()  # wall indices on convex hull edges
    
    for i in range(len(convex_polygon)):
        v1 = convex_polygon[i]
        v2 = convex_polygon[(i + 1) % len(convex_polygon)]
        
        # Walls collinear with this edge
        for wall_idx, wall in enumerate(walls):
            if is_wall_on_edge(wall, v1, v2):
                walls_on_convex.add(wall_idx)
    
    # Step 4: replace hull edges with wall paths
    final_polygon = []
    all_original_points = points # unique_points
    
    for i in range(len(convex_polygon)):
        start_vertex = convex_polygon[i]
        end_vertex = convex_polygon[(i + 1) % len(convex_polygon)]
        
        # Check if edge fully covered by walls
        is_fully_covered = is_edge_fully_covered(start_vertex, end_vertex, walls)
        
        # Append start vertex
        final_polygon.append(start_vertex)
        
        # If not fully covered, try wall path
        if not is_fully_covered:
            # Wall path from start_vertex to end_vertex
            # Search only walls not already on convex hull
            available_walls = [w for idx, w in enumerate(walls) if idx not in walls_on_convex]
            wall_path = find_wall_path(start_vertex, end_vertex, available_walls, wall_graph)
            
            if wall_path and len(wall_path) > 2:
                # Found wall path
                # Verify all points remain enclosed
                # Temporary polygon for enclosure test
                temp_polygon = []
                # 1. Already fixed vertices
                temp_polygon.extend(final_polygon)
                # 2. Intermediate path vertices
                temp_polygon.extend(wall_path[1:-1])
                # 3. Remaining hull edges for closed test loop
                for k in range(i + 1, len(convex_polygon)):
                    temp_polygon.append(convex_polygon[k])
                
                # Check all original points enclosed
                all_enclosed = True
                for p in all_original_points:
                    if not is_point_in_or_on_polygon(p, temp_polygon):
                        all_enclosed = False
                        break
                
                if all_enclosed:
                    # Use path only if enclosure holds
                    for j in range(1, len(wall_path) - 1):
                        final_polygon.append(wall_path[j])
                    # print(f"  - edge {start_vertex}->{end_vertex} replaced by wall path with enclosure preserved")
                else:
                    pass
                    # print(f"  - candidate path for {start_vertex}->{end_vertex} fails enclosure, fallback to line")
    
    # Remove duplicate vertices
    final_polygon = remove_consecutive_duplicates(final_polygon)
    
    # At least 3 vertices
    if len(final_polygon) < 3:
        return convex_polygon
    
    # Must be simple (non self-intersecting)
    if not is_simple_polygon(final_polygon):
        # Self-intersection => fall back to convex hull
        print("⚠️  Generated concave polygon is self-intersecting; falling back to convex hull")
        return convex_polygon
    
    return final_polygon


def remove_consecutive_duplicates(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Remove consecutive duplicate vertices

    Args:
        vertices: Vertex list

    Returns:
        Deduplicated vertex list
    """
    if len(vertices) <= 1:
        return vertices
    
    result = [vertices[0]]
    for i in range(1, len(vertices)):
        # Small tolerance for float comparison
        if not (abs(vertices[i][0] - result[-1][0]) < 1e-6 and 
                abs(vertices[i][1] - result[-1][1]) < 1e-6):
            result.append(vertices[i])
    
    # Check first/last duplicate
    if len(result) > 1:
        if abs(result[0][0] - result[-1][0]) < 1e-6 and abs(result[0][1] - result[-1][1]) < 1e-6:
            result.pop()
    
    return result


def build_wall_graph(walls: List[Dict]) -> Dict[Tuple[float, float], List[Tuple[int, Tuple[float, float]]]]:
    """
    Build wall adjacency graph

    Dict: vertex -> [(wall index, neighbor vertex)]
    """
    graph = {}
    
    for wall_idx, wall in enumerate(walls):
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        
        if s not in graph:
            graph[s] = []
        if e not in graph:
            graph[e] = []
        
        graph[s].append((wall_idx, e))
        graph[e].append((wall_idx, s))
    
    return graph


def snap_wall_endpoints(walls: List[Dict], threshold: float = 0.3) -> List[Dict]:
    """
    Snap wall endpoints.
    For each endpoint, intersect the wall line with other wall segments;
    if distance < threshold, move endpoint to intersection.
    """
    if not walls:
        return walls
        
    new_walls = []
    # Convert to numeric arrays
    for w in walls:
        new_walls.append({
            "s": np.array(w["s"][:2], dtype=np.float64),
            "e": np.array(w["e"][:2], dtype=np.float64),
            "height": w["height"]
        })
        
    for i in range(len(new_walls)):
        for key in ["s", "e"]:
            curr_p = new_walls[i][key]
            # Wall direction vector
            other_key = "e" if key == "s" else "s"
            v = new_walls[i][other_key] - curr_p
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-6:
                continue
            dir_v = v / norm_v
            
            best_snap_p = None
            min_dist = threshold
            
            # Intersections with all other walls
            for j in range(len(new_walls)):
                if i == j:
                    continue
                
                # Wall j segment A-B
                A = new_walls[j]["s"]
                B = new_walls[j]["e"]
                
                # Line (curr_p, dir_v) vs segment AB
                # Line: P = curr_p + t * dir_v
                # Segment: P = A + u * (B - A), 0 <= u <= 1
                # Solve curr_p + t * dir_v = A + u * (B - A)
                # t * dir_v - u * (B - A) = A - curr_p
                
                W = B - A
                det = dir_v[0] * (-W[1]) - dir_v[1] * (-W[0])
                
                if abs(det) < 1e-6: # parallel or collinear
                    # If collinear, snap to nearest point on segment
                    if are_collinear(tuple(curr_p), tuple(A), tuple(B)):
                        tA = np.dot(A - curr_p, dir_v)
                        tB = np.dot(B - curr_p, dir_v)
                        t_min = min(tA, tB)
                        t_max = max(tA, tB)
                        # Project t=0 (current endpoint) onto segment range
                        t_clamp = min(max(0.0, t_min), t_max)
                        dist = abs(t_clamp)
                        if dist < min_dist:
                            min_dist = dist
                            best_snap_p = curr_p + t_clamp * dir_v
                    continue
                
                rhs = A - curr_p
                t = (rhs[0] * (-W[1]) - rhs[1] * (-W[0])) / det
                u = (dir_v[0] * rhs[1] - dir_v[1] * rhs[0]) / det
                
                if 0 <= u <= 1:
                    dist = abs(t)
                    if dist < min_dist:
                        min_dist = dist
                        best_snap_p = curr_p + t * dir_v
            
            if best_snap_p is not None:
                new_walls[i][key] = best_snap_p
                
    # Convert back to original format
    result = []
    for w in new_walls:
        result.append({
            "s": [float(w["s"][0]), float(w["s"][1])],
            "e": [float(w["e"][0]), float(w["e"][1])],
            "height": float(w["height"])
        })
    return result


def is_wall_on_edge(wall: Dict, edge_start: Tuple[float, float], edge_end: Tuple[float, float]) -> bool:
    """
    Check if wall lies on convex hull edge (collinear and within edge)
    """
    wall_s = tuple(wall["s"])
    wall_e = tuple(wall["e"])
    
    # Both endpoints on edge
    if not (point_on_segment(wall_s, edge_start, edge_end) and 
            point_on_segment(wall_e, edge_start, edge_end)):
        return False
    
    # Three-point collinearity
    return are_collinear(edge_start, edge_end, wall_s) and are_collinear(edge_start, edge_end, wall_e)


def is_edge_fully_covered(edge_start: Tuple[float, float], edge_end: Tuple[float, float], walls: List[Dict]) -> bool:
    """
    Check if a convex hull edge is fully covered by walls.
    Collect wall segments on edge, merge intervals, verify full coverage.
    """
    on_edge_walls = []
    for wall in walls:
        if is_wall_on_edge(wall, edge_start, edge_end):
            # Project endpoints to 1D distance from edge_start
            d1 = np.sqrt((wall["s"][0] - edge_start[0])**2 + (wall["s"][1] - edge_start[1])**2)
            d2 = np.sqrt((wall["e"][0] - edge_start[0])**2 + (wall["e"][1] - edge_start[1])**2)
            on_edge_walls.append((min(d1, d2), max(d1, d2)))
    
    if not on_edge_walls:
        return False
    
    # Merge intervals
    on_edge_walls.sort()
    merged = []
    if on_edge_walls:
        curr_start, curr_end = on_edge_walls[0]
        for next_start, next_end in on_edge_walls[1:]:
            if next_start <= curr_end + 1e-6: # tolerance
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))
    
    # Check full coverage
    total_dist = np.sqrt((edge_end[0] - edge_start[0])**2 + (edge_end[1] - edge_start[1])**2)
    
    # First interval starts at 0, last ends at total_dist
    if not merged:
        return False
    
    return merged[0][0] < 1e-6 and merged[-1][1] > total_dist - 1e-6


def is_point_in_or_on_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if point is inside or on polygon boundary
    """
    if point_in_polygon(point, polygon):
        return True
    
    # Check each edge
    for i in range(len(polygon)):
        if point_on_segment(point, polygon[i], polygon[(i + 1) % len(polygon)]):
            return True
    return False


def are_collinear(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> bool:
    """
    Check three-point collinearity

    Cross product: (p2-p1) x (p3-p1) = 0
    """
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]
    
    cross_product = dx1 * dy2 - dy1 * dx2
    return abs(cross_product) < 1e-6


def find_wall_path(start: Tuple[float, float], end: Tuple[float, float], 
                   available_walls: List[Dict], wall_graph: Dict) -> List[Tuple[float, float]]:
    """
    BFS wall path from start to end among available walls

    Args:
        start: Start vertex
        end: End vertex
        available_walls: Usable walls
        wall_graph: Wall graph

    Returns:
        Vertex path or None
    """
    from collections import deque
    
    # Available wall edge set for lookup
    available_edges = set()
    for wall in available_walls:
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        available_edges.add((s, e))
        available_edges.add((e, s))  # bidirectional
    
    # BFS
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        # Target reached
        if current == end:
            return path
        
        # Neighbors of current vertex
        if current in wall_graph:
            for wall_idx, neighbor in wall_graph[current]:
                edge = (current, neighbor)
                
                # Use available edges only; skip visited
                if edge in available_edges and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    
    # No path
    return None


def brute_force_minimum_polygon(points: List[Tuple[float, float]], walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    Brute-force minimum-area polygon

    Purpose: sub-routine for small point sets
    Implementation: enumerate vertex sets/orderings; minimize area while enclosing walls

    Complexity: O(2^n * n!); small n only
    """
    from itertools import combinations

    min_area = float('inf')
    best_polygon = None

    # Try vertex counts >= 3
    for num_vertices in range(3, len(points) + 1):
        for vertex_combination in combinations(points, num_vertices):
            # Best vertex ordering
            polygon = find_best_ordering(list(vertex_combination))
            if polygon and polygon_contains_all_walls(polygon, walls):
                area = polygon_area(polygon)
                if area < min_area:
                    min_area = area
                    best_polygon = polygon

    return best_polygon if best_polygon else points


def heuristic_minimum_polygon(points: List[Tuple[float, float]], walls: List[Dict]) -> List[Tuple[float, float]]:
    """
    Heuristic minimum polygon

    Purpose: sub-routine for large point sets
    Implementation:
        1. Convex hull as initial solution
        2. Drop vertices while walls remain enclosed
    """
    points_array = np.array(points)
    hull = ConvexHull(points_array)
    hull_points = [points[i] for i in hull.vertices]

    # Optimize polygon
    optimized = optimize_polygon(hull_points, walls, points)
    return optimized if optimized else hull_points


def find_best_ordering(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Best vertex order for a simple (non self-intersecting) polygon

    Purpose: sub-routine of brute_force_minimum_polygon
    Implementation:
        1. Angle from centroid
        2. Sort by angle
        3. Test simplicity; fall back to hull order
    """
    if len(vertices) < 3:
        return vertices

    # Centroid
    centroid_x = sum(p[0] for p in vertices) / len(vertices)
    centroid_y = sum(p[1] for p in vertices) / len(vertices)

    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid_y, point[0] - centroid_x)

    sorted_vertices = sorted(vertices, key=angle_from_centroid)

    # Simple polygon test
    if is_simple_polygon(sorted_vertices):
        return sorted_vertices

    # Else convex hull order
    try:
        points_array = np.array(vertices)
        hull = ConvexHull(points_array)
        return [vertices[i] for i in hull.vertices]
    except:
        return sorted_vertices


def is_simple_polygon(vertices: List[Tuple[float, float]]) -> bool:
    """
    Test simple polygon (no edge self-intersection)

    Purpose: sub-routine of find_best_ordering
    Implementation: test all edge pairs
    """
    n = len(vertices)
    if n < 3:
        return True

    # Edge intersection test
    for i in range(n):
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:  # skip adjacent edges
                continue
            if segments_intersect(vertices[i], vertices[(i + 1) % n],
                                vertices[j], vertices[(j + 1) % n]):
                return False
    return True


def segments_intersect(p1: Tuple[float, float], q1: Tuple[float, float],
                      p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """
    Test segment intersection

    Purpose: sub-routine of is_simple_polygon
    Implementation: orientation test
    """
    def orientation(p, q, r):
        """Orientation of three points: 0=collinear, 1=CW, 2=CCW"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        """True if q on segment pr (assumes collinear)"""
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case: segments straddle
    if o1 != o2 and o3 != o4:
        return True

    # Special case: point on segment
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def polygon_contains_all_walls(polygon: List[Tuple[float, float]], walls: List[Dict]) -> bool:
    """
    Test whether polygon encloses all walls

    Purpose: sub-routine of brute_force_minimum_polygon; validate candidate polygon
    Implementation: both wall endpoints inside or on boundary
    """
    for wall in walls:
        if not (point_in_or_on_polygon(tuple(wall["s"]), polygon) and
                point_in_or_on_polygon(tuple(wall["e"]), polygon)):
            return False
    return True


def point_in_or_on_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if point is inside or on polygon boundary

    Purpose: sub-routine of polygon_contains_all_walls
    Implementation: edge test then ray casting
    """
    # On boundary edge
    for i in range(len(polygon)):
        seg_start = polygon[i]
        seg_end = polygon[(i + 1) % len(polygon)]
        if point_on_segment(point, seg_start, seg_end):
            return True

    # Inside (ray casting)
    return point_in_polygon(point, polygon)


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting point-in-polygon test

    Purpose: sub-routine of point_in_or_on_polygon
    Implementation: ray to +X; odd intersection count => inside
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_on_segment(point: Tuple[float, float], seg_start: Tuple[float, float],
                    seg_end: Tuple[float, float]) -> bool:
    """
    Test point on segment

    Purpose: sub-routine of point_in_or_on_polygon
    Implementation: dist to endpoints sums to segment length
    """
    dist_to_start = np.sqrt((point[0] - seg_start[0])**2 + (point[1] - seg_start[1])**2)
    dist_to_end = np.sqrt((point[0] - seg_end[0])**2 + (point[1] - seg_end[1])**2)
    seg_length = np.sqrt((seg_end[0] - seg_start[0])**2 + (seg_end[1] - seg_start[1])**2)

    # Tolerance for float precision
    return abs(dist_to_start + dist_to_end - seg_length) < 1e-6


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """
    Polygon area

    Purpose: compare polygons in brute_force_minimum_polygon
    Implementation: shoelace formula

    Formula: Area = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    if len(polygon) < 3:
        return 0

    area = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2


def optimize_polygon(polygon: List[Tuple[float, float]], walls: List[Dict],
                    all_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Optimize polygon by removing unnecessary vertices

    Purpose: reduce hull vertices in heuristic_minimum_polygon
    Implementation: try removing each vertex if walls still enclosed
    """
    optimized = polygon.copy()
    changed = True

    while changed and len(optimized) > 3:
        changed = False
        for i in range(len(optimized)):
            # Try removing vertex i
            test_polygon = optimized[:i] + optimized[i+1:]
            if polygon_contains_all_walls(test_polygon, walls):
                optimized = test_polygon
                changed = True
                break

    return optimized


def find_closest_wall(center: List[float], walls_dict: Dict) -> str:
    """
    Find wall closest to a center point

    Purpose: auto-attach door/window to nearest wall
    Implementation: min distance over all walls

    Args:
        center: Door/window center [x, y, z]
        walls_dict: Walls {wall_id: wall_data}

    Returns:
        Closest wall unique_id
    """
    point = (center[0], center[1])
    min_distance = float('inf')
    closest_wall_id = None

    for wall_id, wall in walls_dict.items():
        distance, _ = point_to_line_distance(point, tuple(wall["s"]), tuple(wall["e"]))
        if distance < min_distance:
            min_distance = distance
            closest_wall_id = wall_id

    return closest_wall_id


def find_closest_wall_from_list(center: List[float], walls_list: List[Tuple]) -> str:
    """
    Closest wall from a wall list
    
    Args:
        center: Point center [x, y, z]
        walls_list: [(wall_id, p, q), ...]
    
    Returns:
        Closest wall ID
    """
    min_distance = float('inf')
    closest_wall_id = None
    
    point = np.array(center[:2])  # XY only
    
    for wall_id, p, q in walls_list:
        # Point-to-segment distance
        wall_start = np.array(p[:2])
        wall_end = np.array(q[:2])
        
        # Segment vector
        wall_vec = wall_end - wall_start
        wall_length_sq = np.dot(wall_vec, wall_vec)
        
        if wall_length_sq == 0:
            # Degenerate wall (point)
            distance = np.linalg.norm(point - wall_start)
        else:
            # Projection parameter t
            t = max(0, min(1, np.dot(point - wall_start, wall_vec) / wall_length_sq))
            projection = wall_start + t * wall_vec
            distance = np.linalg.norm(point - projection)
        
        if distance < min_distance:
            min_distance = distance
            closest_wall_id = wall_id
    
    return closest_wall_id


def snap_to_wall(center: List[float], wall: Dict) -> List[float]:
    """
    Snap point onto wall line (not clamped to segment)

    Purpose: project door/window center onto wall line even if slightly beyond segment
    Implementation: point_to_line_distance with clamp=False; preserve z

    Args:
        center: Original center [x, y, z]
        wall: Wall data with s and e

    Returns:
        Snapped center [x', y', z]
    """
    point = (center[0], center[1])
    # clamp=False: use line projection when outside segment
    _, closest_point = point_to_line_distance(point, tuple(wall["s"]), tuple(wall["e"]), clamp=False)
    return [closest_point[0], closest_point[1], center[2]]


def calculate_miter_joints(walls: Dict[str, Any], wall_thickness: float) -> Dict[str, Any]:
    """Precompute outer base points for exterior wall miter joints."""
    boundary_walls = {wid: w for wid, w in walls.items() if not w.get("is_partition", False)}

    pt_to_walls: Dict[Tuple[float, float], List[str]] = {}
    for wall_id, wall in boundary_walls.items():
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        for pt in (s, e):
            pt_to_walls.setdefault(pt, []).append(wall_id)

    wall_out_normals = {
        wall_id: -np.array(wall["orientation"], dtype=float)[:2]
        for wall_id, wall in boundary_walls.items()
    }

    def _get_miter_point(pt, wid1, wid2):
        n1 = wall_out_normals[wid1]
        n2 = wall_out_normals[wid2]
        n_avg = n1 + n2
        n_avg_norm = np.linalg.norm(n_avg)
        if n_avg_norm < 1e-4:
            return np.array(pt, dtype=float)[:2] + n1 * wall_thickness
        n_avg /= n_avg_norm
        cos_half_theta = np.dot(n_avg, n1)
        if abs(cos_half_theta) < 1e-4:
            return np.array(pt, dtype=float)[:2] + n1 * wall_thickness
        length = wall_thickness / cos_half_theta
        length = min(length, wall_thickness * 10)
        return np.array(pt, dtype=float)[:2] + n_avg * length

    wall_outer_points = {}
    for wall_id, wall in boundary_walls.items():
        s = tuple(wall["s"])
        e = tuple(wall["e"])
        outer_s, outer_e = None, None
        neighbors_s = [wid for wid in pt_to_walls.get(s, []) if wid != wall_id]
        if neighbors_s:
            outer_s = _get_miter_point(s, wall_id, neighbors_s[0])
        neighbors_e = [wid for wid in pt_to_walls.get(e, []) if wid != wall_id]
        if neighbors_e:
            outer_e = _get_miter_point(e, wall_id, neighbors_e[0])
        wall_outer_points[wall_id] = (outer_s, outer_e)
    return wall_outer_points


def _walls_at_vertex(vertex, boundary_walls: Dict[str, Any], tol: float = 1e-5) -> List[str]:
    """Return exterior wall ids meeting at vertex."""
    v = np.array(vertex[:2], dtype=float)
    matched = []
    for wall_id, wall in boundary_walls.items():
        for pt in (wall["s"], wall["e"]):
            if np.linalg.norm(v - np.array(pt[:2], dtype=float)) < tol:
                matched.append(wall_id)
                break
    return matched


def calculate_floor_polygon_with_wall_thickness(
    vertices: List[Tuple[float, float]],
    walls: Dict[str, Any],
    wall_thickness: float,
) -> List[Tuple[float, float]]:
    """
    Floor/ceiling horizontal outline from exterior wall base (thickness + miters).

    meta.vertices is inner wall loop; floor/ceiling slab extends to outer base
    so walls sit fully on the slab.
    """
    if len(vertices) < 3:
        return [(float(v[0]), float(v[1])) for v in vertices]

    boundary_walls = {wid: w for wid, w in walls.items() if not w.get("is_partition", False)}
    if not boundary_walls:
        return [(float(v[0]), float(v[1])) for v in vertices]

    wall_outer_points = calculate_miter_joints(boundary_walls, wall_thickness)
    wall_out_normals = {
        wall_id: -np.array(wall["orientation"], dtype=float)[:2]
        for wall_id, wall in boundary_walls.items()
    }

    def _miter_at_corner(pt, wid1: str, wid2: str) -> np.ndarray:
        n1 = wall_out_normals[wid1]
        n2 = wall_out_normals[wid2]
        n_avg = n1 + n2
        n_avg_norm = np.linalg.norm(n_avg)
        base = np.array(pt[:2], dtype=float)
        if n_avg_norm < 1e-4:
            return base + n1 * wall_thickness
        n_avg /= n_avg_norm
        cos_half_theta = np.dot(n_avg, n1)
        if abs(cos_half_theta) < 1e-4:
            return base + n1 * wall_thickness
        length = min(wall_thickness / cos_half_theta, wall_thickness * 10)
        return base + n_avg * length

    floor_vertices: List[Tuple[float, float]] = []
    for vertex in vertices:
        meeting = _walls_at_vertex(vertex, boundary_walls)
        if len(meeting) >= 2:
            outer_pt = _miter_at_corner(vertex, meeting[0], meeting[1])
        elif len(meeting) == 1:
            wid = meeting[0]
            outer_pt = np.array(vertex[:2], dtype=float) + wall_out_normals[wid] * wall_thickness
            outer_s, outer_e = wall_outer_points.get(wid, (None, None))
            v_arr = np.array(vertex[:2], dtype=float)
            if outer_s is not None and np.linalg.norm(v_arr - np.array(boundary_walls[wid]["s"][:2])) < 1e-5:
                outer_pt = np.array(outer_s, dtype=float)
            elif outer_e is not None and np.linalg.norm(v_arr - np.array(boundary_walls[wid]["e"][:2])) < 1e-5:
                outer_pt = np.array(outer_e, dtype=float)
        else:
            outer_pt = np.array(vertex[:2], dtype=float)

        floor_vertices.append((float(outer_pt[0]), float(outer_pt[1])))

    return floor_vertices


def create_floor_mesh(vertices: List[Tuple[float, float]], bounds: List[float], 
                      texture_scale: float = 2.0) -> trimesh.Trimesh:
    """
    Create floor mesh (extrude solid with thickness)

    Args:
        vertices: Floor vertices [(x1, y1), ...]
        bounds: [x_min, y_min, x_max, y_max]
        texture_scale: UV scale, default 2.0

    Returns:
        trimesh.Trimesh with physical thickness
    """
    texture_scale = texture_scale if texture_scale  else 2.0
    if len(vertices) < 3:
        return trimesh.Trimesh()

    # 1. Extract 2D coords; ensure CCW winding
    pts2_raw = [(float(v[0]), float(v[1])) for v in vertices]
    pts_arr = np.array(pts2_raw)
    xs, ys = pts_arr[:, 0], pts_arr[:, 1]
    # Signed 2*area for winding
    area2 = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    if area2 < 0:
        pts2_raw = pts2_raw[::-1]

    # 2. Shapely Polygon (concave OK)
    poly = Polygon(pts2_raw)
    
    # 3. Extrude 0.1m downward
    # Auto-triangulation: top, bottom, sides
    # trimesh API: extrude_polygon
    floor_mesh = trimesh.creation.extrude_polygon(poly, height=-0.1)

    # 4. Planar UV mapping
    all_verts = floor_mesh.vertices
    
    # Physical X/Y spans
    x_length = bounds[2] - bounds[0]
    y_length = bounds[3] - bounds[1]
    max_length = max(x_length, y_length)
    
    # Normalize by max span to avoid stretch
    # Square texels even for rectangular rooms
    u = (all_verts[:, 0] - bounds[0]) / max_length * texture_scale
    v = (all_verts[:, 1] - bounds[1]) / max_length * texture_scale
    
    # Assign TextureVisuals
    floor_mesh.visual = trimesh.visual.TextureVisuals(uv=np.column_stack((u, v)))
    
    return floor_mesh


def create_ceiling_mesh(vertices: List[Tuple[float, float]], bounds: List[float], z_height: float) -> trimesh.Trimesh:
    """
    Create ceiling mesh (normals down; ear-clipping for concave polygons)
    
    Args:
        vertices: Same as floor [(x1, y1), ...]
        bounds: [x_min, y_min, x_max, y_max]
        z_height: Ceiling height (z)
    
    Returns:
        trimesh.Trimesh with UV; normals face -Z
    """
    if len(vertices) < 3:
        return trimesh.Trimesh()
    
    texture_scale = 2.0
    # Vertices in XY at z_height
    ceiling_verts = np.array([[v[0], v[1], z_height] for v in vertices], dtype=float)
    
    # CW from below so normals point -Z
    # Opposite of floor: flip if CCW
    if ceiling_verts.shape[0] >= 3:
        xs = ceiling_verts[:, 0]
        ys = ceiling_verts[:, 1]
        # Signed 2*area
        area2 = np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
        if area2 > 0:
            # Reverse to CW (from -Z) for downward normals
            ceiling_verts = ceiling_verts[::-1]
    
    # UV from current vertex order
    ceiling_uvs = [[(v[0] - bounds[0]) / (bounds[2] - bounds[0]) * texture_scale,
                    (v[1] - bounds[1]) / (bounds[3] - bounds[1]) * texture_scale] for v in ceiling_verts]
    
    # Ear-clipping triangulation for concave polygons
    def _is_convex(a, b, c):
        # Convex corner test for (a,b,c)
        # Invert for CW ceiling winding
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < -1e-9
    
    def _point_in_triangle(pt, a, b, c):
        # barycentric technique
        v0 = np.array(c) - np.array(a)
        v1 = np.array(b) - np.array(a)
        v2 = np.array(pt) - np.array(a)
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            return False
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1 + 1e-9)
    
    pts2 = [(float(x), float(y)) for x, y in ceiling_verts[:, :2]]
    idx_list = list(range(len(pts2)))
    ceiling_faces = []
    
    # Guard for degenerate small polygons
    if len(idx_list) == 3:
        ceiling_faces = [[0, 1, 2]]
    else:
        safety = 0
        while len(idx_list) > 3 and safety < 10000:
            made_cut = False
            n = len(idx_list)
            for i in range(n):
                i_prev = idx_list[(i - 1) % n]
                i_curr = idx_list[i]
                i_next = idx_list[(i + 1) % n]
                a = pts2[i_prev]
                b = pts2[i_curr]
                c = pts2[i_next]
                
                if not _is_convex(a, b, c):
                    continue
                
                # Any other vertex inside triangle
                any_inside = False
                for j in idx_list:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if _point_in_triangle(pts2[j], a, b, c):
                        any_inside = True
                        break
                
                if any_inside:
                    continue
                
                # Ear clip: remove b
                ceiling_faces.append([i_prev, i_curr, i_next])
                idx_list.remove(i_curr)
                made_cut = True
                break
            
            if not made_cut:
                # Fallback fan triangulation to avoid infinite loop
                ceiling_faces = [[0, i, i + 1] for i in range(1, len(pts2) - 1)]
                break
            safety += 1
        
        if len(idx_list) == 3:
            ceiling_faces.append([idx_list[0], idx_list[1], idx_list[2]])
    
    # process=False to keep our winding
    ceiling_mesh = trimesh.Trimesh(vertices=ceiling_verts, faces=np.array(ceiling_faces, dtype=int), process=False)
    
    # Set UV
    ceiling_mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(ceiling_uvs))
    
    return ceiling_mesh


def create_opening_box(center: List[float], width: float, height: float,
                      wall_orientation: List[float], wall_thickness: float) -> trimesh.Trimesh:
    """
    Create door/window box for boolean subtraction
    
    Args:
        center: Door/window center [x,y,z] (face mode center)
        width: Opening width
        height: Opening height
        wall_orientation: Inward wall normal
        wall_thickness: Wall thickness
    
    Returns:
        Positioned/rotated box mesh
    """
    # 1. Axis-aligned box (default orientation)
    # extents: [x_size, y_size, z_size]
    # Y is thickness direction
    box = trimesh.creation.box(extents=[width, wall_thickness + 0.02, height])
    
    # 2. Rotate box +Y normal to wall orientation
    # Wall orientation [nx,ny] -> 3D [nx,ny,0]
    target_normal = np.array([wall_orientation[0], wall_orientation[1], 0.0])
    target_normal /= np.linalg.norm(target_normal) + 1e-9
    
    # Default box normal +Y
    default_normal = np.array([0.0, 1.0, 0.0])
    
    # Rotation axis (cross) and angle (dot)
    rotation_axis = np.cross(default_normal, target_normal)
    rotation_angle = np.arccos(np.clip(np.dot(default_normal, target_normal), -1.0, 1.0))
    
    # Apply rotation if not parallel
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            rotation_angle, rotation_axis, point=[0, 0, 0]
        )
        box.apply_transform(rotation_matrix)
    
    # 3. Shift center outward (-orientation) by thickness/2
    adjusted_center = np.array(center) - np.array([
        wall_orientation[0] * wall_thickness / 2,
        wall_orientation[1] * wall_thickness / 2,
        0.0
    ])
    
    # 4. Translate to final position
    box.apply_translation(adjusted_center)
    
    return box


def create_single_wall_mesh(start: List[float], end: List[float], height: float, 
                            orientation: List[float], chip: bool = False,
                            openings: List[Dict] = None, wall_thickness: float = 0.1,
                            texture_scale: float = 2.0) -> trimesh.Trimesh:
    """
    Create mesh for one wall. Default: solid wall extruded outward with thickness.
    
    Args:
        start: Wall start [x, y]
        end: Wall end [x, y]
        height: Wall height
        orientation: Inward normal
        chip: If True, return legacy single-sided face
        openings: [{"center": [x,y,z], "width": w, "height": h}, ...]
        wall_thickness: Wall thickness in meters, default 0.1
        texture_scale: UV scale, default 2.0
    """
    if height == 0:
        height = 0.01
    texture_scale = texture_scale if texture_scale  else 2.0
    # Base wall vectors
    wall_vec = np.array(end[:2]) - np.array(start[:2])
    wall_length = np.linalg.norm(wall_vec)
    if wall_length == 0: wall_length = 1.0
    unit_wall_vec = wall_vec / wall_length

    if chip:
        # --- Legacy: single-sided face ---
        wall_verts = np.array([
            [start[0], start[1], 0.0],
            [end[0], end[1], 0.0],
            [end[0], end[1], height],
            [start[0], start[1], height]
        ], dtype=float)

        u_scale = wall_length / (height if height != 0 else 1.0)
        wall_uvs = np.array([[0.0, 0.0], [u_scale, 0.0], [u_scale, 1.0], [0.0, 1.0]])

        v0, v1, v2 = wall_verts[0], wall_verts[1], wall_verts[2]
        default_normal = np.cross(v1 - v0, v2 - v0)
        default_normal /= (np.linalg.norm(default_normal) + 1e-9)
        dot = np.dot(default_normal, np.array([orientation[0], orientation[1], 0.0]))

        wall_faces = np.array([[0, 1, 2], [0, 2, 3]] if dot > 0 else [[0, 2, 1], [0, 3, 2]], dtype=int)
        wall_mesh = trimesh.Trimesh(vertices=wall_verts, faces=wall_faces, process=False)
        wall_mesh.visual = trimesh.visual.TextureVisuals(uv=wall_uvs)
        return wall_mesh
    else:
        # --- Solid wall via extrude ---
        thickness = wall_thickness
        # Outward offset (-orientation)
        off_x, off_y = -orientation[0] * thickness, -orientation[1] * thickness
        
        # Footprint quad: p1,p2 inner base; p3,p4 outer base
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]), float(end[1]))
        p3 = (p2[0] + off_x, p2[1] + off_y)
        p4 = (p1[0] + off_x, p1[1] + off_y)
        
        # Extrude footprint polygon upward
        poly = Polygon([p1, p2, p3, p4])
        # trimesh API: extrude_polygon
        wall_mesh = trimesh.creation.extrude_polygon(poly, height=height)
        
        # --- Boolean openings ---
        if openings:
            for opening in openings:
                try:
                    opening_box = create_opening_box(
                        center=opening["center"],
                        width=opening["width"],
                        height=opening["height"],
                        wall_orientation=orientation,
                        wall_thickness=thickness
                    )
                    wall_mesh = wall_mesh.difference(opening_box)
                except Exception as e:
                    print(f"⚠️  Boolean operation failed (opening at {opening.get('center')}): {e}")
        
        # --- UV per project projection rules ---
        all_verts = wall_mesh.vertices
        
        # Normalize UV by max(wall_length, height)
        max_dimension = max(wall_length, height)
        
        # u: projection along wall from p1
        rel_vecs = all_verts[:, :2] - np.array(p1)
        u = (np.dot(rel_vecs, unit_wall_vec) / max_dimension) * texture_scale
        
        # v: height fraction on Z
        v = (all_verts[:, 2] / max_dimension) * texture_scale
        
        # Apply UV
        wall_mesh.visual = trimesh.visual.TextureVisuals(uv=np.column_stack((u, v)))
        
        return wall_mesh


def create_face_wall_mesh(vertices: List[Tuple[float, float]], height: float) -> trimesh.Trimesh:
    """
    Thin wall mesh (double-sided, with UV)

    Args:
        vertices: Wall loop CCW [(x1,y1), ...]
        height: Wall height

    Returns:
        trimesh.Trimesh visible from both sides with UV
    """
    n = len(vertices)
    if n < 3:
        return trimesh.Trimesh()

    # Bottom ring + top ring vertices
    mesh_vertices = []

    # Bottom ring z=0
    for v in vertices:
        mesh_vertices.append([v[0], v[1], 0.0])

    # Top ring z=height
    for v in vertices:
        mesh_vertices.append([v[0], v[1], height])

    # UV unwrap along wall perimeter
    # Cumulative edge length for U
    cumulative_length = [0.0]
    for i in range(n):
        i_next = (i + 1) % n
        v_curr = np.array(vertices[i])
        v_next = np.array(vertices[i_next])
        segment_length = np.linalg.norm(v_next - v_curr)
        cumulative_length.append(cumulative_length[-1] + segment_length)

    total_length = cumulative_length[-1]
    if total_length == 0:
        total_length = 1.0  # avoid div by zero

    # Per-vertex UV
    # U: normalized perimeter (may repeat)
    # V: height 0=bottom 1=top
    texture_repeat = total_length / height  # similar H/V texel scale

    uvs = []
    for i in range(n):
        u = cumulative_length[i] / height  # height as unit length
        uvs.append([u, 0.0])  # bottom
    for i in range(n):
        u = cumulative_length[i] / height
        uvs.append([u, 1.0])  # top

    # Double-sided wall faces
    # Vertex index layout:
    # 0..n-1 bottom ring
    # n..2n-1 top ring
    faces = []

    # Two triangles per wall segment
    for i in range(n):
        i_next = (i + 1) % n

        # Four corner indices
        b_curr = i          # current bottom
        b_next = i_next     # next bottom
        t_curr = n + i      # current top
        t_next = n + i_next # next top

        # Face 1: normal inward (CCW vertices)
        # CCW when viewed from inside
        faces.append([b_curr, t_curr, t_next])
        faces.append([b_curr, t_next, b_next])

        # Face 2: outward normal (reversed winding)
        # CCW when viewed from outside
        faces.append([b_curr, t_next, t_curr])
        faces.append([b_curr, b_next, t_next])

    # process=True for correct normals
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=faces, process=True)

    # Store UV on mesh
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs))

    return mesh


def create_wall_ring_mesh(vertices: List[Tuple[float, float]], walls: Dict, height: float, thickness: float) -> trimesh.Trimesh:
    """
    Build combined wall mesh from wall loop vertices

    Args:
        vertices: Wall loop [(x1,y1), (x2,y2), ...]
        walls: Wall dict with orientation
        height: Wall height
        thickness: Wall thickness

    Returns:
        trimesh.Trimesh
    """
    n = len(vertices)
    if n < 3:
        return trimesh.Trimesh()

    # Outer offset per loop vertex
    # At v[i], adjacent edges v[i-1]->v[i] and v[i]->v[i+1]
    # Each edge has a wall; bisector of outer normals gives corner offset
    #
    # Geometry:
    # Unit outer normals n1, n2, angle theta
    # n1+n2 bisector, ||n1+n2|| = 2*cos(theta/2)
    # Perpendicular outward offset thickness along bisector:
    # offset = (n1+n2) * 2*thickness / ||n1+n2||^2
    vertex_offsets = []

    for i in range(n):
        v_curr = np.array(vertices[i])
        v_prev = np.array(vertices[(i - 1) % n])
        v_next = np.array(vertices[(i + 1) % n])

        # Walls for the two adjacent edges
        edge_prev = (tuple(v_prev), tuple(v_curr))  # previous edge
        edge_curr = (tuple(v_curr), tuple(v_next))  # current edge

        offset_sum = np.array([0.0, 0.0])

        for wall in walls.values():
            s = np.array(wall["s"])
            e = np.array(wall["e"])
            orientation = np.array(wall.get("orientation", [0, 0]))

            # Match wall to edge
            if (np.allclose(s, v_prev, atol=1e-6) and np.allclose(e, v_curr, atol=1e-6)) or \
               (np.allclose(s, v_curr, atol=1e-6) and np.allclose(e, v_prev, atol=1e-6)):
                # Wall for previous edge
                outer_normal = -orientation
                offset_sum += outer_normal

            if (np.allclose(s, v_curr, atol=1e-6) and np.allclose(e, v_next, atol=1e-6)) or \
               (np.allclose(s, v_next, atol=1e-6) and np.allclose(e, v_curr, atol=1e-6)):
                # Wall for current edge
                outer_normal = -orientation
                offset_sum += outer_normal

        # Corner offset from angle geometry
        # offset_sum = sum of unit outer normals along bisector
        # offset = offset_sum * (2 * thickness / ||offset_sum||^2)
        norm = np.linalg.norm(offset_sum)
        if norm > 1e-6:
            # Ensures perpendicular outward offset thickness per wall
            offset = offset_sum * (2.0 * thickness / (norm * norm))
        else:
            # Degenerate 180 deg: zero offset
            offset = np.array([0.0, 0.0])

        vertex_offsets.append(offset)

    # Build inner/outer vertex rings
    mesh_vertices = []
    # Bottom inner ring
    for v in vertices:
        mesh_vertices.append([v[0], v[1], 0.0])
    # Bottom outer ring
    for i, v in enumerate(vertices):
        offset = vertex_offsets[i]
        mesh_vertices.append([v[0] + offset[0], v[1] + offset[1], 0.0])
    # Top inner ring
    for v in vertices:
        mesh_vertices.append([v[0], v[1], height])
    # Top outer ring
    for i, v in enumerate(vertices):
        offset = vertex_offsets[i]
        mesh_vertices.append([v[0] + offset[0], v[1] + offset[1], height])

    # Four wall surface groups
    # Vertex index layout:
    # 0..n-1 bottom inner
    # n..2n-1 bottom outer
    # 2n..3n-1 top inner
    # 3n..4n-1 top outer
    faces = []

    # Part 1: inner faces (inward)
    # Quads along inner ring
    for i in range(n):
        i_next = (i + 1) % n

        # Inner quad corners
        b_curr = i
        b_next = i_next
        t_curr = 2 * n + i
        t_next = 2 * n + i_next

        # Normal inward: CCW from inside
        faces.append([b_curr, b_next, t_next])
        faces.append([b_curr, t_next, t_curr])

    # Part 2: outer faces
    # Iterate wall loop
    for i in range(n):
        i_next = (i + 1) % n

        # Four outer-face corners
        b_curr_outer = n + i
        b_next_outer = n + i_next
        t_curr_outer = 3 * n + i
        t_next_outer = 3 * n + i_next

        # Normal outward: reversed winding
        faces.append([b_next_outer, b_curr_outer, t_curr_outer])
        faces.append([b_next_outer, t_curr_outer, t_next_outer])

    # Part 3: top cap (+Z)
    # Iterate wall loop
    for i in range(n):
        i_next = (i + 1) % n

        # Top strip quad
        t_inner_curr = 2 * n + i
        t_inner_next = 2 * n + i_next
        t_outer_curr = 3 * n + i
        t_outer_next = 3 * n + i_next

        # Normal +Z: CCW from above
        faces.append([t_inner_curr, t_inner_next, t_outer_next])
        faces.append([t_inner_curr, t_outer_next, t_outer_curr])

    # Part 4: bottom cap (-Z)
    # Iterate wall loop
    for i in range(n):
        i_next = (i + 1) % n

        # Bottom strip quad
        b_inner_curr = i
        b_inner_next = i_next
        b_outer_curr = n + i
        b_outer_next = n + i_next

        # Normal -Z: CCW from below
        faces.append([b_outer_next, b_outer_curr, b_inner_curr])
        faces.append([b_outer_next, b_inner_curr, b_inner_next])

    return trimesh.Trimesh(vertices=mesh_vertices, faces=faces)


def create_wall_mesh(wall: Dict, thickness: float) -> trimesh.Trimesh:
    """
    Create 3D wall mesh

    Purpose: 3D geometry per wall in construct_floor
    Implementation:
        1. Eight corners from start, end, height, thickness
        2. Thickness extends outward (-orientation)
        3. Twelve triangles (two per face)
        4. Build trimesh

    Args:
        wall: s, e, height, inward orientation
        thickness: Wall thickness

    Returns:
        trimesh.Trimesh
    """
    s = wall["s"]
    e = wall["e"]
    height = wall["height"]
    orientation = wall.get("orientation", [0, 0])  # inward normal

    # Outer normal = -orientation
    # Full thickness outward
    outer_normal = np.array([-orientation[0], -orientation[1], 0]) * thickness

    # Eight vertex layout:
    # Bottom: 0(inner s) -- 1(inner e)  Top: 4 -- 5
    #        |          |                |          |
    #       3(outer s) -- 2(outer e)      7 -- 6
    vertices = [
        # Bottom face
        [s[0], s[1], 0],  # 0: start inner
        [e[0], e[1], 0],  # 1: end inner
        [e[0] + outer_normal[0], e[1] + outer_normal[1], 0],  # 2: end outer
        [s[0] + outer_normal[0], s[1] + outer_normal[1], 0],  # 3: start outer
        # Top face
        [s[0], s[1], height],  # 4: start inner top
        [e[0], e[1], height],  # 5: end inner top
        [e[0] + outer_normal[0], e[1] + outer_normal[1], height],  # 6: end outer top
        [s[0] + outer_normal[0], s[1] + outer_normal[1], height],  # 7: start outer top
    ]

    # Twelve triangles (six quads)
    # Normals point outward from wall solid
    faces = [
        # Bottom (-Z): CW 0->3->2->1 from below
        [0, 3, 2], [0, 2, 1],
        # Top (+Z): CW 4->5->6->7 from above
        [4, 5, 6], [4, 6, 7],
        # Inner (+orientation): CCW from room
        [1, 0, 4], [1, 4, 5],
        # Outer (-orientation): CW from outside
        [2, 3, 7], [2, 7, 6],
        # Start cap: 0->3->7->4
        [0, 3, 7], [0, 7, 4],
        # End cap: 1->5->6->2
        [1, 5, 6], [1, 6, 2],
    ]

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def cut_opening_from_wall(wall_mesh: trimesh.Trimesh, wall: Dict, opening: Dict,
                         thickness: float) -> trimesh.Trimesh:
    """
    Cut door/window opening from wall mesh

    Purpose: openings in construct_floor
    Implementation: boolean difference with opening box

    Note: placeholder; production may need richer boolean handling

    Args:
        wall_mesh: Original wall mesh
        wall: Wall data
        opening: center, width, height
        thickness: Wall thickness

    Returns:
        Wall mesh after cut
    """
    # Stub: return original mesh
    # Production: trimesh.boolean.difference
    return wall_mesh


def build_asset_search_paths(config: Dict, default_key: str = "model_path") -> List[str]:
    """Search order: extra_path -> [hole_extra_path if door/window] -> default -> generate (deduped, stable)."""
    candidates = []
    extra = config.get("model_extra_path")
    if extra:
        candidates.append(extra)
    if default_key == "model_hole_path":
        hole_extra = config.get("hole_extra_path")
        if hole_extra:
            candidates.append(hole_extra)
    default = config.get(default_key)
    if default:
        candidates.append(default)
    generate = config.get("model_generate_path")
    if generate:
        candidates.append(generate)
    seen = set()
    ordered = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def load_mesh(asset_id: int, config: Dict):
    """
    Load furniture mesh

    Purpose: load GLTF/GLB in construct_scene
    Implementation:
        1. Prefer GLB
        2. Fall back to GLTF
        3. trimesh.load

    Args:
        asset_id: Model ID
        config: model_path and optional model_extra_path

    Returns:
        trimesh.Scene or Trimesh, or None
    """
    search_paths = build_asset_search_paths(config, default_key="model_path")

    for base_path in search_paths:
        glb_path = os.path.join(base_path, f"{asset_id}.glb")
        gltf_path = os.path.join(base_path, f"{asset_id}.gltf")

        # Try paths in order
        for path in [glb_path, gltf_path]:
            if os.path.exists(path):
                try:
                    return trimesh.load(path)
                except Exception:
                    continue

    return None


def create_door_or_window_mesh(center: List[float], width: float, height: float, 
                               wall_data: Dict, texture_path: str) -> trimesh.Trimesh:
    """
    Create door/window mesh flush on wall
    
    Args:
        center: Center [x,y,z]
        width: Width
        height: Height
        wall_data: Wall with s, e, orientation
        texture_path: Texture path
        
    Returns:
        trimesh.Trimesh with texture and correct normals
    """
    # Wall start/end
    s = np.array(wall_data["s"], dtype=float)
    e = np.array(wall_data["e"], dtype=float)
    orientation = np.array(wall_data["orientation"], dtype=float)
    
    # Direction along wall
    wall_dir = e - s
    wall_length = np.linalg.norm(wall_dir)
    if wall_length < 1e-9:
        return trimesh.Trimesh()
    wall_dir_norm = wall_dir / wall_length
    
    # Door/window center projected on wall
    center_2d = np.array([center[0], center[1]], dtype=float)
    
    # Local coords along wall
    to_center = center_2d - s
    along_wall = np.dot(to_center, wall_dir_norm)
    
    # Four corners in 3D
    # Along wall: +/- width/2
    # Vertical: center[2] +/- height/2
    
    half_width = width / 2
    half_height = height / 2
    z_bottom = center[2] - half_height
    z_top = center[2] + half_height
    
    # Positions along wall
    pos_left = along_wall - half_width
    pos_right = along_wall + half_width
    
    # Four vertex positions
    vertices = np.array([
        # bottom-left
        [s[0] + pos_left * wall_dir_norm[0], s[1] + pos_left * wall_dir_norm[1], z_bottom],
        # bottom-right
        [s[0] + pos_right * wall_dir_norm[0], s[1] + pos_right * wall_dir_norm[1], z_bottom],
        # top-right
        [s[0] + pos_right * wall_dir_norm[0], s[1] + pos_right * wall_dir_norm[1], z_top],
        # top-left
        [s[0] + pos_left * wall_dir_norm[0], s[1] + pos_left * wall_dir_norm[1], z_top],
    ], dtype=float)
    
    # Slight inward offset to avoid z-fighting
    offset = 0.001
    offset_vec = np.array([orientation[0], orientation[1], 0], dtype=float) * offset
    vertices += offset_vec
    
    # Full-texture quad UV
    uvs = np.array([
        [0.0, 0.0],  # bottom-left
        [1.0, 0.0],  # bottom-right
        [1.0, 1.0],  # top-right
        [0.0, 1.0],  # top-left
    ])
    
    # Normal aligned with wall
    # Same winding test as wall
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    default_normal = np.cross(edge1, edge2)
    default_normal = default_normal / (np.linalg.norm(default_normal) + 1e-9)
    
    # Target inward normal
    target_normal_3d = np.array([orientation[0], orientation[1], 0.0])
    dot = np.dot(default_normal, target_normal_3d)
    
    # Winding from normal dot
    if dot > 0:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    else:
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=int)
    
    # Build mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Load texture
    try:
        from PIL import Image
        if os.path.exists(texture_path):
            texture_image = Image.open(texture_path)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture_image)
        else:
            print(f"⚠️  Texture file not found: {texture_path}")
            # Default color fallback
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    except Exception as e:
        print(f"⚠️  Failed to load texture {texture_path}: {e}")
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
    return mesh


def create_windows_and_doors(walls_data: Dict, door_texture_path: str, 
                             window_texture_path: str, config: Dict = None) -> List[Dict]:
    """
    Create meshes for all doors/windows on walls
    
    Args:
        walls_data: {wall_id: {s,e,height,orientation,doors,windows}}
        door_texture_path: Door texture path
        window_texture_path: Window texture path
        config: Optional config dict
        
    Returns:
        List [{type: 'door'/'window', id, mesh, ...}]
    """
    result = []
    
    for wall_id, wall in walls_data.items():
        # Doors
        for door_id, door_data in wall.get("doors", {}).items():
            door_mesh = create_door_or_window_mesh(
                center=door_data["center"],
                width=door_data["width"],
                height=door_data["height"],
                wall_data=wall,
                texture_path=door_texture_path
            )
            result.append({
                "type": "door",
                "id": door_id,
                "mesh": door_mesh,
                "wall_id": wall_id
            })
        
        # Windows
        for window_id, window_data in wall.get("windows", {}).items():
            window_mesh = create_door_or_window_mesh(
                center=window_data["center"],
                width=window_data["width"],
                height=window_data["height"],
                wall_data=wall,
                texture_path=window_texture_path
            )
            result.append({
                "type": "window",
                "id": window_id,
                "mesh": window_mesh,
                "wall_id": wall_id
            })
    
    return result


def segment_intersects_wall(segment_start: np.ndarray, segment_end: np.ndarray, 
                            wall_data: dict) -> bool:
    """
    Test segment vs wall rectangle (3D line vs wall face)
    
    Args:
        segment_start: Segment start [x,y,z]
        segment_end: Segment end [x,y,z]
        wall_data: Wall with s, e, height
        
    Returns:
        bool: intersection flag
    """
    # Wall rectangle corners
    s = np.array(wall_data["s"] + [0], dtype=float)  # bottom start
    e = np.array(wall_data["e"] + [0], dtype=float)  # bottom end
    height = wall_data["height"]
    
    # Four wall corners
    p1 = s  # bottom-left
    p2 = e  # bottom-right
    p3 = np.array([e[0], e[1], height], dtype=float)  # top-right
    p4 = np.array([s[0], s[1], height], dtype=float)  # top-left
    
    # Wall plane normal
    wall_dir = e - s
    wall_normal = np.array([-wall_dir[1], wall_dir[0], 0])
    wall_normal = wall_normal / np.linalg.norm(wall_normal)
    
    # Segment direction
    line_dir = segment_end - segment_start
    line_length = np.linalg.norm(line_dir)
    
    if line_length < 1e-9:
        return False
        
    line_dir_norm = line_dir / line_length
    
    # Line-plane intersection
    denom = np.dot(line_dir_norm, wall_normal)
    
    if abs(denom) < 1e-9:
        # Segment parallel to wall
        return False
    
    # Plane: dot(P - p1, wall_normal) = 0
    t = np.dot(p1 - segment_start, wall_normal) / denom
    
    # Intersection within segment
    if t < 0 or t > line_length:
        return False
    
    # Intersection point
    intersection = segment_start + t * line_dir_norm
    
    # Point inside wall rectangle
    # Project to wall-local 2D
    # Axes: along wall and +Z
    to_intersection = intersection - p1
    
    # Projection along wall
    wall_dir_norm = wall_dir / np.linalg.norm(wall_dir)
    proj_along_wall = np.dot(to_intersection, wall_dir_norm)
    wall_length = np.linalg.norm(wall_dir)
    
    # Projection along height
    proj_along_height = intersection[2] - p1[2]
    
    # Inside rectangle with epsilon
    epsilon = 0.01
    if (proj_along_wall >= -epsilon and proj_along_wall <= wall_length + epsilon and
        proj_along_height >= -epsilon and proj_along_height <= height + epsilon):
        return True
    
    return False


def find_walls_to_make_transparent(camera_pos_2d: list, look_at_2d: list, 
                                   floor_vertices: List[Tuple[float, float]], 
                                   walls_dict: dict) -> list:
    """
    Pick walls to make transparent from camera pose and view
    
    Algorithm:
    1. Test if camera (x,y) inside concave polygon
    2. If inside: no transparent walls (empty list)
    3. If outside:
       - Ray from camera to each vertex
       - Two extreme-angle rays (view frustum bounds)
       - Front arc between those vertices
       - Walls on front arc become transparent
    
    Args:
        camera_pos_2d: Camera [x, y]
        look_at_2d: Look-at [x, y]
        floor_vertices: Floor polygon (concave OK)
        walls_dict: Walls {wall_id: wall_data}
        
    Returns:
        list: Wall IDs to make transparent
    """
    camera_pos = np.array(camera_pos_2d, dtype=float)
    look_at = np.array(look_at_2d, dtype=float)
    
    # Step 1: camera inside polygon
    camera_tuple = tuple(camera_pos)
    is_inside = point_in_polygon(camera_tuple, floor_vertices)
    
    if is_inside:
        # Inside room: no transparent walls
        print("📍 Camera is inside the room; no transparent walls will be set")
        return []
    
    print("📍 Camera is outside the room; computing walls that should be transparent...")
    
    # Step 2: camera forward (reference)
    camera_direction = look_at - camera_pos
    if np.linalg.norm(camera_direction) < 1e-6:
        # Camera equals look-at: no direction
        print("⚠️  Camera and look-at target coincide; cannot determine orientation")
        return []
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    
    # Step 3: vectors and signed angles to vertices
    vertex_angles = []
    for i, vertex in enumerate(floor_vertices):
        vertex_pos = np.array(vertex, dtype=float)
        to_vertex = vertex_pos - camera_pos
        
        if np.linalg.norm(to_vertex) < 1e-6:
            # Vertex at camera
            angle = 0.0
        else:
            to_vertex_norm = to_vertex / np.linalg.norm(to_vertex)
            
            # Signed angle [-pi, pi]
            # atan2 relative to camera forward
            cos_angle = np.dot(camera_direction, to_vertex_norm)
            # Cross component for left/right
            cross = camera_direction[0] * to_vertex_norm[1] - camera_direction[1] * to_vertex_norm[0]
            angle = np.arctan2(cross, cos_angle)  # [-pi, pi]
        
        vertex_angles.append((i, vertex, angle))
    
    # Step 4: leftmost and rightmost vertices
    vertex_angles.sort(key=lambda x: x[2])
    leftmost_idx, leftmost_vertex, leftmost_angle = vertex_angles[0]
    rightmost_idx, rightmost_vertex, rightmost_angle = vertex_angles[-1]
    
    print(f"   Leftmost vertex: index {leftmost_idx}, angle {np.degrees(leftmost_angle):.1f}°")
    print(f"   Rightmost vertex: index {rightmost_idx}, angle {np.degrees(rightmost_angle):.1f}°")
    
    # Step 5: front arc (shorter path from left to right)
    n_vertices = len(floor_vertices)
    
    # Two candidate arcs
    if leftmost_idx <= rightmost_idx:
        path1 = list(range(leftmost_idx, rightmost_idx + 1))
        path2 = list(range(rightmost_idx, n_vertices)) + list(range(0, leftmost_idx + 1))
    else:
        path1 = list(range(leftmost_idx, n_vertices)) + list(range(0, rightmost_idx + 1))
        path2 = list(range(rightmost_idx, leftmost_idx + 1))
    
    # Mean camera distance per arc
    def avg_distance_to_camera(path):
        if not path:
            return float('inf')
        distances = []
        for idx in path:
            vertex = np.array(floor_vertices[idx], dtype=float)
            dist = np.linalg.norm(vertex - camera_pos)
            distances.append(dist)
        return np.mean(distances)
    
    dist1 = avg_distance_to_camera(path1)
    dist2 = avg_distance_to_camera(path2)
    
    # Pick closer arc as front
    front_vertex_indices = path1 if dist1 < dist2 else path2
    front_vertices_set = set(front_vertex_indices)
    
    print(f"   Path 1 average distance: {dist1:.2f}m, path 2 average distance: {dist2:.2f}m")
    print(f"   Selected {'path 1' if dist1 < dist2 else 'path 2'} as the front arc (closer distance)")
    
    print(f"   Front arc contains {len(front_vertex_indices)} vertices: {front_vertex_indices}")
    print(f"   Front arc vertex coordinates: {[floor_vertices[i] for i in front_vertex_indices]}")
    
    # Step 6: walls on front arc
    # Walls joining adjacent front-arc vertices
    transparent_wall_ids = []
    
    for wall_id, wall_data in walls_dict.items():
        wall_s = tuple(wall_data["s"])
        wall_e = tuple(wall_data["e"])
        
        # Both endpoints in vertex list
        try:
            s_idx = floor_vertices.index(wall_s)
            e_idx = floor_vertices.index(wall_e)
        except ValueError:
            # Endpoint missing from list (unexpected)
            continue
        
        # Both endpoints on front arc
        if s_idx not in front_vertices_set or e_idx not in front_vertices_set:
            continue
        
        # Adjacent on front arc
        # Positions in front_vertex_indices
        try:
            pos_s = front_vertex_indices.index(s_idx)
            pos_e = front_vertex_indices.index(e_idx)
        except ValueError:
            continue
        
        # Adjacency on cyclic arc
        n_front = len(front_vertex_indices)
        is_adjacent = (abs(pos_s - pos_e) == 1 or 
                      abs(pos_s - pos_e) == n_front - 1)
        
        if is_adjacent:
            transparent_wall_ids.append(wall_id)
            print(f"      Transparent wall: {wall_id}, endpoint indices ({s_idx},{e_idx}), positions on front arc ({pos_s},{pos_e})")
    
    print(f"   {len(transparent_wall_ids)} wall(s) will be set transparent")
    
    return transparent_wall_ids


def find_intersecting_walls(camera_position: list, look_at_target: list, mesh_nodes: dict) -> list:
    """
    Find walls intersecting camera-to-target segment (legacy, deprecated)
    
    Args:
        camera_position: Camera [x,y,z]
        look_at_target: Target [x,y,z]
        mesh_nodes: Scene mesh node dict
        
    Returns:
        list: Intersecting wall IDs
    """
    segment_start = np.array(camera_position, dtype=float)
    segment_end = np.array(look_at_target, dtype=float)
    
    intersecting_wall_ids = []
    
    for wall_id, wall_info in mesh_nodes["walls"].items():
        wall_data = wall_info["wall_data"]
        if segment_intersects_wall(segment_start, segment_end, wall_data):
            intersecting_wall_ids.append(wall_id)
    
    return intersecting_wall_ids


def set_mesh_alpha(mesh_nodes: dict, mesh_type: str, mesh_id: str, alpha: float = 0.0):
    """
    Set alpha for a mesh
    
    Args:
        mesh_nodes: Scene mesh node dict
        mesh_type: "walls", "boxes", "doors", "windows"
        mesh_id: Mesh unique ID
        alpha: 0.0-1.0 (0 fully transparent, 1 opaque)
    """
    import pyrender
    
    if mesh_type not in mesh_nodes:
        print(f"⚠️  Unknown mesh type: {mesh_type}")
        return
    
    if mesh_id not in mesh_nodes[mesh_type]:
        print(f"⚠️  {mesh_type} with ID {mesh_id} not found")
        return
    
    mesh_info = mesh_nodes[mesh_type][mesh_id]
    
    # Single or multiple nodes
    nodes_to_update = []
    if "node" in mesh_info:
        nodes_to_update = [mesh_info["node"]]
    elif "nodes" in mesh_info:
        nodes_to_update = mesh_info["nodes"]
    
    for node in nodes_to_update:
        if node and node.mesh:
            # Update all primitive materials
            for primitive in node.mesh.primitives:
                # Stash original material once
                if not hasattr(primitive, '_original_material'):
                    primitive._original_material = primitive.material
                
                # New transparent pyrender material
                if primitive.material:
                    # Copy existing material fields
                    mat = primitive.material
                    baseColorFactor = list(getattr(mat, 'baseColorFactor', [1.0, 1.0, 1.0, 1.0]))
                    baseColorFactor[3] = alpha  # set alpha
                    
                    # pyrender MetallicRoughnessMaterial
                    new_mat = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=baseColorFactor,
                        metallicFactor=getattr(mat, 'metallicFactor', 0.0),
                        roughnessFactor=getattr(mat, 'roughnessFactor', 1.0),
                        alphaMode='BLEND',  # enable alpha blending
                        doubleSided=True
                    )
                    
                    # Preserve texture if any
                    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                        new_mat.baseColorTexture = mat.baseColorTexture
                    
                    primitive.material = new_mat
                else:
                    # Create transparent material if missing
                    primitive.material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.8, 0.8, 0.8, alpha],
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                        alphaMode='BLEND',
                        doubleSided=True
                    )
    
    print(f"✅ Set alpha of {mesh_type}[{mesh_id}] to {alpha}")


def reset_mesh_alpha(mesh_nodes: dict, mesh_type: str, mesh_id: str):
    """
    Restore original mesh material
    
    Args:
        mesh_nodes: Scene mesh node dict
        mesh_type: Mesh category
        mesh_id: Mesh unique ID
    """
    if mesh_type not in mesh_nodes:
        return
    
    if mesh_id not in mesh_nodes[mesh_type]:
        return
    
    mesh_info = mesh_nodes[mesh_type][mesh_id]
    
    nodes_to_update = []
    if "node" in mesh_info:
        nodes_to_update = [mesh_info["node"]]
    elif "nodes" in mesh_info:
        nodes_to_update = mesh_info["nodes"]
    
    for node in nodes_to_update:
        if node and node.mesh:
            for primitive in node.mesh.primitives:
                if hasattr(primitive, '_original_material'):
                    primitive.material = primitive._original_material
                    delattr(primitive, '_original_material')
    
    print(f"✅ Restored original material for {mesh_type}[{mesh_id}]")


def reset_all_alpha(mesh_nodes: dict):
    """
    Restore all mesh materials
    
    Args:
        mesh_nodes: Scene mesh node dict
    """
    for mesh_type in mesh_nodes:
        if isinstance(mesh_nodes[mesh_type], dict):
            for mesh_id in list(mesh_nodes[mesh_type].keys()):
                reset_mesh_alpha(mesh_nodes, mesh_type, mesh_id)


def calculate_optimal_fov(camera_position: np.ndarray, look_at_target: np.ndarray,
                         floor_vertices: List[Tuple[float, float]], z_max: float,
                         bounds: List[float], indoor_fov: float = 120.0, 
                         outdoor_fov_scale: float = 1.05) -> float:
    """
    Compute optimal FOV
    
    Args:
        camera_position: Camera [x,y,z]
        look_at_target: Target [x,y,z]
        floor_vertices: Floor vertices (concave)
        z_max: Room height
        bounds: [x_min, y_min, x_max, y_max]
        indoor_fov: Indoor FOV (degrees)
        outdoor_fov_scale: Outdoor FOV scale factor
        
    Returns:
        fov_y: Vertical FOV (radians)
        
    Algorithm:
        Case 1: camera inside XY+Z bbox -> fixed indoor FOV
        Case 2: z outside [0,z_max] -> aim at bbox center to frame 3D box
        Case 3: z in range but XY outside -> frustum plane vs box edges
            - Vertical plane (forward+up) vs box -> max angle
            - Horizontal plane (forward+right) vs box -> max angle
            - FOV = 2 * max(two angles) * scale_factor
    """
    # Bbox center (when z out of range)
    center_xy = np.array([(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0], dtype=float)
    bbox_center = np.array([center_xy[0], center_xy[1], z_max / 2.0], dtype=float)

    # Camera inside XY+Z bbox
    in_bbox_xy = (bounds[0] <= camera_position[0] <= bounds[2] and 
                  bounds[1] <= camera_position[1] <= bounds[3])
    in_bbox_z = (0 <= camera_position[2] <= z_max)

    if in_bbox_xy and in_bbox_z:
        # Case 1: use indoor_fov
        fixed_fov = indoor_fov
        print(f"📐 Camera is inside bounding box; using fixed FOV: {fixed_fov}°")
        return np.radians(fixed_fov)
    
    camera_z_outside = camera_position[2] < 0 or camera_position[2] > z_max
    if camera_z_outside:
        print("📐 Camera z is outside [0, z_max]; computing FOV toward bounding box center...")
        target_point = bbox_center
    else:
        print("📐 Camera is within vertical range but XY is outside bounding box; computing FOV from frustum planes...")
        target_point = look_at_target

    # Camera basis
    forward = target_point - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # World up = +Z
    world_up = np.array([0, 0, 1], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # forward parallel to up: use Y axis
        right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # Box edges: floor polygon extruded to z_max
    edges_3d = []
    n = len(floor_vertices)
    
    # Bottom edges z=0
    for i in range(n):
        v1 = np.array([floor_vertices[i][0], floor_vertices[i][1], 0.0])
        v2 = np.array([floor_vertices[(i+1)%n][0], floor_vertices[(i+1)%n][1], 0.0])
        edges_3d.append((v1, v2))
    
    # Top edges z=z_max
    for i in range(n):
        v1 = np.array([floor_vertices[i][0], floor_vertices[i][1], z_max])
        v2 = np.array([floor_vertices[(i+1)%n][0], floor_vertices[(i+1)%n][1], z_max])
        edges_3d.append((v1, v2))
    
    # Vertical edges
    for i in range(n):
        v1 = np.array([floor_vertices[i][0], floor_vertices[i][1], 0.0])
        v2 = np.array([floor_vertices[i][0], floor_vertices[i][1], z_max])
        edges_3d.append((v1, v2))
    
    # Intersections of two planes with box
    # Plane 1: forward + up
    plane1_normal = np.cross(forward, up)
    plane1_normal = plane1_normal / np.linalg.norm(plane1_normal)
    
    # Plane 2: forward + right
    plane2_normal = np.cross(forward, right)
    plane2_normal = plane2_normal / np.linalg.norm(plane2_normal)
    
    def plane_edge_intersection(plane_normal, plane_point, edge_start, edge_end):
        """Intersection of plane and segment."""
        # Plane: dot(P - plane_point, plane_normal) = 0
        # Segment: P = edge_start + t * (edge_end - edge_start), t in [0,1]
        
        edge_dir = edge_end - edge_start
        denom = np.dot(edge_dir, plane_normal)
        
        if abs(denom) < 1e-9:
            # Edge parallel to plane
            return None
        
        t = np.dot(plane_point - edge_start, plane_normal) / denom
        
        if 0 <= t <= 1:
            return edge_start + t * edge_dir
        return None
    
    # Collect plane intersections
    intersections_plane1 = []
    intersections_plane2 = []
    
    for edge_start, edge_end in edges_3d:
        pt1 = plane_edge_intersection(plane1_normal, camera_position, edge_start, edge_end)
        if pt1 is not None:
            intersections_plane1.append(pt1)
        
        pt2 = plane_edge_intersection(plane2_normal, camera_position, edge_start, edge_end)
        if pt2 is not None:
            intersections_plane2.append(pt2)
    
    # Max angle per plane
    def calc_max_angle(intersections):
        if len(intersections) == 0:
            return 0.0
        
        max_angle = 0.0
        for pt in intersections:
            to_pt = pt - camera_position
            dist = np.linalg.norm(to_pt)
            if dist < 1e-6:
                continue
            to_pt_norm = to_pt / dist
            cos_angle = np.dot(forward, to_pt_norm)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            max_angle = max(max_angle, angle)
        
        return max_angle
    
    angle1 = calc_max_angle(intersections_plane1)
    angle2 = calc_max_angle(intersections_plane2)
    
    print(f"   Vertical plane intersections: {len(intersections_plane1)}, max angle: {np.degrees(angle1):.1f}°")
    print(f"   Horizontal plane intersections: {len(intersections_plane2)}, max angle: {np.degrees(angle2):.1f}°")
    
    # FOV = 2 * max_angle * scale_factor
    max_angle = max(angle1, angle2)
    fov_y = 2 * max_angle * outdoor_fov_scale
    
    # Clamp FOV range
    fov_y = np.clip(fov_y, np.radians(10), np.radians(120))
    
    print(f"   FOV = 2 × {np.degrees(max_angle):.1f}° × {outdoor_fov_scale} = {np.degrees(fov_y):.1f}°")
    
    return fov_y
def create_wall_edge_lines(start: List[float], end: List[float], height: float,
                           orientation: List[float], edge_color: List[float] = None,
                           offset: float = 0.01):
    """
    Wall edge lines to emphasize floor and corner boundaries
    Offset inward to avoid overlapping adjacent wall edges
    
    Args:
        start: Wall start [x, y]
        end: Wall end [x, y]
        height: Wall height
        orientation: Inward wall normal [nx, ny]
        edge_color: Line color RGBA, default dark gray
        offset: Inward offset in meters, default 0.01
        
    Returns:
        pyrender.Mesh line object
    """
    import pyrender
    # Inward offset along orientation
    offset_vec = np.array([orientation[0] * offset, orientation[1] * offset, 0.0])
    
    # Four wall corners, shifted inward
    vertices = np.array([
        [start[0], start[1], 0.0],        # 0: bottom start
        [end[0], end[1], 0.0],            # 1: bottom end
        [end[0], end[1], height],         # 2: top end
        [start[0], start[1], height]      # 3: top start
    ], dtype=np.float32)
    
    # Shift all vertices inward
    vertices += offset_vec
    
    # Edge line index pairs
    # Four edges: bottom, top, left, right
    edges = np.array([
        [0, 1],  # bottom (wall-floor)
        [2, 3],  # top (wall-ceiling)
        [0, 3],  # left (wall-wall)
        [1, 2],  # right (wall-wall)
    ], dtype=np.uint32)
    
    # Edge color
    if edge_color is None:
        edge_color = [0.1, 0.1, 0.1, 1.0]  # default dark gray
    edge_color = np.array(edge_color, dtype=np.float32)
    
    # Line material
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=edge_color,
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    
    # Line primitive
    primitive = pyrender.Primitive(
        positions=vertices,
        indices=edges,
        mode=pyrender.constants.GLTF.LINES,
        material=material
    )
    
    # Return pyrender.Mesh
    return pyrender.Mesh(primitives=[primitive])


def bbox_2d_from_binary_mask(mask: np.ndarray) -> Optional[List[int]]:
    """Bounding box [x1,y1,x2,y2] from binary mask (pixel coords, top-left origin, inclusive)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# Semantic map rendered without antialiasing; exact RGB index match (no nearest-neighbor).
SEMANTIC_COLOR_MAX_DIST_SQ = 0  # keep name; 0 = exact match only


def attach_semantic_bbox_2d(
    objects: List[Dict[str, Any]],
    rgb: np.ndarray,
    background: Tuple[int, int, int] = SEMANTIC_BACKGROUND,
    max_dist_sq: float = SEMANTIC_COLOR_MAX_DIST_SQ,
) -> None:
    """Fill pixel_num and bbox_2d for each semantic.json object.

    Default: exact color match (requires antialiased-off semantic render).
    max_dist_sq>0: thresholded nearest-neighbor (legacy maps).
    bbox_2d is axis-aligned bounds of all mask pixels; omitted when pixel_num==0.
    """
    for obj in objects:
        obj["pixel_num"] = 0
        obj.pop("bbox_2d", None)

    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb_u8 = np.asarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb_u8.shape[:2]

    indexed_objects: List[Dict[str, Any]] = []
    colors: List[Tuple[int, int, int]] = []
    for obj in objects:
        color = obj.get("color")
        if not color or len(color) != 3:
            continue
        colors.append((int(color[0]), int(color[1]), int(color[2])))
        indexed_objects.append(obj)

    if not indexed_objects:
        return

    if float(max_dist_sq) <= 0:
        for obj, color in zip(indexed_objects, colors):
            mask = np.all(rgb_u8 == np.asarray(color, dtype=np.uint8), axis=-1)
            pixel_num = int(np.count_nonzero(mask))
            obj["pixel_num"] = pixel_num
            if pixel_num == 0:
                continue
            bbox = bbox_2d_from_binary_mask(mask)
            if bbox is not None:
                obj["bbox_2d"] = bbox
        return

    # Legacy maps: thresholded nearest neighbor
    palette: List[Tuple[int, int, int]] = [background] + colors
    palette_arr = np.asarray(palette, dtype=np.float32)
    pixels = rgb_u8.reshape(-1, 3).astype(np.float32)
    dist_sq = np.sum((pixels[:, None, :] - palette_arr[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    min_dist_sq = dist_sq[np.arange(pixels.shape[0]), nearest]
    nearest[min_dist_sq > float(max_dist_sq)] = 0
    nearest = nearest.reshape(h, w)

    for idx, obj in enumerate(indexed_objects):
        mask = nearest == (idx + 1)
        pixel_num = int(np.count_nonzero(mask))
        obj["pixel_num"] = pixel_num
        if pixel_num == 0:
            continue
        bbox = bbox_2d_from_binary_mask(mask)
        if bbox is not None:
            obj["bbox_2d"] = bbox


def bbox_2d_overlay_path(render_path: str) -> str:
    """Sidecar bbox overlay path, e.g. topdown.png -> topdown_bbox_2d.png."""
    base, _ = os.path.splitext(render_path)
    return f"{base}_bbox_2d.png"


def save_bbox_2d_overlay_png(
    render_path: str,
    objects: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    line_width: int = 2,
) -> Optional[str]:
    """Draw semantic bbox_2d and labels on render; save as *_bbox_2d.png."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("⚠️ PIL is unavailable; skipping bbox_2d visualization")
        return None

    if not os.path.isfile(render_path):
        print(f"⚠️ Render image not found; skipping bbox_2d visualization: {render_path}")
        return None

    if output_path is None:
        output_path = bbox_2d_overlay_path(render_path)

    img = Image.open(render_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = img.size

    for obj in objects:
        bbox = obj.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = obj.get("color") or [0, 220, 80]
        rgb = tuple(int(c) for c in color[:3])
        label = str(obj.get("label") or "")

        draw.rectangle((x1, y1, x2, y2), outline=rgb + (255,), width=line_width)

        if not label:
            continue
        text_x = x1 + 2
        text_y = y1 + 2 if (y2 - y1) >= 16 else max(0, y1 - 14)
        if hasattr(draw, "textbbox"):
            text_box = draw.textbbox((text_x, text_y), label)
            pad = 2
            draw.rectangle(
                (text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad),
                fill=rgb + (210,),
            )
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 255))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.alpha_composite(img, overlay).convert("RGB").save(output_path)
    return output_path


def semantic_entity_color(
    entity_key: str,
    used: Optional[Set[Tuple[int, int, int]]] = None,
) -> Tuple[int, int, int]:
    """Semantic color for one entity; optional used set ensures unique colors per scene."""
    used_set = used if used is not None else set()
    salt = 0
    while salt < 4096:
        key = f"{entity_key}\0{salt}" if salt else entity_key
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        hue = digest[0] / 255.0
        sat = 0.70 + (digest[1] / 255.0) * 0.30
        val = 0.65 + (digest[2] / 255.0) * 0.30
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        if (
            color != SEMANTIC_BACKGROUND
            and min(color) >= 64
            and max(color) >= 120
            and color not in used_set
        ):
            used_set.add(color)
            return color
        salt += 1
    fallback = (255, 96, 96)
    used_set.add(fallback)
    return fallback


def compute_depth_encode_scale(depth_m: np.ndarray) -> float:
    """Dynamic uint16 depth scale: n = max_depth * 1.5, scale = 65535 / n."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        n = 1.0
    else:
        n = float(np.max(depth_m[valid])) * 1.5
        if n <= 0:
            n = 1.0
    return 65535.0 / n


def encode_depth_uint16(depth_m: np.ndarray, depth_scale: float) -> np.ndarray:
    """Encode metric depth to uint16 PNG: pixel = depth_m * depth_scale."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_png = np.zeros(depth_m.shape, dtype=np.uint16)
    if np.any(valid):
        scaled = np.rint(depth_m[valid] * float(depth_scale))
        depth_png[valid] = np.clip(scaled, 0, 65535).astype(np.uint16)
    return depth_png


def decode_depth_uint16(depth_png: np.ndarray, depth_scale: float) -> np.ndarray:
    """Decode uint16 depth PNG to meters: depth_m = pixel / depth_scale; 0 invalid."""
    if depth_scale <= 0:
        raise ValueError("depth_scale must be > 0")
    arr = np.asarray(depth_png)
    if arr.ndim == 3:
        arr = arr[..., 0]
    depth_m = arr.astype(np.float64) / float(depth_scale)
    depth_m[arr == 0] = 0.0
    return depth_m


def encode_normal_world_png(normal_01: np.ndarray) -> np.ndarray:
    """World normal in [0,1] -> uint8 RGB PNG (round(c*255) per channel)."""
    arr = np.asarray(normal_01, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[..., :3]
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def encode_normal_directions_uint8_png(normal_vectors: np.ndarray) -> np.ndarray:
    """Normal components in [-1,1] (any frame) -> uint8 RGB PNG."""
    arr = np.asarray(normal_vectors, dtype=np.float64)
    normal_01 = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    return encode_normal_world_png(normal_01)


def encode_normal_world_png_from_cycles_exr(normal_exr: np.ndarray) -> np.ndarray:
    """Cycles Normal pass EXR (world, [-1,1]) -> uint8 RGB PNG."""
    return encode_normal_directions_uint8_png(normal_exr)


def decode_normal_world_uint8(normal_png: np.ndarray) -> np.ndarray:
    """uint8 normal PNG -> unit vectors (H,W,3) in [-1,1] (frame in camera_para.normal_space)."""
    rgb = np.asarray(normal_png, dtype=np.float64)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    rgb = rgb[..., :3] / 255.0
    return rgb * 2.0 - 1.0


def matrix4x4_to_nested_list(matrix: np.ndarray, *, decimals: int = 6) -> List[List[float]]:
    return np.round(np.asarray(matrix, dtype=float), decimals).tolist()


def build_opencv_intrinsic_4x4(
    fov_angle_rad: float,
    width: int,
    height: int,
    *,
    aspect_ratio: Optional[float] = None,
) -> np.ndarray:
    """Pinhole 4x4 intrinsic K (no distortion); matches util_bpy frustum / Blender sensor_fit=AUTO."""
    w = int(width)
    h = int(height)
    aspect = float(aspect_ratio) if aspect_ratio is not None else w / max(h, 1)
    angle = float(fov_angle_rad)
    if aspect >= 1.0:
        tan_x = math.tan(angle * 0.5)
        tan_y = tan_x / max(aspect, 1e-6)
    else:
        tan_y = math.tan(angle * 0.5)
        tan_x = tan_y * aspect
    fx = w / (2.0 * max(tan_x, 1e-6))
    fy = h / (2.0 * max(tan_y, 1e-6))
    cx = w / 2.0
    cy = h / 2.0
    intrinsic = np.eye(4, dtype=float)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def camera_calibration_matrix_fields(
    camera_position,
    look_at_target,
    world_up,
    fov_y_rad: float,
    width: int,
    height: int,
    *,
    aspect_ratio: Optional[float] = None,
    include_intrinsic: bool = True,
) -> Dict[str, Any]:
    """ScanNet/OpenSpatial-style c2w + intrinsic for camera_para.json."""
    try:
        from . import geometry_opencv as geo_cv
    except ImportError:
        import geometry_opencv as geo_cv  # type: ignore

    c2w = geo_cv.build_opencv_c2w_matrix(camera_position, look_at_target, world_up)
    fields: Dict[str, Any] = {
        "camera_convention": "opencv",
        "world_convention": "ssl_z_up",
        "c2w": matrix4x4_to_nested_list(c2w),
    }
    if include_intrinsic:
        intrinsic = build_opencv_intrinsic_4x4(
            fov_y_rad,
            width,
            height,
            aspect_ratio=aspect_ratio,
        )
        fields["intrinsic"] = matrix4x4_to_nested_list(intrinsic)
        fields["intrinsic_model"] = "pinhole_no_distortion"
    return fields


def decode_normal_opencv_uint8(normal_png: np.ndarray) -> np.ndarray:
    """uint8 normal PNG -> OpenCV camera-frame unit normals (H,W,3)."""
    return decode_normal_world_uint8(normal_png)


def normal_map_camera_para_fields() -> Dict[str, Any]:
    """Normal-map metadata for {basename}_camera_para.json (exported with --depth)."""
    return {
        "normal_space": "opencv_camera",
        "normal_axes": {
            "x": "+X image right",
            "y": "+Y image down",
            "z": "+Z along view into scene",
        },
        "normal_encoding": "uint8_rgb",
        "normal_encode": "uint8 = round(((normal_opencv + 1) / 2) * 255)",
        "normal_decode": "normal_opencv = (pixel_rgb / 255.0) * 2.0 - 1.0",
        "normal_invalid_mask": "depth_pixel == 0",
    }


DEFAULT_TOPDOWN_OCCLUDER_BOTTOM_MARGIN_M = 0.25
DEFAULT_TOPDOWN_OCCLUDER_MIN_FLOOR_AREA_RATIO = 0.10
DEFAULT_TOPDOWN_OCCLUDER_LOW_HANGING_MIN_BOTTOM_M = 2.0
DEFAULT_TOPDOWN_OCCLUDER_LOW_HANGING_MAX_HEIGHT_M = 0.35
DEFAULT_TOPDOWN_OCCLUDER_LOW_HANGING_MIN_FLOOR_AREA_RATIO = 0.45
DEFAULT_TOPDOWN_OCCLUDER_ENCLOSURE_MIN_HEIGHT_RATIO = 0.95
DEFAULT_TOPDOWN_OCCLUDER_ENCLOSURE_MIN_FLOOR_AREA_RATIO = 0.95
DEFAULT_TOPDOWN_OCCLUDER_CEILING_SLAB_MAX_HEIGHT_M = 1.0
DEFAULT_TOPDOWN_OCCLUDER_CEILING_SLAB_MIN_FLOOR_AREA_RATIO = 0.95
DEFAULT_TOPDOWN_OCCLUDER_FULL_HEIGHT_MIN_HEIGHT_RATIO = 0.90
DEFAULT_TOPDOWN_OCCLUDER_FULL_HEIGHT_MIN_FLOOR_AREA_RATIO = 0.95
DEFAULT_TOPDOWN_OCCLUDER_FULL_HEIGHT_MAX_BOTTOM_M = 0.15
DEFAULT_TOPDOWN_OCCLUDER_WIDE_FOOTPRINT_MIN_RATIO = 0.90
DEFAULT_TOPDOWN_OCCLUDER_WIDE_FOOTPRINT_MIN_TOP_M = 3.5
DEFAULT_TOPDOWN_OCCLUDER_WIDE_FOOTPRINT_LABEL_KEYWORDS = ("pipe", "ceiling")
DEFAULT_TOPDOWN_OCCLUDER_GEOM_EPS_M = 0.01
TOPDOWN_OCCLUDER_PANEL_LABEL_KEYWORDS = (
    "wall panel",
    "wall partition",
    "room divider",
    "partition wall",
    "decorative panel",
    "隔断",
    "墙板",
    "装饰墙板",
)
TOPDOWN_OCCLUDER_ENCLOSURE_LABEL_KEYWORDS = (
    "sunroom",
    "conservatory",
    "greenhouse",
    "glasshouse",
    "glass cube",
    "glasscube",
    "玻璃房",
    "阳光房",
)
TOPDOWN_OCCLUDER_LABEL_KEYWORDS = ("ceiling", "顶板", "吊顶", "天花板", "顶面")


def _wide_footprint_label_match(label_text: str, keywords: Tuple[str, ...]) -> bool:
    """True when compact label contains any wide-footprint keyword (e.g. pipe, ceiling)."""
    compact = (label_text or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    if not compact:
        return False
    for kw in keywords:
        token = (kw or "").lower().replace("_", "").replace("-", "").replace(" ", "")
        if token and token in compact:
            return True
    return False


def _label_caption_text(box: Dict[str, Any]) -> str:
    return " ".join(
        filter(
            None,
            [
                str(box.get("label") or "").lower(),
                str(box.get("caption") or "").lower(),
            ],
        )
    )


def _is_glass_enclosure_candidate(label_caption: str) -> bool:
    return any(kw in label_caption for kw in TOPDOWN_OCCLUDER_ENCLOSURE_LABEL_KEYWORDS)


def _is_panel_occluder_candidate(label_caption: str, *, label_only: str = "") -> bool:
    """Match wall panel / partition labels.

    ``normalize_scene_context`` may rename labels (e.g. ``wallpanel0``); match compact forms too.
    """
    texts: List[str] = []
    for raw in (label_only, label_caption):
        text = (raw or "").lower()
        if not text:
            continue
        texts.append(text)
        texts.append(text.replace("_", "").replace("-", "").replace(" ", ""))
    for text in texts:
        for kw in TOPDOWN_OCCLUDER_PANEL_LABEL_KEYWORDS:
            k = kw.lower()
            if k in text or k.replace(" ", "") in text:
                return True
    return False


def identify_topdown_occluding_box_ids(
    context: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    """Box IDs to skip in normalized top-down depth rendering for floor-path nav mask.

    Ceiling-mounted objects with a large horizontal footprint block top-down depth
    rays and can zero out the walkable nav mask. Uses wall height (not furniture
    z_max) and compares XY footprint against room floor area.
    """
    config = config or {}
    margin_m = float(
        config.get("topdown_occluder_bottom_margin_m", DEFAULT_TOPDOWN_OCCLUDER_BOTTOM_MARGIN_M)
    )
    min_area_ratio = float(
        config.get(
            "topdown_occluder_min_floor_area_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_MIN_FLOOR_AREA_RATIO,
        )
    )
    keyword_min_ratio = float(
        config.get("topdown_occluder_keyword_min_floor_area_ratio", 0.05)
    )

    walls = context.get("walls") or {}
    wall_z_max = max((float(w.get("height", 0)) for w in walls.values()), default=0.0)
    if wall_z_max <= 0:
        return set()

    meta = context.get("meta") or {}
    span = meta.get("span") or [0.0, 0.0]
    floor_area = float(span[0]) * float(span[1])
    if floor_area <= 1e-6:
        bounds = meta.get("bounds")
        if bounds and len(bounds) >= 4:
            floor_area = max(
                (float(bounds[2]) - float(bounds[0])) * (float(bounds[3]) - float(bounds[1])),
                1e-6,
            )
        else:
            floor_area = 1.0

    min_xy_area = floor_area * min_area_ratio
    keyword_min_xy = floor_area * keyword_min_ratio
    bottom_threshold = wall_z_max - margin_m
    thin_ceiling_max_height = margin_m * 2.0
    low_hanging_min_bottom_m = float(
        config.get(
            "topdown_occluder_low_hanging_min_bottom_m",
            DEFAULT_TOPDOWN_OCCLUDER_LOW_HANGING_MIN_BOTTOM_M,
        )
    )
    low_hanging_max_height_m = float(
        config.get(
            "topdown_occluder_low_hanging_max_height_m",
            DEFAULT_TOPDOWN_OCCLUDER_LOW_HANGING_MAX_HEIGHT_M,
        )
    )
    low_hanging_min_area_ratio = float(
        config.get(
            "topdown_occluder_low_hanging_min_floor_area_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_LOW_HANGING_MIN_FLOOR_AREA_RATIO,
        )
    )
    low_hanging_min_xy = floor_area * low_hanging_min_area_ratio
    enclosure_min_area_ratio = float(
        config.get(
            "topdown_occluder_enclosure_min_floor_area_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_ENCLOSURE_MIN_FLOOR_AREA_RATIO,
        )
    )
    enclosure_min_xy = floor_area * enclosure_min_area_ratio
    ceiling_slab_max_height_m = float(
        config.get(
            "topdown_occluder_ceiling_slab_max_height_m",
            DEFAULT_TOPDOWN_OCCLUDER_CEILING_SLAB_MAX_HEIGHT_M,
        )
    )
    ceiling_slab_min_area_ratio = float(
        config.get(
            "topdown_occluder_ceiling_slab_min_floor_area_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_CEILING_SLAB_MIN_FLOOR_AREA_RATIO,
        )
    )
    ceiling_slab_min_xy = floor_area * ceiling_slab_min_area_ratio
    full_height_min_height_ratio = float(
        config.get(
            "topdown_occluder_full_height_min_height_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_FULL_HEIGHT_MIN_HEIGHT_RATIO,
        )
    )
    full_height_min_area_ratio = float(
        config.get(
            "topdown_occluder_full_height_min_floor_area_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_FULL_HEIGHT_MIN_FLOOR_AREA_RATIO,
        )
    )
    full_height_max_bottom_m = float(
        config.get(
            "topdown_occluder_full_height_max_bottom_m",
            DEFAULT_TOPDOWN_OCCLUDER_FULL_HEIGHT_MAX_BOTTOM_M,
        )
    )
    full_height_min_xy = floor_area * full_height_min_area_ratio
    wide_footprint_min_ratio = float(
        config.get(
            "topdown_occluder_wide_footprint_min_ratio",
            DEFAULT_TOPDOWN_OCCLUDER_WIDE_FOOTPRINT_MIN_RATIO,
        )
    )
    wide_footprint_min_top_m = float(
        config.get(
            "topdown_occluder_wide_footprint_min_top_m",
            DEFAULT_TOPDOWN_OCCLUDER_WIDE_FOOTPRINT_MIN_TOP_M,
        )
    )
    wide_footprint_label_keywords = tuple(
        config.get(
            "topdown_occluder_wide_footprint_label_keywords",
            DEFAULT_TOPDOWN_OCCLUDER_WIDE_FOOTPRINT_LABEL_KEYWORDS,
        )
    )
    geom_eps_m = float(
        config.get("topdown_occluder_geom_eps_m", DEFAULT_TOPDOWN_OCCLUDER_GEOM_EPS_M)
    )

    excluded: Set[str] = set()
    for box_id, box in (context.get("boxes") or {}).items():
        center = box.get("center")
        scale = box.get("scale")
        if not center or not scale or len(center) < 3 or len(scale) < 3:
            continue

        bottom_z = float(center[2]) - float(scale[2]) / 2.0
        top_z = float(center[2]) + float(scale[2]) / 2.0
        height_z = top_z - bottom_z
        xy_area = abs(float(scale[0]) * float(scale[1]))
        label = str(box.get("label") or box.get("caption") or "").lower()
        label_caption = _label_caption_text(box)

        footprint_ratio = xy_area / floor_area if floor_area > 1e-6 else 0.0
        if (
            footprint_ratio > wide_footprint_min_ratio - 1e-9
            and top_z > wide_footprint_min_top_m + geom_eps_m
            and _wide_footprint_label_match(label, wide_footprint_label_keywords)
        ):
            excluded.add(box_id)
            continue

        # Ceiling-mounted: bottom in upper band. Tall floor furniture (bookshelf, wardrobe)
        # can reach the ceiling (top_z in band) but must not be excluded.
        bottom_in_upper_band = bottom_z >= bottom_threshold - geom_eps_m
        thin_ceiling_fixture = (
            top_z >= bottom_threshold - geom_eps_m
            and height_z <= thin_ceiling_max_height + geom_eps_m
        )
        if (bottom_in_upper_band or thin_ceiling_fixture) and xy_area >= min_xy_area:
            excluded.add(box_id)
            continue

        # Large flat ceiling slab: top near ceiling, huge footprint, modest thickness.
        # Catches mis-scaled ceiling fixtures (e.g. full-room light panel) whose bottom
        # sits slightly below the upper band but still blocks the entire top-down view.
        if (
            top_z >= bottom_threshold - geom_eps_m
            and height_z <= ceiling_slab_max_height_m + geom_eps_m
            and xy_area >= ceiling_slab_min_xy
        ):
            excluded.add(box_id)
            continue

        # Labeled wall panel / room divider covering most of the room (e.g. KTV decorative
        # wall panel modeled as a floor-to-ceiling slab). Label-gated to avoid full-height
        # wardrobes, beds, and kitchen units that also reach the ceiling.
        if (
            _is_panel_occluder_candidate(label_caption, label_only=label)
            and xy_area >= full_height_min_xy
            and bottom_z <= full_height_max_bottom_m + geom_eps_m
            and height_z >= wall_z_max * full_height_min_height_ratio - geom_eps_m
        ):
            excluded.add(box_id)
            continue

        # Large thin panels hung mid/high (e.g. dropped ceiling soffits ~2m+ above floor).
        if (
            height_z <= low_hanging_max_height_m
            and bottom_z >= low_hanging_min_bottom_m
            and xy_area >= low_hanging_min_xy
        ):
            excluded.add(box_id)
            continue

        # Glass sunroom / conservatory: label match only + near full wall height + huge footprint.
        enclosure_min_height_ratio = float(
            config.get(
                "topdown_occluder_enclosure_min_height_ratio",
                DEFAULT_TOPDOWN_OCCLUDER_ENCLOSURE_MIN_HEIGHT_RATIO,
            )
        )
        if (
            _is_glass_enclosure_candidate(label_caption)
            and xy_area >= enclosure_min_xy
            and height_z >= wall_z_max * enclosure_min_height_ratio
        ):
            excluded.add(box_id)
            continue

        large_footprint_ratio = float(
            config.get("topdown_occluder_large_footprint_ratio", 0.25)
        )
        near_ceiling_top = wall_z_max - margin_m * 4
        if (
            xy_area >= floor_area * large_footprint_ratio
            and top_z >= near_ceiling_top
            and (bottom_in_upper_band or thin_ceiling_fixture)
        ):
            excluded.add(box_id)
            continue

        if any(kw in label for kw in TOPDOWN_OCCLUDER_LABEL_KEYWORDS):
            kw_band = wall_z_max - margin_m * 2
            bottom_in_kw_band = bottom_z >= kw_band - geom_eps_m
            thin_in_kw_band = (
                top_z >= kw_band - geom_eps_m
                and height_z <= thin_ceiling_max_height + geom_eps_m
            )
            if (bottom_in_kw_band or thin_in_kw_band) and xy_area >= keyword_min_xy:
                excluded.add(box_id)

    return excluded


def describe_topdown_occluding_boxes(
    context: Dict[str, Any],
    box_ids: Set[str],
) -> List[str]:
    """Human-readable labels for excluded top-down occluder boxes."""
    boxes = context.get("boxes") or {}
    labels: List[str] = []
    for box_id in sorted(box_ids):
        box = boxes.get(box_id, {})
        name = box.get("label") or box.get("caption") or box.get("class") or box_id[:8]
        labels.append(str(name))
    return labels


def resolve_topdown_exclude_box_ids(
    context: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    *,
    enabled: bool = True,
    log_prefix: str = "top-down view",
) -> Optional[Set[str]]:
    """Return box IDs to skip when building a downward-looking scene (or None)."""
    if not enabled:
        return None
    excluded = identify_topdown_occluding_box_ids(context, config)
    if excluded:
        labels = describe_topdown_occluding_boxes(context, excluded)
        print(
            f"🚫 Skipping {len(excluded)} ceiling occluder(s) for {log_prefix}: "
            + ", ".join(labels)
        )
        return excluded
    return None
