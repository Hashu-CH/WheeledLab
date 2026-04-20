import os
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from .procedural_track_gen.curriculum import CurriculumConfig, generate_track
from .traversability_utils import *


def compute_map_size(num_envs, env_size):
    """Map dimensions (in cells) to pack `num_envs` non-overlapping sub-envs in a square-ish grid.

    `num_envs` must be a power of 2. The exponent is split evenly so the env grid is
    32x32 for 1024, 16x32 for 512, etc.
    """
    if num_envs < 1 or (num_envs & (num_envs - 1)) != 0:
        raise ValueError(f"num_envs must be a power of 2, got {num_envs}")
    env_num_rows, env_num_cols = env_size
    exponent = num_envs.bit_length() - 1
    grid_rows = 1 << (exponent // 2)
    grid_cols = 1 << (exponent - exponent // 2)
    return grid_rows * env_num_rows, grid_cols * env_num_cols


def generated_colored_track_plane(map_size, spacing, env_size, color_sampling):
    """Generate a colored plane with rasterised curriculum tracks."""
    num_rows, num_cols = map_size
    row_spacing, col_spacing = spacing
    env_num_rows, env_num_cols = env_size

    width = num_rows * row_spacing
    height = num_cols * col_spacing

    if num_rows % env_num_rows != 0 or num_cols % env_num_cols != 0:
        raise ValueError("Map size must be a multiple of the sub environment size.")

    num_env_rows = num_rows // env_num_rows
    num_env_cols = num_cols // env_num_cols

    xs = np.linspace(-width / 2, width / 2, num_rows) - row_spacing / 2
    ys = np.linspace(-height / 2, height / 2, num_cols) - col_spacing / 2
    xx, yy = np.meshgrid(xs, ys)

    vertices = [(x, y, 0) for x, y in zip(xx.ravel(), yy.ravel())]

    def color_sampler(r, g, b, range):
        r = np.random.uniform(r - range // 2, r + range // 2) / 255.
        g = np.random.uniform(g - range // 2, g + range // 2) / 255.
        b = np.random.uniform(b - range // 2, b + range // 2) / 255.
        return Gf.Vec3f(r, g, b)

    if color_sampling:
        colors = [
            color_sampler(30, 30, 30, 30),
            color_sampler(220.0, 220.0, 220.0, 30),
        ]
    else:
        colors = [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 1.0),
        ]

    faces = []
    face_counts = []
    traversability_hashmap = np.zeros((num_rows, num_cols)).astype(bool)
    for row_index in range(num_rows - 1):
        for col_index in range(num_cols - 1):
            v0 = row_index * num_cols + col_index
            v1 = v0 + 1
            v2 = v0 + num_cols
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]

    # Track difficulty is randomized per env in [0.1, 0.9]
    for i in range(num_env_rows):
        for j in range(num_env_cols):
            start_row = i * env_num_rows
            end_row = (i + 1) * env_num_rows
            start_col = j * env_num_cols
            end_col = (j + 1) * env_num_cols
            grid, _ = generate_track(
                env_size=env_size,
                config=CurriculumConfig(difficulty=np.random.uniform(.10, .90)),
            )
            traversability_hashmap[start_row:end_row, start_col:end_col] = grid

    face_colors = [
        colors[int(traversability_hashmap[row_index, col_index])]
        for row_index in range(num_rows - 1)
        for col_index in range(num_cols - 1)
    ]
    face_colors_triangle = []
    for color in face_colors:
        face_colors_triangle += [color, color]

    return vertices, faces, face_counts, face_colors_triangle, traversability_hashmap


def create_track_geometry(file_path, map_size, spacing, env_size, color_sampling=False):
    """Create a USD file with a rasterised curriculum track plane."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(file_path)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())

    plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')

    vertices, faces, face_counts, face_colors, traversability_hashmap = \
        generated_colored_track_plane(map_size, spacing, env_size, color_sampling)

    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors)

    UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")

    stage.GetRootLayer().Save()

    traversability_hashmap = traversability_hashmap.tolist()
    TraversabilityHashmapUtil().set_traversability_hashmap(
        traversability_hashmap, map_size, spacing,
    )
    return traversability_hashmap


def generate_random_poses(num_poses, row_spacing, col_spacing, traversability_hashmap, margin=0.1):
    """Sample random traversable (x, y, yaw) poses from a hashmap."""
    H, W = len(traversability_hashmap), len(traversability_hashmap[0])
    pose_candidates = np.array(traversability_hashmap).nonzero()
    idxs = np.random.choice(len(pose_candidates[0]), num_poses)
    ys, xs = pose_candidates[0][idxs], pose_candidates[1][idxs]
    poses = []
    for i in range(len(xs)):
        x = (float(xs[i]) - W // 2) * row_spacing
        y = (float(ys[i]) - H // 2) * col_spacing
        angle = np.random.uniform(0, 360.0)
        poses.append((x, y, angle))
    return poses
