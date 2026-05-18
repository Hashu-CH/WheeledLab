"""
Generate orange_cone.usda and blue_cone.usda as simple procedural cone meshes.

Run once (outside Isaac Sim) via:
    python generate_cones.py

Or run inside Isaac Sim Python if OmniPBR materials are needed.  The script
uses only core USD libraries so it works standalone; materials are set via
displayColor primvar which Isaac Sim respects for basic shading.

Cone geometry: scaled FSAE-style cone — 0.10 m tall, 0.04 m base radius
(~1/3 the width of the MuSHR car so cones read clearly in the policy camera
without dominating the scene).
"""

import math
import os
import numpy as np
from pxr import Usd, UsdGeom, Gf


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cone_mesh_data(base_radius: float, height: float, segments: int):
    """Return (points, face_counts, face_indices) for a closed cone."""
    points = []
    # base ring
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        points.append(Gf.Vec3f(base_radius * math.cos(angle), base_radius * math.sin(angle), 0.0))
    # apex
    apex_idx = segments
    points.append(Gf.Vec3f(0.0, 0.0, height))
    # base centre
    base_centre_idx = segments + 1
    points.append(Gf.Vec3f(0.0, 0.0, 0.0))

    face_counts = []
    face_indices = []

    # lateral faces (triangles from ring to apex)
    for i in range(segments):
        next_i = (i + 1) % segments
        face_counts.append(3)
        face_indices.extend([i, next_i, apex_idx])

    # base cap (fan from base centre)
    for i in range(segments):
        next_i = (i + 1) % segments
        face_counts.append(3)
        face_indices.extend([base_centre_idx, next_i, i])

    return points, face_counts, face_indices


def _write_cone_usd(path: str, color: tuple[float, float, float]):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Cone")
    stage.SetDefaultPrim(xform.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/Cone/cone_mesh")

    points, face_counts, face_indices = _cone_mesh_data(
        base_radius=0.04, height=0.10, segments=16
    )
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # Uniform display color (one value applied to all faces)
    color_primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    color_primvar.Set([Gf.Vec3f(*color)])

    # No collision — cones are visual-only.
    stage.Save()
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    _write_cone_usd(os.path.join(out_dir, "orange_cone.usda"), color=(1.0, 0.45, 0.0))
    _write_cone_usd(os.path.join(out_dir, "blue_cone.usda"),   color=(0.0, 0.25, 0.85))
