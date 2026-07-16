from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
LOCAL_USD_RUNTIME = SCRIPTS_ROOT / ".codex_usd"
if LOCAL_USD_RUNTIME.is_dir() and str(LOCAL_USD_RUNTIME) not in sys.path:
    sys.path.insert(0, str(LOCAL_USD_RUNTIME))


SIMULATION_APP = None
try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
except ModuleNotFoundError as exc:
    if exc.name != "pxr":
        raise
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    SIMULATION_APP = app_launcher.app
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


COLLISION_TYPES = {
    "bounding_box": "boundingCube",
    "convex_hull": "convexHull",
    "convex_decomposition": "convexDecomposition",
    "triangle_mesh": "none",
}
BODY_TYPES = {"static", "dynamic"}
RESULT_PREFIX = "PHYSICS_USD_RESULT="


def create_physics_asset(
    usdz_path: str | Path,
    *,
    mass: float,
    collision_type: str,
    friction: float,
    body_type: str,
) -> dict[str, Any]:
    source_path = Path(usdz_path).expanduser().resolve()
    if source_path.suffix.lower() != ".usdz" or not source_path.is_file():
        raise ValueError("必须选择一个存在的 USDZ 模型。")

    collision_type = str(collision_type or "").strip().lower()
    body_type = str(body_type or "").strip().lower()
    if collision_type not in COLLISION_TYPES:
        raise ValueError(f"不支持的碰撞体类型: {collision_type or '空'}。")
    if body_type not in BODY_TYPES:
        raise ValueError("刚体类型只能选择 static 或 dynamic。")

    mass = float(mass)
    friction = float(friction)
    if mass <= 0:
        raise ValueError("质量必须大于 0 kg。")
    if not 0 <= friction <= 10:
        raise ValueError("摩擦系数必须在 0 到 10 之间。")
    if body_type == "dynamic" and collision_type == "triangle_mesh":
        raise ValueError("动态刚体不能使用三角网格碰撞，请选择包围盒、凸包或凸分解。")

    output_path = source_path.with_suffix(".usd")
    temp_path = source_path.parent / f".{source_path.stem}.physics.tmp.usda"
    temp_path.unlink(missing_ok=True)
    try:
        collider_paths = _write_physics_stage(
            source_path,
            temp_path,
            mass=mass,
            collision_type=collision_type,
            friction=friction,
            body_type=body_type,
        )
        temp_path.replace(output_path)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"USD file was not created: {output_path}")
        validation_stage = Usd.Stage.Open(str(output_path))
        if validation_stage is None or not validation_stage.GetDefaultPrim():
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Generated USD could not be reopened: {output_path}")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "usdz_path": str(source_path),
        "usd_path": str(output_path),
        "mass": mass,
        "friction": friction,
        "collision_type": collision_type,
        "body_type": body_type,
        "collider_paths": collider_paths,
        "size_bytes": output_path.stat().st_size,
    }


def _write_physics_stage(
    source_path: Path,
    temp_path: Path,
    *,
    mass: float,
    collision_type: str,
    friction: float,
    body_type: str,
) -> list[str]:
    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"无法打开 USDZ 模型: {source_path}")
    source_default_prim = source_stage.GetDefaultPrim()
    if not source_default_prim:
        raise RuntimeError("USDZ 模型没有 Default Prim，无法创建可靠引用。")

    stage = Usd.Stage.CreateNew(str(temp_path))
    if stage is None:
        raise RuntimeError(f"无法创建临时 USD 文件: {temp_path}")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))

    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("sourceAsset", source_path.name)
    root_prim.SetCustomDataByKey("rigidBodyType", body_type)
    root_prim.SetCustomDataByKey("collisionType", collision_type)

    visual_prim = stage.OverridePrim("/Asset/Visual")
    visual_prim.GetReferences().AddReference(
        source_path.name,
        source_default_prim.GetPath(),
    )

    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(mass)
    if body_type == "dynamic":
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(root_prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(False)

    physics_material = UsdShade.Material.Define(stage, "/PhysicsMaterials/PhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(friction)
    material_api.CreateDynamicFrictionAttr(friction)
    material_api.CreateRestitutionAttr(0.0)

    collider_paths: list[str] = []
    if collision_type == "bounding_box":
        collider_prim = _create_bounding_box_collider(stage, source_stage, source_default_prim)
        _bind_physics_material(collider_prim, physics_material)
        collider_paths.append(str(collider_prim.GetPath()))
    else:
        stage.Load()
        approximation = COLLISION_TYPES[collision_type]
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr(True)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr(approximation)
            _bind_physics_material(prim, physics_material)
            collider_paths.append(str(prim.GetPath()))
        if not collider_paths:
            raise RuntimeError("USDZ 中没有找到可添加碰撞属性的 Mesh。")

    stage.GetRootLayer().Save()
    return collider_paths


def _create_bounding_box_collider(
    stage: Usd.Stage,
    source_stage: Usd.Stage,
    source_prim: Usd.Prim,
) -> Usd.Prim:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    aligned_range = bbox_cache.ComputeWorldBound(source_prim).ComputeAlignedRange()
    minimum = aligned_range.GetMin()
    maximum = aligned_range.GetMax()
    if not all(_is_finite(value) for value in (*minimum, *maximum)):
        raise RuntimeError("无法从 USDZ 计算有效的模型包围盒。")

    size = maximum - minimum
    center = (minimum + maximum) * 0.5
    dimensions = Gf.Vec3d(
        max(float(size[0]), 1e-4),
        max(float(size[1]), 1e-4),
        max(float(size[2]), 1e-4),
    )

    collider = UsdGeom.Cube.Define(stage, "/Asset/Collider")
    collider.CreateSizeAttr(1.0)
    collider.AddTranslateOp().Set(Gf.Vec3d(center))
    collider.AddScaleOp().Set(dimensions)
    collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    collision_api = UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    return collider.GetPrim()


def _bind_physics_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim)
    relationship = prim.CreateRelationship("material:binding:physics")
    relationship.SetTargets([material.GetPath()])
    relationship.SetMetadata("bindMaterialAs", UsdShade.Tokens.weakerThanDescendants)


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a physics-enabled USD wrapper for a USDZ asset.")
    parser.add_argument("usdz_path")
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--collision-type", choices=sorted(COLLISION_TYPES), required=True)
    parser.add_argument("--friction", type=float, required=True)
    parser.add_argument("--body-type", choices=sorted(BODY_TYPES), required=True)
    args = parser.parse_args()
    try:
        result = create_physics_asset(
            args.usdz_path,
            mass=args.mass,
            collision_type=args.collision_type,
            friction=args.friction,
            body_type=args.body_type,
        )
        result_json = json.dumps(result, ensure_ascii=False)
        print(
            f"[physics-usd] Created and verified: {result['usd_path']} "
            f"({result['size_bytes']} bytes)",
            file=sys.stderr,
            flush=True,
        )
        print(f"{RESULT_PREFIX}{result_json}", flush=True)
    finally:
        if SIMULATION_APP is not None:
            SIMULATION_APP.close()


if __name__ == "__main__":
    main()
