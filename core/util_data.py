#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import shutil
import time
import copy
import base64
import math
import numpy as np
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
from PIL import Image


def _import_core_util():
    """兼容包内导入与 render_ssl 脚本通过 importlib 加载 util_data 的场景。"""
    try:
        from . import util
        return util
    except ImportError:
        core_dir = os.path.dirname(os.path.abspath(__file__))
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        import util  # type: ignore
        return util


def _get_vlm_api_key() -> str:
    return os.environ.get("PIPE_VLM_API_KEY", "")


def _get_vlm_base_url() -> str:
    return os.environ.get("PIPE_VLM_BASE_URL", "https://oneapi.qunhequnhe.com/v1")


def _ensure_qwen_path() -> None:
    path = str(Path(__file__).resolve().parents[2] / "Qwen3-VL-Embedding")
    if path not in sys.path:
        sys.path.insert(0, path)


def _ensure_react_path() -> None:
    path = "/data-nas/data/experiments/mushui/projects/react"
    if path not in sys.path:
        sys.path.insert(0, path)


def _get_edit_image_with_qunhe():
    _ensure_react_path()
    try:
        from qunhe_api.nano_gemini import edit_image_with_qunhe
        return edit_image_with_qunhe
    except ImportError:
        return None


def _get_hunyuan_gen():
    _ensure_react_path()
    try:
        from qunhe_api.gen_hunyuan import hunyuan_gen
        return hunyuan_gen
    except ImportError:
        return None


def _get_qwen3_embedder_class():
    _ensure_qwen_path()
    try:
        from qwen3_vl_embedding import Qwen3VLEmbedder
        return Qwen3VLEmbedder
    except ImportError:
        return None


_gemini_model = None


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        from agentscope.model import OpenAIChatModel
        _gemini_model = OpenAIChatModel(
            api_key=_get_vlm_api_key(),
            model_name="gemini-3.1-pro-preview",
            stream=True,
            client_kwargs={"base_url": _get_vlm_base_url()},
            generate_kwargs={
                "temperature": 1,
                "max_tokens": 65000,
            },
        )
    return _gemini_model


def _create_orientation_agent():
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    return ReActAgent(
        name="Cooler",
        sys_prompt="You are a helpful canonical orientation estimator.",
        model=_get_gemini_model(),
        formatter=OpenAIChatFormatter(),
    )

def clean_str(content) -> str:
    """将模型返回的原始 content 转换为带 Python 结构但换行符被还原的干净字符串。"""
    if not content:
        return ""
    s = str(content)
    # 还原转义字符，使其变为真实的换行和引号
    return s.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\").strip()


def get_base64_data(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_data = base64.b64encode(image_data).decode("utf-8")
    return base64_data
async def run_conversation(agent, prompt, *image_paths) -> None:
    """运行与模型的对话，支持传入多张图片。"""
    from agentscope.message import Msg, ImageBlock, Base64Source, TextBlock
    contents = [
        ImageBlock(
            type="image",
            source=Base64Source(type="base64", media_type="image/png", data=get_base64_data(path)),
        )
        for path in image_paths
    ]
    contents.append(TextBlock(type="text", text=prompt))
    msg = Msg(name="user", role="user", content=contents)
    resp = await agent(msg)
    return resp.content





def correct_single_asset(
    asset_path: str, 
    label: str = None, 
    caption: str = None,
    correct_tilt: bool = False,
    bbox_cropped_path: Optional[str] = None):
    import asyncio
    import pyrender
    import trimesh
    from jinja2 import Template

    _ = (label, caption)  # keep signature compatibility
    asset_file = Path(asset_path)
    suffix = asset_file.suffix.lower()
    if suffix not in {".glb", ".gltf"}:
        raise ValueError(f"Only .glb/.gltf are supported, got: {asset_path}")

    loaded = trimesh.load(asset_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        scene = trimesh.Scene(loaded)

    if scene.is_empty:
        raise ValueError(f"Empty scene/mesh: {asset_path}")

    # Normalize asset pose: move bbox center to world origin.
    initial_bounds = scene.bounds
    initial_center = initial_bounds.mean(axis=0)
    move_to_origin = trimesh.transformations.translation_matrix(-initial_center)
    scene.apply_transform(move_to_origin)

    bounds = scene.bounds
    center = bounds.mean(axis=0)  # [x, y, z]
    extent = bounds[1] - bounds[0]  # [a, b, c]
    m = float(np.max(extent))
    x, y, z = center.tolist()

    camera_positions = {
        "front": np.array([x, y, z + 2*m], dtype=np.float64),
        # "right": np.array([x + 2*m, y, z], dtype=np.float64),
        "up": np.array([x, y + 2*m, z], dtype=np.float64),
    }

    def camera_pose_look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Trimesh camera convention: camera looks toward -Z in its local frame.
        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(z_axis, up_hint)) > 0.999:
            up_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        x_axis = np.cross(up_hint, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = eye
        return pose

    x_fov_deg, y_fov_deg = 60.0, 45.0
    aspect_ratio = float(
        np.tan(np.deg2rad(x_fov_deg / 2.0)) / np.tan(np.deg2rad(y_fov_deg / 2.0))
    )
    width = 800
    height = int(round(width / aspect_ratio))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    out_dir = asset_file.parent
    stem = asset_file.stem
    rendered = []
    try:
        for name, eye in camera_positions.items():
            py_scene = pyrender.Scene.from_trimesh_scene(
                scene,
                bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
                ambient_light=np.array([0.15, 0.15, 0.15], dtype=np.float32),
            )

            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(y_fov_deg),
                aspectRatio=aspect_ratio,
            )
            camera_pose = camera_pose_look_at(eye=eye, target=center)
            py_scene.add(camera, pose=camera_pose)

            light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            py_scene.add(light, pose=camera_pose)

            color, _ = renderer.render(py_scene)
            out_path = out_dir / f"{stem}_{name}.png"
            Image.fromarray(color).save(out_path)
            rendered.append(str(out_path))
    finally:
        renderer.delete()

    print("Rendered images:")
    for p in rendered:
        print(p)

    #! 第一轮,  给定前视图和俯视图, 通过判断哪个是物体的俯视图, 确定物体的 y 轴也就是 world_up(防止类似于地毯的物体立起来)
    agent = _create_orientation_agent()

    direction_prompt = Template('''
    现在给你看两张图片, 依次是一个三维物体从两个方向看去的渲染图片, 请你从中找出一个最可能是从该物体的上方看过去的, 在你判断的过程中, 需要参考以下规则: 
    - 所谓物体上方是指物体在自然摆放的时候的上方的位置, 例如一个桌子, 床, 椅子, 柜子, 衣架, 灯等家具
    - 对于枕头等可以通过靠着床头立起来的物体, 也要将它在床上放平自然摆放后, 判定此时的上方, 也就是最大的那一面为上方
    - 对于墙上的物体, 例如挂画, 门窗等比较薄的物体, 判断其上方的方式是其正常镶嵌在墙里或者挂在墙上时上方的位置, 也就是从上方看可能是一个细条或者一道线, 只有从正面看才能看到全貌
    如果物体的 label 和 caption 不为 none, 你也可以作为参考, 否则请忽略, 该物体的 label 是 {{label}},  caption 是 {{caption}}.
    你的输出结果是 1, 或者 2, 分别表示第一张图片和第二张图片中哪个是最可能是从该物体的上方看过去的, 请输出你的分析过程并最终严格按照三星号包裹的格式给出最终答案***1*** 或 ***2***.
    ''').render(label=label, caption=caption)


    max_retry = 20
    result = None
    try:
        for attempt in range(1, max_retry + 1):
            try:
                content = asyncio.run(run_conversation(agent, direction_prompt, rendered[0], rendered[1]))
                content = clean_str(content)
                matches = re.findall(r"(\*\*\*1\*\*\*|\*\*\*2\*\*\*)", content)
                result = int(matches[-1].strip("*"))
                assert result in {1, 2}, f"Invalid direction result: {result}"
                break
            except Exception as e:
                print(f"Error in attempt {attempt}: {e}")
                if attempt == max_retry:
                    raise RuntimeError(
                        f"Failed after {max_retry} attempts for asset: {asset_path}"
                    ) from e
                continue
        if result == 1:
            # Clockwise 90 deg around +X axis => right-hand angle -90 deg.
            rot_x_cw_90 = trimesh.transformations.rotation_matrix(
                angle=-np.pi / 2.0,
                direction=[1.0, 0.0, 0.0],
                point=center,
            )
            scene.apply_transform(rot_x_cw_90)
            print("Applied clockwise 90deg rotation around X.")
        else:
            print("No rotation applied.")
    finally:
        for p in rendered:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception as e:
                print(f"Failed to delete temp image {p}: {e}")
    
    # Recompute geometry stats after potential transform for further processing.
    bounds = scene.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    m = float(np.max(extent))
    x, y, z = center.tolist()

    #! 第二轮: 通过前视图+侧视图, 判断物体的正面
    camera_positions = {
        "front": np.array([x, y, z + 2*m], dtype=np.float64),
        "right": np.array([x + 2*m, y, z], dtype=np.float64),
    }
    
    # Judge which of {front, right} is most likely the canonical front.
    front_candidates = []
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        for name, eye in camera_positions.items():
            py_scene = pyrender.Scene.from_trimesh_scene(
                scene,
                bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
                ambient_light=np.array([0.15, 0.15, 0.15], dtype=np.float32),
            )
            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(y_fov_deg),
                aspectRatio=aspect_ratio,
            )
            camera_pose = camera_pose_look_at(eye=eye, target=center)
            py_scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            py_scene.add(light, pose=camera_pose)

            color, _ = renderer.render(py_scene)
            out_path = out_dir / f"{stem}_{name}_front_judge.png"
            Image.fromarray(color).save(out_path)
            front_candidates.append((name, str(out_path)))
    finally:
        renderer.delete()

    print("Front judge images:")
    for idx, (name, path) in enumerate(front_candidates, start=1):
        print(f"{idx}. {name}: {path}")

    front_prompt = Template('''
    现在给你看两张图片, 依次是一个三维物体从两个方向看去的渲染图片。
    请判断哪一张最可能是该物体的正面视角:
    - 1 表示第一张
    - 2 表示第二张
    参考规则:
    - 家具常见“正面”往往是功能面(门板/抽屉/屏幕/把手/开口)更明显的一侧
    - 若两者都不明显, 则选择更宽的那一面作为正面, 如果两者宽度相同, 则选择更符合人类直觉“朝前”的一张, 并可说明不确定性
    - L 形家具的正面应该正对着较宽的那条边
    如果物体的 label 和 caption 不为 none, 可作为参考, 否则忽略。该物体 label={{label}}, caption={{caption}}。
    最终答案必须严格是: ***1*** 或 ***2***.
    ''').render(label=label, caption=caption)

    max_retry = 20
    front_result = None
    try:
        for attempt in range(1, max_retry + 1):
            try:
                content = asyncio.run(
                    run_conversation(
                        agent,
                        front_prompt,
                        front_candidates[0][1],
                        front_candidates[1][1],
                    )
                )
                content = clean_str(content)
                matches = re.findall(r"(\*\*\*1\*\*\*|\*\*\*2\*\*\*)", content)
                front_result = int(matches[-1].strip("*"))
                assert front_result in {1, 2}, f"Invalid front result: {front_result}"
                break
            except Exception as e:
                print(f"Front-judge error in attempt {attempt}: {e}")
                if attempt == max_retry:
                    raise RuntimeError(
                        f"Front-direction judgment failed after {max_retry} attempts "
                        f"for asset: {asset_path}"
                    ) from e
                continue
    finally:
        for _, p in front_candidates:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception as e:
                print(f"Failed to delete temp image {p}: {e}")

    if front_result == 2:
        # Clockwise 90 deg around +Y axis => right-hand angle -90 deg.
        rot_y_cw_90 = trimesh.transformations.rotation_matrix(
            angle=-np.pi / 2.0,
            direction=[0.0, 1.0, 0.0],
            point=center,
        )
        scene.apply_transform(rot_y_cw_90)
        print("Applied clockwise 90deg rotation around Y.")
    else:
        print("No rotation applied.")

    #! 第三轮: 通过前视图+侧视图+俯视图, 判断物体三个方向是否有倾斜

    if correct_tilt:
        # 第三轮处理
        bounds = scene.bounds
        center = bounds.mean(axis=0)
        extent = bounds[1] - bounds[0]
        m = float(np.max(extent))
        x, y, z = center.tolist()

        camera_positions = {
            "front": np.array([x, y, z + 2 * m], dtype=np.float64),
            "right": np.array([x + 2 * m, y, z], dtype=np.float64),
            "up": np.array([x, y + 2 * m, z], dtype=np.float64),
        }
        # Render 3 views for tilt estimation and correction.
        tilt_candidates = []
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            for name, eye in camera_positions.items():
                py_scene = pyrender.Scene.from_trimesh_scene(
                    scene,
                    bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
                    ambient_light=np.array([0.15, 0.15, 0.15], dtype=np.float32),
                )
                camera = pyrender.PerspectiveCamera(
                    yfov=np.deg2rad(y_fov_deg),
                    aspectRatio=aspect_ratio,
                )
                camera_pose = camera_pose_look_at(eye=eye, target=center)
                py_scene.add(camera, pose=camera_pose)
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
                py_scene.add(light, pose=camera_pose)

                color, _ = renderer.render(py_scene)
                out_path = out_dir / f"{stem}_{name}_tilt_judge.png"
                Image.fromarray(color).save(out_path)
                tilt_candidates.append((name, str(out_path)))
        finally:
            renderer.delete()

        print("Tilt judge images:")
        for idx, (name, path) in enumerate(tilt_candidates, start=1):
            print(f"{idx}. {name}: {path}")

        tilt_prompt = Template('''
        现在给你看同一个三维物体的 3 张渲染图，顺序固定为:
        1) front: 从 z 轴正方向看向负方向
        2) right: 从 x 轴正方向看向负方向
        3) up: 从 y 轴正方向看向负方向

        任务: 判断物体是否有“明显倾斜”。只有在倾斜非常明显时才给非零修正角，否则三个轴都输出 0。
        角度定义:
        - 正角度表示逆时针旋转可以摆正
        - 负角度表示顺时针旋转可以摆正
        - 单位是度

        请给出绕 x/y/z 三个轴的修正角(用于把物体摆正)。
        如果你不确定，请保守输出 0，避免过度修正。
        如果物体 label/caption 可用，可以参考: label={{label}}, caption={{caption}}。

        最后必须严格输出如下格式(只允许一行):
        ***X=<float>,Y=<float>,Z=<float>***
        例如: ***X=0,Y=-12.5,Z=3***
        ''').render(label=label, caption=caption)

        max_retry = 20
        tilt_x = tilt_y = tilt_z = 0.0
        try:
            for attempt in range(1, max_retry + 1):
                try:
                    content = asyncio.run(
                        run_conversation(
                            agent,
                            tilt_prompt,
                            tilt_candidates[0][1],
                            tilt_candidates[1][1],
                            tilt_candidates[2][1],
                        )
                    )
                    content = clean_str(content)
                    angle_match = re.findall(
                        r"\*\*\*X\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*Y\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*Z\s*=\s*([+-]?\d+(?:\.\d+)?)\*\*\*",
                        content,
                        flags=re.IGNORECASE,
                    )
                    if not angle_match:
                        raise ValueError(f"Cannot parse tilt angles from response: {content}")

                    tilt_x, tilt_y, tilt_z = map(float, angle_match[-1])
                    break
                except Exception as e:
                    print(f"Tilt-judge error in attempt {attempt}: {e}")
                    if attempt == max_retry:
                        raise RuntimeError(
                            f"Tilt estimation failed after {max_retry} attempts "
                            f"for asset: {asset_path}"
                        ) from e
                    continue
        finally:
            for _, p in tilt_candidates:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Failed to delete temp image {p}: {e}")

        print(f"Estimated tilt correction(deg): X={tilt_x}, Y={tilt_y}, Z={tilt_z}")

        # Apply non-zero axis corrections only, in X->Y->Z order.
        applied_axes = []
        if abs(tilt_x) > 1e-8:
            rot_x = trimesh.transformations.rotation_matrix(
                angle=np.deg2rad(tilt_x), direction=[1.0, 0.0, 0.0], point=center
            )
            scene.apply_transform(rot_x)
            applied_axes.append("X")
        if abs(tilt_y) > 1e-8:
            rot_y = trimesh.transformations.rotation_matrix(
                angle=np.deg2rad(tilt_y), direction=[0.0, 1.0, 0.0], point=center
            )
            scene.apply_transform(rot_y)
            applied_axes.append("Y")
        if abs(tilt_z) > 1e-8:
            rot_z = trimesh.transformations.rotation_matrix(
                angle=np.deg2rad(tilt_z), direction=[0.0, 0.0, 1.0], point=center
            )
            scene.apply_transform(rot_z)
            applied_axes.append("Z")

        if applied_axes:
            print(f"Applied tilt correction on axes: {', '.join(applied_axes)} (not saved yet).")
        else:
            print("All tilt angles are zero, skipped tilt correction.")
    else:
        print("Skip third-round tilt correction (correct_tilt=False).")

    #! 第四轮处理：对比 asset 俯视图与 bbox crop，估计绕 y 轴的朝向修正角(主要是对齐图里面的内容, 防止前面的流程 orientation 判断错)
    if bbox_cropped_path and os.path.exists(bbox_cropped_path):
        bounds = scene.bounds
        center = bounds.mean(axis=0)
        extent = bounds[1] - bounds[0]
        m = float(np.max(extent))
        x, y, z = center.tolist()

        top_eye = np.array([x, y + 2 * m, z], dtype=np.float64)
        top_render_path = out_dir / f"{stem}_top_compare.png"
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            py_scene = pyrender.Scene.from_trimesh_scene(
                scene,
                bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
                ambient_light=np.array([0.15, 0.15, 0.15], dtype=np.float32),
            )
            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(y_fov_deg),
                aspectRatio=aspect_ratio,
            )
            camera_pose = camera_pose_look_at(eye=top_eye, target=center)
            py_scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            py_scene.add(light, pose=camera_pose)
            color, _ = renderer.render(py_scene)
            Image.fromarray(color).save(top_render_path)
        finally:
            renderer.delete()

        compare_prompt = Template('''
        在给定的俯视图设定中，已知 y 轴垂直于水平地面向上，x 轴在图片中水平向右，z 轴指向图片上方（但不是真正的水平面上方）。
        现在给你两张图片：一张是目标资产的俯视图，一张是从目标场景俯视图中把目标资产裁剪出来的局部俯视图（尽管我们的视角在场景中心的正上方，但可能不在物体的正上方）。
        - 第 1 张：从单个 3D asset 的正上方俯视渲染图（从 y 轴正方向看向负方向）。
        - 第 2 张：原图里直接按 bbox 裁剪得到的目标局部图。原图是场景中心正上方 (x_center, y_camera, z_center) 看向场景地面中心 (x_center, 0, z_center) 的透视相机拍摄，图 2 是其中目标物体的裁剪区域。

        重要：图 2 来自透视投影。当物体靠近画面边缘时，透视会导致物体与地面垂直的面在 2D 图像中看起来倾斜，但这不代表其真实 3D 朝向一定有偏差。此时应优先根据物体与支撑面接触处的底线在俯视平面上的投影与 x 轴的角度来判断物体是否真的有倾斜。例如：门朝向画面下方时，若门靠近边缘，透视会让门面在图中呈斜线，此时门真实朝向应根据门与地面相交的底线在俯视平面上的投影是否平行于 x 轴来判断, 或者根据物体的顶部纹理来判断(不能是侧面)。若物体没有清晰的底线（如柜子、桌子）, 也看不到物体顶部纹理，难以可靠判断时，输出 0。对于门窗等看不到顶部纹理的物体就只能通过底线判断了.

        任务：判断 asset 的真实 3D 朝向与图 2 中物体的真实朝向是否一致；若不一致，需要绕 y 轴旋转多少度。
        角度定义：
        - 逆时针旋转为正角度
        - 顺时针旋转为负角度
        - 单位是度，可为小数
        - 如果无法可靠判断，输出 0
        - 如果两张图主朝向差异不明显（例如接近、模糊、近似对齐、对称导致难判断），一律输出 0
        - 若图 2 中的倾斜很可能是透视变形造成（物体靠近边缘、仰视/俯视导致的投影变形），一律输出 0
        - 只有当你能通过底线和顶部纹理明确区分「真实朝向偏差」与「透视造成的视觉倾斜」，且确认存在真实偏差时，才输出非 0 角度

        参考信息（若非 none）：label={{label}}, caption={{caption}}。
        你可以先分析，但最终必须严格按以下格式输出（仅一行）：
        ***Y=<float>***
        例如：***Y=90*** 或 ***Y=-22.5*** 或 ***Y=0***。
        ''').render(label=label, caption=caption)

        max_retry = 20
        yaw_correction = 0.0
        try:
            for attempt in range(1, max_retry + 1):
                try:
                    content = asyncio.run(
                        run_conversation(
                            agent,
                            compare_prompt,
                            str(top_render_path),
                            bbox_cropped_path,
                        )
                    )
                    content = clean_str(content)
                    angle_match = re.findall(
                        r"\*\*\*Y\s*=\s*([+-]?\d+(?:\.\d+)?)\*\*\*",
                        content,
                        flags=re.IGNORECASE,
                    )
                    if not angle_match:
                        raise ValueError(f"Cannot parse Y angle from response: {content}")
                    yaw_correction = float(angle_match[-1])
                    break
                except Exception as e:
                    print(f"Top-vs-bbox yaw-judge error in attempt {attempt}: {e}")
                    if attempt == max_retry:
                        raise RuntimeError(
                            f"Top-vs-bbox yaw estimation failed after {max_retry} attempts "
                            f"for asset: {asset_path}"
                        ) from e
                    continue

            print(f"Estimated top-vs-bbox yaw correction(deg): Y={yaw_correction}")
            if yaw_correction == 0.0:
                print("Top-vs-bbox yaw correction is 0, skipped.")
            elif abs(yaw_correction) > 1e-8:
                bounds = scene.bounds
                center = bounds.mean(axis=0)
                rot_y = trimesh.transformations.rotation_matrix(
                    angle=np.deg2rad(yaw_correction), direction=[0.0, 1.0, 0.0], point=center
                )
                scene.apply_transform(rot_y)
                print(f"Applied top-vs-bbox yaw correction around Y: {yaw_correction} deg.")
            else:
                print("Top-vs-bbox yaw correction is 0, skipped.")
        finally:
            try:
                Path(top_render_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"Failed to delete temp image {top_render_path}: {e}")
    else:
        print("Skip top-vs-bbox yaw correction (bbox_cropped_path missing).")

    # save and cover the asset
    scene.export(asset_path)
    print(f"Saved and covered asset: {asset_path}")


def _coerce_float_list(value, size: Optional[int] = None) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = [float(x) for x in value]
    else:
        out = [float(value)]
    if size is not None:
        while len(out) < size:
            out.append(0.0)
        out = out[:size]
    return out


def _normalize_wall_item(w: Dict[str, Any]) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "p": _coerce_float_list(w.get("p"), 3),
        "q": _coerce_float_list(w.get("q"), 3),
        "height": float(w.get("height", 2.8)),
    }
    if w.get("id") is not None:
        item["id"] = w["id"]
    if w.get("room_id") is not None:
        item["room_id"] = w["room_id"]
    return item


def _normalize_opening_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "center": _coerce_float_list(item.get("center"), 3),
        "width": float(item.get("width", 0.0)),
        "height": float(item.get("height", 0.0)),
    }
    for key in (
        "id", "wall_id", "asset_id", "category", "label", "caption",
        "mesh_id", "bbox_2d", "category_zh", "category_id",
    ):
        if item.get(key) is not None:
            out[key] = item[key]
    return out


def _normalize_bbox_item(item: Dict[str, Any]) -> Dict[str, Any]:
    label = item.get("label") or item.get("category_zh") or item.get("category") or ""
    out: Dict[str, Any] = {
        "center": _coerce_float_list(item.get("center"), 3),
        "angle_z": float(item.get("angle_z", 0.0)),
        "scale": _coerce_float_list(item.get("scale"), 3),
        "label": label,
    }
    for key in (
        "id", "room_id", "asset_id", "caption", "mesh_id", "bbox_2d",
        "category", "category_zh", "category_id",
    ):
        if item.get(key) is not None:
            out[key] = item[key]
    return out


def _normalize_room(room: Any) -> Dict[str, Any]:
    if isinstance(room, str):
        return {"room_type": room}
    if not isinstance(room, dict):
        return {"room_type": "unknown"}
    out: Dict[str, Any] = {
        "room_type": room.get("room_type") or room.get("type") or "unknown",
    }
    for key in ("id", "room_id", "type_id"):
        if room.get(key) is not None:
            out[key] = room[key]
    return out


def normalize_scene_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """将外部 JSON 场景 dict 规范化为 render_ssl 使用的 scene_json 结构。"""
    walls = obj.get("wall")
    if walls is None:
        walls = obj.get("walls", [])
    doors = obj.get("door")
    if doors is None:
        doors = obj.get("doors", [])
    windows = obj.get("window")
    if windows is None:
        windows = obj.get("windows", [])
    bboxes = obj.get("bbox")
    if bboxes is None:
        bboxes = obj.get("bboxes", obj.get("boxes", []))

    return {
        "wall": [_normalize_wall_item(w) for w in (walls or [])],
        "door": [_normalize_opening_item(d) for d in (doors or [])],
        "window": [_normalize_opening_item(w) for w in (windows or [])],
        "bbox": [_normalize_bbox_item(b) for b in (bboxes or [])],
        "room": _normalize_room(obj.get("room", {})),
    }


def parse_scene_input(scene_text: str) -> Dict[str, Any]:
    """解析场景输入：支持标准 SSL 文本，或含 wall/door/window/bbox/room 的 JSON 字符串。"""
    stripped = (scene_text or "").strip()
    if not stripped:
        raise ValueError("场景输入为空")
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 场景解析失败: {exc}") from exc
        if isinstance(payload, dict):
            if any(k in payload for k in ("wall", "walls", "door", "doors", "window", "windows", "bbox", "bboxes", "boxes")):
                return normalize_scene_json(payload)
            raise ValueError("JSON 缺少 wall/door/window/bbox 等场景字段")
        raise ValueError("JSON 场景必须是 object（dict）")
    return parse_ssl_to_json(stripped)


def parse_ssl_to_json(ssl_text: str) -> Dict[str, Any]:
    """
    解析SSL格式场景描述并返回指定的JSON结构
    """
    data = {
        "wall": [],
        "door": [],
        "window": [],
        "bbox": [],
        "room": {"room_type": "unknown"}
    }

    # 提取属性的正则辅助函数
    def get_attr(pattern, text, default=None):
        match = re.search(pattern, text)
        return match.group(1) if match else default

    def get_list_attr(pattern, text, default=None):
        match = re.search(pattern, text)
        if match:
            return [float(x.strip()) for x in match.group(1).split(',')]
        return default

    def add_if_present(target_dict, key, value):
        if value is not None:
            target_dict[key] = value

    # 存储解析出的 caption，以便后续匹配
    captions_map = {}

    for line in ssl_text.strip().split('\n'):
        line = line.strip()
        if not line: continue

        # 解析 Caption(id="...", caption="...")
        if line.startswith('Caption('):
            cid = get_attr(r'id="([^"]+)"', line)
            cval = get_attr(r'caption="([^"]+)"', line)
            if cid and cval:
                captions_map[cid] = cval
            continue

        # Room(id="...", room_type="...")
        if line.startswith('Room('):
            add_if_present(data["room"], "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(data["room"], "room_type", get_attr(r'room_type="([^"]+)"', line))

        # Wall(id="...", room_id="...", p=[...], q=[...], height=...)
        elif line.startswith('Wall('):
            wall_item = {
                "p": get_list_attr(r'p=\[([^\]]+)\]', line, []),
                "q": get_list_attr(r'q=\[([^\]]+)\]', line, []),
                "height": float(get_attr(r'height=([\d.]+)', line, "2.8"))
            }
            add_if_present(wall_item, "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(wall_item, "room_id", get_attr(r'room_id="([^"]+)"', line))
            add_if_present(wall_item, "caption", get_attr(r'caption="([^"]+)"', line))
            data["wall"].append(wall_item)

        # Door(id="...", wall_id="...", center=[...], width=..., height=..., asset_id="...", category="...", label="...", caption="...", bbox_2d=[...], mesh_id="...")
        elif line.startswith('Door(') or line.startswith('Window('):
            is_door = line.startswith('Door(')
            item = {
                "center": get_list_attr(r'center=\[([^\]]+)\]', line, []),
                "width": float(get_attr(r'width=([\d.]+)', line, "0")),
                "height": float(get_attr(r'height=([\d.]+)', line, "0")),
            }
            add_if_present(item, "category", get_attr(r'category="([^"]+)"', line))
            add_if_present(item, "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(item, "wall_id", get_attr(r'wall_id="([^"]+)"', line))
            add_if_present(item, "mesh_id", get_attr(r'mesh_id="([^"]+)"', line))
            add_if_present(item, "label", get_attr(r'label="([^"]+)"', line))
            add_if_present(item, "caption", get_attr(r'caption="([^"]+)"', line))
            add_if_present(item, "bbox_2d", get_list_attr(r'bbox_2d=\[([^\]]+)\]', line))
            add_if_present(item, "asset_id", get_attr(r'asset_id="([^"]+)"', line))

            if is_door:
                data["door"].append(item)
            else:
                data["window"].append(item)

        # Bbox(id="...", room_id="...", asset_id="...", center=[...], angle_z=..., scale=[...], caption="...", label="...", bbox_2d=[...], mesh_id="...")
        elif line.startswith('Bbox('):
            bbox_item = {
                "center": get_list_attr(r'center=\[([^\]]+)\]', line, []),
                "angle_z": float(get_attr(r'angle_z=([\d.-]+)', line, "0")),
                "scale": get_list_attr(r'scale=\[([^\]]+)\]', line, []),
                "label": get_attr(r'label="([^"]+)"', line, "")
            }
            add_if_present(bbox_item, "id", get_attr(r'id="([^"]+)"', line))
            add_if_present(bbox_item, "room_id", get_attr(r'room_id="([^"]+)"', line))
            add_if_present(bbox_item, "mesh_id", get_attr(r'mesh_id="([^"]+)"', line))
            add_if_present(bbox_item, "asset_id", get_attr(r'asset_id="([^"]+)"', line))
            add_if_present(bbox_item, "caption", get_attr(r'caption="([^"]+)"', line))
            add_if_present(bbox_item, "bbox_2d", get_list_attr(r'bbox_2d=\[([^\]]+)\]', line))
            data["bbox"].append(bbox_item)

    # 将独立的 Caption 匹配回对应的物体
    for category in ["bbox", "door", "window"]:
        if category in data:
            for item in data[category]:
                item_id = item.get("id")
                if item_id in captions_map:
                    item["caption"] = captions_map[item_id]

    return data

def _format_asset_prefix(mesh_id: Any, timestamp: str, label: str) -> str:
    """基于分组 mesh_id 生成模型 asset_id 前缀: 三位mesh_id_时间戳"""
    try:
        mesh_num = int(mesh_id)
    except (TypeError, ValueError):
        try:
            mesh_num = int(float(mesh_id))
        except (TypeError, ValueError):
            mesh_num = 0
    return f"{mesh_num:03d}_{timestamp}"


def process_image_for_generation(
    generated_asset_id: str,
    label: str,
    caption: str,
    output_dir: str,
) -> str:
    """
    使用 nanobanana (edit_image_with_qunhe) 将裁剪后的俯视图转换为物体正视图。
    """
    # 检查目标文件是否已存在，如果存在则直接返回
    final_output_path = os.path.join(output_dir, f"{generated_asset_id}_{label}.png")
    if os.path.exists(final_output_path):
        print(f"⏭️ 跳过图像生成，已存在: {final_output_path}")
        return final_output_path

    input_tmp_path = os.path.join(output_dir, f"{generated_asset_id}_mask_cropped.png")
    bbox_tmp_path = os.path.join(output_dir, f"{generated_asset_id}_bbox_cropped.png")
    if not os.path.exists(input_tmp_path) and os.path.exists(bbox_tmp_path):
        input_tmp_path = bbox_tmp_path
    edit_image_with_qunhe = _get_edit_image_with_qunhe()
    if edit_image_with_qunhe is None:
        print("Warning: edit_image_with_qunhe not found, saving original.")
        return input_tmp_path if os.path.exists(input_tmp_path) else ""

    # 构建 prompt
    nano_prompt = f"现在是一幅从俯视图裁剪出来的物体图片, 请你根据描述中的物体形状补全残缺的部分并生成物体清晰的, 真实的, 带有丰富纹理细节的图像, 为了突出实体, 背景为黑色, 注意不能改变已有部分的形状, 只能补充缺失的部分并让纹理变得更加清晰 , 注意如果裁剪出的图片只能看到顶面, 对于一些立体物体, 请你将你视角往物体正前方偏移, 让生成的图片有立体感, (例如给定桌子/柜子/家具如果只能顶面看不到侧前方的话, 这时候对于桌子你需要让视角向物体正面偏移能够同时看到桌面和桌腿, 对于柜子/家具也是类似视角偏移与补全让图片更有立体感),对于一些薄片物体则不需要偏移(例如地毯, 画作等), 同时你必须保持原始裁剪图中的物体形状和细节不变,具有高度的一致性和整体性, 同时生成的图像只关注文本描述中的部分,注意不要在物体上留下黑洞, 例如桌面因为其他物体的分割导致出现的黑洞请补齐, 你生成的必须是完整的物体加上黑色背景, 物体不能超出图片边缘被截断,  后面我会给出该物体的文本描述, 但是你要注意文本描述只是对给定图片的补充和参考, 物体描述: \n 该物体是一个{label}, 具体描述为{caption}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 caption 到 {asset_id}_{label}_caption.txt
    caption_path = os.path.join(output_dir, f"{generated_asset_id}_{label}_caption.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption)
    print(f"📄 Caption 已保存至: {caption_path}")
    
    # 使用外层已保存好的 mask 裁剪图，若不存在则回退到 bbox 裁剪图
    if not os.path.exists(input_tmp_path):
        print(f"Warning: cropped input not found: {input_tmp_path}")
        return ""
    return_image_path = input_tmp_path
    
    # 调用 nanobanana
    print(f"🚀 调用 nanobanana 处理 {label}...")
    try:
        raw_res_path = edit_image_with_qunhe(
            prompt=nano_prompt,
            input_image=input_tmp_path,
            output_dir=output_dir,
        )
        
        if raw_res_path and os.path.exists(raw_res_path):
            # 复制并重命名到目标路径 (output_dir/{asset_id}_{label}.png)
            final_output_path = os.path.join(output_dir, f"{generated_asset_id}_{label}.png")
            shutil.copy2(raw_res_path, final_output_path)
            print(f"✅ 结果已保存至: {final_output_path}")
            
            # 安全删除 nano_gemini 产生的临时时间戳目录
            try:
                temp_dir = os.path.dirname(raw_res_path)
                if temp_dir != output_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning cleaning temp dir: {e}")
            
            return_image_path = final_output_path
        else:
            return_image_path = input_tmp_path
            
    except Exception as e:
        print(f"Error calling nanobanana: {e}")
        # 如果失败，尝试将输入图改名为标准前缀文件作为备份
        final_output_path = os.path.join(output_dir, f"{generated_asset_id}_{label}.png")
        if os.path.exists(input_tmp_path):
            if not os.path.exists(final_output_path):
                shutil.copy2(input_tmp_path, final_output_path)
            return_image_path = final_output_path
        else:
            return_image_path = input_tmp_path
    
    # 最终分辨率检查：混元 API 要求最小 128
    if os.path.exists(return_image_path):
        try:
            with Image.open(return_image_path) as img:
                w, h = img.size
                if w < 128 or h < 128:
                    scale = 128 / min(w, h)
                    new_w = int(w * scale) + 1
                    new_h = int(h * scale) + 1
                    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    img_resized.save(return_image_path)
                    print(f"📏 调整低分辨率图片: {w}x{h} -> {new_w}x{new_h}")
        except Exception as e:
            print(f"Error adjusting resolution: {e}")

    
    return return_image_path

def generate_3d_mesh(
    image_path_for_gen: str,
    asset_id: str,
    gen_asset_dir: str,
    scale: List[float],
    label: Optional[str] = None,
    caption: Optional[str] = None,
    bbox_cropped_path: Optional[str] = None,
    correct_tilt: bool = False,
    correct_yaw: bool = True,
    gen_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
) -> str:
    """
    使用 nano_gen 生成 3D 资产，并将结果复制到目标路径并清理临时文件夹。
    """
    print(f"📦 正在为图片 {image_path_for_gen} 生成 3D 资产 (scale={scale})...")
    
    # 确保输出目录存在
    os.makedirs(gen_asset_dir, exist_ok=True)
    
    # 最终路径
    final_glb_path = os.path.join(gen_asset_dir, f"{asset_id}.glb")
    
    max_retries = 5
    hunyuan_gen = _get_hunyuan_gen()
    for attempt in range(max_retries):
        try:
            # 调用资产生成工具
            if hunyuan_gen is None:
                print("Warning: hunyuan_gen not found.")
                return ""

            glb_raw_path = hunyuan_gen(image_path=image_path_for_gen, output_dir=gen_asset_dir, model=gen_model)
            
            if glb_raw_path and os.path.exists(glb_raw_path):
                # 先复制到最终路径，再调用新的位姿矫正流程
                shutil.copy2(glb_raw_path, final_glb_path)
                try:
                    correct_single_asset(
                        final_glb_path,
                        label=label,
                        caption=caption,
                        correct_tilt=correct_tilt,
                        bbox_cropped_path=bbox_cropped_path if correct_yaw else None,
                    )
                    print(f"✅ 已使用新流程完成位姿矫正: {final_glb_path}")
                except Exception as e:
                    print(f"⚠️ 新位姿矫正失败，保留原始资产: {e}")
                
                # 删除 nano_gen 产生的时间戳中间文件夹
                temp_dir = os.path.dirname(glb_raw_path)
                if temp_dir != gen_asset_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(f"⚠️ 清理临时文件夹失败 (可能不为空): {e}")
                
                return final_glb_path
            else:
                print(f"⚠️ 第 {attempt + 1}/{max_retries} 次生成失败: {image_path_for_gen}")
                if attempt < max_retries - 1:
                    time.sleep(2) # 短暂等待后重试
                
        except Exception as e:
            print(f"Error in generate_3d_mesh attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    print(f"❌ 经过 {max_retries} 次尝试后，资产 {asset_id} 生成最终失败。")
    return ""

def get_mesh(
    scene_json: Dict[str, Any], 
    image_path: Optional[str] = None, 
    retrieve_hole: bool = True,
    asset_mode: Literal["none", "retrieve", "generate"] = "none",
    outpaint_image_dir: Optional[str] = None,
    asset_dir: Optional[str] = "/data-nas/data/dataset/qunhe/Manycore-Future/generate",
    gen_3d_model: Literal["hunyuan-3d-rapid", "hunyuan-3d-pro"] = "hunyuan-3d-pro",
    correct_tilt: bool = False,
    correct_yaw: bool = True,
) -> Dict[str, Any]:
    """
    根据 asset_mode 处理资产并更新 scene_json
    """
    if asset_mode == "none":
        return scene_json
    timestamp = Path(image_path).parent.name if image_path else "0000000000"

    # 预加载图片
    base_image = None
    if image_path and os.path.exists(image_path):
        try:
            base_image = Image.open(image_path).convert("RGB").resize((1000, 1000))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    # --- 通用预处理：建立 mesh_id 到 最佳代表物索引 的映射，并加载 Mask ---
    mesh_to_orig_idx = {}
    masks_data = []
    ref_data = []
    if base_image:
        dir_name = os.path.dirname(image_path)
        
        # 1. 建立映射mesh_to_orig_idx (基于体积最大原则) mesh_id -> 属于这个mesh_id的最大体积物体索引
        ref_json_path = os.path.join(dir_name, "g_asset_resp.json")
        if os.path.exists(ref_json_path):
            try:
                with open(ref_json_path, "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                mesh_max_volumes = {}
                for i, item in enumerate(ref_data):
                    mid = item.get("mesh_id")
                    if mid is None: continue
                    bbox = item.get("bbox", [0,0,0,0])
                    hr = item.get("h_range", [0,0])
                    vol = (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) * (hr[1]-hr[0])
                    try: mid_key = int(mid)
                    except: mid_key = mid
                    if mid_key not in mesh_max_volumes or vol > mesh_max_volumes[mid_key]:
                        mesh_max_volumes[mid_key] = vol
                        mesh_to_orig_idx[mid_key] = i
                
                print(f"📊 建立 Mesh ID 到索引的映射: {mesh_to_orig_idx}")
            except Exception as e:
                print(f"Error building mesh mapping: {e}")

        # 2. 加载 Mask 数据
        mask_path = os.path.join(dir_name, "h_masks_amodal.pkl")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(dir_name, "d_masks.pkl")
        if os.path.exists(mask_path):
            try:
                import pickle
                with open(mask_path, 'rb') as f:
                    masks_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading masks: {e}")

    # ------------------ 分支逻辑开始 ------------------

    if asset_mode == "generate":
        assert base_image is not None, "Image must be provided for generate mode"
        print(f"DEBUG: scene_json keys: {list(scene_json.keys())}")
        if "window" in scene_json:
            print(f"DEBUG: Found {len(scene_json['window'])} windows in scene_json")

        
        
        # 按 mesh_id 分组 (包含 bbox, door, window)
        mesh_groups = {}
        # 遍历所有可能的物体类型, 拿到所有存在的 mesh_id
        for key in ["bbox", "door", "window"]:
            for item in scene_json.get(key, []):
                if item.get("asset_id"): continue
                mid = item.get("mesh_id")
                if mid is not None:
                    try: mid_key = int(mid)
                    except: mid_key = mid
                    if mid_key not in mesh_groups: mesh_groups[mid_key] = []
                    mesh_groups[mid_key].append(item)
        # 遍历所有 mesh_id, 从之前建立的 mesh_to_orig_idx 中拿到最大体积物体的信息
        for mid, bboxes in mesh_groups.items():
            if not bboxes: continue
            
            orig_idx = mesh_to_orig_idx.get(mid)
            if orig_idx is not None and orig_idx < len(ref_data):
                # 统一使用“体积最大”实例作为生成代表，确保 Mask 和 Crop 坐标来自同一个物体
                rep_item = ref_data[orig_idx]
                label = rep_item.get("label", "object")
                caption = rep_item.get("caption", "")
                bbox_2d = rep_item.get("bbox")
            else:
                # 回退方案
                first_bbox = bboxes[0]
                label = first_bbox.get("label", "object")
                caption = first_bbox.get("caption", "")
                bbox_2d = first_bbox.get("bbox_2d")
                if bbox_2d is None:
                    bbox_2d = first_bbox.get("bbox")
            
            scale = bboxes[0].get("scale", [1.0, 1.0, 1.0])
            gen_dir = outpaint_image_dir if outpaint_image_dir else "."
            os.makedirs(gen_dir, exist_ok=True)
            generated_asset_id = _format_asset_prefix(mid, timestamp, label)
            target_glb_path = os.path.join(gen_asset_dir, f"{generated_asset_id}.glb") if gen_asset_dir else ""
            if target_glb_path and os.path.exists(target_glb_path):
                print(f"⏭️ 跳过资产生成，已存在: {target_glb_path}")
                for b in bboxes:
                    b["asset_id"] = generated_asset_id
                continue
            
            if bbox_2d and len(bbox_2d) == 4:
                # 使用 Mask 抠图并处理旋转
                orientation = rep_item.get("orientation", 0) if (orig_idx is not None and orig_idx < len(ref_data)) else bboxes[0].get("orientation", 0)

                def _build_bbox_cropped_image(src_img, bbox, orient_deg):
                    # 在原图先裁一个可容纳旋转后的大框(越界补黑)，再顺时针旋转并中心裁到横平竖直框
                    y1, x1, y2, x2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    theta = float(orient_deg or 0.0)
                    rad = np.deg2rad(theta)
                    cos_t, sin_t = np.cos(rad), np.sin(rad)

                    corners = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float64,
                    )
                    rel = corners - np.array([cx, cy], dtype=np.float64)
                    rot_cw = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
                    rel_rot = rel @ rot_cw.T
                    rx_min, ry_min = rel_rot.min(axis=0)
                    rx_max, ry_max = rel_rot.max(axis=0)

                    target_w = max(1, int(np.ceil(rx_max - rx_min)))
                    target_h = max(1, int(np.ceil(ry_max - ry_min)))
                    abs_cos, abs_sin = abs(cos_t), abs(sin_t)
                    pre_w = max(1, int(np.ceil(target_w * abs_cos + target_h * abs_sin)))
                    pre_h = max(1, int(np.ceil(target_w * abs_sin + target_h * abs_cos)))

                    pre_box = (
                        int(np.floor(cx - pre_w / 2.0)),
                        int(np.floor(cy - pre_h / 2.0)),
                        int(np.floor(cx + pre_w / 2.0)),
                        int(np.floor(cy + pre_h / 2.0)),
                    )
                    local_img = src_img.crop(pre_box)
                    if abs(theta) > 1e-8:
                        local_img = local_img.rotate(-theta, expand=True, resample=Image.BICUBIC)

                    rw, rh = local_img.size
                    final_box = (
                        int(np.floor((rw - target_w) / 2.0)),
                        int(np.floor((rh - target_h) / 2.0)),
                        int(np.floor((rw - target_w) / 2.0)) + target_w,
                        int(np.floor((rh - target_h) / 2.0)) + target_h,
                    )
                    return local_img.crop(final_box)
                
                if masks_data and orig_idx is not None and orig_idx < len(masks_data):
                    import numpy as np
                    mask_np = masks_data[orig_idx]
                    mask_np_bool = (mask_np > 127) if mask_np.max() > 1 else mask_np.astype(bool)
                    
                    # 凹包络 + 膨胀 10px（每个物体单独处理）
                    try:
                        import alphashape
                        from shapely.geometry import Polygon, MultiPolygon
                        from PIL import ImageDraw
                    except Exception:
                        alphashape = None
                    
                    if alphashape is not None:
                        coords = np.argwhere(mask_np_bool)
                        if coords.size > 0:
                            points = [(int(x), int(y)) for y, x in coords]
                            area = float(coords.shape[0])
                            alpha = 0.05 * np.sqrt(area)
                            if alpha > 0 and len(points) >= 4:
                                try:
                                    hull = alphashape.alphashape(points, alpha)
                                    if hull is not None and not hull.is_empty:
                                        hull = hull.buffer(10)
                                        if hull is not None and not hull.is_empty:
                                            mask_h, mask_w = mask_np_bool.shape
                                            mask_img = Image.new("L", (mask_w, mask_h), 0)
                                            draw = ImageDraw.Draw(mask_img)
                                            
                                            def draw_polygon(poly):
                                                if poly.is_empty:
                                                    return
                                                exterior = [(int(round(x)), int(round(y))) for x, y in poly.exterior.coords]
                                                draw.polygon(exterior, fill=1)
                                                for interior in poly.interiors:
                                                    interior_pts = [(int(round(x)), int(round(y))) for x, y in interior.coords]
                                                    draw.polygon(interior_pts, fill=0)
                                            
                                            if hull.geom_type == "Polygon":
                                                draw_polygon(hull)
                                            elif hull.geom_type == "MultiPolygon":
                                                for poly in hull.geoms:
                                                    draw_polygon(poly)
                                            
                                            new_mask = np.array(mask_img, dtype=bool)
                                            if new_mask.any():
                                                mask_np_bool = new_mask
                                except Exception:
                                    pass
                    
                    # 1. 寻找原始 mask 的包围盒
                    coords_orig = np.argwhere(mask_np_bool)
                    if coords_orig.size > 0:
                        y1_o, x1_o = coords_orig.min(axis=0)
                        y2_o, x2_o = coords_orig.max(axis=0)
                        
                        # 2. 裁剪出一个较大的局部区域，为旋转留出空间 (取长宽最大值的 1.5 倍)
                        w_o, h_o = x2_o - x1_o, y2_o - y1_o
                        cx_o, cy_o = (x1_o + x2_o) / 2, (y1_o + y2_o) / 2
                        pad = int(max(w_o, h_o) * 1.0) # 这里的 pad 是半径方向的冗余
                        
                        # 构图：应用 mask 并准备裁剪
                        base_np = np.array(base_image)
                        mask_3d = np.repeat(mask_np_bool[:, :, np.newaxis], 3, axis=2)
                        masked_np = np.where(mask_3d, base_np, 0)
                        
                        full_masked_img = Image.fromarray(masked_np.astype(np.uint8))
                        full_mask_img = Image.fromarray((mask_np_bool * 255).astype(np.uint8))
                        
                        # 初次裁剪：获取包含物体的局部图
                        crop_box = (cx_o - pad, cy_o - pad, cx_o + pad, cy_o + pad)
                        obj_img_local = full_masked_img.crop(crop_box)
                        obj_mask_local = full_mask_img.crop(crop_box)
                        
                        # 3. 在局部图中旋转，expand=True 确保所有像素被保留
                        if orientation != 0:
                            obj_img_local = obj_img_local.rotate(-orientation, expand=True, resample=Image.BICUBIC)
                            obj_mask_local = obj_mask_local.rotate(-orientation, expand=True, resample=Image.NEAREST)
                        
                        # 4. 在旋转后的图中寻找新的最小包围盒
                        rotated_mask_np = np.array(obj_mask_local)
                        coords_rot = np.argwhere(rotated_mask_np > 0)
                        
                        if coords_rot.size > 0:
                            ry_min, rx_min = coords_rot.min(axis=0)
                            ry_max, rx_max = coords_rot.max(axis=0)
                            
                            rw, rh = rx_max - rx_min, ry_max - ry_min
                            rcx, rcy = (rx_min + rx_max) / 2, (ry_min + ry_max) / 2
                            
                            # 5. 最终 1.2 倍比例裁剪
                            final_box = (int(rcx - rw*0.6), int(rcy - rh*0.6), int(rcx + rw*0.6), int(rcy + rh*0.6))
                            cropped_img = obj_img_local.crop(final_box)
                        else:
                            cropped_img = obj_img_local
                    else:
                        # Fallback
                        cropped_img = base_image.crop((bbox_2d[1], bbox_2d[0], bbox_2d[3], bbox_2d[2]))
                else:
                    # 无 Mask 情况下的回退逻辑
                    cropped_img = _build_bbox_cropped_image(base_image, bbox_2d, orientation)

                mask_cropped_path = os.path.join(gen_dir, f"{generated_asset_id}_mask_cropped.png")
                cropped_img.save(mask_cropped_path)
                bbox_cropped_path = os.path.join(gen_dir, f"{generated_asset_id}_bbox_cropped.png")
                _build_bbox_cropped_image(base_image, bbox_2d, orientation).save(bbox_cropped_path)

                image_path_for_gen = process_image_for_generation(
                    generated_asset_id, label, caption, gen_dir
                )
                if not image_path_for_gen:
                    image_path_for_gen = mask_cropped_path
                glb_path = generate_3d_mesh(
                    image_path_for_gen=image_path_for_gen,
                    asset_id=generated_asset_id,
                    gen_asset_dir=gen_asset_dir,
                    scale=scale,
                    label=label,
                    caption=caption,
                    bbox_cropped_path=bbox_cropped_path,
                    correct_tilt=correct_tilt,
                    correct_yaw=correct_yaw,
                    gen_model=gen_3d_model,
                )
                
                for b in bboxes: b["asset_id"] = generated_asset_id if glb_path else None
                    
        return scene_json


    # asset_mode == "retrieve" 逻辑
    import lancedb
    import torch

    db_uri = "/data-nas/data/experiments/mushui/projects/utils/fast-scene/scenebuilder/manycore"
    db = lancedb.connect(db_uri)
    
    if retrieve_hole:
        for hole_type in ["door", "window"]:
            if hole_type in scene_json and scene_json[hole_type]:
                try:
                    table = db.open_table(hole_type)
                    for item in scene_json[hole_type]:
                        if not item.get("asset_id"):
                            query_vector = [float(item.get("width", 0)), float(item.get("height", 0))]
                            result = table.search(query_vector).where("exist = true").metric("l2").limit(1).to_list()
                            if result: item["asset_id"] = result[0]["asset_id"]
                except Exception as e: print(f"Error retrieving {hole_type}: {e}")

    if "bbox" in scene_json and scene_json["bbox"]:
        bboxes_to_process = [b for b in scene_json["bbox"] if not b.get("asset_id")]
        if not bboxes_to_process: return scene_json

        Qwen3VLEmbedder = _get_qwen3_embedder_class()
        if not Qwen3VLEmbedder: return scene_json
        embedder = Qwen3VLEmbedder(model_name_or_path="/data-nas/data/experiments/mushui/.cache/huggingface/hub/Qwen/Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        try: furniture_table = db.open_table("furniture")
        except: return scene_json

        batch_inputs, groups_info = [], []
        if base_image:
            mesh_groups = {}
            for bbox in bboxes_to_process:
                mid = bbox.get("mesh_id", f"no_id_{id(bbox)}")
                if mid not in mesh_groups: mesh_groups[mid] = []
                mesh_groups[mid].append(bbox)
            
            for mid, group_bboxes in mesh_groups.items():
                first_bbox = group_bboxes[0]
                label, caption = first_bbox.get("label"), first_bbox.get("caption")
                if not label: continue
                
                # --- Retrieve 同样使用精准代表物索引和 Mask 抠图 ---
                orig_idx = mesh_to_orig_idx.get(mid)
                if masks_data and orig_idx is not None and orig_idx < len(masks_data):
                    import numpy as np
                    mask_np = masks_data[orig_idx]
                    mask_np = (mask_np > 127) if mask_np.max() > 1 else mask_np.astype(bool)
                    base_np, mask_3d = np.array(base_image), np.repeat(mask_np[:, :, np.newaxis], 3, axis=2) if mask_np.ndim == 2 else mask_np.astype(bool)
                    masked_np = np.where(mask_3d, base_np, np.zeros_like(base_np))
                    masked_image = Image.fromarray(masked_np.astype(np.uint8))
                else: masked_image = base_image

                prompt = f"this is a {label}" + (f", the caption is {caption}" if caption else "")
                bbox_2d = first_bbox.get("bbox_2d")
                if bbox_2d and len(bbox_2d) == 4:
                    # 检索也使用 1.2 倍扩张以获取更多形状信息，但核心是黑色背景
                    cx, cy = (bbox_2d[1] + bbox_2d[3]) / 2, (bbox_2d[0] + bbox_2d[2]) / 2
                    w, h = (bbox_2d[3] - bbox_2d[1]), (bbox_2d[2] - bbox_2d[0])
                    cropped = masked_image.crop((int(cx - w*0.6), int(cy - h*0.6), int(cx + w*0.6), int(cy + h*0.6)))
                    batch_inputs.append({"text": prompt, "image": cropped})
                    groups_info.append(group_bboxes)
        else:
            for bbox in bboxes_to_process:
                label, caption = bbox.get("label"), bbox.get("caption")
                if not label: continue
                batch_inputs.append({"text": f"this is a {label}" + (f", the caption is {caption}" if caption else "")})
                groups_info.append([bbox])

        if batch_inputs:
            try:
                all_embeddings = embedder.process(batch_inputs)
                if isinstance(all_embeddings, torch.Tensor):
                    all_embeddings = all_embeddings.float().cpu().numpy().tolist()
                for i, query_vector in enumerate(all_embeddings):
                    result = furniture_table.search(query_vector).where("exist = true").distance_type("cosine").limit(1).to_list()
                    if result:
                        asset_id = result[0]["asset_id"]
                        for b in groups_info[i]: b["asset_id"] = asset_id
            except Exception as e: print(f"Error in batch search: {e}")

    return scene_json

def asset_id_exists(asset_id: Any, search_paths: Optional[List[str]]) -> bool:
    """检查 asset_id 对应 glb/gltf 是否在搜索路径中存在。"""
    if asset_id is None or not search_paths:
        return False
    aid = str(asset_id)
    for root in search_paths:
        if not root:
            continue
        for ext in (".glb", ".gltf"):
            if os.path.exists(os.path.join(root, aid + ext)):
                return True
    return False


def object_ply_stem(label: str, asset_id: Any = None) -> str:
    """点云/可见几何文件名主干：{label} 或 {label}_{asset_id}。"""
    parts = [label]
    if asset_id is not None:
        parts.append(str(asset_id))
    return "_".join(parts)


def build_pointcloud_ply_relpath(
    category: str,
    label: str,
    asset_id: Any = None,
    suffixes: Optional[List[str]] = None,
) -> str:
    """相对 pointcloud 根目录的 ply 路径，如 boxes/sidetable0_56056912_visible.ply。"""
    stem = object_ply_stem(label, asset_id)
    for suffix in suffixes or []:
        if suffix:
            stem = f"{stem}_{suffix}"
    return os.path.join(category, f"{stem}.ply")


def _recompute_context_z_max(context: Dict[str, Any]) -> None:
    wall_heights = [float(w.get("height", 0.0)) for w in context.get("walls", {}).values()]
    z_max = max(wall_heights) if wall_heights else 0.0
    for box in context.get("boxes", {}).values():
        center = box.get("center") or [0.0, 0.0, 0.0]
        scale = box.get("scale") or [0.0, 0.0, 0.0]
        z_max = max(z_max, float(center[2]) + float(scale[2]) / 2.0)
    context.setdefault("meta", {})["z_max"] = z_max


def normalize_scene_context(
    context: Dict[str, Any],
    model_paths: Optional[List[str]] = None,
    hole_paths: Optional[List[str]] = None,
) -> None:
    """统一 context 键与 label；校验 asset_id（bbox 缺失则删，门窗缺失则去 asset_id）。"""
    meta = context.setdefault("meta", {})
    if meta.get("_scene_normalized"):
        return

    model_paths = model_paths or []
    hole_paths = hole_paths or []

    new_walls: Dict[str, Any] = {}
    door_counter = 0
    window_counter = 0
    for wall_index, (_old_wid, wall) in enumerate(context.get("walls", {}).items()):
        wall_label = f"wall{wall_index}"
        new_wall = {k: v for k, v in wall.items() if k not in ("doors", "windows")}
        new_wall["label"] = wall_label
        new_wall["doors"] = {}
        new_wall["windows"] = {}

        for door in wall.get("doors", {}).values():
            door_label = f"door{door_counter}"
            door_counter += 1
            new_door = dict(door)
            new_door["label"] = door_label
            new_door["wall"] = wall_label
            aid = new_door.get("asset_id")
            if aid is not None and not asset_id_exists(aid, hole_paths):
                new_door.pop("asset_id", None)
                print(f"⚠️ {door_label}: asset_id={aid} 不存在，保留空洞但不加载模型")
            new_wall["doors"][door_label] = new_door

        for window in wall.get("windows", {}).values():
            window_label = f"window{window_counter}"
            window_counter += 1
            new_window = dict(window)
            new_window["label"] = window_label
            new_window["wall"] = wall_label
            aid = new_window.get("asset_id")
            if aid is not None and not asset_id_exists(aid, hole_paths):
                new_window.pop("asset_id", None)
                print(f"⚠️ {window_label}: asset_id={aid} 不存在，保留空洞但不加载模型")
            new_wall["windows"][window_label] = new_window

        new_walls[wall_label] = new_wall
    context["walls"] = new_walls

    new_boxes: Dict[str, Any] = {}
    label_counts: Dict[str, int] = {}
    for _old_id, box in context.get("boxes", {}).items():
        aid = box.get("asset_id")
        if aid is not None and not asset_id_exists(aid, model_paths):
            print(
                f"⚠️ 删除 bbox (asset_id={aid} 不存在): "
                f"{box.get('label') or box.get('class') or _old_id}"
            )
            continue
        slug = _normalize_bbox_label_slug(box.get("label") or box.get("class") or "object")
        ordinal = label_counts.get(slug, 0)
        label_counts[slug] = ordinal + 1
        box_label = f"{slug}{ordinal}"
        new_box = dict(box)
        new_box["label"] = box_label
        new_boxes[box_label] = new_box
    context["boxes"] = new_boxes

    _recompute_context_z_max(context)
    meta["_scene_normalized"] = True


def _ssl_fmt_num(value: float) -> str:
    text = f"{float(value):g}"
    return text


def _ssl_fmt_list(values) -> str:
    parts = [_ssl_fmt_num(v) if isinstance(v, (int, float, np.floating)) else str(v) for v in values]
    return "[" + ", ".join(parts) + "]"


def _normalize_bbox_label_slug(raw_label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", (raw_label or "object").lower())
    return slug or "object"


def compute_topdown_camera_pose(context: Dict[str, Any]):
    """与 topdown_view 首帧相机一致：位于场景中心正上方，看向地面中心。"""
    meta = context["meta"]
    center = meta["center"]
    span = meta["span"]
    z_max = meta["z_max"]
    height = z_max + max(span) * 1.5
    camera_position = [center[0], center[1], height]
    look_at_target = [center[0], center[1], 0.0]
    return camera_position, look_at_target


def ssl_xy_to_image_xy(ssl_x: float, ssl_y: float) -> tuple:
    """SSL 坐标 (Y 北) → SpatialFactory 图像坐标 (Y 南)。"""
    return float(ssl_x), float(-ssl_y)


def image_xy_to_ssl_xy(image_x: float, image_y: float) -> tuple:
    """图像坐标 → SSL 坐标。"""
    return float(image_x), float(-image_y)


def compute_pixel2real_ratio(camera_z: float, fov_y_rad: float, image_half: float = 500.0) -> float:
    """1 像素 = ratio 米；与 SpatialFactory Stage 1 一致。"""
    return float(camera_z) * math.tan(float(fov_y_rad) / 2.0) / float(image_half)


def compute_pixel_align_translation_ssl(
    camera_position_ssl,
    fov_y_rad: float,
    image_half: float = 500.0,
) -> tuple:
    """计算像素对齐平移量，使图像主点落在 (ratio*half, ratio*half) 像素。

    返回 (pixel2real_ratio, dx_ssl, dy_ssl, target_image_x, target_image_y)。
    """
    cam = np.asarray(camera_position_ssl, dtype=float)
    z = float(cam[2])
    ratio = compute_pixel2real_ratio(z, fov_y_rad, image_half)
    img_x, img_y = ssl_xy_to_image_xy(float(cam[0]), float(cam[1]))
    target_x = ratio * image_half
    target_y = ratio * image_half
    dx_img = target_x - img_x
    dy_img = target_y - img_y
    return ratio, dx_img, -dy_img, target_x, target_y


def _round_xy(values, round_decimals: Optional[int]):
    x, y = float(values[0]), float(values[1])
    if round_decimals is not None:
        x, y = round(x, round_decimals), round(y, round_decimals)
    return [x, y]


def _round_xyz(values, round_decimals: Optional[int]):
    xy = _round_xy(values[:2], round_decimals)
    z = float(values[2]) if len(values) > 2 else 0.0
    if round_decimals is not None:
        z = round(z, round_decimals)
    return [xy[0], xy[1], z]


def apply_context_xy_translation(
    context: Dict[str, Any],
    dx_ssl: float,
    dy_ssl: float,
    *,
    round_decimals: Optional[int] = 2,
) -> None:
    """将场景 context 在 SSL XY 平面平移 (dx_ssl, dy_ssl)，并重算 meta。"""
    for wall in context.get("walls", {}).values():
        wall["s"] = _round_xy([wall["s"][0] + dx_ssl, wall["s"][1] + dy_ssl], round_decimals)
        wall["e"] = _round_xy([wall["e"][0] + dx_ssl, wall["e"][1] + dy_ssl], round_decimals)
        for door in wall.get("doors", {}).values():
            door["center"] = _round_xyz(
                [door["center"][0] + dx_ssl, door["center"][1] + dy_ssl, door["center"][2]],
                round_decimals,
            )
        for window in wall.get("windows", {}).values():
            window["center"] = _round_xyz(
                [window["center"][0] + dx_ssl, window["center"][1] + dy_ssl, window["center"][2]],
                round_decimals,
            )
    for box in context.get("boxes", {}).values():
        box["center"] = _round_xyz(
            [box["center"][0] + dx_ssl, box["center"][1] + dy_ssl, box["center"][2]],
            round_decimals,
        )
    recompute_context_meta_from_walls(context)


def build_pixel_aligned_camera_para(
    camera_position_ssl,
    look_at_target_ssl,
    fov_y_rad: float,
    pixel2real_ratio: float,
    width: int = 1000,
    height: int = 1000,
    *,
    round_decimals: int = 2,
) -> Dict[str, Any]:
    """SpatialFactory 兼容的 camera_para（图像坐标系 + pixel2real_ratio）。"""
    cam = np.asarray(camera_position_ssl, dtype=float)
    look = np.asarray(look_at_target_ssl, dtype=float)
    cam_img_x, cam_img_y = ssl_xy_to_image_xy(float(cam[0]), float(cam[1]))
    look_img_x, look_img_y = ssl_xy_to_image_xy(float(look[0]), float(look[1]))
    cam_z = round(float(cam[2]), round_decimals)
    look_z = round(float(look[2]), round_decimals)
    return {
        "camera_position": [
            round(cam_img_x, round_decimals),
            round(cam_img_y, round_decimals),
            cam_z,
        ],
        "look_at_target": [
            round(look_img_x, round_decimals),
            round(look_img_y, round_decimals),
            look_z,
        ],
        "fov_y": float(fov_y_rad),
        "aspectRatio": float(width) / float(height),
        "pixel2real_ratio": float(pixel2real_ratio),
        "image_size": [int(width), int(height)],
        "coordinate_system": "image",
    }


def write_standard_ssl_to_path(context: Dict[str, Any], ssl_path: str) -> str:
    os.makedirs(os.path.dirname(ssl_path) or ".", exist_ok=True)
    ssl_text = format_standard_ssl(context)
    with open(ssl_path, "w", encoding="utf-8") as f:
        f.write(ssl_text)
    print(f"✅ SSL 已导出: {ssl_path}")
    return ssl_path


def prepare_pixel_aligned_topdown_context(
    context: Dict[str, Any],
    *,
    width: int = 1000,
    height: int = 1000,
    indoor_fov: float = 160.0,
    outdoor_fov_scale: float = 1.05,
    round_decimals: int = 2,
) -> Dict[str, Any]:
    """按 SpatialFactory Stage 1 规则平移 context，并返回渲染/导出参数。"""
    util = _import_core_util()

    image_half = float(width) / 2.0
    world_cam_w, world_look_w = compute_topdown_camera_pose(context)
    meta = context["meta"]
    fov_y = util.calculate_optimal_fov(
        np.asarray(world_cam_w, dtype=float),
        np.asarray(world_look_w, dtype=float),
        meta["vertices"],
        meta["z_max"],
        meta["bounds"],
        indoor_fov,
        outdoor_fov_scale,
    )
    ratio, dx_ssl, dy_ssl, _, _ = compute_pixel_align_translation_ssl(
        world_cam_w, fov_y, image_half=image_half
    )
    apply_context_xy_translation(context, dx_ssl, dy_ssl, round_decimals=round_decimals)
    aligned_cam, aligned_look = compute_topdown_camera_pose(context)
    return {
        "pixel2real_ratio": ratio,
        "dx_ssl": float(dx_ssl),
        "dy_ssl": float(dy_ssl),
        "fov_y": float(fov_y),
        "camera_position_ssl": aligned_cam,
        "look_at_target_ssl": aligned_look,
        "camera_para": build_pixel_aligned_camera_para(
            aligned_cam,
            aligned_look,
            fov_y,
            ratio,
            width=width,
            height=height,
            round_decimals=round_decimals,
        ),
    }


def apply_scene_json_xy_translation(
    scene_json: Dict[str, Any],
    dx_ssl: float,
    dy_ssl: float,
    *,
    round_decimals: Optional[int] = 2,
) -> None:
    """将 scene_json 的墙/门窗/家具坐标与 context 同步做 XY 平移。"""
    for wall in scene_json.get("wall", []):
        p = wall.get("p") or [0.0, 0.0, 0.0]
        q = wall.get("q") or [0.0, 0.0, 0.0]
        wall["p"] = _round_xyz([p[0] + dx_ssl, p[1] + dy_ssl, p[2] if len(p) > 2 else 0.0], round_decimals)
        wall["q"] = _round_xyz([q[0] + dx_ssl, q[1] + dy_ssl, q[2] if len(q) > 2 else 0.0], round_decimals)
    for cat in ("door", "window", "bbox"):
        for item in scene_json.get(cat, []):
            center = item.get("center") or [0.0, 0.0, 0.0]
            item["center"] = _round_xyz(
                [center[0] + dx_ssl, center[1] + dy_ssl, center[2] if len(center) > 2 else 0.0],
                round_decimals,
            )


def view_ssl_transform_params(
    camera_position,
    look_at_target,
) -> tuple:
    """视角 SSL 变换：原点平移到相机地面投影 (a,b,0)，+Y 对齐 look_at 在 XY 上的方向。

    返回 (origin_xy, theta_rad)。theta 为 look 方向相对 +Y 的方位角；点坐标绕 Z 旋转 +theta 后与视角 SSL 对齐。
    """
    cam = np.asarray(camera_position, dtype=float)
    look = np.asarray(look_at_target, dtype=float)
    origin_xy = cam[:2]
    delta = look[:2] - origin_xy
    if float(np.linalg.norm(delta)) < 1e-9:
        return origin_xy, 0.0
    theta = float(np.arctan2(delta[0], delta[1]))
    return origin_xy, theta


def _transform_xy_view_ssl(xy, origin_xy, theta: float) -> List[float]:
    rel = np.asarray(xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return [c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]


def _transform_xyz_view_ssl(xyz, origin_xy, theta: float) -> List[float]:
    xy = _transform_xy_view_ssl(xyz[:2], origin_xy, theta)
    z = float(xyz[2]) if len(xyz) > 2 else 0.0
    return [xy[0], xy[1], z]


def _transform_direction_view_ssl(xyz, theta: float) -> List[float]:
    """方向向量：仅绕 Z 旋转，不平移。"""
    v = np.asarray(xyz, dtype=float)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], float(v[2])]


def view_ssl_camera_pose_fields(
    camera_position,
    look_at_target,
    world_up,
    origin_xy,
    theta: float,
    *,
    reference_frame: bool = False,
) -> Dict[str, List[float]]:
    """世界/视角 SSL 双套相机位姿。

    reference_frame=True（首帧/单视角）时视角下相机为 (0,0,h)，look_at 为 (0,x,e)。
    序列后续帧则写入完整视角 SSL 坐标。
    """
    cam = np.asarray(camera_position, dtype=float)
    look = np.asarray(look_at_target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    cam_view = _transform_xyz_view_ssl(cam, origin_xy, theta)
    look_view = _transform_xyz_view_ssl(look, origin_xy, theta)
    up_view = _transform_direction_view_ssl(up, theta)

    if reference_frame:
        camera_position_view = [0.0, 0.0, float(cam[2])]
        look_at_target_view = [0.0, float(look_view[1]), float(look[2])]
    else:
        camera_position_view = [float(v) for v in cam_view]
        look_at_target_view = [float(v) for v in look_view]

    return {
        "camera_position": cam.tolist(),
        "look_at_target": look.tolist(),
        "world_up": up.tolist(),
        "camera_position_view": camera_position_view,
        "look_at_target_view": look_at_target_view,
        "world_up_view": up_view,
    }


def build_camera_para_dict(
    camera_position,
    look_at_target,
    world_up,
    fov_y: float,
    aspect_ratio: float,
    *,
    view_origin_xy=None,
    view_theta: Optional[float] = None,
    reference_frame: bool = False,
    depth_scale: Optional[float] = None,
    include_normal_fields: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    include_intrinsic: bool = True,
) -> Dict[str, Any]:
    """构建 camera_para.json 内容（含世界坐标与视角 SSL 坐标）。"""
    if view_origin_xy is None or view_theta is None:
        view_origin_xy, view_theta = view_ssl_transform_params(camera_position, look_at_target)

    para: Dict[str, Any] = {
        **view_ssl_camera_pose_fields(
            camera_position,
            look_at_target,
            world_up,
            view_origin_xy,
            view_theta,
            reference_frame=reference_frame,
        ),
        "aspectRatio": float(aspect_ratio),
        "fov_y": float(fov_y),
    }
    if width is not None and height is not None:
        try:
            from . import util
        except ImportError:
            import util  # type: ignore
        para.update(
            util.camera_calibration_matrix_fields(
                camera_position,
                look_at_target,
                world_up,
                fov_y,
                width,
                height,
                aspect_ratio=aspect_ratio,
                include_intrinsic=include_intrinsic,
            )
        )
        para["image_size"] = [int(width), int(height)]
    if depth_scale is not None:
        para["depth_unit"] = "meter"
        para["depth_scale"] = float(depth_scale)
        para["is_metric_depth"] = True
        if include_normal_fields:
            try:
                from . import util
            except ImportError:
                import util  # type: ignore
            para.update(util.normal_map_camera_para_fields())
    return para


def recompute_context_meta_from_walls(context: Dict[str, Any]) -> None:
    """视角 SSL 变换后，根据墙段重算 meta（bounds/center/span/vertices/z_max）及墙 orientation。"""
    try:
        from . import util
    except ImportError:
        import util  # type: ignore
    walls = context.get("walls", {})
    if not walls:
        return
    walls_converted = [
        {"s": list(w["s"]), "e": list(w["e"]), "height": w["height"]}
        for w in walls.values()
    ]
    all_points = [p for w in walls_converted for p in [w["s"], w["e"]]]
    x_coords, y_coords = zip(*all_points)
    vertices, _ = util.calculate_minimum_area_polygon_and_partitions(walls_converted)
    z_max = max(w["height"] for w in walls_converted) if walls_converted else 0.0
    for box in context.get("boxes", {}).values():
        z_max = max(z_max, float(box["center"][2]) + float(box["scale"][2]) / 2.0)
    context.setdefault("meta", {}).update({
        "bounds": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
        "center": [(min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2],
        "span": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)],
        "z_max": z_max,
        "vertices": vertices,
    })
    for wall in walls.values():
        if wall.get("is_partition"):
            continue
        wall["orientation"] = util.calculate_wall_orientation(
            tuple(wall["s"]), tuple(wall["e"]), vertices
        )


def _normalize_angle_z_deg(angle: float) -> float:
    """归一化到 (-180, 180]。"""
    angle = float(angle) % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


def _bbox_forward_xy(angle_z_deg: float):
    """SSL 约定：angle_z=0 时朝向 −Y，逆时针为正。返回 XY 单位朝向向量。"""
    rad = np.radians(float(angle_z_deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([s, -c], dtype=float)


def apply_view_ssl_transform_with_params(
    context: Dict[str, Any],
    origin_xy,
    theta: float,
) -> None:
    """原地将 context 变换到视角 SSL 坐标系。

    不变：bbox scale、墙 height、门窗 width/height。
    变：墙 p/q、门窗/bbox center（平移 + 绕 Z 旋转 +θ）、bbox angle_z（+θ 度）。
    """
    theta_deg = float(np.degrees(theta))
    for wall in context.get("walls", {}).values():
        wall["s"] = _transform_xy_view_ssl(wall["s"], origin_xy, theta)
        wall["e"] = _transform_xy_view_ssl(wall["e"], origin_xy, theta)
        for door in wall.get("doors", {}).values():
            door["center"] = _transform_xyz_view_ssl(door["center"], origin_xy, theta)
        for window in wall.get("windows", {}).values():
            window["center"] = _transform_xyz_view_ssl(window["center"], origin_xy, theta)
    for box in context.get("boxes", {}).values():
        box["center"] = _transform_xyz_view_ssl(box["center"], origin_xy, theta)
        box["angle_z"] = _normalize_angle_z_deg(float(box.get("angle_z", 0.0)) + theta_deg)
    recompute_context_meta_from_walls(context)


def apply_view_ssl_transform_inplace(
    context: Dict[str, Any],
    camera_position,
    look_at_target,
):
    origin_xy, theta = view_ssl_transform_params(camera_position, look_at_target)
    apply_view_ssl_transform_with_params(context, origin_xy, theta)
    return origin_xy, theta


def reference_view_camera_pose(
    world_camera,
    world_look_at,
    origin_xy,
    theta: float,
):
    """首帧视角 SSL 相机：(0,0,h) 与 (0,x,e)。"""
    cam = np.asarray(world_camera, dtype=float)
    look_view = _transform_xyz_view_ssl(world_look_at, origin_xy, theta)
    return [0.0, 0.0, float(cam[2])], [0.0, float(look_view[1]), float(np.asarray(world_look_at)[2])]


def world_pose_to_view_ssl(
    camera_position,
    look_at_target,
    origin_xy,
    theta: float,
):
    cam_v = _transform_xyz_view_ssl(camera_position, origin_xy, theta)
    look_v = _transform_xyz_view_ssl(look_at_target, origin_xy, theta)
    return [float(v) for v in cam_v], [float(v) for v in look_v]


def write_view_ssl_from_context(context: Dict[str, Any], output_dir: str) -> str:
    """context 已是视角 SSL 坐标时直接导出 ssl.txt。"""
    os.makedirs(output_dir, exist_ok=True)
    ssl_text = format_standard_ssl(context)
    ssl_path = os.path.join(output_dir, "ssl.txt")
    with open(ssl_path, "w", encoding="utf-8") as f:
        f.write(ssl_text)
    print(f"✅ 视角 SSL 已导出: {ssl_path}")
    return ssl_path


class ViewSslSession:
    """视角 SSL 临时 context：enabled 时变换 owner.context，退出时恢复。"""

    def __init__(self, owner, enabled: bool):
        self.owner = owner
        self.enabled = bool(enabled)
        self._backup = None
        self.origin_xy = None
        self.theta = None
        self.world_camera = None
        self.world_look_at = None
        self.world_up = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end()
        return False

    def setup(self, world_camera, world_look_at, world_up=None):
        self.world_camera = [float(x) for x in world_camera]
        self.world_look_at = [float(x) for x in world_look_at]
        up = np.asarray(world_up if world_up is not None else [0.0, 0.0, 1.0], dtype=float)
        up_norm = float(np.linalg.norm(up))
        self.world_up = (up / up_norm if up_norm > 1e-6 else np.array([0.0, 0.0, 1.0])).tolist()
        if not self.enabled:
            return
        self._backup = copy.deepcopy(self.owner.context)
        self.origin_xy, self.theta = apply_view_ssl_transform_inplace(
            self.owner.context,
            self.world_camera,
            self.world_look_at,
        )
        if self.owner.mesh_nodes.get("floor") is not None:
            self.owner.clear_scene()

    def end(self):
        if not self.enabled or self._backup is None:
            return
        self.owner.context = self._backup
        self._backup = None
        self.origin_xy = None
        self.theta = None
        if hasattr(self.owner, "clear_scene"):
            self.owner.clear_scene()

    def write_ssl(self, output_dir: str):
        if self.enabled:
            write_view_ssl_from_context(self.owner.context, output_dir)

    def render_camera_pose(self, world_camera, world_look_at, *, reference_frame=True):
        if not self.enabled:
            return list(world_camera), list(world_look_at)
        if reference_frame:
            return reference_view_camera_pose(
                world_camera, world_look_at, self.origin_xy, self.theta
            )
        return world_pose_to_view_ssl(
            world_camera, world_look_at, self.origin_xy, self.theta
        )

    def render_world_up(self, world_up=None):
        up = np.asarray(world_up if world_up is not None else self.world_up, dtype=float)
        up_norm = float(np.linalg.norm(up))
        up = up / up_norm if up_norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=float)
        if not self.enabled:
            return up.tolist()
        return _transform_direction_view_ssl(up, self.theta)

    def build_camera_para(
        self,
        world_camera,
        world_look_at,
        world_up,
        fov_y: float,
        aspect_ratio: float,
        *,
        reference_frame: bool = False,
        depth_scale: Optional[float] = None,
        include_normal_fields: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        include_intrinsic: bool = True,
    ) -> Dict[str, Any]:
        if not self.enabled:
            para: Dict[str, Any] = {
                "camera_position": [float(x) for x in world_camera],
                "look_at_target": [float(x) for x in world_look_at],
                "world_up": [float(x) for x in world_up],
                "aspectRatio": float(aspect_ratio),
                "fov_y": float(fov_y),
            }
            if width is not None and height is not None:
                try:
                    from . import util
                except ImportError:
                    import util  # type: ignore
                para.update(
                    util.camera_calibration_matrix_fields(
                        world_camera,
                        world_look_at,
                        world_up,
                        fov_y,
                        width,
                        height,
                        aspect_ratio=aspect_ratio,
                        include_intrinsic=include_intrinsic,
                    )
                )
                para["image_size"] = [int(width), int(height)]
            if depth_scale is not None:
                para["depth_unit"] = "meter"
                para["depth_scale"] = float(depth_scale)
                para["is_metric_depth"] = True
                if include_normal_fields:
                    try:
                        from . import util
                    except ImportError:
                        import util  # type: ignore
                    para.update(util.normal_map_camera_para_fields())
            return para
        return build_camera_para_dict(
            world_camera,
            world_look_at,
            world_up,
            fov_y,
            aspect_ratio,
            view_origin_xy=self.origin_xy,
            view_theta=self.theta,
            reference_frame=reference_frame,
            depth_scale=depth_scale,
            include_normal_fields=include_normal_fields,
            width=width,
            height=height,
            include_intrinsic=include_intrinsic,
        )


def transform_context_to_view_ssl(
    context: Dict[str, Any],
    camera_position,
    look_at_target,
) -> Dict[str, Any]:
    """深拷贝 context 并变换为视角 SSL 坐标（不修改原 context）。"""
    origin_xy, theta = view_ssl_transform_params(camera_position, look_at_target)
    out = copy.deepcopy(context)
    apply_view_ssl_transform_with_params(out, origin_xy, theta)
    return out


def write_view_ssl(
    context: Dict[str, Any],
    output_dir: str,
    camera_position,
    look_at_target,
) -> str:
    """导出视角 SSL 到 output_dir/ssl.txt（场景坐标经相机首帧位姿变换）。"""
    os.makedirs(output_dir, exist_ok=True)
    view_context = transform_context_to_view_ssl(context, camera_position, look_at_target)
    ssl_text = format_standard_ssl(view_context)
    ssl_path = os.path.join(output_dir, "ssl.txt")
    with open(ssl_path, "w", encoding="utf-8") as f:
        f.write(ssl_text)
    print(f"✅ 视角 SSL 已导出: {ssl_path}")
    return ssl_path


def format_standard_ssl(context: Dict[str, Any]) -> str:
    """将加载后的 scene context 导出为简洁标准 SSL（无 id/room_id，label 规范化）。"""
    lines: List[str] = []
    room_type = context.get("meta", {}).get("scene_type", "unknown")
    lines.append(f'Room(room_type="{room_type}")')

    for wall_id, wall in context.get("walls", {}).items():
        wall_label = wall.get("label", wall_id)
        p = list(wall["s"]) + [0.0]
        q = list(wall["e"]) + [0.0]
        parts = [
            f'label="{wall_label}"',
            f'p={_ssl_fmt_list(p)}',
            f'q={_ssl_fmt_list(q)}',
            f'height={_ssl_fmt_num(wall["height"])}',
        ]
        wall_caption = wall.get("caption")
        if wall_caption:
            parts.append(f'caption="{wall_caption}"')
        lines.append(f'Wall({", ".join(parts)})')

    for wall in context.get("walls", {}).values():
        for door in wall.get("doors", {}).values():
            parts = [
                f'label="{door.get("label", "door")}"',
                f'center={_ssl_fmt_list(door["center"])}',
                f'width={_ssl_fmt_num(door["width"])}',
                f'height={_ssl_fmt_num(door["height"])}',
            ]
            caption = door.get("caption")
            if caption:
                parts.append(f'caption="{caption}"')
            wall_label = door.get("wall")
            if wall_label:
                parts.append(f'wall="{wall_label}"')
            if door.get("asset_id") is not None:
                parts.append(f'asset_id="{door["asset_id"]}"')
            lines.append(f'Door({", ".join(parts)})')

    for wall in context.get("walls", {}).values():
        for window in wall.get("windows", {}).values():
            parts = [
                f'label="{window.get("label", "window")}"',
                f'center={_ssl_fmt_list(window["center"])}',
                f'width={_ssl_fmt_num(window["width"])}',
                f'height={_ssl_fmt_num(window["height"])}',
            ]
            caption = window.get("caption")
            if caption:
                parts.append(f'caption="{caption}"')
            wall_label = window.get("wall")
            if wall_label:
                parts.append(f'wall="{wall_label}"')
            if window.get("asset_id") is not None:
                parts.append(f'asset_id="{window["asset_id"]}"')
            lines.append(f'Window({", ".join(parts)})')

    for box in context.get("boxes", {}).values():
        box_label = box.get("label", "object0")
        parts = [
            f'label="{box_label}"',
            f'center={_ssl_fmt_list(box["center"])}',
            f'angle_z={_ssl_fmt_num(box["angle_z"])}',
            f'scale={_ssl_fmt_list(box["scale"])}',
        ]
        caption = box.get("caption")
        if caption:
            parts.append(f'caption="{caption}"')
        if box.get("asset_id") is not None:
            parts.append(f'asset_id="{box["asset_id"]}"')
        lines.append(f'Bbox({", ".join(parts)})')

    return "\n".join(lines) + "\n"


def update_ssl_with_asset_id(ssl_text: str, scene_json: Dict[str, Any]) -> str:
    asset_map = {}
    for cat in ["door", "window", "bbox"]:
        for item in scene_json.get(cat, []):
            if item.get("id") and item.get("asset_id"):
                asset_map[(cat.capitalize() if cat != "bbox" else "Bbox", item["id"])] = item["asset_id"]
    new_lines = []
    for line in ssl_text.splitlines():
        trimmed = line.strip()
        if not trimmed: new_lines.append(line); continue
        obj_type = next((t for t in ["Door", "Window", "Bbox"] if trimmed.startswith(t + "(")), None)
        if obj_type:
            id_match = re.search(r'id="([^"]+)"', trimmed)
            if id_match:
                asset_id = asset_map.get((obj_type, id_match.group(1)))
                if asset_id:
                    if 'asset_id=' in trimmed: line = re.sub(r'asset_id="[^"]*"', f'asset_id="{asset_id}"', line)
                    else:
                        r_idx = line.rfind(')')
                        if r_idx != -1:
                            prefix = line[:r_idx].strip()
                            line = line[:r_idx] + ('' if prefix.endswith('(') else ', ') + f'asset_id="{asset_id}"' + line[r_idx:]
        new_lines.append(line)
    return "\n".join(new_lines)

def generate_texture(ctx: Any, image_path: str):
    edit_image_with_qunhe = _get_edit_image_with_qunhe()
    if edit_image_with_qunhe is None:
        print("Warning: edit_image_with_qunhe not found, skip texture generation.")
        return False

    print(f"📦 正在为图片 {image_path} 生成纹理...")
    image_dir = os.path.dirname(image_path)
    texture_dir = os.path.join(image_dir, "texture")
    os.makedirs(texture_dir, exist_ok=True)
    floor_texture_path = os.path.join(texture_dir, "floor_texture.png")
    wall_texture_path = os.path.join(texture_dir, "wall_texture.png")
    ceiling_texture_path = os.path.join(texture_dir, "ceiling_texture.png")
    
    floor_prompt = "这是一张场景透视图, 帮我生成这个房间可能的地板纹理图片, 你反思自己在生成时候是否已经排除了墙体, 以及房间中其他物体的影响, 只生成地板本身纹理, 同时这个纹理必须是符合这个房间风格的可重复的简单纹理(seamless tileable texture)"
    wall_prompt = "这是一张场景透视图, 帮我生成这个房间可能的墙体纹理图片, 你需要反思自己在生成时候是否已经排除了地板, 墙上的挂画和挂饰, 以及房间中其他物体的影响, 只关注墙面本身纹理, 同时这个纹理必须是符合这个房间风格的可重复的简单纹理(seamless tileable texture)"
    ceiling_prompt = "这是一张场景透视图, 帮我生成这个房间可能的天花板纹理图片, 你需要反思自己在生成时候是否已经排除了墙体, 地板以及房间中其他物体的影响, 只生成天花板本身纹理, 同时这个纹理必须是符合这个房间风格的可重复的简单纹理(seamless tileable texture)"

    floor_texture_path_tmp = None
    wall_texture_path_tmp = None
    ceiling_texture_path_tmp = None

    if os.path.exists(floor_texture_path):
        print(f"⏭️ 跳过地板纹理生成，已存在: {floor_texture_path}")
    else:
        floor_texture_path_tmp = edit_image_with_qunhe(floor_prompt, image_path, output_dir=texture_dir)

    if os.path.exists(wall_texture_path):
        print(f"⏭️ 跳过墙体纹理生成，已存在: {wall_texture_path}")
    else:
        wall_texture_path_tmp = edit_image_with_qunhe(wall_prompt, image_path, output_dir=texture_dir)

    if os.path.exists(ceiling_texture_path):
        print(f"⏭️ 跳过天花板纹理生成，已存在: {ceiling_texture_path}")
    else:
        ceiling_texture_path_tmp = edit_image_with_qunhe(ceiling_prompt, image_path, output_dir=texture_dir)
    
    if floor_texture_path_tmp:
        shutil.copy2(floor_texture_path_tmp, floor_texture_path)
        floor_texture_path_tmp_dir = os.path.dirname(floor_texture_path_tmp)
        if floor_texture_path_tmp_dir != texture_dir and os.path.exists(floor_texture_path_tmp_dir):
            shutil.rmtree(floor_texture_path_tmp_dir)
    if wall_texture_path_tmp:
        shutil.copy2(wall_texture_path_tmp, wall_texture_path)
        wall_texture_path_tmp_dir = os.path.dirname(wall_texture_path_tmp)
        if wall_texture_path_tmp_dir != texture_dir and os.path.exists(wall_texture_path_tmp_dir):
            shutil.rmtree(wall_texture_path_tmp_dir)
    if ceiling_texture_path_tmp:
        shutil.copy2(ceiling_texture_path_tmp, ceiling_texture_path)
        ceiling_texture_path_tmp_dir = os.path.dirname(ceiling_texture_path_tmp)
        if ceiling_texture_path_tmp_dir != texture_dir and os.path.exists(ceiling_texture_path_tmp_dir):
            shutil.rmtree(ceiling_texture_path_tmp_dir)


    ctx.set_wall_blender_texture_path(wall_texture_path)
    ctx.set_floor_blender_texture_path(floor_texture_path)
    ctx.set_ceiling_blender_texture_path(ceiling_texture_path)

    return True