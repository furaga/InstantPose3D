# NOTE: blneder2.93のGUI上で実行すること
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Matrix

# https://www.mixamo.com/
mesh_root = Path(r"D:\workspace\InstantPose3D\avatars\mixamo\characters")
motion_root = Path(r"D:\workspace\InstantPose3D\avatars\mixamo\motions")
out_root = Path(r"D:\workspace\InstantPose3D\train")

done = False

num_frame = 5000


def get_intrinsic(scene, camdata, mode="complete"):
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    if mode == "simple":

        aspect_ratio = height / width
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2) * aspect_ratio
        K[1][1] = height / 2.0 / np.tan(camdata.angle / 2)
        K[0][2] = width / 2.0
        K[1][2] = height / 2.0
        K[2][2] = 1.0
        K.transpose()

    if mode == "complete":

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if camdata.sensor_fit == "VERTICAL":
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = (
                scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            )
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array(
            [[alpha_u, skew, u_0], [0, alpha_v, v_0], [0, 0, 1]], dtype=np.float32
        )

    return K


def look_at(camera, target):
    ray = np.subtract(target, camera.location)
    ray_xy = np.array([ray[0], ray[1], 0])
    x = np.array([-ray[2], +0, ray[1]])
    y = np.array([np.linalg.norm(ray_xy), 0, -ray[0]])
    camera.rotation_euler = np.arctan2(y, x)


def matrix_to_str(m):
    m = np.array(m)
    text = f"{m.shape[0]},{m.shape[1]};"
    text += ",".join(str(v) for v in m.ravel())
    return text


def euler_to_str(e):
    return matrix_to_str([[e.x, e.y, e.z]])


def resize_height(amt, mesh, target_height):
    mat = mesh.matrix_world
    positions = np.array([mat @ v.co for v in mesh.data.vertices])
    max_pos = np.max(positions, axis=0)
    min_pos = np.min(positions, axis=0)
    height = (max_pos - min_pos)[2]
    scale = target_height / height

    amt.scale = (amt.scale[0] * scale, amt.scale[1] * scale, amt.scale[2] * scale)
    print("scale", scale)


def get_children(parent):
    children = []
    for ob in bpy.data.objects:
        if ob.parent == parent:
            children.append(ob)
    return children


def join_meshes(meshes):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) >= 2:
        for mesh in meshes:
            mesh.select_set(True)
        bpy.ops.object.join()
    return meshes[0]


def setup_materials():
    for mat in bpy.data.materials:
        mat.use_nodes = True
        node_tree = mat.node_tree
        principled = node_tree.nodes["Principled BSDF"]

        to_remove = []
        for l in node_tree.links:
            if l.to_socket == principled.inputs["Metallic"]:
                to_remove.append(l)
            if l.to_socket == principled.inputs["Specular"]:
                to_remove.append(l)
            if l.to_socket == principled.inputs["Roughness"]:
                to_remove.append(l)
            if l.to_socket == principled.inputs["Alpha"]:
                to_remove.append(l)

        for l in to_remove:
            node_tree.links.remove(l)

        principled.inputs["Metallic"].default_value = 0
        principled.inputs["Specular"].default_value = 0.1
        principled.inputs["Roughness"].default_value = 0.5
        principled.inputs["Alpha"].default_value = 1


def look_at(camera, target):
    ray = np.subtract(target, camera.location)
    ray_xy = np.array([ray[0], ray[1], 0])
    x = np.array([-ray[2], +0, ray[1]])
    y = np.array([np.linalg.norm(ray_xy), 0, -ray[0]])
    camera.rotation_euler = np.arctan2(y, x)


def move_camera():
    import random

    camera = bpy.data.objects["Camera"]

    r = 7.0

    # 20 ~ 100
    min_theta = np.pi * 20 / 180
    max_theta = np.pi * 100 / 180
    t = random.random()
    theta = min_theta * (1 - t) + max_theta * t
    phi = random.random() * 2 * np.pi
    camera_position = [
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ]
    target = [0, 0, 1]
    camera = bpy.data.objects["Camera"]
    camera.location = camera_position

    look_at(camera, target)


@persistent
def load_post_callback(dummy):
    global done
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    bpy.ops.import_scene.fbx(filepath=str(motion_path))
    bpy.app.handlers.load_post.remove(load_post_callback)

    # アニメーションをリターゲット
    try:
        bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]
        bpy.context.scene.mix_source_armature = bpy.data.objects["Armature.001"]
        bpy.ops.mr.make_rig()
        bpy.ops.mr.import_anim_to_rig()
    except Exception as e:
        print(e)

    for scene in bpy.data.scenes:
        scene.render.engine = "CYCLES"
        for vl in bpy.context.scene.view_layers:
            vl.use_pass_normal = True
        scene.render.resolution_x = 448
        scene.render.resolution_y = 448
        scene.render.film_transparent = True
        scene.render.resolution_percentage = 100
        scene.cycles.progressive = "BRANCHED_PATH"

    name = fbx_path.stem + "_" + motion_path.stem
    amt = bpy.data.objects["Armature"]

    # join meshes
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = amt
    amt.select_set(True)
    bpy.ops.object.select_grouped(extend=True, type="CHILDREN_RECURSIVE")
    meshes = get_children(amt)
    mesh = join_meshes(meshes)

    # モデルのサイズを同じくらいに調整
    bpy.context.scene.frame_set(0)
    resize_height(amt, mesh, 1.6)

    setup_materials()
    
    # カメラランダム移動
    move_camera()

    # JPEG
    # bpy.context.scene.render.image_settings.file_format = "JPEG"
    bpy.context.scene.render.image_settings.file_format = "PNG"

    camera = bpy.data.objects["Camera"]

    kps_hist = []
    for i_frame in range(0, num_frame, 1):
        if i_frame % 10 == 0:
            print(f"  Frame {i_frame}")

        bpy.context.scene.frame_set(i_frame)
        bpy.context.view_layer.update()

        # PARAM
        param_path = out_root / "PARAMS_RAW" / name / f"{i_frame:05d}.txt"
        param_path.parent.mkdir(exist_ok=True, parents=True)

        kps = []
        with open(param_path, "w") as f:
            extrinsic = camera.matrix_world
            intrinsic = get_intrinsic(
                bpy.context.scene, bpy.data.objects["Camera"].data
            )
            f.write("extrinsic=" + matrix_to_str(extrinsic) + "\n")
            f.write("extrinsic_euler=" + euler_to_str(extrinsic.to_euler()) + "\n")
            f.write("intrinsic=" + matrix_to_str(intrinsic) + "\n")
            for b in amt.pose.bones:
                head_pos = amt.matrix_world @ b.head
                tail_pos = amt.matrix_world @ b.tail
                f.write(f"{b.name},head,{head_pos.x},{head_pos.y},{head_pos.z}\n")
                f.write(f"{b.name},tail,{tail_pos.x},{tail_pos.y},{tail_pos.z}\n")
                kps.append([head_pos.x, head_pos.y, head_pos.z])
                kps.append([tail_pos.x, tail_pos.y, tail_pos.z])
        kps = np.array(kps)
        kps_hist.append(kps)

        # RENDER
        out_img_path = out_root / "RENDER" / name / f"{i_frame:05d}.png"
        out_img_path.parent.mkdir(exist_ok=True, parents=True)
        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(filepath=str(out_img_path))

        if len(kps_hist) >= 3:
            diff1 = np.linalg.norm(kps_hist[-1] - kps_hist[-2])
            diff2 = np.linalg.norm(kps_hist[-2] - kps_hist[-3])
            if diff1 < 1e-8 and diff2 < 1e-8:
                break

    done = True


fbx_path = None
motion_path = None


def main():
    global fbx_path, motion_path, done

    for i, fbx_path in enumerate(mesh_root.glob("*.fbx")):
        print(f"[{i:03d}] {str(fbx_path)}")

        for j, motion_path in enumerate(motion_root.glob(f"{fbx_path.stem}/*.fbx")):
            done = False
            print(f"  ({j:03d}) {str(motion_path)}")
            bpy.ops.wm.read_homefile(use_empty=True)
            bpy.app.handlers.load_post.append(load_post_callback)
            bpy.ops.wm.open_mainfile(filepath=str(mesh_root / "../environment.blend"))


main()
