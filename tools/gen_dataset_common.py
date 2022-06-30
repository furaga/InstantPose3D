import numpy as np


def get_intrinsic(scene, camdata, mode="simple"):
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
