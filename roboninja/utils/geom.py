import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def binary2sdf(binary_mask):
    assert len(binary_mask.shape) == 2
    sdf = cv2.distanceTransform(
        src=binary_mask.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE
    )
    return sdf / binary_mask.shape[0] # normalization


def compute_camera_angle(cam_pos, cam_lookat):
    cam_dir = np.array(cam_lookat) - np.array(cam_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-cam_dir[0], -cam_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(cam_dir[1], np.linalg.norm([cam_dir[0], cam_dir[2]]))
    
    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])


def get_camera_matrix(cam_pos, cam_size, cam_fov, cam_angle=None, cam_lookat=None):
    if cam_angle is None:
        assert cam_lookat is not None
        cam_angle = compute_camera_angle(cam_pos, cam_lookat)
    focal_length = cam_size[0] / 2 / np.tan(cam_fov / 2)
    cam_intrinsics = np.array([[focal_length, 0, float(cam_size[1])/2],
                               [0, focal_length, float(cam_size[0])/2],
                               [0, 0, 1]])
    cam_pose = np.eye(4)
    rotation_matrix = Rotation.from_euler('xyz', [cam_angle[1], np.pi - cam_angle[0], np.pi], degrees=False).as_matrix()
    cam_pose[:3, :3] = rotation_matrix
    cam_pose[:3, 3] = cam_pos

    return cam_intrinsics, cam_pose


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


def project_pts_to_2d(pts, camera_view_matrix, camera_intrisic):
    """Project points to 2D.
    Args:
        pts: Nx3 float array of 3D points in world coordinates.
        camera_view_matrix: 4x4 float array. A wrd2cam transformation defining camera's totation and translation.
        camera_intrisic: 3x3 float array. [ [f,0,0],[0,f,0],[0,0,1] ]. f is focal length.
    Returns:
        coord_2d: Nx3 float array of 2D pixel. (w, h, d) the last one is depth
    """
    pts_c = transform_pointcloud(pts, camera_view_matrix[0:3, :])
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = transform_pointcloud(pts_c, rot_algix) # Nx3
    coord_2d = np.dot(camera_intrisic, pts_c.T) # 3xN
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[:, 2]
    coord_2d = np.array([coord_2d[1], coord_2d[0], coord_2d[2]])
    return coord_2d.T