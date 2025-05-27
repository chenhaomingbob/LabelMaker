import numpy as np
# from scene.colmap_loader import Image


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def depth_to_point_cloud(depth_map, intrinsic_matrix):
    """
    将深度图和相机内参转换为3D点云。
    :param depth_map: 深度图 (H x W)
    :param intrinsic_matrix: 相机内参矩阵 (3x3)
    :return: Open3D 点云
    """
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 生成像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))  # (288,512)

    # 计算归一化相机坐标
    z = depth_map.flatten()
    x = (u.flatten() - cx) * z / fx
    y = (v.flatten() - cy) * z / fy

    # 构建点云
    points = np.vstack((x, y, z)).T
    points = points[z > 0]  # 移除无效点（深度为0）

    # 使用 Open3D 构建点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    return points


def point_cloud_to_voxel_grid(points, voxel_size, grid_shape, points_sem=None):
    """
    Convert a point cloud to a voxel grid.

    Args:
        points (numpy.ndarray): Nx3 array of points (x, y, z).
        voxel_size (float): Size of each voxel.
        grid_shape (tuple): Shape of the voxel grid (X, Y, Z).

    Returns:
        numpy.ndarray: Voxel grid of shape (X, Y, Z) where each element is 1 if the voxel is occupied, 0 otherwise.
    """
    # Initialize the voxel grid with zeros
    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

    # Calculate the minimum and maximum coordinates of the points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Calculate the offset to align points with the voxel grid
    offset = min_coords

    # Calculate the indices of each point in the voxel grid
    indices = ((points - offset - (voxel_size / 2.0)) / voxel_size).astype(int)  # voxel_size / 2.0 体素中心的偏移
    # indices = ((points - offset) / voxel_size).astype(int)

    # Clip indices to ensure they are within the bounds of the voxel grid
    indices = np.clip(indices, 0, np.array(grid_shape) - 1)

    if points_sem is None:
        # Mark the voxels as occupied
        voxel_grid[tuple(indices.T)] = 1
    else:
        # 根据最大投票来判断每个voxel的语义类别
        unique_voxels, inverse_indices = np.unique(indices, axis=0, return_inverse=True)
        # 初始化一个数组来存储每个体素的投票情况
        num_voxels = unique_voxels.shape[0]
        num_classes = np.max(points_sem) + 1
        voxel_votes = np.zeros((num_voxels, num_classes), dtype=int)
        # 更新每个体素的投票数
        np.add.at(voxel_votes, (inverse_indices, points_sem), 1)
        # 确定每个体素的类别
        voxel_classes = np.argmax(voxel_votes, axis=1) + 1
        # 0:empty
        # 0:ceiling => 1:ceiling

        voxel_grid[tuple(unique_voxels.T)] = voxel_classes

    return voxel_grid, min_coords


def point_cloud_to_voxel_grid_unbounded(points, voxel_size, points_sem=None):
    """
    Convert a point cloud to a voxel grid without unbounded

    Args:
        points (numpy.ndarray): Nx3 array of points (x, y, z).
        voxel_size (float): Size of each voxel.
        # grid_shape (tuple): Shape of the voxel grid (X, Y, Z).

    Returns:
        numpy.ndarray: Voxel grid of shape (X, Y, Z) where each element is 1 if the voxel is occupied, 0 otherwise.
    """
    # Initialize the voxel grid with zeros
    # voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

    # Calculate the minimum and maximum coordinates of the points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    grid_shape = ((max_coords - min_coords) / voxel_size).astype(int) + 1

    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)

    # Calculate the offset to align points with the voxel grid
    offset = min_coords

    # Calculate the indices of each point in the voxel grid
    indices = ((points - offset - (voxel_size / 2.0)) / voxel_size).astype(int)  # voxel_size / 2.0 体素中心的偏移
    # indices = ((points - offset) / voxel_size).astype(int)

    # Clip indices to ensure they are within the bounds of the voxel grid
    indices = np.clip(indices, 0, np.array(grid_shape) - 1)

    if points_sem is None:
        # Mark the voxels as occupied
        voxel_grid[tuple(indices.T)] = 1
    else:
        # 根据最大投票来判断每个voxel的语义类别
        unique_voxels, inverse_indices = np.unique(indices, axis=0, return_inverse=True)
        # 初始化一个数组来存储每个体素的投票情况
        num_voxels = unique_voxels.shape[0]
        num_classes = np.max(points_sem) + 1
        voxel_votes = np.zeros((num_voxels, num_classes), dtype=int)
        # 更新每个体素的投票数
        np.add.at(voxel_votes, (inverse_indices, points_sem), 1)
        # 确定每个体素的类别
        voxel_classes = np.argmax(voxel_votes, axis=1) + 1
        # 0:empty
        # 0:ceiling => 1:ceiling

        voxel_grid[tuple(unique_voxels.T)] = voxel_classes

    return voxel_grid, min_coords


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)

    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1], indexing='ij') # 按z,y,x的升序排列
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float64)

    coords_grid = (coords_grid * resolution) + resolution / 2

    # temp = np.copy(coords_grid)
    # temp[:, 1] = coords_grid[:, 0]
    # temp[:, 0] = coords_grid[:, 1]
    # coords_grid = np.copy(temp)

    return coords_grid


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
            break

    return params
