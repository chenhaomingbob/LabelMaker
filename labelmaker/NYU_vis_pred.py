import pickle
import os
from omegaconf import DictConfig
import numpy as np
import hydra
from mayavi import mlab


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)

    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float64)

    coords_grid = (coords_grid * resolution) + resolution / 2

    # temp = np.copy(coords_grid)
    # temp[:, 0] = coords_grid[:, 1]
    # temp[:, 1] = coords_grid[:, 0]
    # coords_grid = np.copy(temp)

    return coords_grid


def draw(
        voxels,
        cam_pose,
        vox_origin,
        voxel_size=0.08,
        d=0.75,  # 0.75m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    y = d * 480 / (2 * 518.8579)
    x = d * 640 / (2 * 518.8579)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )  # camera to world
    tri_points = np.hstack([tri_points, np.ones((5, 1))])

    tri_points = (cam_pose @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    # grid_coords = get_grid_coords(
    #     [voxels.shape[0], voxels.shape[2], voxels.shape[1]], voxel_size
    # )
    #
    # # Attach the predicted class to every voxel
    # grid_coords = np.vstack(
    #     (grid_coords.T, np.moveaxis(voxels, [0, 1, 2], [0, 2, 1]).reshape(-1))
    # ).T  #

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack((grid_coords.T, voxels.reshape(-1))).T  #

    # Remove empty and unknown voxels
    occupied_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]
    # occupied_voxels = grid_coords[(grid_coords[:, 3] == 0) & (grid_coords[:, 3] < 255)]  # empty
    # occupied_voxels = grid_coords[(grid_coords[:, 3] >= 0) & (grid_coords[:, 3] < 255)]  # empty & semantic
    # occupied_voxels = grid_coords[(grid_coords[:, 3] == 255)]  # unknown
    # occupied_voxels[:, 3] = 0
    figure = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))

    # Draw the camera
    mlab.triangular_mesh(
        x,
        y,
        z,
        triangles,
        representation="wireframe",
        color=(0, 0, 0),
        line_width=5,
    )

    # Draw occupied voxels
    plt_plot = mlab.points3d(
        occupied_voxels[:, 0],
        occupied_voxels[:, 1],
        occupied_voxels[:, 2],
        occupied_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.1 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=12,
    )

    colors = np.array(
        [
            [22, 191, 206, 255],
            [214, 38, 40, 255],
            [43, 160, 43, 255],
            [158, 216, 229, 255],
            [114, 158, 206, 255],
            [204, 204, 91, 255],
            [255, 186, 119, 255],
            [147, 102, 188, 255],
            [30, 119, 181, 255],
            [188, 188, 33, 255],
            [255, 127, 12, 255],
            [196, 175, 214, 255],
            [153, 153, 153, 255],
        ]
    )

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    # mlab.show()

    #
    x = np.append(x, np.mean(x[1:]))  # 添加
    y = np.append(y, np.mean(y[1:]))
    z = np.append(z, np.mean(z[1:]))

    x = np.append(x, 0)  # 添加世界原点
    y = np.append(y, 0)
    z = np.append(z, 0)

    import math

    # 调整视角，让相机中心始终在屏幕前方
    new_azimuth = math.atan2(y[-2] - y[0], x[-2] - x[0]) / math.pi * 180  # 调整方位角
    # new_elevation = math.atan2(z[-2] - z[0], x[-2] - x[0]) / math.pi * 180  # 调整方位角
    # print(mlab.view()[0], mlab.view()[1])
    mlab.view(azimuth=new_azimuth - 180)
    # print(mlab.view()[0], mlab.view()[1])

    # mlab.savefig(os.path.join(output_dir, f"{name}.png"))
    # mlab.close()
    mlab.show()


@hydra.main(config_path=None)
def main(config: DictConfig):
    scan = config.file

    if 'voxel_size' in config:
        voxel_size = config.voxel_size
    else:
        voxel_size = 0.08

    with open(scan, "rb") as handle:
        b = pickle.load(handle)

    print('voxel_size', voxel_size)

    cam_pose = b["cam_pose"]
    vox_origin = b["voxel_origin"]
    ##

    print(cam_pose)
    print(vox_origin)
    gt_scene = b["target"]  # (60,36,60)
    # gt_scene = b["target_1_4"]  # (60,36,60)
    print(gt_scene.shape)  # (60,36,60)
    # gt_scene = np.swapaxes(gt_scene, 1, 2)
    gt_scene = np.swapaxes(gt_scene, 0, 1)
    print(gt_scene.shape)  # (60,36,60)

    scan = os.path.basename(scan)[:12]

    draw(
        gt_scene,
        cam_pose,
        vox_origin,
        voxel_size=voxel_size,
        d=0.75,
    )


if __name__ == "__main__":
    main()
