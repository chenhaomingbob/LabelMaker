import numpy as np
import open3d as o3d

def drawOpen3dCylLines(bbListIn, col=None):
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    line_sets = []

    for bb in bbListIn:
        points = bb
        if col is None:
            col = [0, 0, 1]
        colors = [col for i in range(len(lines))]

        line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        line_sets = line_mesh1_geoms[0]
        for l in line_mesh1_geoms[1:]:
            line_sets = line_sets + l

    return line_sets

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)
        # line_segments_unit += np.array([1e-2, 0, 0])

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        # assert line_segments_unit.shape[0] == 12
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5  # + np.array([1e-2, 0, 0])
            # create cylinder and apply transformations
            # assert line_length > 0
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                aa = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))  # , center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[5, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[6, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[7, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners

def normalize_point(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_bdb_from_corners(corners, planexy=[3, 2, 6, 7]):
    """
    get coeffs, basis, centroid from corners
    :param corners: 8x3 numpy array
        corners of a 3D bounding box
    :return: bounding box parameters
    """
    up_max = np.max(corners[:, 1])
    up_min = np.min(corners[:, 1])

    points_2d = corners[planexy, :]
    points_2d = points_2d[np.argsort(points_2d[:, 0]), :]

    vector2 = np.array([points_2d[1, 0] - points_2d[0, 0], 0, points_2d[1, 2] - points_2d[0, 2]])
    vector1 = np.array([points_2d[2, 0] - points_2d[0, 0], 0, points_2d[2, 2] - points_2d[0, 2]])

    coeff1 = np.linalg.norm(vector1)
    coeff2 = np.linalg.norm(vector2)
    vector1 = normalize_point(vector1)
    vector2 = np.cross(vector1, [0, 1, 0])
    centroid = np.array(
        [points_2d[0, 0] + points_2d[3, 0], float(up_max) + float(up_min), points_2d[0, 2] + points_2d[3, 2]]) * 0.5

    basis = np.array([vector1, [0, 1, 0], vector2])
    coeffs = np.array([coeff1, up_max - up_min, coeff2]) * 0.5
    return centroid, basis.T, coeffs