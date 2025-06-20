import numpy as np
from scipy.spatial.transform import Rotation

def matrix_to_euler(R):
    # 转换为欧拉角（ZYX顺序）
    return Rotation.from_matrix(R).as_euler('zyx', degrees=True)

def matrix_to_axis_angle(R):
    # 转换为轴角表示
    rotvec = Rotation.from_matrix(R).as_rotvec()
    angle = np.linalg.norm(rotvec)
    axis = rotvec / angle if angle > 0 else np.array([0, 0, 1])
    return axis, np.degrees(angle)
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return R
R = np.array([
    [0.95461623, -0.29783862 , 0.],
    [0.29783862 , 0.95461623,  0.],
    [0,0,1],
])

print("欧拉角:", matrix_to_euler(R))
axis, angle = matrix_to_axis_angle(R)
print("旋转轴:", axis, "角度:", angle)

R = np.array([
    [0.99961417, -0.02777603 , 0.],
    [0.02777603 ,0.99961417 ,  0.],
    [0,0,1],
])

print("欧拉角:", matrix_to_euler(R))
axis, angle = matrix_to_axis_angle(R)
print("旋转轴:", axis, "角度:", angle)


q = [-0.000136242408188991, 0.0010197659721598, 0.0040252492763102, 0.999991357326507]
R = quaternion_to_rotation_matrix(q)
print("欧拉角:", matrix_to_euler(R))
axis, angle = matrix_to_axis_angle(R)
print("旋转轴:", axis, "角度:", angle)

q = [0.0,0.0,0.000220499112933778,0.99999997569007]
R = quaternion_to_rotation_matrix(q)
print("欧拉角:", matrix_to_euler(R))
axis, angle = matrix_to_axis_angle(R)
print("旋转轴:", axis, "角度:", angle)