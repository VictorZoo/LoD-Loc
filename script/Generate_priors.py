import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import argparse

transf = np.array([
                    [1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1.],
                ])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def random_angle_offset(n_yaw):
    # 随机生成在范围内的角度偏差
    return np.random.uniform(-n_yaw, n_yaw)

def random_xyz_offset(n_xyz):
        # 随机生成在范围内的角度偏差
    return np.random.uniform(-n_xyz, n_xyz)

def add_random_offset(yaw_deg, pitch, roll, x, y, z, n_xy, n_z ,n_yaw, n_pi, n_ro):

    # 随机生成每个角度的偏差
    yaw_offset = random_angle_offset(n_yaw)
    pi_offset = random_angle_offset(n_pi)
    ro_offset = random_angle_offset(n_ro)
    x_offset = random_xyz_offset(n_xy)
    y_offset = random_xyz_offset(n_xy)
    z_offset = random_xyz_offset(n_z)

    # 在偏差范围内随机初始化新的欧拉角
    new_yaw_deg = np.clip(yaw_deg + yaw_offset, yaw_deg - n_yaw, yaw_deg + n_yaw)
    new_pi_deg = np.clip(pitch + pi_offset, pitch - n_pi, pitch + n_pi)
    new_ro_deg = np.clip(roll + ro_offset, roll - n_ro, roll + n_ro)
    new_x = np.clip(x + x_offset, x - n_xy, x + n_xy)
    new_y = np.clip(y + y_offset, y - n_xy, y + n_xy)
    new_z = np.clip(z + z_offset, z - n_z, z + n_z)
    
    return new_yaw_deg,new_pi_deg,new_ro_deg, new_x, new_y , new_z


def trans_eulerTo4x4(degree, xyz):
    ret_2 = R.from_euler('xyz', degree, degrees=True)
    R_ = ret_2.as_matrix()
    T = np.identity(4)
    T[0:3,0:3] = R_
    T[0:3,3] = xyz
    T = T @ np.linalg.inv(transf)# C2W

    return T

def seed_pose(txt_pth, save_pth, n_xy, n_z ,n_yaw, n_pi, n_ro):
    pose = np.loadtxt(txt_pth)

    pose = pose @ transf
    initial_R = pose[:3, :3]
    initial_xyz = pose[:3, 3]

    ret_init = R.from_matrix(initial_R)
    initial_euler = ret_init.as_euler('xyz',degrees=True)
    pitch , roll , yaw = initial_euler
    x, y ,z = initial_xyz

    new_yaw_deg, new_pi_deg, new_ro_deg, new_x, new_y , new_z = add_random_offset(yaw, pitch, roll, x, y ,z, n_xy, n_z ,n_yaw, n_pi, n_ro)

    new_euler = np.array([new_pi_deg , new_ro_deg , new_yaw_deg])
    print("New,", new_euler)
    print("Initial,",initial_euler)
    new_xyz = np.array([new_x, new_y, new_z])
    new_T = trans_eulerTo4x4(new_euler, new_xyz)
    np.savetxt(save_pth, new_T )


def main(txt_folder, save_folder):
    n_xy = 10
    n_z = 30
    n_yaw = 7.5
    n_pi = 1.0
    n_ro = 1.0

    list = os.listdir(txt_folder)
    for item in list :
        txt_pth = os.path.join(txt_folder, item)
        save_pth = os.path.join(save_folder, item)

        seed_pose(txt_pth, save_pth, n_xy, n_z ,n_yaw, n_pi, n_ro)
        print("Done with", item)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate the sensor priors")
    parser.add_argument("--GT_folder", default="./GT_folder")
    parser.add_argument("--save_folder", default="./Priors_folder")

    args = parser.parse_args()
    main(args.GT_folder, args.save_folder)
