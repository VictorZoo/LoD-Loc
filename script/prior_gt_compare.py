import numpy as np
import os

from scipy.spatial.transform import Rotation as R


import argparse
import matplotlib.pyplot as plt

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

def qvec2rotmat(qvec):  #!wxyz
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotation_to_quat (rotation_metrix):
    a = []
    # for _,v1 in rotation_metrix.items():
    for v1 in rotation_metrix:
        a.append(float(v1))
    a_np = np.array(a)
    a_np = a_np.reshape(3, 3)
    a_qvec = rotmat2qvec(a_np)# w,x,y,z
    return a_qvec, a_np



def parse_pose_list(path, origin_coord):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = (data[0].split('/')[-1]).split('.')[0]
            if len(data) > 1:
                q, t = np.split(np.array(data[1:], float), [4])
                
                R = np.asmatrix(qvec2rotmat(q))   
                t = -R.T @ t
                R = R.T
                
                T = np.identity(4)
                T[0:3,0:3] = R
                T[0:3,3] = t   #!  c2w

                # if origin_coord is not None:
                #     origin_coord = np.array(origin_coord)
                #     T[0:3,3] -= origin_coord
                transf = np.array([
                    [1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1.],
                ])
                T = T @ transf
                
                poses[name] = T
            
    
    assert len(poses) > 0
    return poses

def pose2euler(pose):
    pose = pose @ transf
    xyz , R_ = pose[0:3,3], pose[0:3,0:3]
    ret_init = R.from_matrix(R_)
    initial_euler = ret_init.as_euler('xyz',degrees=True)
    return xyz, initial_euler #Pitch Roll Yaw

def main(gt_pth, prior_pth, origin_coord, save_pth):
    # 生成gt.txt
    # generate_gt_txt(gt_xml, path)
    # 生成先验.txt
    # write_path = path + '/' + 'pose1.txt'
    # generate_prior_txt(image_path, write_path)
    # 比较
    # gt_poses = parse_pose_list(gt_txt, origin_coord)
    # prior_poses = parse_pose_list(prior_txt, origin_coord)
    # names = list(gt_poses.keys())
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)
    x_error = []
    y_error = []
    z_error = []
    yaw_e = []
    pitch_e = []
    roll_e = []

    # 可视化内容
    vis_img = {}

    item_list = os.listdir(prior_pth)
    
    for name in item_list:
################ load pose

        GPS_pth = os.path.join(prior_pth, name)
        # item =item.split("_pose")[0]+'.txt'
        GT_pth = os.path.join(gt_pth, name)

        GPS_pose = np.loadtxt(GPS_pth)
        GT_pose = np.loadtxt(GT_pth)
        # intrinsic = intrinsics[name]
        
        t_prior, euler_prior = pose2euler(GPS_pose)
        t_gt, euler_gt = pose2euler(GT_pose)
        
        # R_gt = pose_frame_gt[:3, :3]
        # ret_gt = R.from_matrix(R_gt)
        # euler_gt = ret_gt.as_euler('xyz',degrees=True)
        # t_gt = list(pose_frame_gt[:3, 3])

        # R_prior = pose_frame_prior[:3, :3]
        # ret_prior = R.from_matrix(R_prior)
        # euler_prior = ret_prior.as_euler('xyz',degrees=True)
        # t_prior = list(pose_frame_prior[:3, 3])
        
        # if abs(euler_gt[2]- euler_prior[2]) < 200 and abs(t_gt[2]-t_prior[2]) <=6.5 :
        if abs(euler_gt[2]- euler_prior[2]) < 50 and  abs(euler_gt[0]- euler_prior[0]) < 50:

            print(name, t_gt[0]-t_prior[0], t_gt[1]-t_prior[1], t_gt[2]-t_prior[2], euler_gt[0]- euler_prior[0],\
                euler_gt[1]- euler_prior[1], euler_gt[2]- euler_prior[2])

            x_error.append(t_gt[0]-t_prior[0])
            y_error.append(t_gt[1]-t_prior[1])
            z_error.append(t_gt[2]-t_prior[2])

            pitch_e.append(euler_gt[0]- euler_prior[0])
            yaw_e.append(euler_gt[2]- euler_prior[2])
            roll_e.append(euler_gt[1]- euler_prior[1])

            #x,y,z,pitch,roll,yaw
            # temp = [np.abs(t_gt[0]-t_prior[0]),np.abs(t_gt[1]-t_prior[1]),np.abs(t_gt[2]-t_prior[2]),np.abs(euler_gt[0]- euler_prior[0]),np.abs(euler_gt[1]- euler_prior[1]),np.abs(euler_gt[2]- euler_prior[2])]
            temp = [(t_gt[0]-t_prior[0]),(t_gt[1]-t_prior[1]),(t_gt[2]-t_prior[2]),(euler_gt[0]- euler_prior[0]),(euler_gt[1]- euler_prior[1]),(euler_gt[2]- euler_prior[2])]
            vis_img[name] = temp
    
    txt_file_path = save_pth + "/erro.txt"
    with open(txt_file_path, 'w') as file:
        for key, value in vis_img.items():
            file.write(f"{key}: {value}\n")

    # 标签
    labels_trans = ["X", "Y", "Z"]
    labels_angle = [ "Yaw", "Pitch", "Roll"]
    # X_info
    x_trans = ["Error[m]", "Error[m]", "Error[m]"]
    y_trans = ["Number of photos", " ", " "]
    # Y_info
    x_angle = ["Error[deg]", "Error[deg]", "Error[deg]"]
    y_angle = ["Number of photos", " ", " "]
    # title
    trans_title = "GCP vs RTK"
    angle_title = "GCP vs IMU"

    fig_trans, axes_trans = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        ax = axes_trans[i]
        if i == 0:
            ax.hist(x_error, bins=30)
            ax.set_title(labels_trans[i]+' (mean:'+str(np.mean(np.abs(x_error)))[:5] +")")
        elif i == 1:
            ax.hist(y_error,bins=30)
            ax.set_title(labels_trans[i]+' (mean:'+str(np.mean(np.abs(y_error)))[:5] +")")
        else:
            ax.hist(z_error,bins=30)
            ax.set_title(labels_trans[i]+' (mean:'+str(np.mean(np.abs(z_error)))[:5] +")")
        ax.set_xlabel(x_trans[i])
        ax.set_ylabel(y_trans[i])
    # fig_trans.suptitle(trans_title)
    plt.savefig(save_pth + "/trans.png")

    fig_ang, axes_ang = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        ax = axes_ang[i]
        if i == 0:
            ax.hist(yaw_e, bins=30)
            ax.set_title(labels_angle[i]+ ' (mean:'+str(np.mean(np.abs(yaw_e)))[:5] +")")
        elif i == 1:
            ax.hist(pitch_e, bins=30)
            ax.set_title(labels_angle[i]+ ' (mean:'+str(np.mean(np.abs(pitch_e)))[:5] +")")
        else:
            ax.hist(roll_e, bins=30)
            ax.set_title(labels_angle[i]+ ' (mean:'+str(np.mean(np.abs(roll_e)))[:5] +")")
        
        ax.set_xlabel(x_angle[i])
        ax.set_ylabel(y_angle[i])
    # fig_trans.suptitle(angle_title)
    plt.savefig(save_pth + "/angle.png")


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="write EXIF information (name qw qx qy qz x y z) in txt file")
    parser.add_argument("--gt_pth", default="dataset/Swiss-EPFL/inPlace/GT_pose")
    parser.add_argument("--prior_pth", default="/home/ubuntu/code/dataset_upload/Swiss-EPFL/inPlace/GPS_pose")
    parser.add_argument("--save_pth", default="/home/ubuntu/code/dataset_upload/Swiss-EPFL/inPlace/")
    parser.add_argument("--orgin", default=[0, 0, 0])

    args = parser.parse_args()
    main(args.gt_pth,  args.prior_pth, args.orgin, args.save_pth)