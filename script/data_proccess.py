import numpy as np
import torch
import pickle
import os
import argparse
from pathlib import Path
import json
from utils import \
    (read_objFile, 
    extract_pair, 
    wireframe_extract, 
    write_pose,
    write_intrinsic,
    mk_dir)

data_root_dir = "dataset"

def blender_engine(
    blender_path,
    project_path,
    script_path,
    intrinscs_path,
    extrinsics_path,
    save_path,
):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {} {}'.format(
        blender_path,
        project_path,
        script_path,
        intrinscs_path,
        extrinsics_path,
        save_path, 
        
    )
    os.system(cmd)

def sample_3Dlines(vertex_pairs, normal_pairs, sampling_interval=0.8):
    # 将输入数据转换为PyTorch张量并移动到GPU上
    vertex_pairs = torch.tensor(vertex_pairs, dtype=torch.float32)#.cuda()
    normal_pairs = torch.tensor(normal_pairs, dtype=torch.float32)#.cuda()

    # 计算直线的方向向量
    direction_vectors = vertex_pairs[:, 1] - vertex_pairs[:, 0]

    # 计算两点之间的距离作为线段长度
    line_lengths = torch.norm(direction_vectors, dim=1)

    # 找到非零长度的线段索引
    nonzero_indices = torch.nonzero(line_lengths).squeeze()

    # 仅处理非零长度的线段
    direction_vectors = direction_vectors[nonzero_indices]
    line_lengths = line_lengths[nonzero_indices]
    vertex_pairs = vertex_pairs[nonzero_indices]
    normal_pairs = normal_pairs[nonzero_indices]

    # 计算采样点数量
    num_samples = torch.ceil(line_lengths / sampling_interval).to(torch.int32) + 1

    # 生成采样点和插值得到的法向量
    sampled_points = []
    interpolated_normals = []
    sampled_points_dict = {}
    for i, num_sample in enumerate(num_samples):
        t_values = torch.linspace(0, 1, num_sample, dtype=torch.float32).unsqueeze(1).to(direction_vectors.device)
        sampled_points_i = vertex_pairs[i, 0].unsqueeze(0) + t_values * direction_vectors[i].unsqueeze(0)
        interpolated_normals_i = normal_pairs[i, 0] + t_values * (normal_pairs[i, 1] - normal_pairs[i, 0]).unsqueeze(0)
        sampled_points.append(sampled_points_i)
        interpolated_normals.append(interpolated_normals_i)
        sampled_points_dict[i] = {
            'original_points': vertex_pairs[i].cpu().numpy(),  # 保存原始坐标信息
            'sampled_points': sampled_points_i.cpu().numpy(),  # 保存采样点
            'interpolated_normals': interpolated_normals_i.cpu().numpy()  # 保存插值得到的法向量
        }

    # 将采样点张量和插值得到的法向量张量转换为NumPy数组
    sampled_points_array = torch.cat(sampled_points, dim=0).cpu().numpy()
    interpolated_normals_array = torch.cat(interpolated_normals, dim=0).cpu().numpy()

    return sampled_points_array, interpolated_normals_array, sampled_points_dict

def face_VetexIndex(face_vertex_list, face_normal_list):
    face_dict = {}
    for i ,item in enumerate(zip(face_vertex_list, face_normal_list)):
        face_dict[i] = {
            'vertex_index': item[0],  # 保存原始坐标信息
            'normal_index': item[1],  # 保存采样点
        }
    return 

def remove_redundancy_2(input_list, final_sampled_normal):
    # 将列表转换为 NumPy 数组
    input_array = np.array(input_list)

    # 使用 NumPy 的 unique 函数去除重复项并保持原始顺序
    unique_array, indices = np.unique(input_array, axis=0, return_index=True)

    # 按照原始顺序重新排列
    sorted_indices = np.argsort(indices)
    unique_list = unique_array[sorted_indices]
    final_sampled_normal = final_sampled_normal[sorted_indices]

    return np.array(unique_list), np.array(final_sampled_normal)

def line_sample(obj_path: str,
         save_path: str,
         sampling_interval: float
         ):
    
    '''
    obj_path: LoD model in .obj format
    save_path: save path for .npy,
    sampling_interval: sampling interval
    '''
    if os.path.isfile(os.path.join(save_path, 'sampled_points3D.npy')):
        return np.load(os.path.join(save_path, 'sampled_points3D.npy'))
    
    mk_dir(save_path)

    print("Ready to generate .npy")
    vertex_pair = []
    normal_pair = []
    face_vertex_list = []
    face_normal_list = []
    sampled_points_array = np.array([]).reshape(0, 3)
    sampled_normal_array = np.array([]).reshape(0, 3)

    # 载入obj，并读取每一行，并存储顶点v及其对应序号（注：从1开始），和面f的顶点信息
    v_list, v_normal, v_face = read_objFile(obj_path)


    # 处理f的顶点信息，两两组合，去掉冗余，去掉视锥外线段，则得到线段信息（三维点位置、法向量）
    # 处理成 Nx2x3 的数组，便于计算
    for face_i in v_face:
        vertex_index = [int(item.split('/')[0]) for item in face_i]
        normal_index = [int(item.split('/')[2]) for item in face_i]

        # vetex in face
        face_vertex_list.append(vertex_index)
        face_normal_list.append(normal_index)
        # vertex pairs 
        extract_pair(vertex_index, v_list, vertex_pair)
        extract_pair(normal_index, v_normal, normal_pair)
    
    vertex_pair = np.array(vertex_pair)
    normal_pair = np.array(normal_pair)
    face_dict = face_VetexIndex(face_vertex_list, face_normal_list)
    sampled_points_array, sampled_normal_array, sampled_points_dict  = sample_3Dlines(vertex_pair, normal_pair, sampling_interval)

    sampled_points_array, sampled_normal_array = remove_redundancy_2(sampled_points_array, sampled_normal_array)
    
    # xyz + normal
    points3D_all = np.hstack([sampled_points_array, sampled_normal_array])

    with open(os.path.join(save_path,'face_VetexIndex.pkl'), 'wb') as f:
        pickle.dump(face_dict, f)

    np.save(os.path.join(save_path, 'sampled_points3D.npy'), points3D_all)

    with open(os.path.join(save_path,'sampled_points_dict.pkl'), 'wb') as f:
        pickle.dump(sampled_points_dict, f)
    print("Done with generating .npy")
    return points3D_all

def folder_prepare(GPS_pth, GT_pth, intri_pth, bpy_pth, depth_dir, config):
    GPS_folder = Path(os.path.join(config["dataset_pth"], 'GPS_pose'))
    write_pose(GPS_pth, GPS_folder)
    print('Done for GPS_folder')

    GT_folder = Path(os.path.join(config["dataset_pth"], 'GT_pose'))
    write_pose(GT_pth, GT_folder)
    print('Done for GT_folder')

    intri_folder = Path(os.path.join(config["dataset_pth"], 'intrinsic'))
    write_intrinsic(intri_pth, intri_folder)
    print('Done for intri_folder')

    blender_engine(config["blender_pth"], bpy_pth, config["script_path"], intri_pth, GPS_pth, depth_dir)
    print('Done for depths')


def main(config):
    for scene_name in config["scene_name"]:

        config["dataset_pth"] = Path(os.path.join(data_root_dir, config["data_name"],  scene_name))
        GPS_pth = Path(os.path.join(config["dataset_pth"], 'GPS_pose.txt'))
        GT_pth = Path(os.path.join(config["dataset_pth"], 'GT_pose.txt'))
        intri_pth = Path(os.path.join(config["dataset_pth"], 'intrinsic.txt'))
        q_img_pth = Path(os.path.join(config["dataset_pth"], 'Query_image'))
        bpy_pth = Path(os.path.join(config["model_pth"], config["data_name"] + '.blend'))
        depth_dir = Path(os.path.join(config["dataset_pth"], 'depth/'))


        # Prepare folder for poses, intrinsics and depths.
        folder_prepare(GPS_pth, GT_pth, intri_pth, bpy_pth, depth_dir, config)
        
        # Discrete wireframe points extract #
        obj_dir = Path(os.path.join(config["model_pth"].split('.')[0], config["data_name"] + '.obj'))
        save_dir = Path(os.path.join(config["model_pth"].split('.')[0], config["data_name"]))

        for sampling_interval in config["interval"]:
            npy_dir = Path(os.path.join(save_dir, str(sampling_interval) + 'm'))
            points3D_all = line_sample(obj_dir, npy_dir, sampling_interval)
            
            wireframe_points_save_pth = Path(os.path.join(config["dataset_pth"], 'Points', str(sampling_interval) + 'm'))
            wireframe_extract(points3D_all, wireframe_points_save_pth, depth_dir, str(GPS_pth).split('.')[0], intri_pth, q_img_pth, config["if_save"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config/data_preprocess.json', type=str,
                        help='Configure file for data preprocess')
    args = parser.parse_args()

    with open(args.config_file) as fp:
        config = json.load(fp)

    config["script_path"] = "script/depth_render.py"

    main(config)


    
