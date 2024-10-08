import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os 
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

e = 0.3

def mk_dir(path):
    path.mkdir(exist_ok=True, parents=True)

def qvec2rotmat(qvec):
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

def point_proj (sampled_points_array, rt_4x4, K):
    homogeneous_points = np.concatenate((sampled_points_array, np.ones((sampled_points_array.shape[0], 1))),axis = 1)
    rt_4x4 = np.linalg.inv(rt_4x4) # w2c

    transformed_points = np.matmul(rt_4x4, homogeneous_points.transpose(1,0))
    projected_points = np.matmul(K, transformed_points)
    projected_points = projected_points.transpose(1,0)
    pixel_coordinates = projected_points[:, :2] / projected_points[:, 2:]
    pixel_Cz = projected_points[:, 2:]

    return pixel_coordinates, pixel_Cz

def plot_points_on_image(image_path, points, output_path):
    """
    在彩色图像上绘制二维点，并保存结果。

    参数:
    - image_path (str): 输入彩色图像文件的路径。
    - points (numpy.ndarray): 二维点的坐标，形状为 (N, 2)。
    - output_path (str): 输出图像的文件路径，默认为 'output_image.jpg'。
    """

    image = Image.open(image_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(image) 
    # lawngreen
    plt.scatter(points[:, 0], points[:, 1], color='lawngreen', marker='o', s=0.2, label='Points of Wireframe')
    
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.cla()
    plt.close("all")
    # plt.show()  

def interpolate_depth(pos, depth):
    
    ids = torch.arange(0, pos.shape[0])
    depth = depth[:,:,0]
    h, w = depth.size()
    
    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners 验证坐标是否越界
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]

    # Valid depth验证深度
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    # 深度有效的点的index
    ids = ids[valid_depth]
    
    # Interpolation 插值深度
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #插值出来的深度
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return interpolated_depth, pos, ids

def read_objFile(obj_path):
    """
    Input: .obj path
    Output: v_list, v_normal, v_face
    """
    v_list = []
    v_normal = []
    v_face = []
    with open(obj_path,"r") as obj_file:
        for data in obj_file.read().rstrip().split('\n'):
            symbol = str(data.split()[0])   # symbol is v, vt, f or others
            if symbol == "v": # vertex
                v_list.append([float(item) for item in data.split()[1:]])
            elif symbol == "vn": # normal
                v_normal.append([float(item) for item in data.split()[1:]])
            elif symbol == "f": # face
                v_face.append([str(item) for item in data.split()[1:]])

    return v_list, v_normal, v_face

def extract_pair(index, data, pairs):

    paired_list = [[index[i], index[(i + 1) % len(index)]] for i in range(len(index))]

    
    for item in paired_list:
        index_pair = [item[0]-1, item[1]-1]
        data_pair = [data[index_pair[0]], data[index_pair[1]]]
        pairs.append(data_pair)

    return 0

def write_pose(path, save_folder):
    mk_dir(save_folder)
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split("/")[-1].split(".")[0]
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose()   #c2w
            Pose_c2w = np.identity(4)
            Pose_c2w[0:3,0:3] = R
            Pose_c2w[0:3, 3] = -R.dot(t)
            path_ = os.path.join(save_folder, name+"_pose.txt")

            np.savetxt(path_, Pose_c2w, delimiter=' ')


def write_intrinsic(intrinsc_path, save_folder):
    mk_dir(save_folder)
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            name = data_line[0].split("/")[-1].split(".")[0]
            _,_,fx,fy,cx,cy = list(map(float,data_line[2:]))[:]
            K_w2c = np.array([
            [fx,0.0,cx,0],
            [0.0,fy,cy,0],
            [0.0,0.0,1.0,0],
            ])

            path = os.path.join(save_folder, name+"_intrinsic.txt")

            np.savetxt(path, K_w2c, delimiter=' ')
                

def display_grayscale_image(image_path):
    # 打开图像并将其转换为灰度
    image = Image.open(image_path).convert('L')
    return np.array(image)

def read_instrincs(intrinsc_path):
    all_K = {}
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            
        #     print(data_line)
            img_name = data_line[0].split(".")[0]
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:]))[:]
            # cx, cy = w/2.0, h/2.0
            K_w2c = np.array([
            [fx,0.0,cx,0],
            [0.0,fy,cy,0],
            [0.0,0.0,1.0,0],
            ]) 
            # all_name.append(img_name)
            all_K[img_name] = K_w2c
    
    return all_K, w, h

def parse_pose_list(path):
    all_pose_c2w = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split("/")[-1].split(".")[0]
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose()   #c2w
            # w2c_t = -R @ t
            # w2c_t = torch.from_numpy(w2c_t)
            # w2c_t_ = [float(w2c_t[0][0]), float(w2c_t[0][1]), float(w2c_t[0][2])]
            Pose_c2w = np.identity(4)
            Pose_c2w[0:3,0:3] = R
            Pose_c2w[0:3, 3] = -R.dot(t) #t  ##!!! 改
            # all_pose_c2w.append([name, Pose_c2w, q, w2c_t_])
            all_pose_c2w[name] = Pose_c2w
        
    return all_pose_c2w

def create_folder_if_not_exists(save_path: str,
                                folder_item: list):
    for item in folder_item:
        if not os.path.exists(os.path.join(save_path, item)):
            os.makedirs(os.path.join(save_path, item))

def read_depth(depth_exr):
    depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
    depth = torch.tensor(depth)
    return depth

def clip(w, h, i_intr, i_pose, z_near, z_far, vertex_pair):
    R_c2w = i_pose[0:3,0:3]
    t_c2w = i_pose[0:3, 3]
    t_c2w = t_c2w.reshape([3,1])
    K_c2w = np.linalg.inv(i_intr[:,:3])

    # z_far
    p_corner = np.array([[0,w,0,w], [0,0,h,h], [1,1,1,1]])* z_far
    world_corner = R_c2w @ (K_c2w @ p_corner) + t_c2w

    X_max_far = np.amax(world_corner[0, :])
    X_min_far = np.amin(world_corner[0, :])

    Y_max_far = np.amax(world_corner[1, :])
    Y_min_far = np.amin(world_corner[1, :])

    # z_near 
    p_corner = np.array([[0,w,0,w], [0,0,h,h], [1,1,1,1]])* z_near
    world_corner = R_c2w @ (K_c2w @ p_corner) + t_c2w

    X_max_near = np.amax(world_corner[0, :])
    X_min_near = np.amin(world_corner[0, :])

    Y_max_near = np.amax(world_corner[1, :])
    Y_min_near = np.amin(world_corner[1, :])

    # min max
    X_min = min(X_min_far, X_min_near)
    X_max = max(X_max_far, X_max_near)

    Y_min = min(Y_min_far, Y_min_near)
    Y_max = max(Y_max_far, Y_max_near)

    mask_X = np.logical_and(vertex_pair[:, 0] >= X_min, vertex_pair[:, 0] <= X_max)
    mask_Y = np.logical_and(vertex_pair[:, 1] >= Y_min, vertex_pair[:, 1] <= Y_max)
    final_mask = np.logical_and(mask_X, mask_Y)
    
    return final_mask

def point_proj (sampled_points_array, rt_4x4, K):

    homogeneous_points = np.concatenate((sampled_points_array, np.ones((sampled_points_array.shape[0], 1))),axis = 1)
    rt_4x4 = np.linalg.inv(rt_4x4) # w2c


    transformed_points = np.matmul(rt_4x4, homogeneous_points.transpose(1,0))
    projected_points = np.matmul(K, transformed_points)
    projected_points = projected_points.transpose(1,0)
    pixel_coordinates = projected_points[:, :2] / projected_points[:, 2:]
    pixel_Cz = projected_points[:, 2:]


    return pixel_coordinates, pixel_Cz


def read_valid_depth(depth, pixel_coords):

    pixel_coords = torch.tensor(pixel_coords)
    coords_a = torch.unsqueeze(pixel_coords[:,0],0)
    coords_b =  torch.unsqueeze(pixel_coords[:,1],0)
    
    coords_inter = torch.cat((coords_b ,coords_a),0).transpose(1,0)

    depth, _, valid = interpolate_depth(coords_inter, depth)
    return depth, valid

def wireframe_extract(sampled_points3D, save_path, depth_path, pose_path, intrisic_path, image_path, if_save=False):
    mk_dir(save_path)
    z_near, z_far = 1, 300

    folder_item = ['proj_image']
    create_folder_if_not_exists(save_path, folder_item)

    all_k, w, h = read_instrincs(intrisic_path)
    sampled_points, sampled_normal = sampled_points3D[:,:3], sampled_points3D[:, 3:]

    list_ = os.listdir(pose_path)
    for name in list_:
        i_name = name.split("_pose")[0]
        img_pth = os.path.join(image_path, i_name.split('.')[0]+".jpg")

        depth_name = i_name.split('.')[0]+"0001.exr"
    
        depth_pth = os.path.join(depth_path, depth_name)
        pose_pth = os.path.join(pose_path, i_name + "_pose.txt")

        i_pose = np.loadtxt(pose_pth)
        i_intr = all_k[i_name]
        i_intr_depth = (i_intr.copy())
        i_intr_depth[0,2], i_intr_depth[1, 2] = w/2.0, h/2.0

        depth = read_depth(depth_pth)

        # clip
        cilp_mask = clip(w, h, i_intr, i_pose, z_near, z_far, sampled_points)
        i_sampled_points = sampled_points[cilp_mask]
        i_sampled_normal = sampled_normal[cilp_mask]

        pixel_depth, pixel_Cz = point_proj(i_sampled_points, i_pose.copy(), i_intr_depth)

        pixel_depth = pixel_depth - 0.5
        depth_values, valid = read_valid_depth(depth, pixel_depth)

        depth_values = depth_values.numpy()
        valid = valid.numpy()

        pixel_depth = pixel_depth[valid]
        i_sampled_points = i_sampled_points[valid]
        i_sampled_normal = i_sampled_normal[valid]
        pixel_Cz = pixel_Cz[valid]

        comparison_result = pixel_Cz.squeeze() <= (depth_values + e) 

        final_sampled_points = i_sampled_points[comparison_result]
        final_sampled_uv, _ = point_proj(final_sampled_points, i_pose.copy(), i_intr)

        np.save(os.path.join(save_path, '{}_points.npy'.format(i_name.split('_img')[0])), final_sampled_points)
        # breakpoint()
        if if_save:
            try:
                plot_points_on_image(img_pth, final_sampled_uv, os.path.join(save_path, "proj_image/{}".format(i_name)))
            except:
                continue

        print("Done with ", i_name)
