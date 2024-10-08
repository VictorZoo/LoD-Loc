import bpy 
import numpy as np
import sys
import mathutils
from mathutils import Vector
import os
def parse_image_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            model, width, height, *params = data
            params = np.array(params, float)
            images[os.path.basename(name).split('.')[0]] = (model, int(width), int(height), params)
  
    assert len(images) > 0
    return images
def parse_render_image_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = (data_line[0].split('/')[-1]).split('.')[0]
            w, h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K = [w, h,fx,fy,cx,cy]
            images[name] = K
  
    return images
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

                if origin_coord is not None:
                    origin_coord = np.array(origin_coord)
                    T[0:3,3] -= origin_coord
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

def prepare_scene(resolution=[760, 760], device='GPU'):
    bpy.data.scenes["Scene"].render.resolution_x = resolution[0]
    bpy.data.scenes["Scene"].render.resolution_y = resolution[1]
#    bpy.data.scenes["Scene"].cycles.device = device
    try:
        #delete_object('Cube')
        delete_object('Light')
    except:
        pass

def prepare_camera(camera_fov=40, camera_location=[0., 0., -15.], camera_angles=[0., 0., 0.], clip=[0.1, 100000000.], intensity=2., world_color=[0.3, 0.3, 0.3]):
    """
    Definition to prepare Blender for renderings.
    """
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.objects["Camera"].data.type = 'PERSP'
    bpy.context.scene.objects["Camera"].data.lens = camera_fov
    bpy.context.scene.objects["Camera"].data.clip_start = clip[0]
    bpy.context.scene.objects["Camera"].data.clip_end = clip[1]
    bpy.context.scene.objects["Camera"].location[0] = camera_location[0]
    bpy.context.scene.objects["Camera"].location[1] = camera_location[1]
    bpy.context.scene.objects["Camera"].location[2] = camera_location[2]

    bpy.context.scene.objects["Camera"].rotation_euler[0] = camera_angles[0]
    bpy.context.scene.objects["Camera"].rotation_euler[1] = camera_angles[1]
    bpy.context.scene.objects["Camera"].rotation_euler[2] = camera_angles[2]
    
def add_camera(xyz=(0, 0, 0),
               rot_vec_degree=(0, 0, 0),
               name=None,
               proj_model='PERSP',
               f=35,
               sensor_fit='HORIZONTAL',
               sensor_width=32,
               sensor_height=18,
               clip_start=0.1,
               clip_end=10000):
    bpy.ops.object.camera_add(location=xyz, rotation=rot_vec_degree)
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name
    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end
    return cam    
 
def prepare_world(image_save_path, depth_name):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # 必须设置，否则无法输出深度
    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True


    # 必须设置，否则无法输出法向
    bpy.context.view_layer.use_pass_normal = True 

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    
    scene = bpy.context.scene
    # cam = scene.objects['Camera']
    for output_node in [depth_file_output]: #image_file_output
        output_node.base_path = ''

    # 输出路径            
    scene.render.filepath = image_save_path
    depth_file_output.file_slots[0].path = scene.render.filepath + depth_name
    bpy.ops.render.render(write_still=True)


def delete_object(label):
    """
    Definition to delete an object from the scene.
    Parameters
    ----------
    label          : str
                     String that identifies the object to be deleted.
    """
    bpy.data.objects.remove(bpy.context.scene.objects[label], do_unlink=True)

# Clear all nodes in a mat


image_save_path = str(sys.argv[-1])
input_pose = str(sys.argv[-2])
intput_intrin = str(sys.argv[-3])


origin_coord = [0, 0, 0]
#GET pose
poses = parse_pose_list(input_pose, origin_coord)
intrinsics = parse_render_image_list(intput_intrin)
 
names = list(intrinsics.keys())
index = 0
for name in names:
################ load pose
    pose_frame = poses[name]  
    intrinsic = intrinsics[name]
     
    R = pose_frame[:3, :3]
    t = list(pose_frame[:3, 3])
    
    rot_mat = mathutils.Matrix(list(R))
    rot_eulr = rot_mat.to_euler()   
#################initial camera
    w = int(intrinsic[0])
    h = int(intrinsic[1])
    f_x = np.float32(intrinsic[2])
    f_y = np.float32(intrinsic[3])
    cx = np.float32(intrinsic[4])
    cy = np.float32(intrinsic[5]) 
    
    
    f_mm = 35   #!
    intrin_factor_fx = f_x / f_mm
    intrin_factor_fy = f_y / f_mm
    camera_obj = add_camera(xyz = t, rot_vec_degree = rot_eulr, f=f_mm, sensor_fit='HORIZONTAL',sensor_width=w/intrin_factor_fx, sensor_height=h/intrin_factor_fy, clip_end=10000)

 
    image_save_path = image_save_path  +'/'
    prepare_scene(resolution=[w , h ], device='GPU')
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.camera = camera_obj
    prepare_world(image_save_path, (name.split('/')[-1]).split('.')[0])

