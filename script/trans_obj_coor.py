import numpy as np
from utils import read_objFile
from ReTransform import ReframeTransform
import pyproj

def get_ecef_origin():
    """Shift the origin to make the value of coordinates in ECEF smaller and increase training stability"""
    # Warning: this is dataset specific!
    ori_lon, ori_lat, ori_alt = 6.5668, 46.5191, 390
    ori_x, ori_y, ori_z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(ori_lat, ori_lon, ori_alt)
    print('Origin XYZ: {}, {}, {}'.format(ori_x, ori_y, ori_z))
    origin = np.array([ori_x, ori_y, ori_z], dtype=np.float64)
    return origin

def main(obj_path: str,
        save_path: str,
        ):

    v_list, v_normal, v_face = read_objFile(obj_path)

    r = ReframeTransform()
    xyz = np.array(v_list)
    s_h_srs, s_v_srs, t_h_srs, t_v_srs = "lv95", "ln02", "wgs84", "wgs84"  # orgin
    origin_baimo =[2532713, 1152052, 0]
    xyz += origin_baimo
    origin_crossloc = get_ecef_origin()
    new_coors = r.transform_points(xyz, s_h_srs, s_v_srs, t_h_srs, t_v_srs)
    transformer = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978")
    new_coors_ECEF = np.array(list(transformer.itransform(new_coors, switch= True)))
    new_coors_ECEF = new_coors_ECEF[:,[1,0,2]]
    new_coors_ECEF -= origin_crossloc
    new_coors_ECEF = list(new_coors_ECEF)

    with open(save_path, 'w') as file:
        for v in new_coors_ECEF:
            to_write = " ".join(map(str, v))
            file.write(f"v {to_write}\n")

        for v in v_normal:
            to_write = " ".join(map(str, v))
            file.write(f"vn {to_write}\n")

        for v in v_face:
            to_write = " ".join(map(str, v))
            file.write(f"f {to_write}\n")
    return 0



if __name__ == "__main__":
    # obj coordinates system lv95+ln02 -> ECEF
    obj_path = "./simple_YforZup_clip.obj"
    save_path = "./epfl_final/ecef.obj"
    sampling_interval = 0.5
    main(obj_path, save_path)


