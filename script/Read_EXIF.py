import numpy as np
import os
import pandas as pd
from pyproj import Transformer
from pyproj import CRS
from scipy.spatial.transform import Rotation as R

import argparse
from utils import qvec2rotmat,rotmat2qvec
import exifread
from pyexiv2 import Image


def get_dji_exif(exif_file):
    # 打开exif文件
    with open(exif_file, "rb") as f:
        # 读取exif信息
        exif_data = exifread.process_file(f)
        img = Image(exif_file)
        all_info_xmp = img.read_xmp()
        # 获取GPS对应的标签
        gps_latitude_tag = "GPS GPSLatitude"
        gps_latitude_ref_tag = "GPS GPSLatitudeRef"
        gps_longitude_tag = "GPS GPSLongitude"
        gps_longitude_ref_tag = "GPS GPSLongitudeRef"
        gps_altitude_tag = "GPS GPSAltitude"
        yaw_tag = "Xmp.drone-dji.GimbalYawDegree"
        pitch_tag = "Xmp.drone-dji.GimbalPitchDegree"
        roll_tag = "Xmp.drone-dji.GimbalRollDegree"
        if gps_latitude_tag in exif_data and gps_latitude_ref_tag in exif_data and gps_longitude_tag in exif_data and gps_longitude_ref_tag in exif_data:
        
            # 获取GPS纬度和经度的分数值和方向值
            gps_latitude_value = exif_data[gps_latitude_tag].values
            gps_latitude_ref_value = exif_data[gps_latitude_ref_tag].values
            gps_longitude_value = exif_data[gps_longitude_tag].values
            gps_longitude_ref_value = exif_data[gps_longitude_ref_tag].values
            gps_altitude_value = exif_data[gps_altitude_tag].values
            # 将GPS纬度和经度的分数值转换为浮点数值
            gps_latitude = (float(gps_latitude_value[0].num) / float(gps_latitude_value[0].den) +
                            (float(gps_latitude_value[1].num) / float(gps_latitude_value[1].den)) / 60.0 +
                            (float(gps_latitude_value[2].num) / float(gps_latitude_value[2].den)) / 3600.0)
            gps_longitude = (float(gps_longitude_value[0].num) / float(gps_longitude_value[0].den) +
                             (float(gps_longitude_value[1].num) / float(gps_longitude_value[1].den)) / 60.0 +
                             (float(gps_longitude_value[2].num) / float(gps_longitude_value[2].den)) / 3600.0)
            gps_altitude = eval(str(gps_altitude_value[0]).split('/')[0]) / eval(str(gps_altitude_value[0]).split('/')[1])
            
            roll_value = eval(all_info_xmp[roll_tag])
            pitch_value = eval(all_info_xmp[pitch_tag])
            raw_value = eval(all_info_xmp[yaw_tag])
            create_time = exif_data["EXIF DateTimeOriginal"].values
            # 根据GPS纬度和经度的方向值，判断正负号
            if gps_latitude_ref_value != "N":
                gps_latitude = -gps_latitude
            if gps_longitude_ref_value != "E":
                gps_longitude = -gps_longitude
            # 返回这些值
            return roll_value, pitch_value, raw_value, gps_latitude, gps_longitude, gps_altitude, create_time
        else:
            # 如果不存在这些标签，返回None
            return None


def read_exif_data(folder_path):
    """
    读取指定文件夹中每个图像文件的EXIF信息, 并返回一个字典, 其中包含每个图像的EXIF数据
    """
    exif_dict = {}

    for filename in os.listdir(folder_path):
        # if filename == "DJI_20231018101819_0149_D.JPG":
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):

            # 打开图像文件并获取exif信息
            image_path = os.path.join(folder_path, filename)
            result = get_dji_exif(image_path)     

            # 保存exif信息到字典中
            if result:
                exif_dict[filename] = result

    return exif_dict

def main(image_path, txt_write_pose):


    exif_dict = read_exif_data(image_path)
    with open(txt_write_pose, "w") as fd:
        for item in exif_dict.keys():
            roll, pitch, yaw, gps_lat, gps_lon, gps_alt, creat_time = exif_dict[item]
            gps = [gps_lon, gps_lat, gps_alt]

            crs_CGCS2000 = CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')  # degree
            crs_WGS84 = CRS.from_epsg(4326)

            from_crs = crs_WGS84
            to_cgcs = crs_CGCS2000
            transformer = Transformer.from_crs(from_crs, to_cgcs, always_xy=True)
            new_x, new_y, _ = transformer.transform(gps_lon, gps_lat, gps_alt)

            # 欧拉角转四元数
            euler = [yaw,pitch,roll]
            ret = R.from_euler('zxy',[float(euler[0]), 90-float(euler[1]), float(euler[2])],degrees=True)
            R_matrix = ret.as_matrix()
            qw, qx, qy, qz = rotmat2qvec(R_matrix)

            # c2w
            q = [qx, qy, qz, qw]
            R1 = np.asmatrix(qvec2rotmat(q))
            

            T = np.identity(4)
            T[0:3, 0:3] = R1
            T[0:3, 3] = -R1.dot(np.array([new_x, new_y, gps_alt]))

            out_line_str = 'query/' + item + ' ' + str(qw) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(T[0:3, 3][0]) + ' ' + str(T[0:3, 3][1]) + ' ' + str(T[0:3, 3][2]) + '\n'
            fd.write(out_line_str)
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="write EXIF information (name qw qx qy qz x y z) in txt file")
    parser.add_argument("--input_EXIF_photo", default="./images/")
    parser.add_argument("--txt_pose", default="./pose.txt")

    args = parser.parse_args()
    main(args.input_EXIF_photo, args.txt_pose)
