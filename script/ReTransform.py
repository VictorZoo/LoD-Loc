import os
import jpype
import jpype.imports
from jpype.types import *
import sys
import numpy as np
import pyproj
import sys
import os



class ReframeTransform:
    def __init__(self):
        if not jpype.isJVMStarted():
            jpype.startJVM('-ea', classpath=os.path.join(sys.path[0], 'reframeLib.jar'), convertStrings=False)
        from com.swisstopo.geodesy.reframe_lib import Reframe
        from com.swisstopo.geodesy.reframe_lib.IReframe import AltimetricFrame, PlanimetricFrame, ProjectionChange
        self.reframeObj = Reframe()
        self.proj = ProjectionChange
        self.default_v_srs = {"lv03": "ln02",
                              "lv95": "lhn95",
                              "wgs84": "wgs84"
                              }
        self.vframes = {
            "ln02": AltimetricFrame.LN02,
            "lhn95": AltimetricFrame.LHN95,
            "bessel": AltimetricFrame.Ellipsoid,
            "wgs84": AltimetricFrame.Ellipsoid  # we have bessel altitude before/after transformation from/to wgs84
        }
        self.hframes = {
            "lv03": PlanimetricFrame.LV03_Military,
            "lv95": PlanimetricFrame.LV95,
            "wgs84": PlanimetricFrame.LV95  # we have LV95 coordinates before/after transformation from/to wgs84
        }
        self.default_raster_transform_res = 10
    def check_transform_args(self, s_h_srs, s_v_srs, t_h_srs, t_v_srs):
        if s_h_srs is None:
            raise Exception(
                "Source horizontal srs must be provided using -s_h_srs option [script] or in the function call")
        if s_v_srs is None:
            s_v_srs = self.default_v_srs[s_h_srs]
        if t_h_srs is None:
            raise Exception(
                "Target horizontal srs must be provided using -t_h_srs option [script] or in the function call")
        if t_v_srs is None:
            t_v_srs = self.default_v_srs[t_h_srs]
        if s_h_srs == t_h_srs and s_v_srs == t_v_srs:
            raise Exception("Source and target srs are exactly the same, doing nothing")
        if (s_h_srs != 'wgs84' and s_v_srs == 'wgs84') or (t_h_srs != 'wgs84' and t_v_srs == 'wgs84'):
            print("Not recommended to use wgs84 height with swiss system planimetry")

        return s_h_srs, s_v_srs, t_h_srs, t_v_srs

    # Note: wgs84 always in lon, lat format (not lat, lon!)
    def transform(self, coord: list, s_h_srs: str, s_v_srs: str, t_h_srs: str, t_v_srs: str):
        if type(coord) is np.ndarray and coord.ndim > 1:
            for j, elem in enumerate(coord):
                coord[j, :] = self.transform(elem, s_h_srs, s_v_srs, t_h_srs, t_v_srs)
            return coord
        if s_h_srs == 'wgs84':
            if s_v_srs == 'wgs84':
                if t_h_srs == 'wgs84':
                    if t_v_srs != 'wgs84':
                        lv95bessel = self.reframeObj.ComputeGpsref(coord, self.proj.ETRF93GeographicToLV95)
                        if t_v_srs == 'bessel':
                            coord[2] = lv95bessel[2]
                        else:
                            coord[2] = self.reframeObj.ComputeReframe(lv95bessel, self.hframes['lv95'],
                                                                      self.hframes['lv95'], self.vframes['bessel'],
                                                                      self.vframes[t_v_srs])[2]
                else:
                    if t_v_srs == 'wgs84':
                        lv95bessel = self.reframeObj.ComputeGpsref(coord, self.proj.ETRF93GeographicToLV95)
                        if t_h_srs == 'lv95':
                            coord[0:2] = lv95bessel[0:2]
                        else:
                            coord[0:2] = self.reframeObj.ComputeReframe(lv95bessel, self.hframes['lv95'],
                                                                        self.hframes[t_h_srs], self.vframes['bessel'],
                                                                        self.vframes['bessel'])[0:2]
                    else:
                        lv95bessel = self.reframeObj.ComputeGpsref(coord, self.proj.ETRF93GeographicToLV95)
                        if t_h_srs == 'lv95' and t_v_srs == 'bessel':
                            coord = lv95bessel
                        else:
                            coord = self.reframeObj.ComputeReframe(lv95bessel, self.hframes['lv95'],
                                                                   self.hframes[t_h_srs], self.vframes['bessel'],
                                                                   self.vframes[t_v_srs])

            else:  # degenerated --> approximations
                if t_h_srs == 'wgs84':
                    if t_v_srs == 'wgs84':
                        coord[2] = self.reframeObj.ComputeGpsref(coord, self.proj.LV95ToETRF93Geographic)[2]
                    else:
                        if s_v_srs != t_v_srs:
                            xyLV95 = self.reframeObj.ComputeGpsref(coord, self.proj.ETRF93GeographicToLV95)  # approx
                            xyLV95 = self.reframeObj.ComputeReframe([xyLV95[0], xyLV95[1], coord[2]],
                                                                    self.hframes['lv95'], self.hframes['lv95'],
                                                                    self.vframes[s_v_srs], self.vframes[t_v_srs])
                            coord[2] = xyLV95[2]
                else:
                    if t_v_srs == 'wgs84':
                        # source vertical is not wgs, destination horizontal is not wgs
                        xyLV95 = self.reframeObj.ComputeGpsref(coord, self.proj.ETRF93GeographicToLV95)  # approx
                        if s_v_srs == 'bessel':
                            xyLV95 = [xyLV95[0], xyLV95[1], coord[2]]
                        else:
                            xyLV95 = self.reframeObj.ComputeReframe([xyLV95[0], xyLV95[1], coord[2]],
                                                                    self.hframes['lv95'], self.hframes['lv95'],
                                                                    self.vframes[s_v_srs], self.vframes['bessel'])
                        coord[2] = self.reframeObj.ComputeGpsref(xyLV95, self.proj.LV95ToETRF93Geographic)[2]
                        if t_h_srs != 'lv95':
                            coord[0:2] = self.reframeObj.ComputeReframe(xyLV95, self.hframes['lv95'],
                                                                        self.hframes[t_h_srs], self.vframes['bessel'],
                                                                        self.vframes['bessel'])[0:2]
                    else:
                        xyLV95 = self.reframeObj.ComputeGpsref(coord, self.proj.ETRF93GeographicToLV95)  # approx
                        if t_h_srs == 'lv95' and t_v_srs == s_v_srs:
                            coord = [xyLV95[0], xyLV95[1], coord[2]]
                        else:
                            coord = self.reframeObj.ComputeReframe([xyLV95[0], xyLV95[1], coord[2]],
                                                                   self.hframes['lv95'], self.hframes[t_h_srs],
                                                                   self.vframes[s_v_srs], self.vframes[t_v_srs])
        else:  # degenerated
            if s_v_srs == 'wgs84':
                if t_h_srs == 'wgs84':
                    if t_v_srs == 'wgs84':
                        if s_h_srs != 'lv95':
                            coord = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs], self.hframes['lv95'],
                                                                   self.vframes['bessel'], self.vframes['bessel'])
                        coord[0:2] = self.reframeObj.ComputeGpsref(coord, self.proj.LV95ToETRF93Geographic)[
                                     0:2]  # approx
                    else:
                        if s_h_srs != 'lv95':
                            coord = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs], self.hframes['lv95'],
                                                                   self.vframes['bessel'], self.vframes['bessel'])
                        wgs84 = self.reframeObj.ComputeGpsref(coord, self.proj.LV95ToETRF93Geographic)[0:2]  # approx
                        lv95bessel = self.reframeObj.ComputeGpsref([wgs84[0], wgs84[1], coord[2]],
                                                                   self.proj.ETRF93GeographicToLV95)
                        if t_v_srs == 'bessel':
                            h = lv95bessel[2]
                        else:
                            h = self.reframeObj.ComputeReframe(lv95bessel, self.hframes['lv95'], self.hframes['lv95'],
                                                               self.vframes['bessel'], self.vframes[t_v_srs])[2]
                        coord = [wgs84[0], wgs84[1], h]
                else:
                    if t_v_srs == 'wgs84':
                        # horizontal: swiss to swiss, vertical: wgs to wgs
                        coord[0:2] = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs], self.hframes[t_h_srs],
                                                                    self.vframes['lhn95'], self.vframes['lhn95'])[0:2]
                    else:
                        # horizontal: swiss to swiss, vertical: wgs to swiss
                        if s_h_srs != 'lv95':
                            lv95 = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs], self.hframes['lv95'],
                                                                  self.vframes['bessel'], self.vframes['bessel'])
                        else:
                            lv95 = coord
                        wgs84 = self.reframeObj.ComputeGpsref(lv95, self.proj.LV95ToETRF93Geographic)[0:2]  # approx
                        h = self.reframeObj.ComputeGpsref([wgs84[0], wgs84[1], coord[2]],
                                                          self.proj.ETRF93GeographicToLV95)[2]
                        if t_v_srs == 'bessel' and t_h_srs == s_h_srs:
                            coord[2] = h
                        elif t_v_srs == 'bessel' and t_h_srs == 'lv95':
                            coord = [lv95[0], lv95[1], h]
                        else:
                            coord = self.reframeObj.ComputeReframe([coord[0], coord[1], h], self.hframes[s_h_srs],
                                                                   self.hframes[t_h_srs], self.vframes['bessel'],
                                                                   self.vframes[t_v_srs])
            else:  # source = swiss system
                if t_h_srs == 'wgs84':
                    if t_v_srs == 'wgs84':
                        if s_h_srs == 'lv95' and s_v_srs == 'bessel':
                            lv95bessel = coord
                        else:
                            lv95bessel = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs],
                                                                        self.hframes['lv95'], self.vframes[s_v_srs],
                                                                        self.vframes['bessel'])
                        coord = self.reframeObj.ComputeGpsref(lv95bessel, self.proj.LV95ToETRF93Geographic)
                    else:
                        if s_h_srs == 'lv95' and s_v_srs == 'bessel':
                            lv95bessel = coord
                        else:
                            lv95bessel = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs],
                                                                        self.hframes['lv95'], self.vframes[s_v_srs],
                                                                        self.vframes['bessel'])
                        if t_v_srs == 'bessel':
                            coord[2] = lv95bessel[2]
                        elif t_v_srs != s_v_srs:
                            coord[2] = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs],
                                                                      self.hframes[s_h_srs],
                                                                      self.vframes[s_v_srs], self.vframes[t_v_srs])[2]
                        coord[0:2] = self.reframeObj.ComputeGpsref(lv95bessel, self.proj.LV95ToETRF93Geographic)[0:2]
                else:
                    if t_v_srs == 'wgs84':
                        if s_h_srs == 'lv95' and s_v_srs == 'bessel':
                            lv95bessel = coord
                        else:
                            lv95bessel = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs],
                                                                        self.hframes['lv95'], self.vframes[s_v_srs],
                                                                        self.vframes['bessel'])
                        if t_h_srs == 'lv95':
                            coord[0:2] = lv95bessel[0:2]
                        elif t_h_srs != s_h_srs:
                            coord[0:2] = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs],
                                                                        self.hframes[t_h_srs], self.vframes['lhn95'],
                                                                        self.vframes['lhn95'])[0:2]
                        coord[2] = self.reframeObj.ComputeGpsref(lv95bessel, self.proj.LV95ToETRF93Geographic)[2]
                    else:
                        if t_h_srs != s_h_srs or t_v_srs != s_v_srs:
                            coord = self.reframeObj.ComputeReframe(coord, self.hframes[s_h_srs], self.hframes[t_h_srs],
                                                                   self.vframes[s_v_srs], self.vframes[t_v_srs])
        return coord

    def transform_points(self, coors, s_h_srs, s_v_srs, t_h_srs, t_v_srs):
        s_h_srs, s_v_srs, t_h_srs, t_v_srs = self.check_transform_args(s_h_srs, s_v_srs, t_h_srs, t_v_srs)
        resolution = 8 if t_h_srs == 'wgs84' else 2

        coordinates = self.transform(coors, s_h_srs, s_v_srs, t_h_srs, t_v_srs)

        print(coordinates)
        return  coordinates

def get_ecef_origin():
    """Shift the origin to make the value of coordinates in ECEF smaller and increase training stability"""
    # Warning: this is dataset specific!
    ori_lon, ori_lat, ori_alt = 6.5668, 46.5191, 390
    ori_x, ori_y, ori_z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(ori_lat, ori_lon, ori_alt)
    print('Origin XYZ: {}, {}, {}'.format(ori_x, ori_y, ori_z))
    origin = np.array([ori_x, ori_y, ori_z], dtype=np.float64)
    return origin




