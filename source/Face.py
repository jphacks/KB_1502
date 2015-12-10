# -*- coding: UTF-8 -*-

from omoroi_data import OmoroiData
from graph_drawer import Graph
import numpy as np

class GeoInfo(object):

    def _center(self):
        return tuple([self.rect[0] + self.rect[2] / 2.0, self.rect[1] + self.rect[3] / 2.0])
    def _coordinates(self):
        return tuple([tuple(self.rect[0:2]),tuple(self.rect[0:2]+self.rect[2:4])])
    def _length(self):
        return tuple(self.rect[2:4])

    def __init__(self,rect):
        self.rect = rect
        self.coordinates = self._coordinates()
        self.length = self._length()
        self.center = self._center()


class FaceImageArray(object):
    """
    顔が一旦消えて、また現れた時に、以前の履歴を引き継ぐために顔画像を保存する。
    再び現れた時に照合して、同一人物であるかを判定する。
    """

    # 保存する顔画像の最大枚数。多分そこまで、測度に影響はないと思う。多いほうが精度は高いと思う。
    max_image_number = 20
    # この枚数より保存数が小さい時は、カメラの前にいる時間が短すぎるので、履歴の対象から除外する
    min_image_number = 3

    def __init__(self):
        self.images = []

    def add_face_image(self, face_image):
        """
        顔画像を保存する。保存している画像の枚数が最大枚数に達している場合は、何もしない
        """
        if len(self.images) < self.max_image_number:
            self.images.append(face_image)

    def clear_face_images(self):
        self.images = []

    def is_enough_images(self):
        return len(self.images) >= self.min_image_number

class Face(object):

    def __init__(self,geoinfo,speech=""):
        self.geoinfo = geoinfo
        self.is_smiling = False
        self.speech = speech
        self.smile_sequence = []
        self.omoroi_data = OmoroiData()
        self.graph = Graph(
            ylim=[self.omoroi_data.omoroi_min-1.0,self.omoroi_data.omoroi_max+1.0],
            ylabel="Omorosa",scale=80,figsize=(2,2)
        )


        self.face_images = FaceImageArray()

    def update(self):
        self.omoroi_data.update_omoroi_sequence(self.is_smiling,0)
        length = 20
        omoroi_subsequence = self.omoroi_data.get_subsequence(self.omoroi_data.omoroi_sequence,length)
        pos = (self.geoinfo.coordinates[0][0]+self.geoinfo.length[0],
               self.geoinfo.coordinates[0][1]-self.geoinfo.length[1]/2)
        self.graph.set_graph_data(np.arange(len(omoroi_subsequence)),
                                  omoroi_subsequence,
                                  pos = pos)
