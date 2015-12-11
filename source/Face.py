# -*- coding: UTF-8 -*-

from collections import deque
from omoroi_data import OmoroiData
from graph_drawer import Graph
import numpy as np
from PIL import Image
import cv2

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
    max_image_number = 10
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


class MouthImageArray(object):
    """
    口元の領域（決め打ち）の画像を連続して保存する。
    連続する２つの画像をそれぞれ比較して、変化量の合計が一番大きい人物が話しているとする。
    """
    # 保存する画像の最大数。多すぎると話し終えたのに、まだ話していることになる。調整してね。
    max_image_number = 7

    def __init__(self):
        self.images = deque(maxlen=self.max_image_number)
        # images[i]とimages[i+1]の差をdifference_array[i]に入れる
        self.difference_array = deque(maxlen=self.max_image_number - 1)

    def _fit_image_size(self, base_image, image):
        """imageをbase_imageの大きさに縮尺を変更する"""
        new_image = image.resize(base_image.size)
        return new_image

    def _compute_difference(self, image1, image2):
        """
        ２つの画像の差を計算する

        画像の大きさに変化量が依存するのはおかしいと思うので、
        依存しない平均事情誤差で測る
        """
        tmp1 = Image.fromarray(np.uint8(image1))
        tmp2 = Image.fromarray(np.uint8(image2))
        new_image2 = np.asarray(self._fit_image_size(tmp1, tmp2))

        return np.mean((image1 - new_image2) ** 2)

    def compute_variability(self):
        return np.sum(self.difference_array)

    def add_mouth_image(self, image):
        if len(self.images) == self.max_image_number:
            # 上限まで保存しているので、一番古い画像を捨てる
            self.images.popleft()
            self.difference_array.popleft()

        # どちらかと言うと形の変化が重要なので、たぶん色の細かな変化は余計である
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 追加する
        self.images.append(gray_image)

        n = len(self.images)
        if n == 1:
            # 一枚しかないので、変化を計算しようがない
            return
        # 変化を計算
        self.difference_array.append(self._compute_difference(self.images[n - 2], self.images[n - 1]))

    def clear_mouth_images(self):
        self.images.clear()
        self.difference_array.clear()


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
        self.mouth_images = MouthImageArray()

    def update(self):
        self.omoroi_data.update_omoroi_sequence(self.is_smiling)
        length = 20
        omoroi_subsequence = self.omoroi_data.get_subsequence(self.omoroi_data.omoroi_sequence,length)
        pos = (self.geoinfo.coordinates[0][0]+self.geoinfo.length[0],
               self.geoinfo.coordinates[0][1]-self.geoinfo.length[1]/2)
        self.graph.set_graph_data(np.arange(len(omoroi_subsequence)),
                                  omoroi_subsequence,
                                  pos = pos)
