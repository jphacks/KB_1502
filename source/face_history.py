# -*- coding: UTF-8 -*-
__author__ = 'TakeruMiyato'

"""
OpenCV を使用して、写真に特定の人物が写っているか判定する[http://symfoware.blog68.fc2.com/blog-entry-1525.html]を参考に
1.画面に現れた人物の顔画像を幾つか保存して、
2.特徴量（キーポイント）を計算し保存、
3.候補の顔画像と比較することで、
同一人物であるかを判定する

1.で保存する画像は１枚でもいいかもしれない。3.で一致度が得られるので、どのように判定するかに依存する。
"""

import sys
import numpy as np
import cv2

# OpenCV 2.4.9 使用できるDetector,Extractor,Matcher[http://symfoware.blog68.fc2.com/blog-entry-1523.html]
# 検出器のアルゴリズム
detectors = ['FAST','ORB','BRISK','MSER','GFTT','HARRIS','Dense', 'SIFT', 'SURF']
# 抽出器のアルゴリズム
extractors = ['ORB','BRISK','BRIEF','FREAK', 'SIFT', 'SURF']
# 特徴量を比較するアルゴリズム
matchers = ['BruteForce','BruteForce-L1','BruteForce-SL2','BruteForce-Hamming','BruteForce-Hamming(2)']


class FeatureSettings(object):
    """detector,extractor,matcherを保存するクラス

    画像に対して処理するたびにdetector,extractor,matcherを作成するのは良くないと思うので、保存する
    """
    def __init__(self, detector_name, extractor_name, matcher_name):
        """引数はそれぞれ上のdetectors, extractors, matchersから一つ選ぶ"""
        self.detector = cv2.FeatureDetector_create(detector_name)
        self.descriptor = cv2.DescriptorExtractor_create(extractor_name)
        self.matcher = cv2.DescriptorMatcher_create(matcher_name)


class FaceFeatures(object):
    """顔の特徴量（キーポイント）のクラス

    カメラで新たに得られた顔画像と比較することで、同一人物であるかを判定する
    """
    def __init__(self, face_image, feature_settings):
        self.features = []

        # 顔画像を登録
        self.register_face_image(face_image, feature_settings)

    def _compute_features(self, face_image, feature_settings):
        """
        face_image: 顔画像（１枚） <- カラー画像でいいのかわからない、白黒画像が必要な場合はこの関数内で変換する
        feature_settings: FeatureSettingsのインスタンス

        特徴量の計算をここで行う。
        """
        gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        # キーポイントを設定
        keypoints = feature_settings.detector.detect(gray_image)
        # キーポイントの特徴量を計算
        keypoint_data, discription_data = feature_settings.descriptor.compute(gray_image, keypoints)

        return keypoint_data, discription_data

    def register_face_image(self, face_image, feature_settings):
        """
        顔画像のテンプレートを登録する。特徴量の計算もここで行う。
        複数枚登録するときは、同じ人物の画像をちゃんと渡してね。トラッキングに依存すると思う。
        """
        # キーポイントの特徴量を計算
        keypoint_data, discription_data = self._compute_features(face_image, feature_settings)

        # マッチングに必要なデータはdiscription_dataだけなんだけど、あったほうが便利だと思うので、保存しておく
        self.features.append((face_image, keypoint_data, discription_data))

    def compare_face_image(self, face_image, feature_settings):
        """
        カメラで新たに得られた画像の人物がここに登録されている人物と同じであるかを判定する
        同一人物であると判定した場合は、True、そうでない場合は、Falseを返す。
        """
        # キーポイントの特徴量を計算
        keypoint_data, discription_data = self._compute_features(face_image, feature_settings)

        # print 'matching data type: %s %s' % (type(discription_data), discription_data.dtype)
        matching_results = [feature_settings.matcher.match(discription, discription_data)
                            for (_, _, discription) in self.features]

        # 同一人物であるかを判定する
        # 判定方法、しきい値はいい感じに改良してね。
        min_mean = sys.float_info.max
        for result in matching_results:
            distances = [m.distance for m in result]
            min_mean = min((sum(distances) / len(distances)), min_mean)
        print 'the distance is %f' % min_mean
        return min_mean < 100


class FaceHistories(object):
    """
    過去にカメラの前に現れていて、一旦消えて、また現れた時に、以前のデータを引き継ぐためのクラス

    FaceFeaturesとFaceのインスタンスの２つのデータのタプルの配列を持つ
    """
    def __init__(self):
        self.histories = []
        self.feature_settings = FeatureSettings('Dense', 'BRISK', 'BruteForce-Hamming')

    def get_history(self, face_image, face):
        """
        履歴を検索して、一致する人物のデータが有れば、それを返す。
        無ければ、引数のfaceを返す。
        """
        print '----- get history -----'
        if len(self.histories) == 0:
            return face

        index = self._search_face_history(face_image)
        print '----- get history: the size of histories: %d' % len(self.histories)
        if index < 0:
            return face
        else:
            stored_face = self.histories[index][1]
            # 顔の座標が移動しているので、書き換える
            stored_face.geoinfo = face.geoinfo

            # 履歴から削除
            del self.histories[index]

            return stored_face

    def _search_face_history(self, face_image):
        """
        顔画像に該当するデータが有るかを調べる。
        あれば、そのインデックス、なければ、-1を返す。
        """
        search_result = [face_features.compare_face_image(face_image, self.feature_settings)
                         for (face_features, _) in self.histories]
        if any(search_result):
            # 複数で該当した場合でも、配列の先頭の履歴を返す
            i = search_result.index(True)
            print '----- found the face: index=%d -----' % i
            return i
        else:
            print '----- not found the face -----'
            return -1

    def set_history(self, face_image, face):
        """
        履歴を保存する
        """
        print '----- set history -----'
        self._register_new_face(face_image, face)

        print '----- set history: the size of histories: %d' % len(self.histories)

    def _register_new_face(self, face_image, face):
        """
        FaceFeaturesとFaceのインスタンスを保存
        """
        print '----- register new face -----'
        self.histories.append((FaceFeatures(face_image, self.feature_settings), face))
