import numpy
import os
import sys
from sen2vec import Sen2VecByWord2Vec


class SentencePredictor(object):

    def __init__(self):
        self.model_type="nn"
        self.model = self.load_model()
        self.s2v = Sen2VecByWord2Vec()
    def load_model(self):
        print "load_model"
        return None

    def pred(self,sentence):
        """
        :param sentence:　分かち書きされた文章(単語のリスト)
        :return: ポジネガ判定(1:ポジティブ,0:ネガティブ)
        """


