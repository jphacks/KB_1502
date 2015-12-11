
import numpy
from gensim.models import Word2Vec,Doc2Vec
from gensim.models.doc2vec import TaggedDocument


model_filename = "../dataset/w2v_model"

class Sen2VecByWord2Vec(object):

    def __init__(self):
        self.w2v_model = Word2Vec.load(model_filename)

    def _w2v(self,word):
        try:
            return self.w2v_model[word]
        except:
            return None

    def sen2vec(self,sentence):
        distributed_words = []
        for word in sentence:
            ret = self._w2v(word)
            if ret is not None:
                distributed_words.append(ret)

        if (len(distributed_words) ):
            return numpy.array(distributed_words).astype("float32").mean(axis=0)
        else:
            return None

    def sens2vec(self,sentences):
        sens = []
        for sentence in sentences:
            sens.append(self.sen2vec(sentence))

        return sens

class Sen2VecByDoc2Vec(object):

    def __init__(self):
        self.d2v_model = Doc2Vec.load(model_filename)

    def sen2vec(self,sentence):
        distributed_words = []
        for word in sentence:
            ret = self._w2v(word)
            if ret is not None:
                distributed_words.append(ret)

        if (len(distributed_words) ):
            return numpy.array(distributed_words).astype("float32").mean(axis=0)
        else:
            return None

    def sens2vec(self,sentences):
        sens = []
        for sentence in sentences:
            sens.append(self.sen2vec(sentence))

        return sens