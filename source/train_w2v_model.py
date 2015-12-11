#coding:utf-8

import cPickle
from gensim.models import Word2Vec
from preprocessing import *

print "transforming rawtext -> sentences"
rawtext2sentences()
print "transforming sentences -> divided_sentences(wakachigaki)"
sentences2divsentences("unlabeled_sentences")
load_dir_path = "../dataset/sentences_divided/"

sentences = cPickle.load(open(load_dir_path + "unlabeled_sentences.pkl","rb"))

savefilename = "w2v_model"

#size:隠れそうのニューロンの数
#min_count:学習に使う単語の頻度のスレショルド
#worker:並列ワーカー数

print "training word2vec model"
w2v_model = Word2Vec(sentences,min_count=20,size=400,workers=4)

w2v_model.save("../dataset/" + savefilename )




