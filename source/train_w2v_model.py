#coding:utf-8

import cPickle
from gensim.models import Word2Vec
from n_parser import rawtext2sentences,sentences2divsentences

print "transforming rawtext -> sentences"
rawtext2sentences()
print "transforming sentences -> divided_sentences(wakachigaki)"
sentences2divsentences()

load_dir_path = "../training_dataset/sentences_divided/"
all_sentences = []
for i in xrange(1,130):
    sentences = cPickle.load(open(load_dir_path + "sentences_divided" + str(i) + ".pkl","rb"))
    all_sentences.extend(sentences)

savefilename = "w2v_model"

#size:隠れそうのニューロンの数
#min_count:学習に使う単語の頻度のスレショルド
#worker:並列ワーカー数

print "training word2vec model"
w2v_model = Word2Vec(all_sentences,min_count=10,size=200,workers=4)

w2v_model.save("../training_dataset/" + savefilename )




