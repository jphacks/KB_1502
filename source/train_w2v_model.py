#coding:utf-8

import cPickle
from gensim.models import Word2Vec,Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from preprocessing import *


print "transforming rawtext -> sentences"
rawtext2sentences()
divide_sentence_and_label()
print "transforming sentences -> divided_sentences(wakachigaki)"
sentences2divsentences("labeled_sentences")
sentences2divsentences("unlabeled_sentences")

load_dir_path = "../dataset/sentences_divided/"

sentences = cPickle.load(open(load_dir_path + "unlabeled_sentences.pkl","rb"))

#word2vec学習
print "training word2vec model"
savefilename = "w2v_model"
#size:隠れそうのニューロンの数
#min_count:学習に使う単語の頻度のスレショルド
#worker:並列ワーカー数
w2v_model = Word2Vec(min_count=20,size=300,workers=4,window=8)
w2v_model.build_vocab(sentences)
for epoch in xrange(50):
    print epoch
    w2v_model.train(sentences)

w2v_model.save("../dataset/" + savefilename )


"""
print "training doc2vec model"
savefilename = "d2v_model"

tagged_docs = []
for i, sentence in enumerate(sentences):
    tagged_docs.append(TaggedDocument(words=sentence,tags="sentence"+str(i)))

d2v_model = Doc2Vec(size=300,min_alpha=0.025,window=7)
d2v_model.build_vocab(tagged_docs)
for epoch in xrange(10):
    print epoch
    d2v_model.train(tagged_docs)

d2v_model.save("../dataset/" + savefilename )
"""
