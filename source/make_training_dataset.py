

import cPickle
import numpy
from preprocessing import remove_short_sentences
from sen2vec import Sen2VecByWord2Vec,Sen2VecByDoc2Vec

load_dir_path = "../dataset/sentences_divided/"
save_dir_path = "../dataset/training_dataset/"

unlabeled_sentences = cPickle.load(open(load_dir_path + "unlabeled_sentences.pkl","rb"))
labeled_sentences = cPickle.load(open(load_dir_path + "labeled_sentences.pkl","rb"))

s2v = Sen2VecByWord2Vec()

unlabeled_vecs = s2v.sens2vec(unlabeled_sentences)
labeled_vecs = s2v.sens2vec(labeled_sentences)

labels = cPickle.load(open(save_dir_path + "labels.pkl","rb"))

print len(labeled_vecs),len(labels),len(unlabeled_vecs)

index =0
while index < len(unlabeled_vecs) :
    if unlabeled_vecs[index] is None :
        del unlabeled_vecs[index]
    else:
        index += 1

index =0
while index < len(labeled_vecs) :
    if labeled_vecs[index] is None :
        del labeled_vecs[index]
        del labels[index]
    else:
        index += 1

x = numpy.array(labeled_vecs).astype("float32")
y = numpy.array(labels).astype("float32")
ul_x = numpy.array(unlabeled_vecs).astype("float32")
cPickle.dump((x,y), open(save_dir_path + "dataset.pkl","wb"))
cPickle.dump((ul_x,), open(save_dir_path + "ul_dataset.pkl","wb"))




