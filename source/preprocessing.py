#coding:utf-8

import numpy
import re
import os
import errno
import sys
import MeCab
import cPickle

def make_sure_path_exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

def divide_sentences_and_label():
    load_dir_path = "../dataset/sentiment/"
    save_sentences_dir_path = "../dataset/sentences/"
    save_trdata_dir_path = "../dataset/training_dataset/"
    make_sure_path_exists(save_sentences_dir_path)
    make_sure_path_exists(save_trdata_dir_path)
    labeled_text_file = open(load_dir_path + "labeled_text.txt", "r")
    sentences = open(save_sentences_dir_path + "labeled_sentences.txt", "wr")
    labels=[]

    inline = labeled_text_file.readline()
    while inline:
        label,sentence = inline.split(":")
        labels.append(label)
        sentences.writelines(sentence)
        inline = labeled_text_file.readline()
    cPickle.dump(numpy.array(labels).astype("float32"),open(save_trdata_dir_path + "labels.pkl","wb"))


def rawtext2sentences():
    load_dir_path = "../dataset/nuc/"
    save_dir_path = "../dataset/sentences/"

    make_sure_path_exists(save_dir_path)

    senteneces = open(save_dir_path + "unlabeled_sentences.txt", "wr")
    for doc_index in xrange(1, 130):
        text_file = open(load_dir_path + "data%03d.txt" % doc_index, "r")

        inline = text_file.readline()
        while inline:
            if not re.search(r"＠", inline):

                # 会話文の取得
                sentence = inline.split("：")[-1]
                senteneces.writelines(sentence)

            inline = text_file.readline()

    senteneces.close()

def remove_specific_symbols(sentence_divided):
    i = 0
    while(i < len(sentence_divided)):
        w = sentence_divided[i]
        if w is "<" or w is "(":
            j = i + 1
            while(j < len(sentence_divided)):
                w = sentence_divided[j]
                if w is ">" or w is ")":
                    del sentence_divided[i:j+1]
                else:
                    j += 1
        else:
            i += 1

def sentences2divsentences(filename):
    def make_sure_path_exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                    raise

    load_dir_path = "../dataset/sentences/"
    save_dir_path = "../dataset/sentences_divided/"

    make_sure_path_exists(save_dir_path)

    t = MeCab.Tagger("-Owakati")

    sentences = open(load_dir_path + filename + '.txt', "r")
    line = sentences.readline()
    sentences_divided = []
    while line:
        print line
        sentence_divided = t.parse(line).split(" ")
        #後ろの3文字[。,\r,\n]は除去
        sentence_divided = sentence_divided[:-3]
        remove_specific_symbols(sentence_divided)
        sentences_divided.append(sentence_divided)
        line = sentences.readline()
    cPickle.dump(sentences_divided, open(save_dir_path + filename + '.pkl', "wb"))
    sentences.close()







