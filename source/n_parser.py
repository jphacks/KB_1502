#coding:utf-8

import numpy
import re
import os
import errno
import sys
import MeCab
import cPickle


def rawtext2sentences():
    def make_sure_path_exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    load_dir_path = "../training_dataset/nuc/"
    save_dir_path = "../training_dataset/sentences/"

    make_sure_path_exists(save_dir_path)

    for doc_index in xrange(1, 130):
        text_file = open(load_dir_path + "data%03d.txt" % doc_index, "r")
        text = open(save_dir_path + "sentences" + str(doc_index) + ".txt", "wr")

        inline = text_file.readline()
        while inline:
            if not re.search(r"＠", inline):

                # 会話文の取得
                sentence = inline.split("：")[-1]
                text.writelines(sentence)

            inline = text_file.readline()

        text.close()

def sentences2divsentences():
    def make_sure_path_exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                    raise

    load_dir_path = "../training_dataset/sentences/"
    save_dir_path = "../training_dataset/sentences_divided/"

    make_sure_path_exists(save_dir_path)

    t = MeCab.Tagger("-Owakati")

    for doc_index in xrange(1, 130):
        sentences = open(load_dir_path + "sentences%d.txt" % doc_index, "r")
        line = sentences.readline()
        sentences_divided = []
        while line:
            print line
            sentence_divided = t.parse(line).split(" ")
            #後ろの3文字[。,\r,\n]は除去
            sentence_divided = sentence_divided[:-3]
            sentences_divided.append(sentence_divided)
            line = sentences.readline()
        cPickle.dump(sentences_divided, open(save_dir_path + "sentences_divided%d.pkl" % doc_index,"wb"))
        sentences.close()


