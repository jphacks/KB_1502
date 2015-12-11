#coding:utf-8

import numpy
import re
import os
import sys

file_path = "../dataset/nuc/"
LABELED_DATA_NUM = 1000

if os.path.exists("../dataset/sentiment/unlabeled_text.txt"):
    print "上書きしちゃうので終了"
    sys.exit()

# sentenceの辞書を作る
sentences = []
sentence_num = 0
for doc_index in xrange(1, 101):
    text_file = open(file_path + "data%03d.txt" % doc_index, "r")
    inline = text_file.readline()

    while inline:
        if not re.search(r"＠", inline):
            sentence = inline.split("：")[-1]
            sentences.append(sentence)
            sentence_num += 1
        inline = text_file.readline()

print sentence_num


index_list = [i for i in xrange(sentence_num)]
random_index_list = numpy.random.permutation(index_list)

labeled_text = open("../dataset/sentiment/labeled_text.txt", "w")
unlabeled_text = open("../dataset/sentiment/unlabeled_text.txt", "w")
for i, random_index in enumerate(random_index_list):
    sentence = sentences[random_index]
    if i < LABELED_DATA_NUM:
        print ""
        print sentence
        print "保存しますか？(保存する:1, 0.5で保存する:0, 保存しない:Enter):"
        save_flag = raw_input()

        # ラベルあり
        if save_flag == "1":
            print "ラベルを入力してください:"
            label = raw_input()

            labeled_text.writelines("%s:" % label + sentence)
            print "保存しました."

        # ラベル無し
        elif save_flag == "0":
            label = 0.5
            labeled_text.writelines("%s:" % label + sentence)
            print "ニュートラルで保存しました."
    else:
        unlabeled_text.writelines(sentence)

labeled_text.close()
unlabeled_text.close()
