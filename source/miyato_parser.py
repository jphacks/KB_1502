#coding:utf-8

import numpy
import re
import os
import sys

file_path = "../training_dataset/nuc/"

for doc_index in xrange(1, 2):
    text_file = open(file_path + "data%03d.txt" % doc_index, "r")
    # トレーニングデータを保持するテキスト
    text = open("../training_dataset/sentences_" + str(doc_index) + ".txt", "wr")

    inline = text_file.readline()
    while inline:
        if not re.search(r"＠", inline):

            # 会話文の取得
            sentence = inline.split("：")[-1]
            text.writelines(sentence)

        inline = text_file.readline()

    text.close()



