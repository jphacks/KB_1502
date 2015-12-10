#coding:utf-8

import numpy
import re
import os
import sys

file_path = "../training_dataset/nuc/"

if os.path.exists("../training_dataset/sentiment/labeled_text.txt"):
    print "上書きしちゃうので終了"
    sys.exit()

for doc_index in xrange(1, 100):
    text_file = open(file_path + "data%03d.txt" % doc_index, "r")
    inline = text_file.readline()

    if doc_index == 1:
        # トレーニングデータを保持するテキスト
        labeled_text = open("../training_dataset/sentiment/labeled_text.txt", "w")
        unlabeled_text = open("../training_dataset/sentiment/unlabeled_text.txt", "w")
    else:
        unlabeled_text = open("../training_dataset/sentiment/unlabeled_text.txt", "a")

    sentences = []
    sentence_index = 1
    while inline:
        if not re.search(r"＠", inline):

            # 会話文の取得
            sentence = inline.split("：")[-1]
            if doc_index == 1:
                if sentence_index < 101:
                    labeled_text = open("../training_dataset/sentiment/labeled_text.txt", "a")
                    print sentence
                    print "ラベルを入力してください:"
                    label = raw_input()
                    print "保存しますか？(保存する:1, 保存しない:0):"
                    save_flag = raw_input()
                    if save_flag == "1":
                        labeled_text.writelines("%s:" % label + sentence)
                        print "保存しました."
                    labeled_text.close()
                else:
                    break
            else:
                unlabeled_text.writelines(sentence)

            sentences.append(sentence)

            sentence_index += 1
        inline = text_file.readline()

    labeled_text.close()
    unlabeled_text.close()

