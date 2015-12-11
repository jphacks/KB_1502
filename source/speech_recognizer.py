#! /usr/bin/python
# -*- coding: UTF-8 -*-

import threading
import socket
from execute_background import ExecuteBackground
import time
import os
import time

import threading


local_source_dir = os.getcwd()

dictation_path = local_source_dir + "/../local/dictation-kit-v4.3.1-osx/"

# You need to download dicatation-kit for julius from http://julius.osdn.jp/index.php?q=dictation-kit.html

# specify the path of dictation-kit
# dictatation_path = os.environ[JDICT_PATH]

# specify host,port and bufsize to communicate with julius
host = 'localhost'
port = 10500
bufsize = 2048
reset_sec = 1.0

class SpeechRecognizer(object):

    def __init__(self):
        self.speech = ""
        self.stop_event = threading.Event()
        self.recogflg = False

    def get_speech(self):
        """
        認識した言葉をgetするメソッド
        utf-8形式からunicodeへ変換
        """
        return unicode(self.speech,'utf-8')


    def reset_speech(self):
        self.speech = ""

    def start(self):
        """
        音声認識をスタートするメソッド
        juliusとjuliusの認識した言葉をparseするスレッドを作成
        """

        # define the bash command for launching julius
        dic = { 'id'  : 1,
         'cmd' : [dictation_path + 'bin/julius','-C',dictation_path+'main.jconf','-C',dictation_path+'am-gmm.jconf','-module'],
         'cwd' : '/', }
        self.thread_julius = ExecuteBackground(**dic)

        # launch julius
        print("Launching Julius...")
        self.thread_julius.start()
        time.sleep(3.)

        self.thread_parser = threading.Thread(target=self.parse)
        # run the parser for the message from Julius
        self.thread_parser.start()


    def parse(self):
        """
        juliusがsendしてくるxmlを処理するメソッド
        self.speechに認識した言葉を代入する
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host,port))
        recogouts=''
        while not self.stop_event.is_set():
            recv_data = sock.recv(bufsize)
            if('<RECOGOUT>' in recv_data):
                self.recogflg = True
                recogouts=''
            if(self.recogflg):
                recogouts += recv_data

            if('</RECOGOUT>' in recv_data):
                ret = ''
                for line in recogouts.split('\n'):
                    index = line.find('WORD=')
                    if(index>-1):
                        ret += line[index+len('WORD=')+1:].split('"')[0]

                self.speech = ret
                print("Recognition results:" + ret)
                threading.Timer(reset_sec,self.reset_speech).start()

                self.recogflg = False


    def stop(self):
        """
        音声認識の終了
        TO DO: juliusの終了処理
        """
        self.stop_event.set()
        self.thread_parser.join()

