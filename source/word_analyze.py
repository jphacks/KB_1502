# coding: utf-8
import json
import urllib2

class WordAnalyze(object):
    def __init__(self):
        pass

    def morphological_analysis(self, sentence):
        data = {"app_id":"462c21263bd43b4489f562f136060ccee6a7cfaf9671c8cdcf0cb988198b4981", "sentence":"日本語を分析します"}

        data = json.dumps(data)
        url = 'https://labs.goo.ne.jp/api/morph'
        req = urllib2.Request(url, data, {'Content-Type': 'application/json'})
        f = urllib2.urlopen(req)
        response = f.read()

        return response
