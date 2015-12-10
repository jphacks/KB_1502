from omoroi_data import OmoroiData
from graph_drawer import Graph
import numpy as np

class GeoInfo(object):

    def _center(self):
        return tuple([self.rect[0] + self.rect[2] / 2.0, self.rect[1] + self.rect[3] / 2.0])
    def _coordinates(self):
        return tuple([tuple(self.rect[0:2]),tuple(self.rect[0:2]+self.rect[2:4])])
    def _length(self):
        return tuple(self.rect[2:4])

    def __init__(self,rect):
        self.rect = rect
        self.coordinates = self._coordinates()
        self.length = self._length()
        self.center = self._center()

class Face(object):

    def __init__(self,geoinfo,speech=""):
        self.geoinfo = geoinfo
        self.is_smiling = False
        self.speech = speech
        self.smile_sequence = []
        self.omoroi_data = OmoroiData()
        self.graph = Graph(
            ylim=[self.omoroi_data.omoroi_min-1.0,self.omoroi_data.omoroi_max+1.0],
            ylabel="Omorosa",scale=80,figsize=(2,2)
        )



    def update(self):
        self.omoroi_data.update_omoroi_sequence(self.is_smiling,0)
        length = 20
        omoroi_subsequence = self.omoroi_data.get_subsequence(self.omoroi_data.omoroi_sequence,length)
        pos = (self.geoinfo.coordinates[0][0]+self.geoinfo.length[0],
               self.geoinfo.coordinates[0][1]-self.geoinfo.length[1]/2)
        self.graph.set_graph_data(np.arange(len(omoroi_subsequence)),
                                  omoroi_subsequence,
                                  pos = pos)
