from omoroi_data import OmoroiData
from graph_drawer import GraphDrawer
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
        self.graph_drawer = GraphDrawer(ylabel="Omorosa",scale=80,figsize=(2,2))
        self.omoroi_data = OmoroiData()


    def update(self,image_data,color_num):
        self.omoroi_data.update_omoroi_sequence(self.is_smiling,0)
        length = 20
        omoroi_subsequence = self.omoroi_data.get_subsequence(self.omoroi_data.omoroi_sequence,length)
        self.graph_drawer.update_plot1d(
            np.arange(len(omoroi_subsequence)),
            omoroi_subsequence,
            ylim=[self.omoroi_data.omoroi_min-1.0,self.omoroi_data.omoroi_max+1.0],
            color_num=color_num)
        pos = (self.geoinfo.coordinates[0][0]+self.geoinfo.length[0],
               self.geoinfo.coordinates[0][1]-self.geoinfo.length[1]/2)
        return self.graph_drawer.paste_graph_image(image_data=image_data,pos=pos)