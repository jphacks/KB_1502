import numpy as np
import matplotlib.pyplot as plt
from fig2img import fig2img,fig2data
from PIL import Image
import threading

import matplotlib 
matplotlib.use('TkAgg')


sp_color = (1.0,0.0,1.0)

colors = ( (1.0,0.0,0.0),
           (0.0,1.0,0.0),
           (0.0,0.0,1.0),
           (0.0,1.0,1.0),
           (1.0,1.0,0.0) )


class Graph(object):
    def __init__(self,ylim=(0,10),color=None,xlabel="",ylabel="",title="",figsize=(4,4),scale=80):
        self.x = None
        self.y = None
        self.pos = None
        self.ylim = ylim
        self.color = color if color is not None else colors[np.random.randint(0,4)]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.figsize= figsize
        self.boxsize = (figsize[0]*scale,figsize[1]*scale)
        self.fontsize = 0.2*scale*figsize[0]/4
        self.image = None

    def set_image(self,image):
        self.image = image

    def set_graph_data(self,x,y,pos):
        self.x = x
        self.y = y
        self.pos = pos


class GraphDrawer(object):

    def __init__(self):
        self.graphs = []
        self.stop_event = threading.Event()
        self.thread_plot_routine = threading.Thread(target=self.plot_routine)
        self.thread_plot_routine.start()

    def stop(self):
        self.stop_event.set()
        self.thread_plot_routine.join()

    """
    def get_color(self,n):
        if (n < 0):
            return sp_color
        else:
            return colors[n%len(colors)]
    """

    def init_graphs(self):
        self.graphs=[]

    def append_graphs(self,graph):
        self.graphs.append(graph)

    def draw_graphs(self,org_img_array):
        print "draw_graphs"
        org_img = Image.fromarray(org_img_array)
        for graph in self.graphs:
            if graph.image is not None:
                print "graph"
                box = (graph.pos[0],graph.pos[1],graph.pos[0]+graph.boxsize[0],graph.pos[1]+graph.boxsize[1])
                org_img.paste(graph.image,box)
        return np.asarray(org_img)

    def plot_routine(self):
        while not self.stop_event.is_set():
            self.datas2imgs()

    def _get_graph_image(self,graph):
        fig = plt.figure(figsize=graph.figsize)
        plt.plot(graph.x,graph.y,c=graph.color)
        plt.ylim(graph.ylim)

        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        plt.xlabel(graph.xlabel,fontsize=graph.fontsize)
        plt.ylabel(graph.ylabel,fontsize=graph.fontsize)

        plt.title(graph.title,fontsize=graph.fontsize)
        plt.grid(10)
        data = fig2data(fig)
        return Image.fromarray(data)

    def datas2imgs(self):
        for graph in self.graphs:
            if( graph.x is not None):
                graph.set_image(self._get_graph_image(graph))





