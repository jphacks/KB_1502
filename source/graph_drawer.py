import numpy
import matplotlib.pyplot as plt
from fig2img import fig2img,fig2data
from PIL import Image
import threading

import matplotlib 
matplotlib.use('TkAgg')

draw_freq = 0.5

sp_color = (1.0,0.0,1.0)

colors = ( (1.0,0.0,0.0),
           (0.0,1.0,0.0),
           (0.0,0.0,1.0),
           (0.0,1.0,1.0),
           (1.0,1.0,0.0))


class GraphDrawer(object):

    def __init__(self,xlabel="",ylabel="",title="",figsize=(4,4),scale=80):
        self.perfume = 0
        self.graph_image = None
        self.figsize = figsize
        self.boxsize = (figsize[0]*scale,figsize[1]*scale)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.fontsize = 0.2*scale*figsize[0]/4
        self.plotable = True

    def get_color(self,n):
        if (n < 0):
            return sp_color
        else:
            return colors[n%len(colors)]


    def paste_graph_image(self,image_data,pos):
        print self.graph_image
        box = (pos[0],pos[1],pos[0]+self.boxsize[0],pos[1]+self.boxsize[1])
        image = Image.fromarray(image_data)
        image.paste(self.graph_image,box)
        return numpy.asarray(image)

    def enable_plot(self):
        self.plotable = True

    def update_plot1d(self,x,y,ylim=None,color_num=0):
        color = self.get_color(color_num)
        if(not self.plotable):
            return

        fig = plt.figure(figsize=self.figsize)
        plt.plot(x,y,c=color)

        if ylim is not None:
            print ylim
            plt.ylim(ylim)

        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        #plt.yticks((ylim[0],ylim[1]),('min','max'))
        plt.xlabel(self.xlabel,fontsize=self.fontsize)
        plt.ylabel(self.ylabel,fontsize=self.fontsize)
        plt.title(self.title,fontsize=self.fontsize)
        plt.grid(10)
        data = fig2data(fig)
        self.graph_image = Image.fromarray(data)
        self.plotable=False
        threading.Timer(draw_freq,self.enable_plot).start()





