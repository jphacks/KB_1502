import numpy
from PIL import Image
# import matplotlib
# matplotlib.use('TKAgg')

def fig2data ( fig ):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring( fig.canvas.tostring_rgb(),dtype=numpy.uint8)
    buf.shape = ( w, h, 3 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):

    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) )
