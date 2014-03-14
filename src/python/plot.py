
"""
A library for plotting pretty images of all kinds.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

                                                    
class InteractiveImshow(object):
    """
    A brief extension to matplotlib's imshow that puts a colorbar next to 
    the image that you can click on to scale the maximum numeric value
    displayed.
    
    Based on code from pyana_misc by Ingrid Ofte.
    
    Parameters
    ----------
    inarr : np.ndarray
        The array to imshow()
        
    filename : {str, None}
        The filename to call the file if it is saved. If `None`, disable saving
        ability.
    """
    
    def __init__(self, inarr, filename=None, fig=None, ax=None):
        """
        Parameters
        ----------
        inarr : np.ndarray
            The array to imshow()

        filename : {str, None}
            The filename to call the file if it is saved. If `None`, disable saving
            ability.
            
        fig : pyplot.figure
            A figure object to draw on.
            
        ax : pyplot.axes
            An axes canvas to draw on.
        """
        self.inarr = inarr
        self.filename = filename
        self.fig = fig
        self.ax = ax
        self.cmax = self.inarr.max()
        self.cmin = self.inarr.min()
        self._draw_img()
        

    def _on_keypress(self, event):
        
        if event.key == 's':
            if not self.filename:
                self.filename = raw_input('Saving. Enter filename: ')
            plt.savefig(self.filename)
            logger.info("Saved image: %s" % self.filename)
            
        elif event.key == 'r':
            logger.info("Reset plot")
            colmin, colmax = self.orglims
            plt.clim(colmin, colmax)
            plt.draw()
            

    def _on_click(self, event):
        if event.inaxes:
            lims = self.im.get_clim()
            colmin = lims[0]
            colmax = lims[1]
            rng = colmax - colmin
            value = colmin + event.ydata * rng
            if event.button is 1:
                if value > colmin and value < colmax :
                    colmax = value
            elif event.button is 2:
                colmin, colmax = self.orglims
            elif event.button is 3:
                if value > colmin and value < colmax:
                    colmix = value
            self.im.set_clim(colmin, colmax)
            plt.draw()
            
            
    def _on_scroll(self, event):
        lims = self.im.get_clim()
        speed = 1.1
        
        if event.button == 'up':
            colmax = lims[1] / speed
        elif event.button == 'down':
            colmax = lims[1] * speed
            
        self.im.set_clim(lims[0], colmax)
        plt.draw()
            

    def _draw_img(self):
        
        if not self.fig:
            self.fig = plt.figure()
            
        cid1 = self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)
        cid2 = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        cid3 = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        if not self.ax:
            self.ax = self.fig.add_subplot(111)
        
        self.im = self.ax.imshow(self.inarr, vmax=self.cmax, origin='lower')
        self.colbar = plt.colorbar(self.im, pad=0.01)
        self.orglims = self.im.get_clim()
        

def plot_polar_intensities(shot, output_file=None):
    """
    Plot an intensity map in polar coordinates.

    Parameters
    ----------
    shot : odin.xray.Shot
        A shot to plot.
    output_file : str
        The filename to write. If `None`, will display the image on screen and
        not save.
    """

    pi = shot.polar_grid

    colors = shot.polar_intensities # color by intensity
    ax = plt.subplot(111, polar=True)
    c = plt.scatter(pi[:,1], pi[:,0], c=colors, cmap=cm.hsv)
    c.set_alpha(0.75)

    if output_file:
        plt.savefig(output_file)
        logger.info("Saved: %s" % output_file)
    else:
        plt.show()

    return