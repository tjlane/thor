
"""
Functions that are useful in various places, but have no common theme.
"""

import numpy as np

def smooth(x, beta=10.0, window_size=11):
    """
    Apply a Kaiser window smoothing convolution.
    
    Parameters
    ----------
    x : ndarray
        The array to smooth.
    beta : float
        Parameter controlling the strength of the smoothing -- bigger beta 
        results in a smoother function.
    window_size : int
        The size of the Kaiser window to apply, i.e. the number of neighboring
        points used in the smoothing.
    """
    
    # make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # apply the smoothing function
    s = np.r_[x[window_size-1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve( w/w.sum(), s, mode='valid' )
    
    # remove the extra array length convolve adds
    b = (window_size-1) / 2
    smoothed = y[b:len(y)-b]
    
    return smoothed

def arctan3(y, x):
    """ like arctan2, but returns a value in [0,2pi] """
    theta = np.arctan2(y,x)
    theta[theta < 0.0] += 2 * np.pi
    return theta


graphic = """
	                                    .  
	                                  .#   
	           N                     .##   
	          #.                     #  :  
	       .# .                     .# .#  
	       .. |       ..##.         #   #  
	       #  l     .#.    #       #    #  
	             .#         #.    #    #   
	      #    #             #.  #.    #   
	      # .#                ##      #    
	      ##.                #7      :#       ____  _____ _____ _   _ 
	      #                 .# .:   .#       / __ \|  __ \_   _| \ | |
	   .:.  .   .    .      .#..   Z#       | |  | | |  | || | |  \| |
	  .####  . . ...####     #.   #.        | |  | | |  | || | | . ` |
	 # .##   . # . #.#   . ND .##.#         | |__| | |__| || |_| |\  |
	 . .#N   .  .   N.   # D .=#  #          \____/|_____/_____|_| \_|
	#   . ####     .###,  ,      ##        
	#.## .               7#7. # N  #                       Observation
	 .                      ##.. . #       	               Driven
	                          .#   #                       Inference
	                            #7 Z                   of eNsembles
		                             .#        

     ----------------------------------------------------------------------
"""
