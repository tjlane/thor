
"""
Functions that are useful in various places, but have no common theme.
"""

import functools
import sys

from io import StringIO
from pprint import pprint
from argparse import ArgumentParser

import numpy as np


class Capturing(list):
    """
    Context manager
    http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    
    >>> with Capturing() as output:
    >>>    do_something(my_object)
    """
    
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        
        self._stringio = StringIO()
        sys.stdout = self._stringio
        sys.stderr = self._stringio
        
        return self
        
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        

class Parser(ArgumentParser):
    """
    Simple extension of argparse, designed to automatically print stuff
    """
    def parse_args(self):
        print(graphic)
        args = super(Parser, self).parse_args()
        pprint(args.__dict__)
        return args
    
        
def is_iterable(obj):
    """
    Determine if `obj` is iterable. Returns bool.
    """
    try:
        for x in obj:
            pass
        return True
    except:
        return False
    
        
# http://stackoverflow.com/questions/11984684/display-only-one-logging-line
logger_return = '\x1b[80D\x1b[1A\x1b[K'

    
def all_pairs(n):
    """
    Generator that yields all unique pairs formed by the integers {0, ..., n-1}
    """
    for i in range(n):
        for j in range(i+1,n):
            yield (i,j)
    
    
def unique_rows(a):
    """
    For a two-dim array, returns unique rows.
    """
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def make_square_detector(waveLength,detDist,pixSz,detX,detY,a,b):
    """
    Return the qvector object for a square detector perpendicular to the
    forward beam. This object then be passed to e.g. scattering.simulate_shot
    
    Parameters:
    waveLength - float, wavelength of the x-ray beam (angstroms)
    detDist    - float, shortest distance from detector to sample position (meters)
    pixSz      - float, size of pixels, assumed here to be square (meters) 
    detX,detY  - int  , number of pixels in the x,y directions (pixel units)
    a,b        - float, position on detector where forward beam intersects (pixel units)

    Returns:
    qvec       - (N x 3) float array, qx,qy,qz for each pixel on the detector (inverse angstroms) 
                         To assemble the image use
                         >> shot = simulate_shot( traj, numMols, qvec  ). reshape ( (detX, detY ) ) 
    """

    X,Y = np.meshgrid ( np.arange( detX) , np.arange( detY) )
    PR = np.sqrt ( (X-a)**2 + (Y-b)**2 )
    PHI = np.arctan2( Y - b, X - a)
    THETA = 0.5* np.arctan( PR * pixSz / detDist )

    Q  = 4*np.pi*np.sin(THETA)/waveLength
    qx = (Q*np.cos(THETA)*np.cos(PHI) ).flatten()
    qy = (Q*np.cos(THETA)*np.sin(PHI) ).flatten()
    qz = (Q*np.sin(THETA) ).flatten()

    qvec = np.zeros ((qx.shape[0], 3 ) )
    qvec[:,0] = qx
    qvec[:,1] = qy
    qvec[:,2] = qz

    return qvec


def random_pairs(total_elements, num_pairs): #, extra=10):
    """
    Sample `num_pairs` random pairs (i,j) from i,j in [0:total_elements) without
    replacement.
    
    Parameters
    ----------
    total_elements : int
        The total number of elements.
    num_pairs : int
        The number of unique pairs to sample
    
    Returns
    -------
    pairs : np.ndarray, int
        An `num_pairs` x 2 array of the random pairs.
    
    if num_pairs > (total_elements * (total_elements-1)) / 2:
        raise ValueError('Cannot request more than N(N-1)/2 unique pairs')
    
    not_done = True
    
    while not_done:
        n_to_draw = num_pairs + extra
        p = np.random.randint(0, total_elements, size=(num_pairs, 2))
        p.sort(axis=1)
        p = unique_rows(p)
        p = p[ p[:,0] != p[:,1] ] # slice out i == j
        
        if p.shape[0] >= num_pairs:
            p[:num_pairs]
            not_done = False
        else:
            extra += 10
    
    return p[0:num_pairs]
    """
    
    np.random.seed()
    inter_pairs = []
    factor = 2
    while len(inter_pairs) < num_pairs:
        rand_pairs   = np.random.randint( 0, total_elements, (num_pairs*factor,2) )
        unique_pairs = list( set( tuple(pair) for pair in rand_pairs ) )
        inter_pairs  = [x for x in unique_pairs if x[0] != x[1]]
        factor += 1
        
    return np.array(inter_pairs[0:num_pairs])


def maxima(a):
    """
    Returns the indices where `a` is at a local max.
    """
    return np.where(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True] == True)[0]
        
    

    
def memoize(obj):
    """
    An expensive memoizer that works with unhashables
    """
    # stolen unashamedly from pymc3
    cache = obj.cache = {}
    
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = (hashable(args), hashable(kwargs))
        
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
            
        return cache[key]
    
    return memoizer

    
def hashable(a):
    """
    Turn some unhashable objects into hashable ones.
    """
    # stolen unashamedly from pymc3
    if isinstance(a, dict):
        return hashable(iter(a.items()))
    try:
        return tuple(map(hashable,a))
    except:
        return a

        
graphic = """
                           ..........                                  
                            . ....+=.                                  
                  . .. ..,.~~,,,Z+=~:7....                             
                  ......:+,,,+$+++=7?+....                             
                  . ..,:~,~,:D?+=+=?II+:Z...                           
                .....??:::M$?++=?+???+IDO~=. .                         
                  ..+I:??.8?+=+=+=++?I?Z$~~+..                         
              . ..:=?,,?=7I=+=+==Z$++~=$$??=:~..                       
         . .. ..~:~=..=N$7?=++++=Z++===$=+=:+~M:...                    
             ..:::I.=.M$???+=++=+Z+====O=++::IIN8..                    
         . ..~,,?,.=~MD7I??=====~=~~===+7==::=?M:..                    
         ...,:~=..~:???M8O8O?====~~~~===+?=~::M:...                    
         ...+~O,,:?I?+++++++=+?=+=~~==+==+?+~M+....                    
         ..~~,.,.NI?=+=++===+=~==~+==I+=+~Z?MI.. ..                    
    ......~:,,,:MZ??+~=+?+=~++====~~~~7+=~+M~..   .    _____ _   _ ___________          
    ....,~:..=.O??===+=I?+=========~~==$I8O...        |_   _| | | |  _  | ___ \
      .==+,.,:8$M?=~=~~DI+===++=====~==?8+. ..          | | | |_| | | | | |_/ /
  ....:~.,:?~?$Z?7+==8+MI+=====+====~~~M=.....          | | |  _  | | | |    / 
  ...+D:,~?~+7??7D7++++?=+++=======~=+M:..              | | | | | \ \_/ / |\ \
. ..~~:~N??=??8ZZZMO7NZI+++++======+Z~=...              \_/ \_| |_/\___/\_| \_|
  . D77?+7I++Z=?Z$I+?++8???=+=+=+~?M+O8$D......                        
   ..,M+=~~++=O?+DO7I+=+ZZ$+++=+++~+NNDDZ$?...                         
    ...=?=~:~==++NN$I7??I8+87=++Z??M8N8DDNDZ..                         
         ~==::~==++ZMO$$7+???M?MI...8OOONNDDZZ.....                    
         ..?=:~~=++=8I8O$7?=??M?. .. .NNOMDD8O$Z...                    
         ....?=+==+???8M87?=~+,... . ..OO8888DNO7:.                    
              .7+?+=?I?IZZ$M$...     .. .DNO8NDD8ZO.....               
              ..+I?=+IOI,MZ......    .   ..$888DDD8Z+...               
              .. .:7,I?..........    .. .....8OODND8Z77.               
                                            . 78O8DDNOZ7....           
                                            ...,ZZM8D8OO?...           
                                              ....ZOONMNI....          
                                                  .$OMO.               
                                                 . ..  .
		                                
     ----------------------------------------------------------------------
"""
