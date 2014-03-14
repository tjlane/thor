
"""
Functions that are useful in various places, but have no common theme.
"""

import functools

from pprint import pprint
from argparse import ArgumentParser

import numpy as np


class Parser(ArgumentParser):
    """
    Simple extension of argparse, designed to automatically print stuff
    """
    def parse_args(self):
        print graphic
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
    for i in xrange(n):
        for j in xrange(i+1,n):
            yield (i,j)
    
    
def unique_rows(a):
    """
    For a two-dim array, returns unique rows.
    """
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    

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
        inter_pairs  = filter( lambda x:x[0] != x[1], unique_pairs)
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
        return hashable(a.iteritems())
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
