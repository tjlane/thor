
"""
Write output files specific for xray data.

-- CXIdb files
"""

import tables
import numpy as np

from mdtraj import io


def write_cxidb(filename, shotset, sample_name='odinshotset'):
    """
    Write a shotset to disk in CXIdb format.
    
    Parameters
    ----------
    filename : str
        The name of the file!
        
    shotset : odin.xray.ShotSet
        The shotset object to save
        
    Optional Parameters
    -------------------
    sample_name : str
        The name of the sample, to aid future researchers
    
    References
    ----------
    ..[1] http://www.cxidb.org
    """
    
    if not filename.endswith('.cxi'):
        filename += '.cxi'
    
    # this code is based on the diagram on pp 13 of the document "The CXI File 
    # Format for Coherent X-ray Imaging" (v 1.3, F. Maia, 2012)
    
    f = tables.File(filename, mode='w')
    
    # generate atoms
    fa = tables.Atom.from_dtype(np.dtype(np.float64))
    
    # generate all groups
    f.create_group('/', 'entry_1')
    f.create_group('/entry_1', 'sample_1')
    f.create_group('/entry_1', 'data_1')
    f.create_group('/entry_1', 'image_1')
    f.create_group('/entry_1/image_1', 'detector_1')
    f.create_group('/entry_1/image_1', 'source_1')
    
    # cxi verson
    version = 110
    f.create_carray('/', 'cxi_version', obj=[version])
    
    # data name
    f.create_carray('/entry_1/sample_1', 'name', obj=[sample_name])
    
    # save data
    pi_node = f.createEArray(where='/entry_1/image_1', name='intensities',
                             shape=(0, shotset.num_pixels), 
                             atom=fa, filters=io.COMPRESSION,
                             expectedrows=shotset.num_shots)
                               
    for intx in shotset.intensities_iter:
        pi_node.append(intx[None,:])
    
    # link /entry_1/data_1/data to /entry_1/data_1/image_1 (not sure why?)
    f.create_soft_link('/entry_1/data_1', 'data', '/entry_1/image_1/data')
    
    # data attributes
    f.create_carray('/entry_1/image_1', 'data_type', obj=['intensities'])
    f.create_carray('/entry_1/image_1', 'data_space', obj=['diffraction'])
    
    # detector
    # THIS IS NOT CXIdb FORMAT -- but that format for detectors is a bit rough 
    # and I am not going to spend time on it right now
    f.create_carray('/entry_1/image_1/detector_1', 'pixelmap', obj=shotset.detector.xyz)
    if shotset.mask != None:
        f.create_carray('/entry_1/image_1/detector_1', 'mask', obj=shotset.mask)
    
    # energy
    f.create_carray('/entry_1/image_1/source_1', 'energy', obj=[shotset.detector.beam.energy])
    
    # fft shifted (?)
    f.create_carray('/entry_1/image_1', 'is_fft_shifted', obj=[0])
    
    return