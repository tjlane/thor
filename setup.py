u"""
setup.py: Install THOR
"""

import os
import sys
import re
import subprocess
from os.path import join as pjoin
from glob import glob

from distutils.extension import Extension
from distutils.core import setup

from Cython.Distutils import build_ext
import numpy


# ------------------------------------------------------------------------------
# HEADER
# 

VERSION     = "0.0.1"
ISRELEASED  = False
__author__  = "TJ Lane"
__version__ = VERSION

metadata = {
    'name': 'thor',
    'version': VERSION,
    'author': __author__,
    'author_email': 'tjlane@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://github.com/tjlane/thor',
    'download_url': 'https://github.com/tjlane/thor',
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'mdtraj', 
                         'nose', 'cython>=0.16', 'tables'],
    'dependency_links' : ['https://github.com/kif/fabio/tarball/master#egg=fabio-0.1.3'],
    'platforms': ['Linux', 'OSX'],
    'zip_safe': False,
    'test_suite': "nose.collector",
    'description': "Code for Structure Determination",
    'long_description': """THOR is a simulation toolpackage for producing
models of biomolecular structures consistent with a large set of experimental
data."""}


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS -- path finding, git, python version, readthedocs
# 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
    
def print_warning(string):
    print bcolors.WARNING + string + bcolors.ENDC
    

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
    

def get_numpy_include():
    """
    Obtain the numpy include directory. This logic works across numpy versions.
    """
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    return numpy_include
    

def git_version():
    """
    Return the git revision as a string.
    Copied from numpy setup.py
    """
    
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# ------------------------------------------------------------------------------
# GPU FUNCTION WRAPPING -- nvcc support
# python distutils doesn't have NVCC by default
# 

def locate_cuda():
    """
    Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDA_HOME or CUDA_ROOT env variable is in use
    
    CUDA_ENVs = ['CUDA_HOME','CUDA_ROOT']
    
    found_config = False
    for CUDA_ENV in CUDA_ENVs:
        if CUDA_ENV in os.environ:
            home = os.environ[CUDA_ENV].split(':')
            nvcc = map( lambda x:pjoin(x, 'bin', 'nvcc'), home)
        else:
            # otherwise, search the PATH for NVCC
            nvcc = find_in_path('nvcc', os.environ['PATH'])
            if nvcc is None:
                print_warning('The nvcc binary could not be located in your $PATH. '
                              'add it to your path, or set $CUDA_HOME.')
                return False
                
            home = os.path.dirname(os.path.dirname(nvcc))


        cudaconfig_list=[{'home'   : home[x], 
                          'nvcc'   : nvcc[x],
                          'include': pjoin(home[x], 'include'),
                          'lib64'  : pjoin(home[x], 'lib64')} \
                          for x in xrange(len(home)) ]
        
        # be sure all the necessary items are there
        for cudaconfig in cudaconfig_list:
            found_items = 0
            for k, v in cudaconfig.iteritems():
                if not os.path.exists(v):
                    print_warning('The CUDA %s path could not be located in %s' % (k, v))
                elif os.path.exists(v):
                    found_items += 1
            if found_items == len(cudaconfig):
                found_config = True
                break
        if found_config:
            break
            
    if not found_config:
        cudaconfig = {'enabled' : False,
                      'home'    : '', 
                      'nvcc'    : '',
                      'include' : '',
                      'lib64'   : ''}
        
    else:
        cudaconfig['enabled'] = True
        print "Found CUDA config:", cudaconfig
        
    return cudaconfig
    

def customize_compiler_for_nvcc(compiler):
    """
    Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    CUDA = locate_cuda()
    
    # save references to the default compiler_so and _comple methods
    old_compile     = compiler._compile
    old_compiler_so = compiler.compiler_so

    def _new_compile(obj, src, ext, cc_args, postargs, pp_opts):
        """
        `postargs` is usually a list, but here we let it be a dict
        as a sloppy way of choosing both the compiler and args for
        that compiler...
        """

        if type(postargs) == dict:
            if ('nvcc' in postargs.keys()) and CUDA['enabled']:
                compiler.set_executable('compiler_so', CUDA['nvcc'])
                postargs_list = postargs['nvcc']
            else:
                postargs_list = postargs[postargs.keys()[0]]
        else:
            postargs_list = postargs
            
        # call the compile routine, then
        # reset the default compiler_so, which we might have changed for cuda
        old_compile(obj, src, ext, cc_args, postargs_list, pp_opts)
        compiler.compiler_so = old_compiler_so

    # inject our redefined _compile method into the class
    compiler._compile = _new_compile


class custom_build_ext(build_ext, object):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)
        
        
        
def locate_open_mp():
    # openmp support -- disabled by default
    if '--enable-openmp' in sys.argv[2:]:
        sys.argv.remove('--enable-openmp')
        enabled     = True
        omp_compile = ['-fopenmp']
        omp_link    = ['-lgomp']
        print_warning('set --enable-openmp, enabling OPENMP support')
    else:
        enabled = False
        omp_compile = ['-DNO_OMP']
        omp_link    = []
        
    OMP = {'enabled' : enabled,
           'compile' : omp_compile,
           'link'    : omp_link}
        
    return OMP

# -----------------------------------------------------------------------------
# INSTALL C/C++ EXTENSIONS
# 

OMP  = locate_open_mp()
CUDA = locate_cuda()

cppscatter = Extension('thor._cppscatter',
                    sources=['src/scatter/cpp_scatter_wrap.pyx', 'src/scatter/cpp_scatter.cpp'],
                    extra_compile_args={'nvcc' : ['-use_fast_math', '-arch=sm_20', '--ptxas-options=-v', 
                                                 '-c', '--shared', '-Xcompiler=-fPIC']
                                         'gcc': ['-O3', '-fPIC', '-Wall'],
                                         'g++': ['-O3', '-fPIC', '-Wall', '-mmacosx-version-min=10.6'] }},
                                                 
                    library_dirs=[CUDA['lib64']],
                    libraries=['cudart'],
                    runtime_library_dirs=['/usr/lib', '/usr/local/lib', CUDA['lib64']],
                    extra_link_args = ['-lstdc++', '-lm'],
                    include_dirs = [get_numpy_include(), 'src/scatter', CUDA['include']],
                    language='c++')

misc = Extension('thor.misc_ext',
                 sources=['src/misc/misc_wrap.pyx', 'src/misc/solidangle.cpp'],
                 extra_compile_args = ['-O3', '-fPIC', '-Wall'],
                 runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                 extra_link_args = ['-lstdc++', '-lm'],
                 include_dirs = [get_numpy_include(), 'src/misc'],
                 language='c++')

corr = Extension('thor.corr',
                 sources=['src/corr/correlate.pyx', 'src/corr/corr.cpp'],
                 extra_compile_args = ['-O3', '-fPIC', '-Wall'],
                 runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                 extra_link_args = ['-lstdc++', '-lm'],
                 include_dirs = [get_numpy_include(), 'src/corr'],
                 language='c++')


metadata['packages']     = ['thor']
metadata['package_dir']  = {'thor' :         'src/python'}
metadata['ext_modules']  = [cppscatter, misc, corr]
metadata['scripts']      = [s for s in glob('scripts/*') if not s.endswith('__.py')]
metadata['data_files']   = [('reference', glob('./reference/*'))]
metadata['cmdclass']     = {'build_ext': custom_build_ext}

# ------------------------------------------------------------------------------
#
# Finally, print a warning at the *end* of the build if something fails
#

def print_warnings():
        
    if not CUDA_SUCCESS:
        print 
        print '*'*65
        print '* WARNING : CUDA/GPU SUPPORT'
        print '* --------------------------'
        print '* Could not install one or more CUDA/GPU features. Look for'
        print '* warnings in the setup.py output (above) for more details. THOR'
        print '* will function without any GPU-acceleration. EVERYTHING WILL STILL'
        print '* WORK -- just certain things will be a bit slower. Note that for  '
        print '* successful installation of GPU support, you must have an nVidia'
        print '* Fermi-class GPU (or better) and the CUDA toolkit installed. See'
        print '* the nVidia website for more details.'
        print '*'*65
        
    try:
        import pyfftw
    except:
        print 
        print '*'*65
        print '* WARNING : PYFFTW SUPPORT'
        print '* --------------------------'
        print '* Could not load the pyfftw package, EVERYTHING WILL STILL'
        print '* WORK -- just certain things will be a bit slower. Install FFTW'
        print '* and pyfftw if you wish to accelerate any calculation involving '
        print '* FFTs, most notably correlation computations.'
        print '* (https://pypi.python.org/pypi/pyFFTW)'
        print '* (http://www.fftw.org/)'
        print '*'*65
     
    print "\n"

if __name__ == '__main__':
    setup(**metadata) # ** will unpack dictionary 'metadata' providing the values as arguments
    print_warnings()
