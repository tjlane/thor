u"""
setup.py: Install ODIN
"""

import os, sys
from os.path import join as pjoin
from glob import glob

from setuptools import setup, Extension
from distutils.unixccompiler import UnixCCompiler
from distutils.command.install import install as DistutilsInstall
from Cython.Distutils import build_ext

import numpy

import subprocess
from subprocess import CalledProcessError

# ------------------------------------------------------------------------------
# COLORS

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# ------------------------------------------------------------------------------
# HEADER -- metadata for setup()
# ------------------------------------------------------------------------------

VERSION = "0.1a"
ISRELEASED = False
__author__ = "TJ Lane"
__version__ = VERSION

metadata = {
    'name': 'odin',
    'version': VERSION,
    'author': __author__,
    'author_email': 'tjlane@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://github.com/tjlane/odin',
    'download_url': 'https://github.com/tjlane/odin',
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'mdtraj', 
                         'nose'],
    'dependency_links' : ['https://github.com/rmcgibbo/mdtraj/tarball/master#egg=mdtraj-0.0.0'],
    'platforms': ['Linux'],
    'zip_safe': False,
    'test_suite': "nose.collector",
    'description': "Code for Structure Determination",
    'long_description': """ODIN is a simulation toolpackage for producing
models of biomolecular structures consistent with a large set of experimental
data."""}


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS -- path finding, git, python version, readthedocs
# ------------------------------------------------------------------------------

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
    

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


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


def write_version_py(filename='src/python/version.py'):
    
    cnt = """
# THIS FILE IS GENERATED FROM ODIN SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""

    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

    
    
# ------------------------------------------------------------------------------
# GPU FUNCTION WRAPPING -- nvcc support
# python distutils doesn't have NVCC by default
# ------------------------------------------------------------------------------

def locate_cuda():
    """
    Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDA_HOME env variable is in use
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print bcolors.WARNING + 'The nvcc binary could not be located in your $PATH. add it to your path, or set $CUDA_HOME.' + bcolors.ENDC
            return False
            
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    print "CUDA config:", cudaconfig
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            s = 'The CUDA %s path could not be located in %s' % (k, v)
            print bcolors.WARNING + s + bcolors.ENDC
            return False
    return cudaconfig
    
CUDA = locate_cuda()
if CUDA == False:
    CUDA_SUCCESS = False
else:
    CUDA_SUCCESS = True

def customize_compiler_for_nvcc(self):
    """
    Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# ------------------------------------------------------------------------------
# Custom installer, that will allow us to use automake to install the c packages
# in ./depend/, specifically:
#
# -- cbflib & pycbf
#
#

PYCBF_SUCCESS = True # will get toggeled to False if it fails
curdir = os.path.abspath(os.curdir)

# install cbflib & pycbf
try:
    import pycbf
except ImportError as e:
    try:
        print "moving: ./depend/cbflib"
        os.chdir('./depend/cbflib')
        print "calling sh install_cbflib.sh"
        subprocess.check_call('sh install_cbflib.sh', shell=True)
    except:
        PYCBF_SUCCESS = False
        print bcolors.WARNING + 'Error during cbflib/pycbf installation' + bcolors.ENDC

try:
    import pycbf
except ImportError as e:
    print bcolors.WARNING + 'Error during cbflib/pycbf installation' + bcolors.ENDC

print "moving: %s" % curdir
os.chdir(curdir)

# -----------------------------------------------------------------------------
# INSTALL C/C++ EXTENSIONS
# gpuscatter, cpuscatter, bcinterp
# -----------------------------------------------------------------------------


# set up OMP flags -- if NO_OPENMP=1, turn off open mp
if 'NO_OPENMP' in os.environ.keys():
    if (os.environ['NO_OPENMP'] == '1'):
        omp_compile = ['-DNO_OMP']
        omp_link    = []
        print bcolors.WARNING + '$NO_OMP=1, disabling OPENMP support' + bcolors.ENDC
    else:
        omp_compile = ['-fopenmp']
        omp_link    = ['-lgomp']
elif '--no-openmp' in sys.argv[2:]:
    sys.argv.remove('--no-openmp')
    omp_compile = ['-DNO_OMP']
    omp_link    = []
    print bcolors.WARNING + 'set --no-openmp, disabling OPENMP support' + bcolors.ENDC
else:
    omp_compile = ['-fopenmp']
    omp_link    = ['-lgomp']


# install the extensions
if CUDA:
    print "Attempting to install gpuscatter module..."
    gpuscatter = Extension('odin._gpuscatter',
                            sources=['src/gpuscatter/swig_wrap.cpp', 'src/gpuscatter/gpuscatter_mgr.cu'],
                            library_dirs=[CUDA['lib64']],
                            libraries=['cudart'],
                            runtime_library_dirs=[CUDA['lib64']],
                            # this syntax is specific to this build system
                            # we're only going to use certain compiler args with nvcc and not with gcc
                            # the implementation of this trick is in customize_compiler() below
                            extra_compile_args={'gcc': [],
                                                'nvcc': ['-use_fast_math', '-arch=sm_20', '--ptxas-options=-v', 
                                                         '-c', '--compiler-options', "'-fPIC'"]},
                            include_dirs = [numpy_include, CUDA['include'], 'src/gpuscatter'])
                    

cpuscatter = Extension('odin._cpuscatter',
                        sources=['src/cpuscatter/swig_wrap.cpp', 'src/cpuscatter/cpuscatter.cpp'],
                        extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile,
                                            'g++': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile},
                        runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                        extra_link_args = ['-lstdc++', '-lm'] + omp_link,
                        include_dirs = [numpy_include, 'src/cpuscatter'])
                        
                        
bcinterp    = Extension('odin.bcinterp',
                        sources=['src/interp/cyinterp.pyx', 'src/interp/bcinterp.cpp'],
                        extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile,
                                            'g++': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile},
                        runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                        extra_link_args = ['-lstdc++', '-lm'] + omp_link,
                        include_dirs = [numpy_include, 'src/interp'],
                        language='c++')

# check for swig
if find_in_path('swig', os.environ['PATH']):
    subprocess.check_call('swig -Wall -python -c++ -o src/cpuscatter/swig_wrap.cpp src/cpuscatter/cpuscatter.i', shell=True)
    subprocess.check_call('swig -Wall -python -c++ -o src/gpuscatter/swig_wrap.cpp src/gpuscatter/gpuscatter.i', shell=True)
    
    # this could be a bad idea, but try putting the SWIG python files into the python source tree
    subprocess.check_call('cp src/cpuscatter/cpuscatter.py src/python/cpuscatter.py', shell=True)
    subprocess.check_call('cp src/gpuscatter/gpuscatter.py src/python/gpuscatter.py', shell=True)
    
else:
    raise EnvironmentError('the swig executable was not found in your PATH')

metadata['packages']     = ['odin', 'odin.scripts']
metadata['package_dir']  = {'odin' : 'src/python', 'odin.scripts' : 'scripts'}
metadata['ext_modules']  = [bcinterp, cpuscatter]
metadata['scripts']      = [s for s in glob('scripts/*') if not s.endswith('__.py')]
metadata['data_files']   = [('reference', glob('./reference/*'))]
metadata['cmdclass']     = {'build_ext': custom_build_ext}
metadata['zip_safe']     = False

# if we have a CUDA-enabled GPU...
if CUDA:
    metadata['ext_modules'].append(gpuscatter)


# ------------------------------------------------------------------------------
#
# Finally, print a warning at the *end* of the build if something fails
#

def print_warnings():

    if not PYCBF_SUCCESS:
        print 
        print '*'*65
        print '* WARNING : PYCBF'
        print '* ---------------'
        print '* Could not install cbflib/pycbf successfully. If you wish to'
        print '* load/employ cbf (crystallographic binary files), please install'
        print '* cbflib and pycbf manually. Use the script "install_cbflib.sh in'
        print '* odin/depend/cbflib as a template.  Until then, ODIN will'
        print '* function as usual without cbf-reading capailities.'
        print '*'*65
        
    if not CUDA_SUCCESS:
        print 
        print '*'*65
        print '* WARNING : CUDA/GPU SUPPORT'
        print '* --------------------------'
        print '* Could not install one or more CUDA/GPU features. Look for'
        print '* warnings in the setup.py output (above) for more details. ODIN'
        print '* will function without any GPU-acceleration. Note that for  '
        print '* successful installation of GPU support, you must have an nVidia'
        print '* Fermi-class GPU (or better) and the CUDA toolkit installed. See'
        print '* the nVidia website for more details.'
        print '*'*65


if __name__ == '__main__':
    write_version_py()
    setup(**metadata)
    print_warnings()
