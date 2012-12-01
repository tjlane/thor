u"""
setup.py: Install ODIN

Many thanks to Robert McGibbon (rmcgibbo) -- this file is largely stolen from
his MSMBuilder setup.py file.
"""

import os, sys
from os.path import join as pjoin
from glob import glob

# setuptools needs to come before numpy.distutils to get install_requires
import setuptools 
from setuptools import setup

from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
#from distutils.extension import Extension
from distutils.command.build_ext import build_ext

import numpy
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

import subprocess
from subprocess import CalledProcessError



# ------------------------------------------------------------------------------
# HEADER -- metadata for setup()
# ------------------------------------------------------------------------------

VERSION = "0.1a"
ISRELEASED = False
__author__ = "TJ Lane"
__version__ = VERSION

metadata = {
#    'name': 'odin',
    'version': VERSION,
    'author': __author__,
    'author_email': 'tjlane@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://github.com/tjlane/odin',
    'download_url': 'https://github.com/tjlane/odin',
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'mdtraj', 
                         'nose', 'tables'],
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

# TJL commented out -- readthedocs not set up yet anyways
#
# if os.environ.get('READTHEDOCS', None) == 'True' and __name__ == '__main__':
#     # On READTHEDOCS, the service that hosts our documentation, the build
#     # environment does not have numpy and cannot build C extension modules,
#     # so if we detect this environment variable, we're going to bail out
#     # and run a minimal setup. This only installs the python packages, which
#     # is not enough to RUN anything, but should be enough to introspect the
#     # docstrings, which is what's needed for the documentation
#     from distutils.core import setup
#     import tempfile, shutil
#     write_version_py()
#     
#     metadata['name'] = 'odin'
#     metadata['packages'] = ['odin', 'odin.scripts'] # odin.module, ...
#     metadata['scripts'] = [e for e in glob('scripts/*.py') if not e.endswith('__.py')]
# 
#     # dirty, dirty trick to install "mock" packages
#     mockdir = tempfile.mkdtemp()
#     open(os.path.join(mockdir, '__init__.py'), 'w').close()
#     extensions = ['odin._image_wrap'] # c-extensions to the python code here
#     metadata['package_dir'] = {'odin': 'src/python', 'odin.scripts': 'scripts'}
#     metadata['packages'].extend(extensions)
#     for ex in extensions:
#         metadata['package_dir'][ex] = mockdir
#     # end dirty trick :)
# 
#     setup(**metadata)
#     shutil.rmtree(mockdir) #clean up dirty trick
#     sys.exit(1)
    
    
# ------------------------------------------------------------------------------
# GPU FUNCTION WRAPPING -- nvcc support
# python distutils doesn't have NVCC by default
# ------------------------------------------------------------------------------

def locate_cuda():
    """
    Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print
            print '------------------------- WARNING --------------------------'
            print 'The nvcc binary could not be located in your $PATH. Either '
            print 'add it to your path, or set $CUDAHOME. The installation will'
            print 'continue witout CUDA/GPU features.'
            print '------------------------------------------------------------'
            print
            return False
            
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            print
            print '------------------------- WARNING --------------------------'
            print 'The CUDA %s path could not be located in %s' % (k, v)
            print 'The installation will continue witout CUDA/GPU features.'
            print '------------------------------------------------------------'
            print
            return False
    return cudaconfig
    
CUDA = locate_cuda()


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


# run the customize_compiler
if CUDA:
    class custom_build_ext(build_ext):
        def build_extensions(self):
            customize_compiler_for_nvcc(self.compiler)
            build_ext.build_extensions(self)


# -----------------------------------------------------------------------------
# PROCEED TO STANDARD SETUP
# odin, gpuscatter,
# -----------------------------------------------------------------------------

def configuration(parent_package='',top_path=None):
    "Configure the build"

    config = Configuration('odin',
                           package_parent=parent_package,
                           top_path=top_path,
                           package_path='src/python')
    config.set_options(assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False)
    
    #once all of the data is in one place, we can add it with this
    config.add_data_dir('reference')
    
    # add the scipts, so they can be called from the command line
    config.add_scripts([s for s in glob('scripts/*') if not s.endswith('__.py')])
    
    # add scripts as a subpackage (so they can be imported from other scripts)
    config.add_subpackage('scripts', subpackage_path=None)

    print config.todict()
 
    return config


if CUDA:
    print "Attempting to install gpuscatter module..."
    gpuscatter = Extension('_gpuscatter',
                            sources=['src/cuda/swig_wrap.cpp', 'src/cuda/gpuscatter_mgr.cu'],
                            library_dirs=[CUDA['lib64']],
                            libraries=['cudart'],
                            runtime_library_dirs=[CUDA['lib64']],
                            # this syntax is specific to this build system
                            # we're only going to use certain compiler args with nvcc and not with gcc
                            # the implementation of this trick is in customize_compiler() below
                            extra_compile_args={'gcc': [],
                                                'nvcc': ['-use_fast_math', '-arch=sm_20', '--ptxas-options=-v', 
                                                         '-c', '--compiler-options', "'-fPIC'"]},
                            include_dirs = [numpy_include, CUDA['include'], 'src/cuda'])


# check for swig
if find_in_path('swig', os.environ['PATH']):
    subprocess.check_call('swig -python -c++ -o src/cuda/swig_wrap.cpp src/cuda/gpuscatter.i', shell=True)
else:
    raise EnvironmentError('the swig executable was not found in your PATH')

# ADD PACKAGES, MODULES TO metadata

#metadata['packages']    = [] # ['odin']

metadata['py_modules']  = []
metadata['package_dir'] = {} #{'': 'src/python'}
metadata['ext_modules'] = []
metadata['scripts'] = [s for s in glob('scripts/*') if not s.endswith('__.py')]                 
metadata['configuration'] = configuration

# if we have a CUDA-enabled GPU...

if CUDA:
    metadata['package_dir'][''] = 'src/cuda'
    metadata['py_modules'].append('gpuscatter')
    metadata['ext_modules'].append(gpuscatter)

    # inject our custom trigger
    metadata['cmdclass'] = {'build_ext': custom_build_ext}


if __name__ == '__main__':
    write_version_py()
    setup(**metadata)
