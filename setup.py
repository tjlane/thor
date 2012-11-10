u"""
setup.py: Install ODIN

Many thanks to Robert McGibbon (rmcgibbo) -- this file is largely stolen from
his MSMBuilder setup.py file.
"""

import os, sys
from glob import glob

# setuptools needs to come before numpy.distutils to get install_requires
import setuptools 
from setuptools import setup

from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
from distutils.extension import Extension
from distutils.command.build_ext import build_ext

#import numpy
#from numpy.distutils.core import setup, Extension
#from numpy.distutils.misc_util import Configuration

import subprocess
from subprocess import CalledProcessError



# ------------------------------------------------------------------------------
# HEADER -- metadata for setup()
# ------------------------------------------------------------------------------

VERSION = "0.1b"
ISRELEASED = False
__author__ = "TJ Lane"
__version__ = VERSION

metadata = {
    'version': VERSION,
    'author': __author__,
    'author_email': 'tjlane@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://github.com/tjlane/odin',
    'download_url': 'https://simtk.orgc/home/msmbuilder',
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pyyaml',
                         'deap', 'fastcluster==1.1.6'],
    'platforms': ["Linux", "Mac OS X"],
    'zip_safe': False,
    'description': "Code for Structure Determination",
    'long_description': """ODIN is a simulation toolpackage for producing
models of biomolecular structures consistent with a large set of experimental
data."""}


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS -- git, python version, readthedocs
# ------------------------------------------------------------------------------

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
# python distutils doesn't have NVCC by default. This is a total hack.
# ------------------------------------------------------------------------------

subprocess.check_call('swig -python -c++ -o src/swig/swig_wrap.cpp src/swig/swig.i', shell=True)

# make the clean command always run first
#sys.argv.insert(1, 'clean')
#sys.argv.insert(2, 'build')

class MyExtension(Extension):
    """subclass extension to add the kwarg 'glob_extra_link_args'
    which will get evaluated by glob right before the extension gets compiled
    and let the swig shared object get linked against the cuda kernel
    """
    def __init__(self, *args, **kwargs):
        self.glob_extra_link_args = kwargs.pop('glob_extra_link_args', [])
        Extension.__init__(self, *args, **kwargs)

class NVCC(UnixCCompiler):
    src_extensions = ['.cu']
    executables = {'preprocessor' : None,
                   'compiler'     : ["nvcc"],
                   'compiler_so'  : ["nvcc"],
                   'compiler_cxx' : ["nvcc"],
                   'linker_so'    : ["echo"], # TURN OFF NVCC LINKING
                   'linker_exe'   : ["gcc"],
                   'archiver'     : ["ar", "-cr"],
                   'ranlib'       : None,
               }
    def __init__(self):
        # Check to ensure that nvcc can be located
        try:
            subprocess.check_output('nvcc --help', shell=True)
        except CalledProcessError:
            print >> sys.stderr, 'Could not find nvcc, the nvidia cuda compiler'
            sys.exit(1)
        UnixCCompiler.__init__(self)


class custom_build_ext(build_ext):
    """
    this cusom class lets us build one extension with nvcc and one extension with 
    regular gcc basically, it just tries to detect a .cu file ending to trigger 
    the nvcc compiler.
    """
    
    def build_extensions(self):
        # we're going to need to switch between compilers, so lets save both
        self.default_compiler = self.compiler
        self.nvcc = NVCC()
        build_ext.build_extensions(self)

    def build_extension(self, *args, **kwargs):
        extension = args[0]
        # switch the compiler based on which thing we're compiling
        # if any of the sources end with .cu, use nvcc
        if any([e.endswith('.cu') for e in extension.sources]):
            # note that we've DISABLED the linking (by setting the linker to be "echo")
            # in the nvcc compiler
            self.compiler = self.nvcc
        else:
            self.compiler = self.default_compiler

        # evaluate the glob pattern and add it to the link line
        # note, this suceeding with a glob pattern like build/temp*/gpurmsd/RMSD.o
        # depends on the fact that this extension is built after the extension
        # which creates that .o file
        if hasattr(extension, 'glob_extra_link_args'):
            for pattern in extension.glob_extra_link_args:
                unglobbed = glob(pattern)
                if len(unglobbed) == 0:
                    raise RuntimeError("glob_extra_link_args didn't match any files")
                self.compiler.linker_so += unglobbed
        
        # call superclass
        build_ext.build_extension(self, *args, **kwargs)


# this code will get compiled up to a .o file by nvcc. the final .o file(s) that
# it makes will be just one for each input source file. Note that we turned off
# the nvcc linker so that we don't make any .so files.
nvcc_compiled = Extension('nvcc_object_avoider',
                          sources=['src/cuda/gpuscatter_manager.cu'],
                          extra_compile_args=['-arch=sm_20', '--ptxas-options=-v', 
                                              '-c', '--compiler-options', "'-fPIC'"],
                          # we need to include src as an input directory so that 
                          # the header files and device_kernel.cu can be found
                          include_dirs=['/usr/local/cuda/include', 'src'],
                          )

# the swig wrapper for gpuscatter.cu gets compiled, and then linked to scatter.o
swig_wrapper = MyExtension('_scatter',
                           sources=['src/swig/swig_wrap.cpp'],
                           library_dirs=['/usr/local/cuda/lib64'],
                           libraries=['cudart'],
                           # extra bit of magic so that we link this
                           # against the kernels -o file
                           # this picks up the build/temp.linux/src/manager.cu
                           glob_extra_link_args=['build/*/*/manager.o'],
                           # make sure that at runtime we find the linked libraries
                           runtime_library_dirs="$ORIGIN/../lib/")

# inject our custom trigger
metadata['cmdclass'] = {'build_ext': custom_build_ext}


# -----------------------------------------------------------------------------
# PROCEED TO STANDARD SETUP
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
    
    # add the scipts, so they can be called from the command line
    config.add_scripts([e for e in glob('scripts/*.py') if not e.endswith('__.py')])
    
    # add scripts as a subpackage (so they can be imported from other scripts)
    config.add_subpackage('scripts', subpackage_path=None)

    # example for additional packages or extensions
    #config.add_subpackage('package', subpackage_path='src/python/package_name')
    #dist = Extension('msmbuilder._distance_wrap', sources=glob('src/ext/scipy_distance/*.c'))
        
    #for extension in [module_name]:
        # ext.extra_compile_args = ["-O3", "-fopenmp", "-Wall"]
        # ext.extra_link_args = ['-lgomp']
        # ext.include_dirs = [numpy.get_include()]
        # config.ext_modules.append(extension)
    
    return config


metadata['py_modules']  = ['scatter']
metadata['package_dir'] = {'': 'src/python',         # the 'root' package - odin
                           'gpuscatter': 'src/cuda'  # GPU scattering code
                           }
metadata['ext_modules'] = [nvcc_compiled, swig_wrapper]






if __name__ == '__main__':
    write_version_py()
    metadata['configuration'] = configuration
    setup(**metadata)
