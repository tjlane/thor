u"""
setup.py: Install ODIN

Many thanks to Robert McGibbon (rmcgibbo) -- this file is largely stolen from
his MSMBuilder setup.py file.
"""

import os, sys
from glob import glob
import subprocess

VERSION = "0.1b"
ISRELEASED = False
__author__ = "TJ Lane"
__version__ = VERSION

# metadata for setup()
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

# ---------------- 
# HELPER FUNCTIONS 
# ----------------

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


if os.environ.get('READTHEDOCS', None) == 'True' and __name__ == '__main__':
    # On READTHEDOCS, the service that hosts our documentation, the build
    # environment does not have numpy and cannot build C extension modules,
    # so if we detect this environment variable, we're going to bail out
    # and run a minimal setup. This only installs the python packages, which
    # is not enough to RUN anything, but should be enough to introspect the
    # docstrings, which is what's needed for the documentation
    from distutils.core import setup
    import tempfile, shutil
    write_version_py()
    
    metadata['name'] = 'odin'
    metadata['packages'] = ['odin', 'odin.scripts'] # odin.module, ...
    metadata['scripts'] = [e for e in glob('scripts/*.py') if not e.endswith('__.py')]

    # dirty, dirty trick to install "mock" packages
    mockdir = tempfile.mkdtemp()
    open(os.path.join(mockdir, '__init__.py'), 'w').close()
    extensions = ['odin._image_wrap'] # c-extensions to the python code here
    metadata['package_dir'] = {'odin': 'src/python', 'odin.scripts': 'scripts'}
    metadata['packages'].extend(extensions)
    for ex in extensions:
        metadata['package_dir'][ex] = mockdir
    # end dirty trick :)

    setup(**metadata)
    shutil.rmtree(mockdir) #clean up dirty trick
    sys.exit(1)

# -----------------------------------------------------------------------------
# PROCEED TO STANDARD SETUP
# setuptools needs to come before numpy.distutils to get install_requires
# -----------------------------------------------------------------------------

import setuptools 
import numpy
from distutils import sysconfig
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

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

if __name__ == '__main__':
    write_version_py()
    metadata['configuration'] = configuration
    setup(**metadata)
