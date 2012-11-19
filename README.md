ODIN
====

Current Functionality
---------------------

ODIN is a work in progress. Currently, this is what works:

* GPU-based scattering simulations (SAXS/WAXS/CXS)

Description
-----------

Structural ensemble determination from heterogeneous, multiple experimental sources. 

ODIN takes your experimental data and predicts structures, and also structural ensembles in a rigorous way!

ODIN is a flexible software package that implements statistical mechanical theory capable of inferring geometrical structures from almost any experiment. The main focus of the software is on biomolecules studied via x-ray scattering and NMR, but in principle could be adapted to a number of different experimental systems and techniques.

The driving idea behind ODIN is that when an experiment is performed on a system, that experiment contains some quantity of information about the system's geometry. Extracting that information may be difficult. If, however, we possess an algorithm for predicting experiments from structures (often a much easier problem), then a direct comparison is possible. ODIN employs this comparison to generate a structural ensemble that, when run through such an algorithm, is guaranteed to reproduce the experimental data. Further, ODIN works to ensure that this ensemble contains as little "information", as measured by the entropy of the ensemble's Boltzmann distribution, as possible.

Details of the theory and ideas behind the inner workings of ODIN can be found in the references, below, and documentation included in the package.

In addition to providing structure determination functionality, ODIN contains:
* Powerful tools for predicting SAXS and WAXS experiments, and analyzing these experiments using traditional and correlation techniques.

Install
-------

Installation should be very easy. Hopefully, you can pretty much copy and paste the commands below!

1. Install the Enthought Python distribution (alternative: install all the below dependencies marked 'epd'). Should be as easy as running their install script.

	<pre><code>./sh epd-install-script.sh</code></pre>


2. Clone & install MDTraj (https://github.com/rmcgibbo/mdtraj/)

	<pre><code>git clone https://github.com/rmcgibbo/mdtraj.git
	cd mdtraj
	python setup.py install
	</code></pre>

3. Clone & install ODIN

	<pre><code>git clone https://github.com/tjlane/odin.git
	cd odin
	python setup.py install
	</code></pre>

4. Enjoy!

If you wish to run the structure determination part of ODIN, you'll need an nVidia GPU (sorry ATI owners) and also OpenMM. To get OpenMM, get the correct installation script from https://simtk.org/home/openmm, then

	./sh install
	
should do the trick!

We support Linux exclusively. MacOS and Windows *should* work, they are just largely untested. If someone wants to try on either, or has success, please let us know: <tjlane@stanford.edu>.


Dependencies
------------
* python2.7+ (epd)
* numpy      (epd)
* scipy      (epd)
* pytables   (epd)
* pyyaml     (epd)
* matplotlib (epd)
* OpenMM     (https://simtk.org/home/openmm)
* nvcc       (optional additional GPU support, https://developer.nvidia.com/cuda-downloads)


We HIGHLY recommend using the Enthought python distribution (EPD, http://www.enthought.com/products/epd.php) which contains all of the dependancies marked above with (epd) in a single package. It's awesome and free for academics.


Authors and Contributors
------------------------
* TJ Lane
* Derek Mendez
* Yutong Zhao
* Robert McGibbon


License
-------

ODIN is full-GPL. See LICENSE file.


Related Packages
----------------

Various parts of ODIN are derived from, or explicitly borrow from:
* Robert McGibbon & Yutong Zhao's sweet GPU-to-python wrapping scheme: https://github.com/rmcgibbo/npcuda-example
* MSMBuilder & OpenMM PDB & trajectory classes: https://github.com/rmcgibbo/mdtraj
* grid: https://github.com/dermen/grid


