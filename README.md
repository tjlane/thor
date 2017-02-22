Thor
====

[![Build Status](https://travis-ci.org/tjlane/thor.png?branch=master)](https://travis-ci.org/tjlane/thor)

Thor is a package for simulating and analyzing x-ray scattering experiments.

Originally engineered to process CXS (also called fXS, XCCA) correlated scattering experiments, Thor is agnostic in many respects to the x-ray scattering experiment. It is not well-suited for the specific needs of crystallography, but provides a useful base for e.g. single-particle diffraction, speckle/xray correlation spectroscopy, and diffuse scatter in crystallography.

One of the key aspects of Thor is that it provides a unified framework for analyzing experimental data and simulations with the same code. So you can do a simulation in Thor of your x-ray scattering experiment and text the analysis pipeline on it before you do an experiment -- when you can re-use that same code.

An emphasis has been placed on flexibility and usability over serving the specific needs of any of these experiments. You may find that what you want doesn't exist out of the box, but can be thrown together with a few lines of python, the language of the  primary API. Performance is relatively high, with C/C++ and CUDA extensions used to speed bottlenecks. The code is designed with LCLS data throughput rates in mind.

Enjoy! Let me know if you have specific questions.

TJ Lane
Lead Developer
<tjlane@stanford.edu>


Contributors
------------
* Derek Mendez
* Yutong Zhao
* Gundolf Scheck
* Jonas Sellberg
* AP