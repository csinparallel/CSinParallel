# CSinParallel Code

CSinParallel is an NSF-funded project to support instructors seeking to add Parallel and Distributed Computing (PDC) to their undergraduate computer science courses.  

This public repository contains some of the code examples from the teaching modules that we use the most in our courses at various levels of the curriculum. We will be attempting to add to this repository over time.

Most of these code examples are also now available in online dynamic textbooks avaialble by visiting [learnpdc.org](https://www.learnpdc.org/).

## Software requirements

To use all of this sofcodetware on your own server, you will need quite a few software libraries installed. Here is a list of what we install on our Ubuntu systems that back the online textbook and that we have used in our courses.

Ubuntu packages

- [ ] environment-modules
- [ ] openmpi-bin openmpi-doc libopenmpi-dev
- [ ] python3-mpi4py
- [ ] shelltestrunner
- [ ] cmake

Installs from source
- [ ] trng
- [ ] NVidia HPC toolkit

Python installs
- [ ] numpy
- [ ] matplotlib

### Shelltestrunner

We have some shell scripts that use shelltestrunner to build and run several of our code examples to make certain that our software installs are working on the systems that run the PDC code. These shell scripts, typically called test_all.sh inside a directory, also sometimes set up the shell environment using environment modules listed next. On Ubuntu, install shelltestrunner like this:

    sudo apt install shelltestrunner

### Environment Modules

You might not need this package, but some people find it useful when you install more than one version of any of the software listed below.

https://modules.readthedocs.io/en/latest/

It appears that it can be loaded from the 'universe' packages:
https://askubuntu.com/questions/148638/how-do-i-enable-the-universe-repository

	sudo apt install environment-modules

### Python3 libraries

	sudo apt install python3-numpy
	sudo apt install python3-matplotlib

## trng libraries

This is built from code with cmake. The code is obtained from this github repo:

https://github.com/rabauke/trng4/tree/master

I have been using a directory called installs in my home directory.

	mkdir installs
	cd installs/
	git clone https://github.com/rabauke/trng4.git
	cd trng4

From the documentation, build with cmake to place in /usr/local.Note that currently our makefiles have paths to /usr/local for the include and library files.

However, first there a change needed: in the CMakeLists.text file, must change line 15:
from

	option(TRNG_ENABLE_TESTS "Enable/Disable the compilation of the TRNG tests" ON)

to

	option(TRNG_ENABLE_TESTS "Enable/Disable the compilation of the TRNG tests" OFF)

build:

	cd build
	cmake -DCMAKE_INSTALL_PREFIX=/opt/trng ..
	sudo cmake --build . --target install

	
### NVIDIA HPC toolkit

https://docs.nvidia.com/hpc-sdk//hpc-sdk-install-guide/index.html

These instructions place the version that you choose to download into /opt, which is the default location.

There are optional environment module files that you can set up also if you wish. Otherwise, you will need to make sure you are setting the environment variables correctly for each use to find the compilers, nvc, nvcc, and nvc++ and the libraries and include files.

## Brought to you by

The CSinParallel team is:

Dick Brown, St. Olaf College

Libby Shoop, Macalester College

Joel Adams, Calvin University

Suzanne Matthews, USMA West Point

## Sponsors

This work is sponsored in part by U.S. National Science Foundation (NSF) Collaborative Research Grants DUE-1822480/1822486/1855761. *Collaborative Research: CSinParallel: Experiential Learning of Parallel and Distributed Computing through Sight, Sound, and Touch*. 

This work is also sponsored in part by the U.S. Department of Defense.

### Original project site

The [csinparallel.org](https://csinparallel.org/index.html) website is our original resource for PDC teaching materials, including a number of teaching modules for inserting a few days of instruction on a PDC topic to a given course.