higra-distributed
==============

Algorithms for distributed binary partition trees using Higra


Installation
------------

**Requires a C++ 14 compiler and cmake**

 - `pip install ./higra-distributed`

Build a binary wheel
--------------------
 
A binary wheel ease the redistribution of your project and can be installed with *pip* on a client machine without a compiler.

**Create wheel**

 - `cd higra-distributed`
 - `python setup.py bdist_wheel`
 - `pip install ./higra-distributed`
 
 The wheel is created in the directory `higra-distributed/dist`, it will be named `higra_distributed-XXXXX.whl` where `XXXXXX` are name tags identifying the current platform and Python version. 
 
**Install wheel**
 
A wheel can be installed with *pip*:
 
 - `pip install wheel_name.whl`
 
 Note that a binary wheel is specific to a platform and to a python version (a wheel built on Windows with Python 3.5 can only be installed on Windows with Python 3.5).

Tests
-----

Tests are run automatically at the end of a build: the build will fail if tests are not successful. 

Known Issues
------------

Clang on Linux may not work due to ABI compatibilty issues.
