Attempt to install SPIVET on CentOS 7 via Anaconda
# Dependencies
## Create a python 2 env
```
conda create -n py27_new python=2.7 ipykernel
conda activate py27_new
conda install -c conda-forge backports.functools_lru_cache
python -m ipykernel install --user --name py27_new
```
## python packages
Matplotlib may have conflict with package (maybe vtk), so install that first:
`conda install -c conda-forge matplotlib`
Then general packages:
```
conda install -c anaconda numpy scipy netcdf4 pillow
conda install -c conda-forge vtk
```
Since the PIL package required by SPIVET is not maintained any more, nor can it be installed via pip or conda, we install pillow instead.
`pillow` is a fork of PIL, it does not support `import Image` since v1.0 (now v7.2), so we need to manually modify all `import Image` to
`from PIL import Image`
This is similar for modules like `ImageFilter`, `ImageOps`.
We also need to add the correct path in `setup.py` for the pkgs:
```
include_dirs = [
    ibpath,
    ibpath + '/..',# for dependencies like netcdf.h
    lbpath + '/numpy/core/include',
    lbpath + '/numpy/numarray',
    'lib/pivlib/exodusII',
    #'/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers',
    '/usr/include' # for lapack/blas
]

```
## LAPACK and BLAS
SPIVET use LAPACK and BLAS via  `clapack.h`, `cblas.h`, and it seems to be from Accelerate.framework (vecLib) under Mac by default, since it includes:
```
typedef __CLPK_integer lpk_int;
```
which does not exist in lapack or atlas package under CentOS.
On the other hand, MKL uses `mkl_lapack.h`, `mkl_blas.h`, and more variables/types are not in common.

### Attempt to use Atlas
Atlas is more optimized than lapack. In its `clapack.h` it defined `ATL_INT` instead of `__CLPK_integer`.
These are the file structures of atlas and lapack, they can co-exist:
[https://centos.pkgs.org/7/centos-x86_64/lapack-devel-3.4.2-8.el7.x86_64.rpm.html](https://centos.pkgs.org/7/centos-x86_64/lapack-devel-3.4.2-8.el7.x86_64.rpm.html)
[https://centos.pkgs.org/7/centos-x86_64/lapack-3.4.2-8.el7.x86_64.rpm.html](https://centos.pkgs.org/7/centos-x86_64/lapack-3.4.2-8.el7.x86_64.rpm.html)
[https://centos.pkgs.org/7/centos-x86_64/atlas-devel-3.10.1-12.el7.x86_64.rpm.html](https://centos.pkgs.org/7/centos-x86_64/atlas-devel-3.10.1-12.el7.x86_64.rpm.html)
[https://centos.pkgs.org/7/centos-x86_64/atlas-3.10.1-12.el7.x86_64.rpm.html](https://centos.pkgs.org/7/centos-x86_64/atlas-3.10.1-12.el7.x86_64.rpm.html)

Install Atlas:
```
yum install -y atlas atlas-devel
```
replace all `__CLPK_integer` to `ATL_INT` in SPIVET.
Now under root dir of SPIVET:
```
python setup.py build
python setup.py install
```
And then SPIVET can be installed under this python env of anaconda.
Test:
```
cd tests/
python run_tests.py
```
We can pass some tests, but fail some important ones:

```
Ran 149 tests in 23.684s

FAILED (failures=1, errors=37)
======================================================================
EXODUSII Tests Run: 123, Failed 0
```
#### use lapacke for lapack instead of atlas
Install lapacke:
```
yum install -y lapack lapack-devel
```
lapacke uses lapacke/lapacke.h under /usr/include, `lapack_int` instead of `__CLPK_integer`.
Do the changes accordingly, the tests give identical results:
```
Ran 149 tests in 23.709s

FAILED (failures=1, errors=37)
======================================================================
EXODUSII Tests Run: 123, Failed 0
```
### Use MKL instead of Atlas
Change:
`setup.py`:
```
include_dirs = [
    ibpath,
    ibpath + '/..',
    lbpath + '/numpy/core/include',
    lbpath + '/numpy/numarray',
    'lib/pivlib/exodusII',
    #'/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers',
    '/opt/intel/compilers_and_libraries_2020.0.166/linux/mkl/include'
]

```
In `lib/spivet/pivlib/pivlibc.c`:
```
#include <mkl_lapack.h>
#include <mkl_cblas.h>


//#include <lapacke/lapacke.h>
//#include <cblas.h>

typedef MKL_INT lpk_int;
//typedef __CLPK_integer lpk_int;

```
In `lib/spivet/flolib/floftlec.c`:
```
#include <mkl_lapack.h>

//#include <clapack.h>
//#include <lapacke/lapacke.h>

typedef MKL_INT lpk_int;
//typedef lapack_int lpk_int;
//typedef __CLPK_integer lpk_int;
```
build, install,test:
```
Ran 149 tests in 23.353s

FAILED (failures=1, errors=37)
======================================================================
EXODUSII Tests Run: 123, Failed 0

```
It is probably because the deprecated spivet code it self.

## Attempt to solve errors in the code one by one
###  Most common type:
```
File "/home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/spivet/pivlib/pivdata.py", line 1116, in __setitem__
    if ( self.m_eshape == None ):
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

```

Fix: change `==` to `is`, `!=` to `is not`
[https://stackoverflow.com/questions/23086383/how-to-test-nonetype-in-python/23086405](https://stackoverflow.com/questions/23086383/how-to-test-nonetype-in-python/23086405)

related lines:
- lib/spivet/pivlib/pivdata.py: 797,1116,1317
- lib/spivet/pivlib/pivir.py: 114, 140, 226, 498
- lib/spivet/pivlib/pivsim.py: 510
- A lot in lib/spivet/:  `pivlib/pivof.py  tlclib/tlcutil.py steputil.py steps.py pivlib/pivutil.py pivlib/pivsim.py pivlib/pivdata.py pivlib/pivpickle.py flolib/flotrace.py  pivlib/pivpg.py pivlib/pivpgcal.py flolib/flotrace.py flolib/floftle.py pivlib/pivpost.py pivlib/pivir.py tlclib/tlctf.py`(== None, != None replace using %s)
Now errors reduce to 4!
### Second type:

```File "/home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/spivet/pivlib/pivsim.py", line 471, in getVTKRep
    po.GetOutput().Update()
AttributeError: 'vtkCommonDataModelPython.vtkPolyData' object has no attribute 'Update'

  File "/home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/spivet/pivlib/pivsim.py", line 840, in dump2vtk
    dw.SetInput(rpd)
AttributeError: 'vtkIOLegacyPython.vtkPolyDataWriter' object has no attribute 'SetInput'

  File "/home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/spivet/pivlib/pivsim.py", line 504, in getVTKRep
    apd.AddInput(pd)
AttributeError: 'vtkFiltersCorePython.vtkAppendPolyData' object has no attribute 'AddInput'

```
This is due to incompatible updates of VTK6:
[https://vtk.org/Wiki/VTK/VTK_6_Migration/Replacement_of_SetInput#Replacement_of_SetInput.28.29_with_SetInputData.28.29_and_SetInputConnection.28.29](https://vtk.org/Wiki/VTK/VTK_6_Migration/Replacement_of_SetInput#Replacement_of_SetInput.28.29_with_SetInputData.28.29_and_SetInputConnection.28.29)
[https://stackoverflow.com/questions/47776121/vtk-changes-for-getimage-and-update](https://stackoverflow.com/questions/47776121/vtk-changes-for-getimage-and-update)

Fix:
- lib/spivet/pivlib/pivsim.py: 
313,322,471,480,587,596:    po.GetOutput().Update()-->po.Update()
349,500,510,1441,1446? dw.SetInputData(rpd) --->  dw.SetInputData(rpd)
353,354,504,505,600? apd.AddInput ---> apd.AddInputData
### More Errors & Warnings
Warning: will check in the end
```
testCalibration (pivpg_test.test_pivpg) ... /home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/scipy/optimize/minpack.py:447: RuntimeWarning: Number of calls to function has reached maxfev = 1000.
  warnings.warn(errors[info][0], RuntimeWarning)
ok
```
Error:
```
testOctreeVTKINode (pivsim_test.testSimOctree) ... I am here: test-output/octree
ERROR
testoctreeVTKLNode (pivsim_test.testSimOctree) ... ERROR
testOutput (pivsim_test.testTraceRectangle) ... ERROR
testOutput (pivsim_test.testTraceCylinder) ... ERROR
testImagesMatch (pivsim_test.testTraceBitmapRectangle) ... Segmentation fault (core dumped)
```
The summary report is not available due to segmentation fault.
Comment out line 1124 in `pivsim_test.py`
```
suite.addTest( unittest.makeSuite( testTraceBitmapRectangle  ) )
```
rerun the tests

Currently passed tests:
```
    suite.addTest( unittest.makeSuite( testSimObjectSetup        ) )
    suite.addTest( unittest.makeSuite( testSimObjectTransforms   ) )
    suite.addTest( unittest.makeSuite( testSimRay                ) )
    suite.addTest( unittest.makeSuite( testSimCylindricalSurface ) )
    suite.addTest( unittest.makeSuite( testSimRectPlanarSurface  ) )
    suite.addTest( unittest.makeSuite( testSimCircPlanarSurface  ) )
    suite.addTest( unittest.makeSuite( testSimLeafNode           ) )
    //suite.addTest( unittest.makeSuite( testSimOctree             ) )
    suite.addTest( unittest.makeSuite( testSimRefractiveObject   ) )
    suite.addTest( unittest.makeSuite( testSimCamera             ) )
    suite.addTest( unittest.makeSuite( testSimLight              ) )
    suite.addTest( unittest.makeSuite( testSimEnv                ) )
    //suite.addTest( unittest.makeSuite( testTraceRectangle        ) )
    //suite.addTest( unittest.makeSuite( testTraceCylinder         ) )
    //suite.addTest( unittest.makeSuite( testTraceBitmapRectangle  ) )
    //suite.addTest( unittest.makeSuite( testTraceSurfaceRender    ) )
```
Currently failed tests:
```
    suite.addTest( unittest.makeSuite( testSimOctree             ) )
    suite.addTest( unittest.makeSuite( testTraceRectangle        ) )
    suite.addTest( unittest.makeSuite( testTraceCylinder         ) )
    suite.addTest( unittest.makeSuite( testTraceBitmapRectangle  ) )
    suite.addTest( unittest.makeSuite( testTraceSurfaceRender    ) )
```
They are all related to vtk (dump2vtk).
```
======================================================================
ERROR: testOctreeVTKINode (pivsim_test.testSimOctree)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-modified/tests/pivsim_test.py", line 583, in testOctreeVTKINode
    d = abs(tpts -kpts) < self.eps
ValueError: operands could not be broadcast together with shapes (0,3) (600,3) 

======================================================================
ERROR: testoctreeVTKLNode (pivsim_test.testSimOctree)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-modified/tests/pivsim_test.py", line 595, in testoctreeVTKLNode
    d = abs(tpts -kpts) < self.eps
ValueError: operands could not be broadcast together with shapes (0,3) (1890,3) 

======================================================================
ERROR: testOutput (pivsim_test.testTraceRectangle)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-modified/tests/pivsim_test.py", line 925, in testOutput
    d = abs(tpts -kpts) < self.eps
ValueError: operands could not be broadcast together with shapes (0,3) (48,3) 

======================================================================
ERROR: testOutput (pivsim_test.testTraceCylinder)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-modified/tests/pivsim_test.py", line 976, in testOutput
    d = abs(tpts -kpts) < self.eps
ValueError: operands could not be broadcast together with shapes (15,3) (12,3) 

======================================================================
FAIL: testImagesMatch (pivsim_test.testTraceSurfaceRender)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-modified/tests/pivsim_test.py", line 1105, in testImagesMatch
    self.assertEqual(tm.hexdigest(),km.hexdigest())
AssertionError: '57cfefd1c6ac988f11faf064e7e3939f' != '784a96839ef2d92fefce8cea9f807733'



testImagesMatch (pivsim_test.testTraceBitmapRectangle) ... Segmentation fault (core dumped)
```

The segmentation fault is due to python path in the C extension lib/pivlib/pivsimc.c somehow does not include the path we need.
After line 1602, add:
```
PyRun_SimpleString("import sys;sys.path.append('/home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/spivet/pivlib')");
  mod     = PyImport_ImportModule("pivutil");
  if (mod==NULL){
        printf("%s\n","mod null!");
        PyErr_Print();
        fflush(stdout);}
```
`PyErr_Print` will report `no module named pivutil` if we do not add that path.
We will replace `/home/xbao/.conda/envs/py27_new/lib/python2.7/site-packages/` to 
`ibpath = distutils.sysconfig.get_python_inc()` later
After this correction the error turns into FAIL:
```
FAIL: testImagesMatch (pivsim_test.testTraceBitmapRectangle)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-mkl/tests/pivsim_test.py", line 1047, in testImagesMatch
    self.assertEqual(tm.hexdigest(),km.hexdigest())
AssertionError: '9b874624537e026d7dc1394e283ad3eb' != '62e3ea406f87a374587082f0cfc5b02f'
```
The rest of the errors continue to show up even if we modified the code to use the updated api of vtk6 (and later version). We do not know the intermediate output, and the desperate debugging part can be really time-consuming. It is therefore a good idea to try older version of vtk instead. 
## Attempt to install vtk 5.10
clone the conda env py27_piv and uninstall vtk7 from conda source.
```
conda create --clone py27_new --name py27_bak
conda uninstall vtk
```
get vtk5.10 from gitlab
https://gitlab.kitware.com/vtk/vtk/-/archive/v5.10.1/vtk-v5.10.1.zip
8.3.1 is too new, will report lots of error like 
```
CMake Error at CMake/vtkCompilerExtras.cmake:40 (if):   if given arguments:      "gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)    Copyright (C) 2018 Free Software Foundation, Inc.    This is free software" " see the source for copying conditions.  There is   NO    warranty" " not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR   PURPOSE." "VERSION_GREATER" "4.2.0" "AND" "BUILD_SHARED_LIBS" "AND"   "HAVE_GCC_VISIBILITY" "AND" "VTK_USE_GCC_VISIBILITY" "AND" "NOT" "MINGW"   "AND" "NOT" "CYGWIN"    Unknown arguments specified Call Stack (most recent call first):   CMakeLists.txt:73 (INCLUDE)
```
switch to gcc 4.8.5
``` 
module load gcc/8.3.1
module unload gcc/8.3.1
```
cmake(or edit CMakeList.txt):
[https://stackoverflow.com/questions/28761702/getting-error-glintptr-has-not-been-declared-when-building-vtk-on-linux](https://stackoverflow.com/questions/28761702/getting-error-glintptr-has-not-been-declared-when-building-vtk-on-linux)
[https://vtk.org/Wiki/VTK/Configure_and_Build](https://vtk.org/Wiki/VTK/Configure_and_Build)
[https://unix.stackexchange.com/questions/306682/how-to-install-vtk-with-python-wrapper-on-red-hat-enterprise-linux-rhel](https://unix.stackexchange.com/questions/306682/how-to-install-vtk-with-python-wrapper-on-red-hat-enterprise-linux-rhel)
[https://cmake.org/cmake/help/v3.0/module/FindPythonLibs.html](https://cmake.org/cmake/help/v3.0/module/FindPythonLibs.html)
[https://groups.google.com/forum/#!msg/vmtk-users/3jrKrW7qWZA/ejWDwuotOGAJ](https://groups.google.com/forum/#!msg/vmtk-users/3jrKrW7qWZA/ejWDwuotOGAJ)
[http://vtk.1045678.n5.nabble.com/Compiling-VTK-Python-from-source-td5746733.html](http://vtk.1045678.n5.nabble.com/Compiling-VTK-Python-from-source-td5746733.html)


```
cmake -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DVTK_WRAP_PYTHON=ON -DCMAKE_C_FLAGS=-DGLX_GLXEXT_LEGACY -DCMAKE_CXX_FLAGS=-DGLX_GLXEXT_LEGACY -DCMAKE_INSTALL_PREFIX=/home/app/vtk5 -Wno-dev \
-DPYTHON_INCLUDE_DIR=/home/xbao/.conda/envs/py27_bak/include/python2.7 \
-DPYTHON_LIBRARY=/home/xbao/.conda/envs/py27_bak/lib/libpython2.7.so \
../vtk-v5.10.1
```
Then
```
make -j 56
make install
```

some error will show up related to `__cxa_throw_bad_array_new_length`, we need gcc 4.9 or newer :
[https://stackoverflow.com/questions/29230777/program-linking-fails-when-using-custom-built-gcc](https://stackoverflow.com/questions/29230777/program-linking-fails-when-using-custom-built-gcc)

#### As root, install gcc 4.9.2 with gcc 4.8.5 [due-to-c11-errors, we cannot use newer gcc ] (https://stackoverflow.com/questions/41204632/unable-to-build-gcc-due-to-c11-errors)

[https://gist.github.com/craigminihan/b23c06afd9073ec32e0c](https://gist.github.com/craigminihan/b23c06afd9073ec32e0c)
```
sudo yum install libmpc-devel mpfr-devel gmp-devel
#in /usr/syssoft/gcc4.9
curl ftp://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-4.9.2/gcc-4.9.2.tar.bz2 -O
tar xvfj gcc-4.9.2.tar.bz2
cd gcc-4.9.2
md build
../configure  --enable-languages=c,c++,fortran --prefix=/usr/local/gcc4.9.2 --disable-multilib
make -j 56
make install
```
Write a module file like what we did for gcc 8.3.1 in `/etc/modulefiles/gcc/4.9.2`
```
#%Module 1.0
#
#  gcc4.9.2 module built from source with system python3:
#
unsetenv COMP_WORDBREAKS;
prepend-path PATH {/usr/local/gcc4.9.2/bin};
prepend-path MANPATH {/usr/local/gcc4.9.2/share/man};
#append-path PERL5LIB {/opt/rh/devtoolset-8/root//usr/lib64/perl5/vendor_perl};
#append-path PERL5LIB {/opt/rh/devtoolset-8/root/usr/lib/perl5};
#append-path PERL5LIB {/opt/rh/devtoolset-8/root//usr/share/perl5/vendor_perl};
prepend-path LD_LIBRARY_PATH {/usr/local/gcc4.9.2/lib};
prepend-path LD_LIBRARY_PATH {/usr/local/gcc4.9.2/lib64};
#prepend-path LD_LIBRARY_PATH {/opt/rh/devtoolset-8/root/usr/lib/dyninst};
#prepend-path LD_LIBRARY_PATH {/opt/rh/devtoolset-8/root/usr/lib64/dyninst};
prepend-path LD_LIBRARY_PATH {/usr/local/gcc4.9.2/lib};
prepend-path LD_LIBRARY_PATH {/usr/local/gcc4.9.2/lib64};
#prepend-path PKG_CONFIG_PATH {/usr/local/gcc4.9.2/lib64/pkgconfig};
prepend-path INFOPATH {/usr/local/gcc4.9.2/share/info};
#prepend-path PYTHONPATH {/opt/rh/devtoolset-8/root/usr/lib/python2.7/site-packages};
#prepend-path PYTHONPATH {/opt/rh/devtoolset-8/root/usr/lib64/python2.7/site-packages};
```
Test:
```
$ gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/local/gcc4.9.2/libexec/gcc/x86_64-unknown-linux-gnu/4.9.2/lto-wrapper
Target: x86_64-unknown-linux-gnu
Configured with: ../configure --enable-languages=c,c++,fortran --prefix=/usr/local/gcc4.9.2 --disable-multilib
Thread model: posix
gcc version 4.9.2 (GCC)
```
When cmake and make the same ABI error appears.(ABI1.3.9 is required, not present in libstdc++.so.6.0.19 )
Failed attempt
`export CXXFLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'`
`ln -s libstdc++.so.6.0.26 libstdc++.so.6` for `/usr/lib64` or `gcc4.9.2/lib64`

Now install gcc5.5.0 similar to 4.9.2 before.
In its `/usr/local/gcc5.5.0/lib64`:
```
strings libstdc++.so.6|grep CXXABI
CXXABI_1.3
CXXABI_1.3.1
CXXABI_1.3.2
CXXABI_1.3.3
CXXABI_1.3.4
CXXABI_1.3.5
CXXABI_1.3.6
CXXABI_1.3.7
CXXABI_1.3.8
CXXABI_1.3.9
CXXABI_TM_1
CXXABI_FLOAT128
CXXABI_1.3
CXXABI_1.3.2
CXXABI_1.3.6
CXXABI_FLOAT128
CXXABI_1.3.9
CXXABI_1.3.1
CXXABI_1.3.5
CXXABI_1.3.8
CXXABI_1.3.4
CXXABI_TM_1
CXXABI_1.3.7
CXXABI_1.3.3
```
Now
Now in `vtk-build` dir, under py27_bak:
(Note that `CMAKE_INSTALL_PREFIX` should contain bin/python, not a path like `/home/app/vtk5`. You may need to add vtk path to path manually if the prefix path is not correct. [http://blog.sina.com.cn/s/blog_62746020010125e6.html](http://blog.sina.com.cn/s/blog_62746020010125e6.html))
```
module load gcc/5.5.0
cmake -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DVTK_WRAP_PYTHON=ON \
-DCMAKE_C_FLAGS=-DGLX_GLXEXT_LEGACY -DCMAKE_CXX_FLAGS=-DGLX_GLXEXT_LEGACY \
-DCMAKE_INSTALL_PREFIX=/home/xbao/.conda/envs/py27_bak/ -Wno-dev \
-DPYTHON_INCLUDE_DIR=/home/xbao/.conda/envs/py27_bak/include/python2.7 \
-DPYTHON_LIBRARY=/home/xbao/.conda/envs/py27_bak/lib/libpython2.7.so \
../vtk-v5.10.1
make -j 112
make install
```
And it works!
```
#The last-line-output of make
[100%] Built target vtkpython_pyc
```
The cmake version is 3.14.6.
(Use `sudo alternatives --config cmake` to switch!)
Test in python

```
$python
Python 2.7.15 | packaged by conda-forge | (default, Mar  5 2020, 14:56:06)
[GCC 7.3.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import vtk
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/xbao/.conda/envs/py27_bak/lib/python2.7/site-packages/VTK-5.10.1-py2.7.egg/vtk/__init__.py", line 41, in <module>
    from vtkCommonPython import *
ImportError: libvtkCommonPythonD.so.5.10: cannot open shared object file: No such file or directory
```
We need to set `LD_LIBRARY_PATH` to solve this issue. We want vtk5 just for this conda env, follow this [link](https://stackoverflow.com/questions/46826497/conda-set-ld-library-path-for-env-only),  we can find the env dir, and 
```
cd /home/xbao/.conda/envs/py27_bak/
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```
for activate:
```
#!/bin/bash
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/xbao/.conda/envs/py27_bak/lib/vtk-5.10/:${LD_LIBRARY_PATH}
```
for deactivate:
```
#!/bin/bash
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
```
Open a new terminal and enter py_bak env, and verify the vtk can be imported.

it seems that does not work jupyter notebook at least for python2
[https://github.com/Anaconda-Platform/nb_conda_kernels/issues/54](https://github.com/Anaconda-Platform/nb_conda_kernels/issues/54)
[https://stackoverflow.com/questions/37890898/how-to-set-env-variable-in-jupyter-notebook](https://stackoverflow.com/questions/37890898/how-to-set-env-variable-in-jupyter-notebook)
A temporary fix is to add them in jupyter cells
```
import os
os.environ['LD_LIBRARY_PATH']=....
```
or 
[https://github.com/jupyter/notebook/issues/3704](https://github.com/jupyter/notebook/issues/3704)

Or the best way: modify the kernel file:
```
jupyter kernelspec list
cd /home/xbao/.local/share/jupyter/kernels/py27_bak
vi kernel.json
```
Example:
{
 "display_name": "py27_bak",
 "language": "python",
 "argv": [
  "/home/xbao/.conda/envs/py27_bak/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
"env": {"LD_LIBRARY_PATH":"/home/xbao/.conda/envs/py27_bak/lib/vtk-5.10/:${LD_LIBRARY_PATH}"}
}

restart the kernel and `import vtk` now works!


Now use the original pivsim.py (except in line 510 `!=` to `is not`), build, install spivet and rerun the pivsim tests, we can pass more:
```
testOctreeVTKINode (pivsim_test.testSimOctree) ... I am here: test-output/octree
ok
testoctreeVTKLNode (pivsim_test.testSimOctree) ... ok
testCreate (pivsim_test.testSimRefractiveObject) ... ok
testSetOrientation (pivsim_test.testSimRefractiveObject) ... ok
testSetPosition (pivsim_test.testSimRefractiveObject) ... ok
testCenterRayHeading (pivsim_test.testSimCamera) ... ok
testLeftRayHeading (pivsim_test.testSimCamera) ... ok
testRayCount (pivsim_test.testSimCamera) ... ok
testRightRayHeading (pivsim_test.testSimCamera) ... ok
testSource (pivsim_test.testSimCamera) ... ok
testIRayHeading (pivsim_test.testSimLight) ... ok
testIRaySource (pivsim_test.testSimLight) ... ok
testLocalIRayHeading (pivsim_test.testSimLight) ... ok
testLocalIRaySource (pivsim_test.testSimLight) ... ok
testExitingRefraction1 (pivsim_test.testSimEnv) ... ok
testExitingRefraction2 (pivsim_test.testSimEnv) ... ok
testNormalIncidence (pivsim_test.testSimEnv) ... ok
testRefraction1 (pivsim_test.testSimEnv) ... ok
testRefraction2 (pivsim_test.testSimEnv) ... ok
testTotalInternalReflection (pivsim_test.testSimEnv) ... ok
testOutput (pivsim_test.testTraceRectangle) ... ok
testOutput (pivsim_test.testTraceCylinder) ... ERROR
testImagesMatch (pivsim_test.testTraceBitmapRectangle) ... FAIL
testImagesMatch (pivsim_test.testTraceSurfaceRender) ... FAIL

======================================================================
ERROR: testOutput (pivsim_test.testTraceCylinder)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-mkl/tests/pivsim_test.py", line 976, in testOutput
    d = abs(tpts -kpts) < self.eps
ValueError: operands could not be broadcast together with shapes (15,3) (12,3)

======================================================================
FAIL: testImagesMatch (pivsim_test.testTraceBitmapRectangle)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-mkl/tests/pivsim_test.py", line 1047, in testImagesMatch
    self.assertEqual(tm.hexdigest(),km.hexdigest())
AssertionError: '9b874624537e026d7dc1394e283ad3eb' != '62e3ea406f87a374587082f0cfc5b02f'

======================================================================
FAIL: testImagesMatch (pivsim_test.testTraceSurfaceRender)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-mkl/tests/pivsim_test.py", line 1105, in testImagesMatch
    self.assertEqual(tm.hexdigest(),km.hexdigest())
AssertionError: '57cfefd1c6ac988f11faf064e7e3939f' != '784a96839ef2d92fefce8cea9f807733'

----------------------------------------------------------------------
Ran 58 tests in 1.844s

FAILED (failures=2, errors=1)
```

Try use vtk5.2 instead, failed to compile due to too many errors.(under different cmake, gcc, mpicc/gcc, etc.)
## Go back to tests
Now go back to tests/test-output, actually the output figures for testImagesMatch (`surface-render.png` `surface-img-render.png`) look exactly the same as the benchmark results. We can verify this using
[https://stackoverflow.com/questions/5132749/diff-an-image-using-imagemagick](https://stackoverflow.com/questions/5132749/diff-an-image-using-imagemagick)

```
sudo yum install -y imagemagick
compare surface-img-render.png ../data/surface-img-render-known.png -compose src diff.png
compare surface-render.png ../data/surface-render-known.png -compose src diff2.png
```
The two diff.png are transparent.
We can read the images as binary files using other tools:
[cmp](https://blog.csdn.net/ly890700/article/details/52796766)
[vim&xxd](https://blog.csdn.net/hansel/article/details/5097262?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)

```
cmp -l surface-render.png ../data/surface-render-known.png
 37  53 313
 44 143 142
 60 314  14
 61 104   0
 62 254   0
 63 201   0
 64 304 377
 65 202 377
 66 121 142
 67   3  42
 68 107 326
 69  15 100
 70  34 142
 71  65   1
 72 160   0
 73 324   0
 74 300   0
 75 121 377
 76   3 377
 77 207 242
 78 212 272
 81  50   0
 82 376   0
 84 301 377
 85  13 377
 86 233 242
 87 347 272
 88 113 201
 93 111 377
 94 105 377
 95 116 242
 96 104 272
 97 256 201
 98 102   0
 99 140   0
100 202   0
cmp: EOF on surface-render.png
```
```
vim -b surface-render.png
#in vim
:%!xxd
#you will see
0000000: 8950 4e47 0d0a 1a0a 0000 000d 4948 4452  .PNG........IHDR
0000010: 0000 0050 0000 0014 0800 0000 003f b3e5  ...P.........?..
0000020: 0b00 0000 2b49 4441 5478 9c63 6420 1230  ....+IDATx.cd .0
0000030: b240 0033 0b3a 0389 c5c2 c2cc 44ac 81c4  .@.3.:......D...
0000040: 8251 0347 0d1c 3570 d4c0 5103 878a 8100  .Q.G..5p..Q.....
0000050: 28fe 00c1 0b9b e74b 0000 0000 4945 4e44  (......K....IEND
0000060: ae42 6082                                .B`.
```
```
vim -b ../data/surface-render-known.png
#in vim
:%!xxd
#you will see
0000000: 8950 4e47 0d0a 1a0a 0000 000d 4948 4452  .PNG........IHDR
0000010: 0000 0050 0000 0014 0800 0000 003f b3e5  ...P.........?..
0000020: 0b00 0000 cb49 4441 5478 9c62 6420 1230  .....IDATx.bd .0
0000030: b240 0033 0b3a 0389 c5c2 c20c 0000 00ff  .@.3.:..........
0000040: ff62 22d6 4062 0100 0000 ffff a2ba 8100  .b".@b..........
0000050: 0000 00ff ffa2 ba81 0000 0000 ffff a2ba  ................
0000060: 8100 0000 00ff ffa2 ba81 0000 0000 ffff  ................
0000070: a2ba 8100 0000 00ff ffa2 ba81 0000 0000  ................
0000080: ffff a2ba 8100 0000 00ff ffa2 ba81 0000  ................
0000090: 0000 ffff a2ba 8100 0000 00ff ffa2 ba81  ................
00000a0: 0000 0000 ffff a2ba 8100 0000 00ff ffa2  ................
00000b0: ba81 0000 0000 ffff a2ba 8100 0000 00ff  ................
00000c0: ffa2 ba81 0000 0000 ffff a2ba 8100 0000  ................
00000d0: 00ff ffa2 ba81 0000 0000 ffff a2ba 8100  ................
00000e0: 0000 00ff ffa2 ba81 0000 0000 ffff 0300  ................
00000f0: 28fe 00c1 10ff 986d 0000 0000 4945 4e44  (......m....IEND
0000100: ae42 6082                                .B`.
```
However, it is convenient to convert output and known to another format and compare:
```
convert surface-render.png surface-render.jpg
convert ../data/surface-render-known.png surface-render-known.jpg
cmp surface-render-known.jpg surface-render.jpg
convert surface-img-render.png surface-img-render.jpg
convert ../data/surface-img-render-known.png surface-img-render-known.jpg
cmp surface-img-render.jpg surface-img-render-known.jpg
```
No output, means they are exactly the same in terms of data. We have reason to believe this is purely because we replaced `PIL` with `pillow`.
## The last error
```
ERROR: testOutput (pivsim_test.testTraceCylinder)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/xbao/SPIVET-mkl/tests/pivsim_test.py", line 976, in testOutput
    d = abs(tpts -kpts) < self.eps
ValueError: operands could not be broadcast together with shapes (15,3) (12,3)
```
However, we can pass the lnode test in `testTraceCylinder`, means the cylinder has been correctly created. 
It is hard to debug using these numbers in vtk file only, we need to understand the physical image.
Load the lnode and ray vtk files to paraview:
![image.png](https://upload-images.jianshu.io/upload_images/5300880-4b43a3c4463c8bf6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Use Wireframe for lnode and surface for ray, we can see that the rays are not refracted at the second time:
![image.png](https://upload-images.jianshu.io/upload_images/5300880-3cc7917f59eed77b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Set breakpoints and we can locate this issue.
Line `960` in `tests/pivsim_test.py`, when we call `se.image()`, spivet starts to trace the rays and form an image at the camera. 
This relates to line `848` in `lib/spivet/pivlib/pivsim.py`, the method `image` of class `SimEnv`. We then notice line 872 `pivsimc.SimEnv_image(self,rays,chnli,maxrcnt)`, this calls the function `SimEnv_image` in the C extension `lib/spivet/pivlib/pivsimc.c`. 
Line `2495` in `pivsimc.c` finds the intersection between the ray and the object of leaf node, i.e. the cylinder in this case. `SimRefractiveObject_intersect` at line `2068` find the intersection for all the objects in the scene, and also determine the next intersection using the nearest distance `t` between the last ray source point and intersections for surfaces of all objects. Line `2101` will calculate the function that finds intersections for a particular type of surface, and we care about `SimCylindricalSurf_intersect` at line 1118 for cylindrical surface. It turns out for the second refraction, 
```
ptc=0,lsource=-0.00,-2.12,2.12,lhead=0.00,0.98,-0.18,lray->points=-0.00,-2.12,2.12
phv=0.98,-0.18,phs=1.000000
i=0,ptv[i]=8.88178e-16,phv[0]=9.84211e-01
ic=-2.121320,plv[0]=-2.121320,lhead[1]=0.984211,t=0.000000
i=1,ptv[i]=4.92660e+00,phv[0]=9.84211e-01
ic=2.727493,plv[0]=-2.121320,lhead[1]=0.984211,t=4.926600
this 3d t=0.00000
this 3d t=4.92660
```
so the zero distance `t` leads to the issue. We can either add
```
    if (t < PIVSIMC_DEPS) {
      continue;
    }
```
at line `1189`, or better to modify line `1179-1181` from:
```  
for ( i = 0; i < 2; i++ ) {
    if ( ptv[i] < 0. )
      continue;
```
to
```
  for ( i = 0; i < 2; i++ ) {
    if ( ptv[i] < PIVSIMC_DEPS )
      continue;
```
Now rebuild and install, run the test:
```
SimCylindricalSurf_intersect!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ptc=0,lsource=-0.00,-2.12,2.12,lhead=0.00,0.98,-0.18,lray->points=-0.00,-2.12,2.12
phv=0.98,-0.18,phs=1.000000
i=1,ptv[i]=4.92660e+00,phv[0]=9.84211e-01
ic=2.727493,plv[0]=-2.121320,lhead[1]=0.984211,t=4.926600
this 3d t=4.92660
......
testOutput (pivsim_test.testTraceCylinder) ... STARTING: SimEnv.image
 | CAM 0
 | Initializing rays.
 | Tracing ...
 | EXITING: SimEnv.image
ok
```
![image.png](https://upload-images.jianshu.io/upload_images/5300880-7692d25616382e43.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Comparison:
![image.png](https://upload-images.jianshu.io/upload_images/5300880-c1e4465092c92268.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Success!

-------------------------------------------------------------------------------------------------------------------
# To get intermediate carriage (frames for loop_plane_worker):
In `steps.py`:(version saved as crg_steps.py)
- in `loop_plane.excute`:skip all the lines after `tpd    = carriage['pivdata']`
- in `_loop_epoch_worker`: comment the redirect to stdout/stderr, save plane carriage as gcrg, return gcrg in the end
- in 'loop_epoch': add temporary carriage external port `self.m_crg` in `__init__`, unpack `erslt[2]` to `m_crg` in `execute`

# To rotate existing calibration upside down:
At the end of `dewarpimg`:
```
simgchnls[cam][frm][c] = timg
#Xiyuan:rotate cal
if (self.m_config.has_key('rotate_cal')):
                                if (self.m_config['rotate_cal']):
                                        simgchnls[cam][frm][c] = rot90(timg,2)

```

# To make `mkflatplotmm-anim.py` work:
add `from flotrace import mptassmblr` in `flolib/__init__.py`

# Problems during generating calibration 
`pivlib/pivpgcal.py`
## Float cannot be array index in python 2.7
Need to force related variable as int in both `pivlib/pivpgcal.py` and `pivlib/pivutil.py`
## Sometimes cross correlation method finds no intersection 
(`rv` empty)
add `EMPTY rv` case
## also added some debug images
--------------------------------------------------------------------------------------------------------------------
# Speed up SPIVET
## parallel in `ofcomp` failed due to GIL(only 1 CPU is runing)
failed version saved as `fail_parallel_pivof.py`
## parallel epoch, like the NetWorkSpace
We don't want to use NWS because it was last updated in 2007.
Basically `multiprocessing` is good enough. Both 'apply_async' +`Pool` and 'Process' are implemented. The `call_back` feature somehow will fail silently.
Expected processing time for 43 planes x 56 epoch: <2 hours
Till this point, no external module is required.
