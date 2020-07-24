from distutils.core import setup, Extension
import glob
import distutils.sysconfig

# Get location of headers.
ibpath = distutils.sysconfig.get_python_inc()
lbpath = distutils.sysconfig.get_python_lib()

include_dirs = [
    ibpath,
    ibpath + '/..',
    lbpath + '/numpy/core/include',
    lbpath + '/numpy/numarray',
    'lib/pivlib/exodusII',
    #'/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers',
    '/opt/intel/compilers_and_libraries_2020.0.166/linux/mkl/include'
]

# Setup C modules.
ext_modules = []

module = Extension(
    'spivet.pivlib.pivlibc',
    sources=['lib/spivet/pivlib/pivlibc.c'],
    libraries=['lapack','blas'],
    include_dirs=include_dirs
)
ext_modules.append(module)

module = Extension(
    'spivet.pivlib.ex2lib',
    sources=glob.glob('lib/spivet/pivlib/exodusII/*.c'),
    include_dirs=include_dirs,
    libraries=['netcdf']
)
ext_modules.append(module)

module = Extension(
    'spivet.flolib.flotracec',
    sources=['lib/spivet/flolib/flotracec.c'],
    include_dirs=include_dirs
)
ext_modules.append(module)

module = Extension(
    'spivet.flolib.floftlec',
    sources=['lib/spivet/flolib/floftlec.c'],
    libraries=['lapack','blas'],
    include_dirs=include_dirs
)
ext_modules.append(module)

module = Extension(
    'spivet.flolib.flohetc',
    sources=['lib/spivet/flolib/flohetc.c'],
    libraries=['lapack','blas'],
    include_dirs=include_dirs
)
ext_modules.append(module)

module = Extension(
    'spivet.pivlib.pivsimc',
    sources=['lib/spivet/pivlib/pivsimc.c'],
    include_dirs=include_dirs
)
ext_modules.append(module)

module = Extension(
    'spivet.pivlib.pivtpsc',
    sources=['lib/spivet/pivlib/pivtpsc.c'],
    include_dirs=include_dirs
)
ext_modules.append(module)

module = Extension(
    'spivet.tlclib.tlclibc',
    sources=['lib/spivet/tlclib/tlclibc.c'],
    include_dirs=include_dirs
)
ext_modules.append(module)

# Data files.
package_data = {'spivet':['skel/*']}

# Call setup.
setup(
    name = 'SPIVET', 
    version = '1.0', 
    description = 'SPIV Library with Thermochromic Thermometry',
    package_dir = {'': 'lib'},
    ext_modules = ext_modules,
    packages = ['spivet','spivet.pivlib','spivet.tlclib','spivet.flolib'],
    package_data = package_data
)
