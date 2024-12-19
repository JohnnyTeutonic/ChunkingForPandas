from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "chunking_pandas.optimized.cy_chunks",
        ["src/chunking_pandas/optimized/cy_chunks.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    # ... other setup parameters ...
    ext_modules=cythonize(extensions),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'numba>=0.53.0',
        'cython>=0.29.0',
    ],
) 