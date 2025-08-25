import subprocess
import sys
import sysconfig

from setuptools import Extension, setup

if __name__ == "__main__":
    subprocess.call([sys.executable, "-m", "pip", "install", "cython", "numpy"])

    import numpy as np
    from Cython.Build import cythonize

    ext_modules = cythonize([
        Extension(
            "sim3dr_cython",
            sources=[
                "pyface/components/face_depth/Sim3DR/rasterize.pyx",
                "pyface/components/face_depth/Sim3DR/rasterize_kernel.cpp",
            ],
            language="c++",
            include_dirs=[np.get_include(), sysconfig.get_paths()["platinclude"]],
            extra_compile_args=["-std=c++11"],
        )
    ])
    setup(ext_modules=ext_modules)
