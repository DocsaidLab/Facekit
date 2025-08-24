import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

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


# if __name__ == "__main__":
#     prepare_requirements()
#     import numpy as np

#     setup(ext_modules=get_sim3dr_ext())
