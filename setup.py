import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

py_inc = sysconfig.get_paths().get("include")
py_platinc = sysconfig.get_paths().get("platinclude")

inc_dirs = [
    "pyface/components/face_depth/Sim3DR",
    np.get_include(),
]
# 有些 runner 只放其中一個，把能拿到的全加進去
for p in (py_inc, py_platinc):
    if p:
        inc_dirs.append(p)

ext_modules = cythonize(
    [
        Extension(
            "pyface.components.face_depth.Sim3DR.sim3dr_cython",
            sources=[
                "pyface/components/face_depth/Sim3DR/rasterize.pyx",
                "pyface/components/face_depth/Sim3DR/rasterize_kernel.cpp",
            ],
            include_dirs=inc_dirs,
            language="c++",
            extra_compile_args=["-std=c++11"],
        )
    ],
    language_level="3",
)
setup(ext_modules=ext_modules)
