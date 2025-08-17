import subprocess
import sys
import sysconfig

from setuptools import Extension, setup


def prepare_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0", "cython"])


def get_sim3dr_ext():
    return [
        Extension(
            "sim3dr_cython",
            sources=[
                "facekit/components/face_depth/Sim3DR/rasterize.pyx",
                "facekit/components/face_depth/Sim3DR/rasterize_kernel.cpp",
            ],
            language="c++",
            include_dirs=[np.get_include(), sysconfig.get_paths()["platinclude"]],
            extra_compile_args=["-std=c++11"],
        )
    ]


if __name__ == "__main__":
    prepare_requirements()
    import numpy as np

    setup(ext_modules=get_sim3dr_ext())
