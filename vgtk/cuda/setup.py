from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="grouping",
    ext_modules=[CppExtension(
        name="grouping",
        sources=["grouping_cuda.cpp"],
    )],
    cmdclass={
        "build_ext": BuildExtension
    }
)
