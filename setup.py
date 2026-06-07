import os
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


def is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


class BuildExt(build_ext):
    c_opts = {
        "msvc": [
            "/O2",
            "/std:c++17",
        ],
        "unix": [
            "-O3",
            "-DNDEBUG",
            "-funroll-loops",
        ],
    }

    l_opts = {
        "msvc": [],
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        system = platform.system().lower()
        machine = platform.machine().lower()

        opts = self.c_opts.get(ct, []).copy()
        link_opts = self.l_opts.get(ct, []).copy()

        if ct == "unix":
            opts += ["-std=c++17", "-fPIC"]

            if system != "darwin":
                opts += ["-flto"]
                link_opts += ["-flto", "-lm"]

            if machine in ("x86_64", "amd64"):
                opts += ["-mpopcnt"]
                print("** Building LicketySPLIT with x86 POPCNT support")
            elif machine in ("arm64", "aarch64"):
                print("** Building LicketySPLIT on ARM64; skipping x86 -mpopcnt")
            else:
                print(f"** Building LicketySPLIT on unknown Unix arch {machine}; skipping popcount-specific flags")

        elif ct == "msvc":
            print("** Building LicketySPLIT with MSVC safe flags")

        aggressive = is_truthy_env("AGGRESSIVE")

        if aggressive and ct == "unix":
            if machine in ("x86_64", "amd64"):
                opts += [
                    "-mbmi",
                    "-mbmi2",
                    "-mavx2",
                ]
                print("** Building LicketySPLIT with additional aggressive x86 flags")
            elif machine in ("arm64", "aarch64"):
                print("** AGGRESSIVE requested on ARM64; no extra portable flags added")
        elif aggressive and ct != "unix":
            print("** AGGRESSIVE requested on non-Unix compiler; using safe flags")

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "licketysplit._core",
        sources=[
            "src/licketysplit/_core.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/licketysplit/cpp",
        ],
        language="c++",
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)