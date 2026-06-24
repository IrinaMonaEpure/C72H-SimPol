import os
import shutil
from pathlib import Path

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize


def ensure_msvc_on_path() -> None:
    """Make cl.exe discoverable when setuptools found MSVC metadata only partly."""
    if os.name != "nt" or shutil.which("cl"):
        return

    try:
        import setuptools._distutils._msvccompiler as msvccompiler
    except Exception:
        return

    _get_vc_env = msvccompiler._get_vc_env
    vc_env = _get_vc_env("x64")
    for key in ("include", "lib", "libpath"):
        if key in vc_env:
            os.environ[key.upper()] = vc_env[key]

    path_parts = []
    tools_dir = vc_env.get("vctoolsinstalldir")
    if tools_dir:
        path_parts.append(str(Path(tools_dir) / "bin" / "Hostx64" / "x64"))
    sdk_bin = vc_env.get("windowssdkverbinpath") or vc_env.get("windowssdkbinpath")
    if sdk_bin:
        path_parts.append(str(Path(sdk_bin) / "x64"))
    if "path" in vc_env:
        path_parts.append(vc_env["path"])
    path_parts.append(os.environ.get("PATH", ""))
    merged_path = os.pathsep.join(part for part in path_parts if part)
    os.environ["PATH"] = merged_path
    os.environ["Path"] = merged_path
    os.environ["path"] = merged_path

    def patched_get_vc_env(plat_spec):
        env = _get_vc_env(plat_spec)
        env["path"] = merged_path
        env["include"] = os.environ.get("INCLUDE", env.get("include", ""))
        env["lib"] = os.environ.get("LIB", env.get("lib", ""))
        env["libpath"] = os.environ.get("LIBPATH", env.get("libpath", ""))
        return env

    msvccompiler._get_vc_env = patched_get_vc_env


ensure_msvc_on_path()


sparse_extension = Extension(
    name="direct_social_belief_sim_cy_sparse",
    sources=["direct_social_belief_sim_cy_sparse.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["/O2", "/GL-"],
    extra_link_args=["/MANIFEST:NO"],
)


setup(
    name="direct-social-belief-sim-cy",
    ext_modules=cythonize(
        [sparse_extension],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
