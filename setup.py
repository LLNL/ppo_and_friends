from setuptools import setup, find_packages

version  = "0.0"
packages = find_packages()

package_dirs = {}
for pkg in packages:
    name               = "ppo_and_friends." + pkg
    directory          = pkg.replace(".", "/")
    package_dirs[name] = directory

package_dirs["ppo_and_friends"] = "."

dependencies = [
    'mujoco-py<2.2,>=2.1',
    'pillow',
    'matplotlib',
    'opencv-python',
    'pygame',
    'gym',
    'box2d-py',
    'numpy',
    'mpi4py',
    'moviepy',
    'torch>=1.10.2',
]

setup(name             = "ppo_and_friends",
      version          = version,
      description      = "Proximal Policy Optimization and friends",
      author           = "Alister Maguire",
      license          = "MIT",
      package_dir      = package_dirs,
      packages         = list(package_dirs.keys()),
      install_requires = dependencies,
      zip_safe         = False)
