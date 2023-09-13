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
    'gymnasium',
    'gymnasium[mujoco]',
    'pillow',
    'rware',
    'matplotlib',
    'opencv-python',
    'pygame',
    'box2d-py',
    'numpy',
    'dill',
    'mpi4py',
    'moviepy',
    'torch>=1.10.2,<2.0',
    'swig',
    'pettingzoo==1.23',
    'pymunk',
    'packaging',
    # FIXME: abmarl currently causes prblems in github's
    # CI, so I'm commenting it out until this is fixed.
    #'abmarl',
]

setup(name             = "ppo_and_friends",
      version          = version,
      description      = "Proximal Policy Optimization and friends",
      author           = "Alister Maguire",
      license          = "MIT",
      package_dir      = package_dirs,
      packages         = list(package_dirs.keys()),
      package_data     = {"" : ["environments/abmarl/envs/maze.txt",
                                "environments/abmarl/envs/large_maze.txt"]},
      install_requires = dependencies,
      entry_points     = {
          'console_scripts' : ['ppoaf=ppo_and_friends.ppoaf_cli:cli']
      },
      zip_safe         = False)
