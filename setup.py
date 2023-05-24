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
    'mujoco<=2.3.3',
    'pillow',
    'rware',
    #'lbforaging',#FIXME
    'matplotlib',
    'opencv-python',
    'pygame',
    'gymnasium',
    'box2d-py',
    'numpy',
    'dill',
    'mpi4py',
    'moviepy',
    'torch>=1.10.2,<2.0',
    'swig',
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
          'console_scripts' : ['ppoaf-baselines=ppo_and_friends.train_baseline:train_baseline']
      },
      zip_safe         = False)
