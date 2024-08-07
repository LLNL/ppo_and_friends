"""
    Housing for EnvRunner class decorators.
"""

def ppoaf_runner(runner):
    """
    Add a _ppoaf_tag to a decorated class.
    """
    runner._ppoaf_runner_tag = "ppo-af-env-runner"
    return runner
