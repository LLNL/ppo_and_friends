from utils import run_training, high_score_test
import pytest

@pytest.mark.skip(reason="Need Abmarl's update to gymansium")
def test_abmarl_maze_mpi(num_ranks):
    num_timesteps = 50000
    passing_scores = {"navigator" : 0.0}

    run_training(
        baseline_runner = 'abmarl_maze.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '--test-explore')

    high_score_test('mpi abmarl maze',
        'abmarl_maze.py', 10, passing_scores)


#FIXME: find good settings for this
@pytest.mark.skip(reason="Need Abmarl's update to gymansium")
def test_mat_abmarl_reach_the_target(num_ranks):
    num_timesteps = 50000
    passing_scores = {"runner" : 0.0}

    run_training(
        baseline_runner = 'mat_abmarl_reach_the_target.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '--test-explore')

    high_score_test('mpi abmarl maze',
        'abmarl_maze.py', 10, passing_scores)
