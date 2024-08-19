from utils import run_training, high_score_test
import pytest

def test_abmarl_maze_mpi(num_ranks):
    num_timesteps = 50000
    passing_scores = {"navigator" : 0.0}

    run_training(
        baseline_type   = 'abmarl', 
        baseline_runner = 'abmarl_maze.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mpi abmarl maze',
        'abmarl_maze.py', 10, passing_scores)

def test_mat_abmarl_reach_the_target(num_ranks):
    num_timesteps = 50000
    passing_scores = {"runner" : 0.0}

    run_training(
        baseline_type   = 'abmarl', 
        baseline_runner = 'abmarl_reach_the_target.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '--policy mat')

    high_score_test('mat abmarl rtt',
        'abmarl_reach_the_target.py', 10, passing_scores)
