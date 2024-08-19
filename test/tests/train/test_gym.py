from utils import run_training, high_score_test

def test_mat_robot_warehouse_tiny(num_ranks):
    num_timesteps = 10000000
    passing_scores = {"single_agent" : 3.0}

    run_training(
        baseline_type   = 'gym',
        baseline_runner = 'robot_warehouse.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '--grid_size tiny --policy mat')

    high_score_test('mat robot warehouse tiny',
        'robot_warehouse.py', 10, passing_scores)
