from utils import run_training, high_score_test

def test_mixed_actions_mpi(num_ranks):
    num_timesteps  = 30000
    passing_scores = {}

    for i in range(10):
        passing_scores[f"agent_{i}"] = -2000

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mixed_action_test_env.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mpi mixed actions',
        'mixed_action_test_env.py', 4, passing_scores)
