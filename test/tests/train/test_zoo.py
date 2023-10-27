from utils import run_training, high_score_test

#
# MPE tests
#
def test_mpe_simple_tag_mpi(num_ranks):
    num_timesteps = 200000
    passing_scores = {"adversary_1" : 100.0}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mpe_simple_tag.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mpi mpe simple tag',
        'mpe_simple_tag.py', 10, passing_scores)

def test_mat_mpe_simple_tag_discrete_mpi(num_ranks):
    num_timesteps = 200000
    passing_scores = {"adversary" : 300.0}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mat_mpe_simple_tag_discrete.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mat mpi mpe simple tag discrete',
        'mat_mpe_simple_tag_discrete.py', 10, passing_scores)

def test_mat_mpe_simple_tag_continuous_mpi(num_ranks):
    num_timesteps = 200000
    passing_scores = {"adversary" : 300.0}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mat_mpe_simple_tag_continuous.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mat mpi mpe simple tag continuous',
        'mat_mpe_simple_tag_continuous.py', 10, passing_scores)
