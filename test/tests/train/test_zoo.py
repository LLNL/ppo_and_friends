from utils import run_training, high_score_test

#
# MPE tests
#
def test_mpe_simple_tag_mpi(num_ranks):
    num_timesteps = 250000
    passing_scores = {"adversary" : 100.0}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mpe_simple_tag.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mpi mpe simple tag',
        'mpe_simple_tag.py', 10, passing_scores, options="--policy_tag adversary_best")

def test_mat_mpe_simple_tag_discrete_mpi(num_ranks):
    num_timesteps = 250000
    passing_scores = {"adversary" : 300.0}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mpe_simple_tag.py',
        options         = '--policy mat --continuous_actions 0',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mat mpi mpe simple tag discrete',
        'mpe_simple_tag.py', 10, passing_scores, options="--policy_tag adversary_best")

def test_mat_mpe_simple_tag_continuous_mpi(num_ranks):
    num_timesteps = 300000
    passing_scores = {"adversary" : 300.0}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mpe_simple_tag.py',
        options         = '--policy mat --continuous_actions 1',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('mat mpi mpe simple tag continuous',
        'mpe_simple_tag.py', 10, passing_scores, options="--policy_tag adversary_best")

def test_agent_shared_icm(num_ranks):
    #
    # I don't care about the scores here. I merely want to make
    # sure the agent shared ICM setting doesn't cause any crashes.
    #
    num_timesteps = 10000
    passing_scores = {"agent" : -100}

    run_training(
        baseline_type   = 'pettingzoo',
        baseline_runner = 'mpe_simple_adversary.py',
        options         = '--policy mat --icm 1',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks)

    high_score_test('agent shared ICM',
        'mpe_simple_adversary.py', 1, passing_scores, options="--policy_tag agent_best")
