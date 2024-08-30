from utils import run_training, high_score_test

def test_cart_pole_serial():

    num_timesteps = 70000
    passing_scores = {"single_agent" : 200.}

    run_training('gymnasium', 'cart_pole.py', num_timesteps)
    high_score_test('serial cart pole',
        'cart_pole.py', 10, passing_scores, options="--policy_tag single_agent_best")

def test_cart_pole_mpi(num_ranks):

    num_timesteps = 70000
    passing_scores = {"single_agent" : 200.}

    run_training('gymnasium', 'cart_pole.py', num_timesteps, num_ranks)
    high_score_test('mpi cart pole',
        'cart_pole.py', 10, passing_scores, options="--policy_tag single_agent_best")

def test_cart_pole_multi_envs():

    num_timesteps = 70000
    passing_scores = {"single_agent" : 200.}

    run_training(
        baseline_type   = 'gymnasium',
        baseline_runner = 'cart_pole.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 0,
        options         = '--envs_per_proc 2')

    high_score_test('multi-env cart pole', 
        'cart_pole.py', 10, passing_scores, options="--policy_tag single_agent_best")

def test_cart_pole_multi_envs_mpi(num_ranks):

    num_timesteps = 70000
    passing_scores = {"single_agent" : 200.}

    run_training(
        baseline_type   = 'gymnasium',
        baseline_runner = 'cart_pole.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '--envs_per_proc 2')

    high_score_test('multi-env mpi cart pole', 
        'cart_pole.py', 10, passing_scores, options="--policy_tag single_agent_best")

def test_binary_cart_pole_serial():

    num_timesteps = 70000
    passing_scores = {"single_agent" : 200.}

    run_training(
        baseline_type   = 'gymnasium',
        baseline_runner = 'binary_cart_pole.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 0,
        options         = '')

    high_score_test('binary cart pole', 
        'binary_cart_pole.py', 10, passing_scores, options="--policy_tag single_agent_best")

#
# LunarLander tests
#
def test_lunar_lander_mpi(num_ranks):
    num_timesteps = 500000
    passing_scores = {"single_agent" : 200.}

    run_training(
        baseline_type   = 'gymnasium',
        baseline_runner = 'lunar_lander.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '')

    high_score_test('mpi lunar lander', 
        'lunar_lander.py', 10, passing_scores, options="--policy_tag single_agent_best")

def test_binary_lunar_lander_mpi(num_ranks):
    num_timesteps = 300000
    passing_scores = {"single_agent" : 100.}

    run_training(
        baseline_type   = 'gymnasium',
        baseline_runner = 'binary_lunar_lander.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '')

    high_score_test('binary lunar lander', 
        'binary_lunar_lander.py', 10, passing_scores, options="--policy_tag single_agent_best")

#
# MountainCar tests
#
# FIXME: ICM takes quite a while to train. We might need
# to figure out a better way of testing these.
# On LC, I was able to converge using the default settings,
# but it was VERY sensitive to the random seed.
#def test_mountain_car_icm(num_ranks):
#    # FIXME: this might take a lot longer now without bs clip
#    num_timesteps = 300000
#    passing_scores = {"single_agent" :-199.}
#
#    run_training(
#        baseline_type   = 'gymnasium',
#        baseline_runner = 'mountain_car.py',
#        num_timesteps   = num_timesteps,
#        num_ranks       = num_ranks,
#        options         = '--enable_icm 1 --bs_clip_min -100000000')
#
#    high_score_test('mountain car icm', 
#        'mountain_car.py', 10, passing_scores, options="--policy_tag single_agent_best")
#
#def test_mountain_car_continous_icm(num_ranks):
#    num_timesteps = 300000
#    passing_scores = {"single_agent" :50.}
#
#    run_training(
#        baseline_type   = 'gymnasium',
#        baseline_runner = 'mountain_car.py',
#        num_timesteps   = num_timesteps,
#        num_ranks       = num_ranks,
#        options         = '--enable_icm 1 --continuous_actions 1 --bs_clip_min -100000000')
#
#    high_score_test('mountain car continuous icm',
#        'mountain_car.py', 10, passing_scores, options="--policy_tag single_agent_best")

def test_mountain_car_bs_clip(num_ranks):
    num_timesteps = 300000
    passing_scores = {"single_agent" :-199.}

    run_training(
        baseline_type   = 'gymnasium',
        baseline_runner = 'mountain_car.py',
        num_timesteps   = num_timesteps,
        num_ranks       = num_ranks,
        options         = '--bs_clip_min 0.01')

    high_score_test('mountain car bs clip',
        'mountain_car.py', 10, passing_scores, options="--policy_tag single_agent_best")
