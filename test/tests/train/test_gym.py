from utils import run_training, high_score_test

def run_cart_pole_test(name, num_test_runs=10):

    cmd  = f"ppoaf-baselines "
    cmd += f"CartPole --test "
    cmd += f"--num-test-runs {num_test_runs} "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 200.}

    high_score_test(name, cmd,
        passing_scores, "CartPole")

def test_cart_pole_serial():

    num_timesteps = 70000
    passing_scores = {"agent0" : 200.}

    run_training('cart_pole.py', num_timesteps)
    high_score_test('serial cart pole', 'cart_pole.py', 10, passing_scores)

def test_cart_pole_mpi():

    num_timesteps = 70000
    passing_scores = {"agent0" : 200.}

    run_training('cart_pole.py', num_timesteps, 2)
    high_score_test('mpi cart pole', 'cart_pole.py', 10, passing_scores)

def test_cart_pole_multi_envs():

    num_timesteps = 70000
    passing_scores = {"agent0" : 200.}

    run_training(
        baseline_runner = 'cart_pole.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 0,
        options         = '--env-per-proc 2')

    high_score_test('multi-env cart pole', 'cart_pole.py', 10, passing_scores)

def test_cart_pole_multi_envs_mpi():

    num_timesteps = 70000
    passing_scores = {"agent0" : 200.}

    run_training(
        baseline_runner = 'cart_pole.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '--env-per-proc 2')

    high_score_test('multi-env mpi cart pole', 'cart_pole.py', 10, passing_scores)

def test_binary_cart_pole_serial():

    num_timesteps = 70000
    passing_scores = {"agent0" : 200.}

    run_training(
        baseline_runner = 'binary_cart_pole.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 0,
        options         = '')

    high_score_test('binary cart pole',
        'binary_cart_pole.py', 10, passing_scores)

#
# LunarLander tests
#
def test_lunar_lander_mpi():
    num_timesteps = 500000
    passing_scores = {"agent0" : 200.}

    run_training(
        baseline_runner = 'lunar_lander.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '')

    high_score_test('mpi lunar lander',
        'lunar_lander.py', 10, passing_scores)

def test_binary_lunar_lander_mpi():
    num_timesteps = 300000
    passing_scores = {"agent0" : 100.}

    run_training(
        baseline_runner = 'binary_lunar_lander.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '')

    high_score_test('binary lunar lander',
        'binary_lunar_lander.py', 10, passing_scores)

#
# MountainCar tests
#
def test_mountain_car_mpi():
    num_timesteps = 300000
    passing_scores = {"agent0" :-199.}

    run_training(
        baseline_runner = 'mountain_car.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '')

    high_score_test('mountain car',
        'mountain_car.py', 10, passing_scores)

def test_mountain_car_continous_mpi():
    num_timesteps = 300000
    passing_scores = {"agent0" :50.}

    run_training(
        baseline_runner = 'mountain_car_continuous.py',
        num_timesteps   = num_timesteps,
        num_ranks       = 2,
        options         = '')

    high_score_test('mountain car continuous',
        'mountain_car_continuous.py', 10, passing_scores)


if __name__ == "__main__":

    test_cart_pole_serial()
    test_cart_pole_mpi()
    test_cart_pole_multi_envs()
    test_cart_pole_multi_envs_mpi()
    test_binary_cart_pole_serial()

    test_lunar_lander_mpi()
    test_binary_lunar_lander_mpi()

    test_mountain_car_mpi()
    test_mountain_car_continous_mpi()

