from utils import run_training, high_score_test

#
# CartPole tests
#
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
    cmd  = f"ppoaf-baselines "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)
    run_cart_pole_test("cart-pole-serial")

def test_cart_pole_mpi():

    num_timesteps = 70000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)
    run_cart_pole_test("cart-pole-mpi")

def test_cart_pole_multi_envs():

    num_timesteps = 70000
    cmd  = f"ppoaf-baselines "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    run_training(cmd)
    run_cart_pole_test("cart-pole-multi-env")

def test_cart_pole_multi_envs_mpi():

    num_timesteps = 70000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    run_training(cmd)
    run_cart_pole_test("cart-pole-multi-env-mpi")

def test_binary_cart_pole_serial():

    num_timesteps = 70000
    cmd  = f"ppoaf-baselines "
    cmd += f"BinaryCartPole --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)
    run_cart_pole_test("cart-pole-binary-serial")

#
# LunarLander tests
#
def test_lunar_lander_mpi():

    num_timesteps = 500000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"LunarLander --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"LunarLander --test --num-test-runs 10 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 200.}

    high_score_test("lunar-lander-mpi", cmd,
        passing_scores, "LunarLander")

def test_binary_lunar_lander_mpi():

    num_timesteps = 300000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"BinaryLunarLander --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"BinaryLunarLander --test --num-test-runs 10 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 100.}

    high_score_test("binary-lunar-lander-mpi", cmd,
        passing_scores, "BinaryLunarLander")

#
# MountainCar tests
#
def test_mountain_car_mpi():

    num_timesteps = 300000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"MountainCar --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"MountainCar --test --num-test-runs 5 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : -199}

    high_score_test("mountain-car-mpi", cmd,
        passing_scores, "MountainCar")

def test_mountain_car_continous_mpi():

    num_timesteps = 300000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"MountainCarContinuous --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"MountainCarContinuous --test --num-test-runs 5 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 50.}

    high_score_test("mountain-car-continuous-mpi", cmd,
        passing_scores, "MountainCarContinuous")


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

