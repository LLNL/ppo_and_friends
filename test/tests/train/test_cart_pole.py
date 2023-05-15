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

    num_timesteps = 100000
    cmd  = f"ppoaf-baselines "
    cmd += f"CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    run_training(cmd)
    run_cart_pole_test("cart-pole-multi-env")

def test_cart_pole_multi_envs_mpi():

    num_timesteps = 90000
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


if __name__ == "__main__":
    test_cart_pole_serial()
    test_cart_pole_mpi()
    test_cart_pole_multi_envs()
    test_cart_pole_multi_envs_mpi()
    test_binary_cart_pole_serial()
