from utils import extrinsic_score_training_test

def test_cart_pole_serial():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps}"

    extrinsic_score_training_test("cart-pole-serial", cmd, 200.)

def test_cart_pole_mpi():

    num_timesteps = 40000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps}"

    extrinsic_score_training_test("cart-pole-mpi", cmd, 200.)

def test_cart_pole_multi_envs():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    extrinsic_score_training_test("cart-pole-multi-env", cmd, 128.)

def test_cart_pole_multi_envs_mpi():

    num_timesteps = 40000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps} "
    cmd += f"--envs-per-proc 2"

    extrinsic_score_training_test("cart-pole-multi-env-mpi", cmd, 128.)

def test_binary_cart_pole_serial():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"-e BinaryCartPole --clobber --num-timesteps {num_timesteps}"

    extrinsic_score_training_test("binary-cart-pole-serial", cmd, 200.)


if __name__ == "__main__":
    test_cart_pole_serial()
    test_cart_pole_mpi()
    test_cart_pole_multi_envs()
    test_cart_pole_multi_envs_mpi()
    test_binary_cart_pole_serial()
