from utils import extrinsic_score_test

def test_cart_pole():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps}"

    extrinsic_score_test("cart-pole", cmd)

def test_cart_pole_mpi():

    num_timesteps = 40000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps}"

    extrinsic_score_test("cart-pole-mpi", cmd)


if __name__ == "__main__":
    test_cart_pole()
    test_cart_pole_mpi()
