from utils import extrinsic_score_test

def test_cart_pole_train():

    num_timesteps = 17000
    cmd  = f"python train_baseline.py "
    cmd += f"-e CartPole --clobber --num-timesteps {num_timesteps}"

    extrinsic_score_test("cart-pole-train", cmd)


if __name__ == "__main__":
    test_cart_pole_train()
