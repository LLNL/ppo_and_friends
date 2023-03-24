from utils import run_training, average_score_test

def test_lunar_lander_mpi():

    num_timesteps = 400000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"-e LunarLander --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"python train_baseline.py "
    cmd += f"-e LunarLander --test --test-explore --num-test-runs 10 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 200.}

    average_score_test("lunar-lander-mpi", cmd,
        passing_scores, "LunarLander")

if __name__ == "__main__":
    test_lunar_lander_mpi()
