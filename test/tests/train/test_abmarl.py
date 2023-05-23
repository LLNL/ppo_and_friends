from utils import run_training, high_score_test

def test_abmarl_maze_mpi():

    num_timesteps = 30000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"AbmarlMaze --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"AbmarlMaze --test --num-test-runs 5 "
    cmd += f"--save-test-scores --test-explore"

    passing_scores = {"navigator" : 0.0}

    high_score_test("abmarl-maze-mpi", cmd,
        passing_scores, "AbmarlMaze")


if __name__ == "__main__":
    test_abmarl_maze_mpi()
