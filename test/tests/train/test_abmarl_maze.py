from utils import run_training, average_score_test

def test_abmarl_blind_maze_mpi():

    num_timesteps = 85000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"AbmarlBlindMaze --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"AbmarlBlindMaze --test --num-test-runs 5 "

    passing_scores = {"navigator" : 0.8}

    average_score_test("abmarl-blind-maze-mpi", cmd,
        passing_scores, "AbmarlBlindMaze")

if __name__ == "__main__":
    test_abmarl_blind_maze_mpi()
