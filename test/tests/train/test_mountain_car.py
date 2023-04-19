from utils import run_training, high_score_test

def test_mountain_car_mpi():

    num_timesteps = 700000
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

    num_timesteps = 700000
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
    test_mountain_car_mpi()
    test_mountain_car_continous_mpi()
