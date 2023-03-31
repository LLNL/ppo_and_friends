from utils import run_training, average_score_test

def test_mountain_car_mpi():

    num_timesteps = 200000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"MountainCar --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"python train_baseline.py "
    cmd += f"MountainCar --test --test-explore --num-test-runs 5 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : -199}

    average_score_test("mountain-car-mpi", cmd,
        passing_scores, "MountainCar")

def test_mountain_car_continous_mpi():

    num_timesteps = 300000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"MountainCarContinuous --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"python train_baseline.py "
    cmd += f"MountainCarContinuous --test --test-explore --num-test-runs 5 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent0" : 50.}

    average_score_test("mountain-car-continuous-mpi", cmd,
        passing_scores, "MountainCarContinuous")

if __name__ == "__main__":
    test_mountain_car_mpi()
    test_mountain_car_continous_mpi()
