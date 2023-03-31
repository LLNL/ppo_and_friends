from utils import run_training, average_score_test

def test_pressure_plate_mpi():

    num_timesteps = 310000
    cmd  = f"mpirun -n 4 python train_baseline.py "
    cmd += f"PressurePlate --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"python train_baseline.py "
    cmd += f"PressurePlate --test --test-explore --num-test-runs 1 "
    cmd += f"--save-test-scores"

    passing_scores = {"agent3" : -512}

    average_score_test("pressure-plate-mpi", cmd,
        passing_scores, "PressurePlate")

if __name__ == "__main__":
    test_pressure_plate_mpi()
