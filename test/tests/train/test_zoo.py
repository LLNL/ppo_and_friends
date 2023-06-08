from utils import run_training, high_score_test

#
# MPE tests
#
def test_mpe_simple_tag_mpi():

    num_timesteps = 30000
    cmd  = f"mpirun -n 2 ppoaf-baselines "
    cmd += f"MPESimpleTag --clobber --num-timesteps {num_timesteps}"

    run_training(cmd)

    cmd  = f"ppoaf-baselines "
    cmd += f"MPESimpleTag --test --num-test-runs 5 "
    cmd += f"--save-test-scores"

    #
    # We don't need a resolved policy. We just need something that's
    # started the learning process.
    #
    passing_scores = {"adversary_1" : 100}

    high_score_test("mpe-simple-tag-mpi", cmd,
        passing_scores, "MPESimpleTag")


if __name__ == "__main__":

    test_mpe_simple_tag_mpi()

