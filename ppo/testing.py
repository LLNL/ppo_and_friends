import torch
from utils.misc import get_action_type, need_action_squeeze

def test_policy(policy,
                env,
                render,
                num_test_runs,
                device):

    action_type    = get_action_type(env)
    action_squeeze = need_action_squeeze(env)

    num_steps = 0
    score     = 0
    policy.eval()

    for _ in range(num_test_runs):

        obs  = env.reset()
        done = False

        while not done:
            num_steps += 1

            if render:
                env.render()

            obs = torch.tensor(obs, dtype=torch.float).to(device).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs).detach().cpu()

            if action_type == "discrete":
                action = torch.argmax(action).numpy()
            else:
                action = action.numpy()

            if action_squeeze:
                action = action.squeeze()

            obs, reward, done, _ = env.step(action)
            score += reward

    print("Ran env {} times.".format(num_test_runs))
    print("Ran {} total time steps.".format(num_steps))
    print("Ran {} time steps on average.".format(num_steps / num_test_runs))
    print("Average score: {}".format(score / num_test_runs))
