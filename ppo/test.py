import torch

def test_policy(policy, env, render, device):

    num_steps = 0
    obs  = env.reset()
    done = False

    while not done:

        if render:
            env.render()

        obs    = torch.tensor(obs).to(device)
        action = policy(obs).detach().cpu().numpy()

        obs, reward, done, _ = env.step(action)

    print("Ran {} steps.".format(num_steps))
