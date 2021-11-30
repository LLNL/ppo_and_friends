import torch

def test_policy(policy,
                env,
                render,
                device,
                action_type):

    num_steps = 0
    score     = 0
    obs  = env.reset()
    done = False
    policy.eval()

    while not done:

        num_steps += 1

        if render:
            env.render()

        obs    = torch.tensor(obs).to(device)
        action = policy(obs).detach().cpu()

        if action_type == "discrete":
            action = np.argmax(action)

        obs, reward, done, _ = env.step(action)
        score += reward

    print("Ran {} steps.".format(num_steps))
    print("Score: {}".format(score))
