import torch
from utils import get_action_type, need_action_squeeze

def test_policy(policy,
                env,
                render,
                device):

    action_type    = get_action_type(env)
    action_squeeze = need_action_squeeze(env)

    num_steps = 0
    score     = 0
    obs       = env.reset()
    done      = False
    policy.eval()

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

        if need_action_squeeze:
            action = action.squeeze()

        obs, reward, done, _ = env.step(action)
        score += reward

    print("Ran {} steps.".format(num_steps))
    print("Score: {}".format(score))
