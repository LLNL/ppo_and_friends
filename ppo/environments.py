from ppo import PPO
import gym
from testing import test_policy

def run_ppo(env,
            action_type,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test):

    ppo = PPO(env         = env,
              device      = device,
              action_type = action_type,
              render      = render,
              load_state  = load_state,
              state_path  = state_path)

    if test:
        test_policy(ppo.actor, env, render, device, action_type)
    else: 
        ppo.learn(num_timesteps)


def cartpole_ppo(state_path,
                 action_type,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('CartPole-v0')

    run_ppo(env,
            action_type,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)


def cartpole_pixel_ppo(state_path,
                       action_type,
                       load_state,
                       render,
                       num_timesteps,
                       device,
                       test = False):

    env = gym.make('CartPole-v0')

    #TODO: create a cartpole solver from pixels.
    #run_ppo(env,
    #        action_type,
    #        state_path,
    #        load_state,
    #        render,
    #        num_timesteps,
    #        device,
    #        test)


def pendulum_ppo(state_path,
                 action_type,
                 load_state,
                 render,
                 num_timesteps,
                 device,
                 test = False):

    env = gym.make('Pendulum-v1')

    run_ppo(env,
            action_type,
            state_path,
            load_state,
            render,
            num_timesteps,
            device,
            test)
