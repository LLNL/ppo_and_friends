import numpy as np
from abmarl.sim.gridworld.base import GridWorldSimulation
from abmarl.sim.gridworld.agent import GridObservingAgent, MovingAgent
from abmarl.sim.gridworld.state import PositionState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.observer import SingleGridObserver
from abmarl.sim.gridworld.observer import AbsolutePositionObserver


class AlternateMazeNavigationSim(GridWorldSimulation):

    def __init__(self, observe="position", max_steps = 512, **kwargs):

        self.agents    = kwargs['agents']
        self.navigator = kwargs['agents']['navigator']
        self.target    = kwargs['agents']['target']
        self.max_steps = max_steps

        # State Components
        self.position_state = PositionState(**kwargs)

        # Action Components
        self.move_actor = MoveActor(**kwargs)

        # Observation Components
        assert observe in ("position", "grid")

        if observe == "grid":
            self.observer = SingleGridObserver(**kwargs)

        elif observe == "position":
            self.observer = AbsolutePositionObserver(**kwargs)
        
        self.prev_pos   = self.navigator.initial_position
        self.step_count = 0

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)

        # Track the rewards
        self.reward = 0

        self.step_count = 0

    def step(self, action_dict, **kwargs):
        # Process moves
        action = action_dict['navigator']

        move_result = self.move_actor.process_action(self.navigator, action, **kwargs)

        self.prev_pos = self.navigator.position.copy()

        # Entropy penalty
        self.reward -= 0.01

        self.step_count += 1

    def get_obs(self, agent_id, **kwargs):
        return {
            **self.observer.get_obs(self.navigator, **kwargs)
        }

    def get_reward(self, agent_id, **kwargs):

        if self.agent_reached_goal():
            self.reward = 1.

        reward = self.reward
        self.reward = 0

        return reward

    def agent_reached_goal(self):
        return np.all(self.navigator.position == self.target.position)

    def get_done(self, agent_id, **kwargs):
        return self.get_all_done()

    def get_all_done(self, **kwargs):
        return self.agent_reached_goal() or self.step_count == 512

    def get_info(self, agent_id, **kwargs):
        return {}

