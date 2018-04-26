import math
import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.state_size = self.action_repeat * 6  # 6 states x 3 repeat = 18 
        self.action_low = 350 # 0
        self.action_high = 700 # 900
        self.action_size = 1 # 4

        # Goal : taking-off, hovering, going to a place or landing.
        self.target_pos = target_pos 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.0 - 0.06 * abs(self.target_pos[2] - self.sim.pose[2]) - 0.04 * abs(self.sim.pose[0:1] - self.target_pos[0:1]).sum() - 0.02 * abs(self.sim.pose[3:6]).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = [rotor_speeds[0], rotor_speeds[0], rotor_speeds[0], rotor_speeds[0]]
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state