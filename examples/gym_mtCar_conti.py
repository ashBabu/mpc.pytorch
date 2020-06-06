import logging
import math
import time

import gym
import numpy as np
import torch
import torch.autograd
from gym import wrappers, logger as gym_log
from mpc import mpc

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "MountainCarContinuous-v0"
    TIMESTEPS = 10  # T
    N_BATCH = 1
    LQR_ITER = 5
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    class MountainCarDynamics(torch.nn.Module):
        def forward(self, state, action):
            power = 0.0015
            min_action, max_action = torch.tensor([[-1.0]]), torch.tensor([[1.0]])
            min_position, max_position = torch.tensor([[-1.2]]), torch.tensor([[0.6]])
            max_speed = torch.tensor([[0.07]])
            # if state.ndim == 1:
            #     state = state[None, :]
            position = state[:, 0].view(-1, 1)
            velocity = state[:, 1].view(-1, 1)
            force = min(max(action[0], min_action), max_action)

            velocity += force * power - 0.0025 * torch.cos(3 * position)
            if velocity > max_speed: velocity = max_speed
            if velocity < -max_speed: velocity = -max_speed
            position += velocity
            if position > max_position: position = max_position
            if position < min_position: position = min_position
            if position == min_position and velocity < 0: velocity = torch.tensor([[0.]])
            state = torch.cat((position, velocity), dim=1)
            return state


    downward_start = False
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = np.array([0., 0.01])
        # env.state = torch.tensor([[-1.2, 0.07]])

    env = wrappers.Monitor(env, '/tmp/box_ddp_mountainCarContinuous/', force=True)
    env.reset()
    if downward_start:
        env.env.state = np.array([0., 0.07])
        # env.env.state = torch.tensor([[-1.2, 0.07]])

    nx = 2
    nu = 1

    u_init = None
    render = False
    retrain_after_iter = 50
    run_iter = 500

    goal_weights = torch.tensor((10., 0.01))  # nx
    goal_state = torch.tensor((0.45, 0.))  # nx  # values from https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
    ctrl_penalty = 0.000001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    # run MPC
    total_reward = 0
    for i in range(run_iter):
        state = env.state.copy()
        state = torch.tensor(state).view(1, -1).float()
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # compute action based on current state, dynamics, and cost
        # nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, PendulumDynamics())
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, MountainCarDynamics())
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.detach().numpy())
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

    logger.info("Total reward %f", total_reward)
