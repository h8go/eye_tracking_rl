# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library used by example_viz.py to generate visualizations.

This file illustrates the following:
  - How to subclass an existing agent to add visualization functionality.
    - For DQN we visualize the cumulative rewards and the Q-values for each
      action (MyDQNAgent).
    - For Rainbow we visualize the cumulative rewards and the Q-value
      distributions for each action (MyRainbowAgent).
  - How to subclass Runner to run in eval mode, lay out the different subplots,
    generate the visualizations, and compile them into a video (MyRunner).
  - The function `run()` is the main entrypoint for running everything.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.utils import agent_visualizer
from dopamine.utils import atari_plotter
from dopamine.utils import bar_plotter
from dopamine.utils import line_plotter
import gin
import numpy as np
import tensorflow as tf
import tf_slim
import pdb
import matplotlib.pyplot as plt


class MyDQNAgent(dqn_agent.DQNAgent):
    """Sample DQN agent to visualize Q-values and rewards."""

    def __init__(self, sess, num_actions, summary_writer=None):
        super(MyDQNAgent, self).__init__(sess, num_actions,
                                         summary_writer=summary_writer)
        self.q_values = [[] for _ in range(num_actions)]
        self.rewards = []

    def step(self, reward, observation, step_number):
        self.rewards.append(reward)
        return super(MyDQNAgent, self).step(reward, observation, step_number)

    def _select_action(self, step_number):
        action = super(MyDQNAgent, self)._select_action(step_number)
        # print("on selectionne ici")
        q_vals = self._sess.run(self._net_outputs.q_values,
                                {self.state_ph: self.state})[0]
        for i in range(len(q_vals)):
            self.q_values[i].append(q_vals[i])
        return action

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        if use_legacy_checkpoint:
            variables_to_restore = atari_lib.maybe_transform_variable_names(
                tf.compat.v1.global_variables(), legacy_checkpoint_load=True)
        else:
            global_vars = set([x.name for x in tf.compat.v1.global_variables()])
            ckpt_vars = [
                '{}:0'.format(name)
                for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            include_vars = list(global_vars.intersection(set(ckpt_vars)))
            variables_to_restore = tf_slim.get_variables_to_restore(
                include=include_vars)
        if variables_to_restore:
            reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
            reloader.restore(self._sess, checkpoint_path)
            logging.info('Done restoring from %s', checkpoint_path)
        else:
            logging.info('Nothing to restore!')

    def get_q_values(self):
        return self.q_values

    def get_rewards(self):
        return [np.cumsum(self.rewards)]


class MyRainbowAgent(rainbow_agent.RainbowAgent):
    """Sample Rainbow agent to visualize Q-values and rewards."""

    def __init__(self, sess, num_actions, summary_writer=None):
        super(MyRainbowAgent, self).__init__(sess, num_actions,
                                             summary_writer=summary_writer)
        self.rewards = []

    def step(self, reward, observation, step_number):
        self.rewards.append(reward)
        return super(MyRainbowAgent, self).step(reward, observation, step_number)

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        if use_legacy_checkpoint:
            variables_to_restore = atari_lib.maybe_transform_variable_names(
                tf.compat.v1.global_variables(), legacy_checkpoint_load=True)
        else:
            global_vars = set([x.name for x in tf.compat.v1.global_variables()])
            ckpt_vars = [
                '{}:0'.format(name)
                for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            include_vars = list(global_vars.intersection(set(ckpt_vars)))
            variables_to_restore = tf_slim.get_variables_to_restore(
                include=include_vars)
        if variables_to_restore:
            reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
            reloader.restore(self._sess, checkpoint_path)
            logging.info('Done restoring from %s', checkpoint_path)
        else:
            logging.info('Nothing to restore!')

    def get_probabilities(self):
        return self._sess.run(tf.squeeze(self._net_outputs.probabilities),
                              {self.state_ph: self.state})

    def get_rewards(self):
        return [np.cumsum(self.rewards)]


class MyRunner(run_experiment.Runner):
    """Sample Runner class to generate visualizations."""

    def __init__(self, base_dir, trained_agent_ckpt_path, create_agent_fn,
                 use_legacy_checkpoint=False):
        self._trained_agent_ckpt_path = trained_agent_ckpt_path
        self._use_legacy_checkpoint = use_legacy_checkpoint
        super(MyRunner, self).__init__(base_dir, create_agent_fn)

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        self._agent.reload_checkpoint(self._trained_agent_ckpt_path,
                                      self._use_legacy_checkpoint)
        self._start_iteration = 0

    def _run_one_iteration(self, iteration):
        statistics = iteration_statistics.IterationStatistics()
        logging.info('Starting iteration %d', iteration)
        _, _ = self._run_eval_phase(statistics)
        return statistics.data_lists

    def _run_one_iteration(self, iteration):
        statistics = iteration_statistics.IterationStatistics()
        logging.info('Starting iteration %d', iteration)

        num_episodes_eval, average_reward_eval = self._run_eval_phase(
            statistics)
        return statistics.data_lists

    def _run_eval_phase(self, statistics):
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True

        _, sum_returns, num_episodes = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logging.info('Average undiscounted return per evaluation episode: %.2f',
                     average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        step_count = 0
        num_episodes = 0
        sum_returns = 0.
        print("min_steps", min_steps)
        while step_count < min_steps:
          print(">>>>> step_count", step_count)
          episode_length, episode_return = self._run_one_episode()
          statistics.append({
              '{}_episode_lengths'.format(run_mode_str): episode_length,
              '{}_episode_returns'.format(run_mode_str): episode_return
          })
          step_count += episode_length
          sum_returns += episode_return
          num_episodes += 1
          # We use sys.stdout.write instead of logging so as to flush frequently
          # without generating a line break.
          sys.stdout.write('Steps executed: {} '.format(step_count) +
                           'Episode length: {} '.format(episode_length) +
                           'Return: {}\r'.format(episode_return))
          sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_one_episode(self):
        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
          observation, reward, is_terminal = self._run_one_step(action, step_number)

          total_reward += reward
          step_number += 1
          print("step_number", step_number)

          if self._clip_rewards:
            # Perform reward clipping.
            reward = np.clip(reward, -1, 1)

          if (self._environment.game_over or
              step_number == self._max_steps_per_episode):
            # Stop the run loop once we reach the true end of episode.
            break
          elif is_terminal:
            # If we lose a life but the episode is not over, signal an artificial
            # end of episode to the agent.
            self._end_episode(reward, is_terminal)
            action = self._agent.begin_episode(observation)
          else:
            action = self._agent.step(reward, observation, step_number)

        self._end_episode(reward, is_terminal)

        return step_number, total_reward

    def _run_one_step(self, action, step_number):
        observation, reward, is_terminal, _ = self._environment.step(action)
        if True:
            if step_number == 100 or step_number == 500 or step_number == 900:
                # self._environment.render('human')
                image = self._environment.render('rgb_array')
                plt.imshow(image)
                plt.savefig("/home/hugo/saliency_maps/Rainbow-Tennis/render/render"+str(step_number)+".png")
        # image = self._environment.render('rgb_array')
        # plt.imshow(image)
        # plt.savefig("/home/hugo/render1.png")
        # sys.exit()
        # pdb.set_trace()
        return observation, reward, is_terminal

def create_dqn_agent(sess, environment, summary_writer=None):
    return MyDQNAgent(sess, num_actions=environment.action_space.n,
                      summary_writer=summary_writer)


def create_rainbow_agent(sess, environment, summary_writer=None):
    return MyRainbowAgent(sess, num_actions=environment.action_space.n,
                          summary_writer=summary_writer)


def create_runner(base_dir, trained_agent_ckpt_path, agent='dqn',
                  use_legacy_checkpoint=False):
    create_agent = create_dqn_agent if agent == 'dqn' else create_rainbow_agent
    return MyRunner(base_dir, trained_agent_ckpt_path, create_agent,
                    use_legacy_checkpoint)





def run(agent, game, num_steps, root_dir, restore_ckpt,
        use_legacy_checkpoint=False):
    """Main entrypoint for running and generating visualizations.

  Args:
    agent: str, agent type to use.
    game: str, Atari 2600 game to run.
    num_steps: int, number of steps to play game.
    root_dir: str, root directory where files will be stored.
    restore_ckpt: str, path to the checkpoint to reload.
    use_legacy_checkpoint: bool, whether to restore from a legacy (pre-Keras)
      checkpoint.
  """
    tf.compat.v1.reset_default_graph()
    config = """
  atari_lib.create_atari_environment.game_name = '{}'
  WrappedReplayBuffer.replay_capacity = 300
  """.format(game)
    base_dir = os.path.join(root_dir, 'agent_viz', game, agent)
    gin.parse_config(config)
    runner = create_runner(base_dir, restore_ckpt, agent, use_legacy_checkpoint)
    iteration = 0
    runner._run_one_iteration(iteration)
