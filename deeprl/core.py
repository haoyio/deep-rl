import logging
import timeit

from copy import deepcopy


class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class)
    by first observing the state of the environment. Based on this observation
    the agent changes the environment by performing an action.
    """
    def __init__(self, processor=None):
        self.logger = logging.getLogger(__name__)
        self.processor = processor

    def train(self,
              env,
              n_episodes,
              min_experiences=0,
              n_simulations=0,
              action_repetition=1,
              max_episode_steps=float('inf'),
              verbose=1):

        start_time = timeit.default_timer()
        is_aborted = False

        history = {
            'n_episode_steps': [],
            'episode_rewards': [],
        }

        if n_simulations > 0:
            history['avg_rewards'] = []

        episode = 0
        episode_step = 0

        observation = None
        episode_reward = None
        episode_step = None

        try:
            # initialize with minimum number of experiences
            if min_experiences > 0:
                if verbose:
                    print("Starting agent initialization...")

                experience = 0

                while experience < min_experiences:
                    # initialize environment
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = \
                            self.processor.process_observation(
                                observation
                            )
                    assert observation is not None

                    done = False
                    episode_step = 0

                    while not done and episode_step < max_episode_steps:
                        initial_observation = deepcopy(observation)
                        episode_step_reward = 0.0

                        # select action
                        action = self.act(
                            observation,
                            is_train=False
                        )
                        if self.processor is not None:
                            action = self.processor.process_action(
                                action
                            )

                        # execute action
                        for _ in xrange(action_repetition):
                            observation, reward, done, _ = \
                                env.step(action)
                            observation = deepcopy(observation)

                            if self.processor is not None:
                                observation = \
                                    self.processor.process_observation(
                                        observation
                                    )
                                reward = self.processor.process_reward(
                                    reward
                                )

                            episode_step_reward += reward

                            if done:
                                break

                        # store experience
                        self.remember(
                            initial_observation,
                            action,
                            episode_step_reward,
                            observation,
                            done
                        )

                        # housekeeping for experience gained
                        experience += 1

                if verbose:
                    print("Completed agent initialization in {} sec.".format(
                        round(timeit.default_timer() - start_time, 2)
                    ))

            # begin training proper
            if verbose:
                print("Starting agent training...")

            while episode < n_episodes:
                # initialize environment
                observation = deepcopy(env.reset())
                if self.processor is not None:
                    observation = self.processor.process_observation(observation)
                assert observation is not None

                done = False
                episode_reward = 0.0
                episode_step = 0

                # interact with the environment until termination
                while not done and episode_step < max_episode_steps:
                    initial_observation = deepcopy(observation)
                    episode_step_reward = 0.0

                    # select action
                    action = self.act(observation, is_train=True)
                    if self.processor is not None:
                        action = self.processor.process_action(action)

                    # execute action
                    for _ in xrange(action_repetition):
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)

                        if self.processor is not None:
                            observation = self.processor.process_observation(
                                observation
                            )
                            reward = self.processor.process_reward(reward)
                            info = self.processor.process_info(info)

                        episode_step_reward += reward

                        if done:
                            break

                    # store experience
                    self.remember(
                        initial_observation,
                        action,
                        episode_step_reward,
                        observation,
                        done
                    )

                    # update agent (experience replay, etc.)
                    self.update()

                    # housekeeping for each episode step
                    episode_reward += episode_step_reward
                    episode_step += 1

                # housekeeping for each episode
                history['n_episode_steps'].append(episode_step)
                history['episode_rewards'].append(episode_reward)

                # simulate current agent for average rewards
                if n_simulations > 0:
                    simulation = 0
                    simulation_rewards = 0.0

                    while simulation < n_simulations:
                        # initialize environment
                        sim_observation = deepcopy(env.reset())
                        if self.processor is not None:
                            sim_observation = \
                                self.processor.process_observation(
                                    sim_observation
                                )
                        assert sim_observation is not None

                        sim_done = False
                        sim_episode_reward = 0.0
                        sim_episode_step = 0

                        while not sim_done and \
                              sim_episode_step < max_episode_steps:
                            sim_observation = deepcopy(sim_observation)
                            sim_episode_step_reward = 0.0

                            # select action
                            sim_action = self.act(
                                sim_observation,
                                is_train=False
                            )
                            if self.processor is not None:
                                sim_action = self.processor.process_action(
                                    sim_action
                                )

                            # execute action
                            for _ in xrange(action_repetition):
                                sim_observation, sim_reward, sim_done, _ = \
                                    env.step(sim_action)
                                sim_observation = deepcopy(sim_observation)

                                if self.processor is not None:
                                    sim_observation = \
                                        self.processor.process_observation(
                                            sim_observation
                                        )
                                    sim_reward = self.processor.process_reward(
                                        sim_reward
                                    )

                                sim_episode_step_reward += sim_reward

                                if sim_done:
                                    break

                            # housekeeping for each simulation step
                            sim_episode_reward += sim_episode_step_reward
                            sim_episode_step += 1

                        # housekeeping for each simulation
                        simulation_rewards += sim_episode_reward
                        simulation += 1

                    # housekeeping for simulations
                    history['avg_rewards'].append(
                        simulation_rewards / simulation
                    )

                episode += 1
                observation = None
                episode_reward = None
                episode_step = None

                # print only every |verbose| episodes
                if verbose and episode % verbose == 0:
                    print("episode={}/{}: episode_reward={}{}".format(
                        episode,
                        n_episodes,
                        history['episode_rewards'][-1],
                        ", avg_reward={}".format(history['avg_rewards'][-1]) \
                            if n_simulations > 0 else ""
                    ))

        except KeyboardInterrupt:
            # catch keyboard interrupts to safely abort training
            self.logger.warning("Aborted agent training manually.")
            is_aborted = True

        history['total_time_sec'] = timeit.default_timer() - start_time
        history['is_aborted'] = is_aborted

        if verbose:
            print("Completed agent training in {} sec{}.".format(
                round(history.get('total_time_sec'), 2),
                " (aborted)" if is_aborted else ""
            ))

        return history

    def test(self,
             n_simulations,
             action_repetition=1,
             max_episode_steps=float('inf'),
             verbose=1):

        if verbose:
            print("Starting agent simulation for {} episodes...".format(
                    n_simulations
            ))

        start_time = timeit.default_timer()

        simulation = 0
        simulation_rewards = 0.0

        while simulation < n_simulations:
            # initialize environment
            sim_observation = deepcopy(env.reset())
            if self.processor is not None:
                sim_observation = \
                    self.processor.process_observation(
                        sim_observation
                    )
            assert sim_observation is not None

            sim_done = False
            sim_episode_reward = 0.0
            sim_episode_step = 0

            while not sim_done and \
                  sim_episode_step < max_episode_steps:
                sim_observation = deepcopy(sim_observation)
                sim_episode_step_reward = 0.0

                # select action
                sim_action = self.act(
                    sim_observation,
                    is_train=False
                )
                if self.processor is not None:
                    sim_action = self.processor.process_action(
                        sim_action
                    )

                # execute action
                for _ in xrange(action_repetition):
                    sim_observation, sim_reward, sim_done, _ = \
                        env.step(sim_action)
                    sim_observation = deepcopy(sim_observation)

                    if self.processor is not None:
                        sim_observation = \
                            self.processor.process_observation(
                                sim_observation
                            )
                        sim_reward = self.processor.process_reward(
                            sim_reward
                        )

                    sim_episode_step_reward += sim_reward

                    if sim_done:
                        break

                # housekeeping for each simulation step
                sim_episode_reward += sim_episode_step_reward
                sim_episode_step += 1

            # housekeeping for each simulation
            simulation_rewards += sim_episode_reward
            simulation += 1

        if verbose:
            print("Completed agent simulation in {} sec.".format(
                round(timeit.default_timer() - start_time, 2)
            ))

        return simulation_rewards / simulation

    def remember(self, observation, action, reward, next_observation, done):
        raise NotImplementedError()

    def act(self, observation, is_train=False):
        """Given the env observation, return the action to be taken."""
        raise NotImplementedError()

    def update(self):
        """Updates the agent by replaying its experiences."""
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError()

    def load_weights(self, filepath):
        raise NotImplementedError()


class Processor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`.
    This can be necessary if your agent has different requirements with respect
    to the form of the observations, actions, and rewards of the environment.
    By implementing a custom processor, you can effectively translate between
    the two without having to change the underlaying implementation of the
    agent or environment.
    """
    def process_observation(self, observation):
        return observation

    def process_action(self, action):
        return action

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info
