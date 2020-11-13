"""
A simple implementation of a Hidden Markov Model
"""
import numpy as np


class TransitionModel(object):

    def __init__(self, transition_probabilities, init_distribution=None,
                 n_states=None):

        self.transition_probabilities = transition_probabilities

        if n_states is not None:
            if n_states != len(transition_probabilities):
                raise ValueError('The number of states is different than the number '
                                 'of transitions')
            self.n_states = int(n_states)
        else:
            self.n_states = len(transition_probabilities)

        self.current_state = None

        if init_distribution is None:
            self.init_distribution = ((1.0 / float(self.n_states)) *
                                      np.ones(self.n_states, dtype=np.float))
        else:
            self.init_distribution = init_distribution


class ObservationModel(object):

    def compute_obs_prob(self, observation, current_state, *args, **kwargs):
        raise NotImplementedError()


class HiddenMarkovModel(object):

    def __init__(self, observation_model, transition_model, update_func=None):

        self.om = observation_model
        self.tm = transition_model
        self.n_states = self.tm.n_states
        self.forward_variable = None
        self.forward_variables = []
        self._update_func = update_func

    def forwardAlgorithm(self, observation):

        obs_prob = self.om.compute_obs_prob(
            observation,
            current_state=self.current_state)

        if self.forward_variable is None:
            self.forward_variable = obs_prob * self.init_distribution

        else:
            self.forward_variable = (
                obs_prob * np.dot(self.tm.transition_probabilities.T,
                                  self.forward_variable))

        self.forward_variable /= self.forward_variable.sum()

        self.forward_variables.append(self.forward_variable)

        self.updateState()
        return self.current_state

    def updateState(self):

        self.current_state = self.forward_variable.argmax()
        if self._update_func is not None:
            self._update_func(self.current_state)

    @property
    def current_state(self):
        return self.tm.current_state

    @current_state.setter
    def current_state(self, value):
        self.tm.current_state = value

    @property
    def init_distribution(self):
        return self.tm.init_distribution

    @init_distribution.setter
    def init_distribution(self, distribution):
        self.tm.init_distribution = distribution
