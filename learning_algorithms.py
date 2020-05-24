import numpy as np
import numpy.random as random
from abc import abstractmethod

class OnlineLearning:
    @staticmethod
    def calculate_regret(payoffs, actions):
        """
        Calculates OPT, then reports regret
        :param payoffs: 2-D k x n numpy array, where the value payoffs[k, n]
                is the payoff at round n with action k
        :param actions: 1-D numpy array of length n where actions[i] is the action
                taken by the learning algorithm at round i
        :return regret: Value of regret
        """
        # Find opt
        opt_action = payoffs.sum(axis=1).argmax()

        actual_payoff = sum([payoffs[action, i] for i, action in enumerate(actions)])
        original_payoff = sum(payoffs[opt_action, :])

        regret = (original_payoff - actual_payoff) / actions.shape[0]

        return regret

    @staticmethod
    def generate_payoffs(n: int, action_probabilities):
        """
        Generates
        :param n: Number of rounds to generate
        :param action_probabilities: 1-D ndarray, where action_probabilities[i]
            is the probability that payoffs[i, j] would be 1
        :return payoffs: 2-D n x k ndarray where payoffs[i, j] is the payoff
            at action i and round j
        """
        payoffs = [
            random.choice(2, size=n, p=[1 - probability, probability])
            for probability in action_probabilities
        ]

        return np.array(payoffs, dtype=int)

    @staticmethod
    def generate_decaying_payoffs(n: int, action_probabilities: np.ndarray, decay):
        if decay <= 0 or decay > 1:
            raise ValueError("Decay must be greater than zero, and less than or equal to 1.")
        payoffs = [
            [
                1 if random.random() < probability * decay ** (round - 1) else 0
                for round in range(n)
            ]
            for probability in action_probabilities
        ]

        return np.array(payoffs, dtype=int)

    def __init__(self, learning_rate: float, max_payoff=1):

        if not (0 <= learning_rate <= 1):
            raise ValueError("Learning rate must be between 0 and 1.")

        self.learning_rate = learning_rate
        self.max_payoff = max_payoff

    @abstractmethod
    def generate_cum_payoffs(self, k, n, h):
        pass

    @abstractmethod
    def generate_action(self, k, n, h, cumulative_payoffs):
        pass

    def run(self, payoffs: np.ndarray):

        """
        Runs Online Learning algorithm on the given payoffs
        :param payoffs: 2-D k x n numpy array, where the value payoffs[k, n]
            is the payoff at round n with action k
        :return actions: 1-D numpy array of length n where actions[i] is the action
            taken by the learning algorithm at round i
        """

        k, n, h = payoffs.shape[0], payoffs.shape[1], self.max_payoff
        cumulative_payoffs = self.generate_cum_payoffs(k=k, n=n, h=h)
        actions = []

        for i in range(n):
            actions.append(self.generate_action(k=k, n=n, h=h, cumulative_payoffs=cumulative_payoffs))
            cumulative_payoffs += payoffs[:, i]

        return np.array(actions, dtype=int)

    def experiment(self, payoffs, _print=True):
        """
        Runs algorithm on payoffs, then reports actions and regret.
        Also prints results
        :param payoffs: 2-D k x n numpy array, where the value payoffs[k, n]
            is the payoff at round n with action k
        :return regret: Regret of algorithm
        """
        actions = self.run(payoffs)
        regret = OnlineLearning.calculate_regret(payoffs, actions)
        if _print:
            print(f"Actions:\n{actions}")
            print(f"The regret is {regret}")
        return actions, regret


class ExponentialWeights(OnlineLearning):

    def generate_cum_payoffs(self, k, n, h):
        return np.zeros(k, dtype=np.float64)

    def get_scaled_probabilities(self, cumulative_payoffs, h):
        payoffs_copy = cumulative_payoffs.copy()
        inf = float("inf")
        sum_probabilities = inf
        # Possible overflow with large values of n
        while sum_probabilities == inf:
            probabilities = np.array([(1 + self.learning_rate) ** (payoff - 1 / h)
                                      for payoff in payoffs_copy])

            sum_probabilities = np.sum(probabilities)
            payoffs_copy /= 10
        return probabilities / sum_probabilities

    def generate_action(self, k, n, h, cumulative_payoffs):
        probabilities = self.get_scaled_probabilities(cumulative_payoffs, h)
        action = random.choice(k, p=probabilities)

        return action