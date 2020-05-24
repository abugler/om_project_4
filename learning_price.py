import numpy as np
import numpy.random as random
from learning_algorithms import ExponentialWeights
import heapq

class OnlineReserve:
    """
    OnlineReserve holds a second-price auction, and
    will learn a discretized reserve price to maximize revenue.
    OnlineReserve uses an exponential weights algorithm
    """
    def __init__(self, learning_rate, k, h=1):
        """
        :param k: Number of discretized prices
        :param h: Maximum bid
        :param learning_rate: Learning Rate of learning algorithm
        """
        self.reserve_prices = np.linspace(0, h, num=k)
        self.h = h
        # If we wanted to, we could add FTPL, but I don't see a reason to.
        self.learning_algorithm = ExponentialWeights(learning_rate, max_payoff=h)

    def run_auctions(self, bids):
        """
        Performs second price auction n times.

        :param bids: ndarray of size m x n of bids between 0 and h, where m is the
            number of bidders, and n is the number of bids.
        :return prices: ndarray of size n. Reserve price for each round
        :return regret: Regret compared to OPT
        """
        revenue = np.empty((self.reserve_prices.shape[0], bids.shape[1]))
        for i in range(self.reserve_prices.shape[0]):
            price = self.reserve_prices[i]
            for j in range(bids.shape[1]):
                max_bid, second_max_bid = heapq.nlargest(2, bids[:, j])
                revenue[i, j] = max(second_max_bid, price) if max_bid >= price else 0

        actions, regret = self.learning_algorithm.experiment(revenue, _print=False)
        prices = self.reserve_prices[actions]
        return prices, regret

if __name__ == "__main__":
    random.seed(0)
    # draw from uniform distribution from 0-1
    n = 100
    bidders = 2
    bids = random.rand(bidders, n)
    reserve = OnlineReserve(.5, 10)
    prices, regret = reserve.run_auctions(bids)
    print("##### Prices #####")
    print(prices)
    print(f"Regret: {regret}")
