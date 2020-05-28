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

    @staticmethod
    def _n_generator(n):
        if isinstance(n, int):
            return lambda: n
        elif isinstance(n, np.ndarray):
            return lambda: random.choice(n)
        else:
            raise ValueError("n must be either int or ndarray")


    def run_auctions(self, bids, n=2):
        """
        Performs nth price auction rounds times.

        :param bids: ndarray of size m x rounds of bids between 0 and h, where m is the
            number of bidders, and rounds is the number of bids.
        :param n: Number of items to sell in an nth price auction. If num_bids is a ndarray, the
            number of items to sell is uniformly randomly chosen from num_bids.
        :return prices: ndarray of size rounds. Reserve price for each round
        :return regret: Regret compared to OPT
        """
        if bids.shape[0] < n:
            raise ValueError("n should be less than the number of bidders")

        revenue = np.empty((self.reserve_prices.shape[0], bids.shape[1]))
        n_gen = OnlineReserve._n_generator(n)

        for j in range(bids.shape[1]):
            n_largest_bids = heapq.nlargest(n_gen(), bids[:, j])

            for i in range(self.reserve_prices.shape[0]):

                reserve_price = self.reserve_prices[i]
                sell_price = max(n_largest_bids[-1], reserve_price)
                # Note: By definition, if b is above sell_price,
                # Then b is in the n_largest_bids
                revenue[i, j] = sum([sell_price if b > sell_price else 0 for b in n_largest_bids])

        actions, regret = self.learning_algorithm.experiment(revenue, _print=False)
        prices = self.reserve_prices[actions]
        revenue = np.array(
            [revenue[action, i] for i, action in enumerate(actions)]
        )
        return prices, revenue, regret

class OnlineExchange:

    def __init__(self, learning_rate, k, h=1):
        """
        :param k: Number of discretized prices
        :param h: Maximum bid
        :param learning_rate: Learning Rate of learning algorithm
        """
        
        self.prices = np.linspace(0, h, num=k)
        self.h = h
        self.learning_algorithm = ExponentialWeights(learning_rate, max_payoff=h)

    def run_auctions(self, bids, n=2):

        #generate uniform value of buyer and seller
        #generate random price to exchange at
        #calculate payoff and opt
        #learn

        if bids.shape[0] < n:
            raise ValueError("n should be less than the number of bidders")

        revenue = np.empty((self.prices.shape[0], bids.shape[1]))
        for j in range(bids.shape[1]):

            sell_value, buy_value = bids[0][j], bids[1][j]
            for i in range(self.prices.shape[0]):

                price = self.prices[i]
                #revenue is price, otherwise it's 0 if buyer < seller or price out of range.
                revenue[i, j] = price if sell_value < price < buy_value else 0

        actions, regret = self.learning_algorithm.experiment(revenue, _print=False)
        result_prices = self.prices[actions]
        revenue = np.array(
            [revenue[action, i] for i, action in enumerate(actions)]
        )
        return result_prices, revenue, regret
        
            


if __name__ == "__main__":
    random.seed(0)
    # draw from uniform distribution from 0-1
    rounds = 100
    bidders = 2
    bids = random.rand(bidders, rounds)
    reserve = OnlineReserve(.6, 111)
    prices, _, regret = reserve.run_auctions(bids)
    print("##### Prices #####")
    print(prices)
    print(f"Regret: {regret}")


    buyer_and_seller_values = random.rand(2, 1000)
    exchange = OnlineExchange(.6, 100)
    prices, _, regret = exchange.run_auctions(buyer_and_seller_values)
    print("##### Prices #####")
    print(prices)
    print(f"Regret: {regret}")