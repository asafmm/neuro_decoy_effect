"""Data classes describing the lotteries and lottery sets used in the study.

A `Lottery` is a single gamble (amount in NIS, probability in %).
A `Set` bundles three lotteries (target, competitor, decoy) that
together form one trinary lottery set, along with the set's behavioral
decoy effect and target-choice shares.
"""

import numpy as np
import matplotlib.pyplot as plt

class Lottery():
    """A single risky lottery defined by an amount and a probability.

    Attributes:
        lottery_id (int): 1-based identifier matching the stimulus CSVs.
        amount (float): Win amount (NIS) if the lottery is realized.
        prob (float): Probability (%) of winning the amount.
        EV (float): Expected value, ``amount * prob / 100``.
    """

    def __init__(self, lottery_id, amount, prob):
        self.lottery_id = lottery_id
        self.amount = amount
        self.prob = prob
        self.EV = self.amount * self.prob / 100

class Set():
    """A trinary lottery set (target, competitor, decoy) with its behavior.

    Attributes:
        set_num (int): 1-based set identifier (1-27 for the analysed sets).
        target / competitor / decoy (Lottery): the three constituent lotteries.
        lottery_ids (np.ndarray): the three lottery IDs in [target, competitor, decoy] order.
        decoy_effect (float): change in target choice share from binary to trinary.
        target_ratio_binary (float): share of A choices in the binary group.
        target_ratio_ternary (float): share of A choices in the trinary group.
    """

    def __init__(self, set_num, target, competitor, decoy, decoy_effect, target_ratio_binary, target_ratio_ternary):
        self.set_num = set_num
        self.target = target
        self.competitor = competitor
        self.decoy = decoy
        self.lottery_ids = np.array([target.lottery_id, competitor.lottery_id, decoy.lottery_id])
        self.decoy_effect = decoy_effect
        self.target_ratio_binary = target_ratio_binary
        self.target_ratio_ternary = target_ratio_ternary

    def __repr__(self):
        return "Set ()"

    def __str__(self):
        return f'Set {self.set_num}: target {self.target.lottery_id}, competitor {self.competitor.lottery_id}, decoy {self.decoy.lottery_id}'

    def overlapping_with(self, other_set):
        """Return ``True`` if any lottery is shared with ``other_set``."""
        overlap = np.intersect1d(self.lottery_ids, other_set.lottery_ids)
        is_overlapping = overlap.size > 0
        return is_overlapping

    def has_lottery(self, lottery_id):
        """Return ``True`` if ``lottery_id`` is one of the three lotteries in this set."""
        return lottery_id in self.lottery_ids

    def plot(self):
        """Scatter-plot the set in amount x probability space.

        Target / competitor / decoy are color-coded and the title reports the
        empirical decoy effect for the set.
        """
        colors = ['#C42934', '#3e8a83', '#f5cbcb']
        scatter_size = 900
        plt.figure(figsize=(8, 8), dpi=150)
        scatter1 = plt.scatter( self.target.amount, self.target.prob, s=scatter_size, 
                                c = colors[0], edgecolor='black', label='Target')
        scatter2 = plt.scatter( self.competitor.amount, self.competitor.prob, s=scatter_size, 
                                c = colors[1], edgecolor='black', label='Competitor')
        scatter3 = plt.scatter( self.decoy.amount, self.decoy.prob, s=scatter_size, 
                                c = colors[2], edgecolor='black', label='Decoy')
        plt.xlabel('Amount', fontsize=28)
        plt.ylabel('Probability', fontsize=28)
        plt.xlim([0, 82])
        plt.ylim([0, 100])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc='best', markerscale=0.7, fontsize=16)
        plt.title(f'Effect: {self.decoy_effect*100:.2f}%', fontsize=36)
        plt.show()