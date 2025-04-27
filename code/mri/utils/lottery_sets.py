import numpy as np
import matplotlib.pyplot as plt

class Lottery():
    def __init__(self, lottery_id, amount, prob):
        self.lottery_id = lottery_id
        self.amount = amount
        self.prob = prob
        self.EV = self.amount * self.prob / 100

class Set():
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
        overlap = np.intersect1d(self.lottery_ids, other_set.lottery_ids)
        is_overlapping = overlap.size > 0
        return is_overlapping
    
    def has_lottery(self, lottery_id):
        return lottery_id in self.lottery_ids
    
    def plot(self):
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