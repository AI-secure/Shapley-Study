from shapley.utils.plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LabelPlotter(Plotter):

    def __init__(self, Label):
        self.name = 'LabelPlotter'
        self.app = Label

    def plot(self):
        res_v = self.app.result
        res_i = np.argsort(-res_v)[::-1]
        cnt = 0
        f = []
        total = 0
        cnt = 0
        for i in range(len(res_i)):
            if self.app.flip[int(res_i[i])] == 1:
                total += 1
        for i in range(len(res_i)):
            if self.app.flip[int(res_i[i])] == 1:
                cnt += 1
            f.append(1.0 * cnt / total)
        x = np.array(range(1, len(res_i) + 1)) / len(res_i) * 100
        x = np.append(x[0:-1:100], x[-1])
        f = np.append(f[0:-1:100], f[-1])
        plt.plot(x, np.array(f) * 100, 's-', color = 'blue', label = "Shapley")

        ran_v = np.random.rand(len(res_v))
        ran_i = np.argsort(-ran_v)[::-1]
        cnt = 0
        f = []
        total = 0
        cnt = 0
        for i in range(len(ran_i)):
            if self.app.flip[int(ran_i[i])] == 1:
                total += 1
        for i in range(len(ran_i)):
            if self.app.flip[int(ran_i[i])] == 1:
                cnt += 1
            f.append(1.0 * cnt / total)
        x = np.array(range(1, len(ran_i) + 1)) / len(ran_i) * 100
        f = x / 100
        plt.plot(x, np.array(f) * 100, '--', color='red', label = "Random", zorder=7)

        plt.xlabel('Fraction of data inspected (%)', fontsize=15)
        plt.ylabel('Fraction of incorrect labels (%)', fontsize=15)
        plt.legend(loc='lower right', prop={'size': 15})
        plt.tight_layout()
        plt.show()
