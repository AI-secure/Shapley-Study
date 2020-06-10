import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set()

x = np.array([10, 100, 200, 400, 800, 1000, 5000, 10000, 20000, 50000])
# knn = np.array([0.0003832538922627767, 0.004426650206247966, 0.01631486415863037, 0.06262378295262655, 0.25503607193628947, 0.3868168075879415, 6.302051556110382, 25.532700236638387, 102.27655944824218]) * 60
# loo = np.array([0.02149387200673421, 0.5834330042203267, 2.0351594130198163, 6.966519888242086, 24.5041117866834, 37.45188350280126]) * 60
# tmc = np.array([0.7461043953895569, 144.0786436120669]) * 60
# g = np.array([0.5796960711479187, 3.785581676165263, 9.595915234088897, 14.533872322241466, 46.74548430840174, 57.338612226645154]) * 60


knn = np.array([0.0769142468770345,   0.677141539255778,   1.653036856651306,   3.4390464584032694,  8.59050339460373,    12.708731484413146]) * 60
loo = np.array([0.7347790956497192,   66.44814310471217]) * 60
tmc = np.array([11.529986302057901]) * 60
g = np.array([0.12539432843526205, 0.9315359711647033, 3.903498136997223, 9.672818299134573, 50.83118432760239,150.22751605113348]) * 60


plt.loglog(x[0:loo.shape[0]], loo, '^-', color = 'olive', label = "Leave-One-Out")
plt.loglog(x[0:tmc.shape[0]], tmc, 's-', color = 'blue', label = "TMC-Shapley")
plt.loglog(x[0:knn.shape[0]], knn, 'o-', color='purple', label = 'KNN-Shapley')
plt.loglog(x[0:g.shape[0]], g, 's-', color = 'orange', label = "G-Shapley")

plt.xlabel('Number of training data points in log scale')
plt.ylabel('Running time in log scale (s)')
plt.legend(loc='lower right')
plt.show()