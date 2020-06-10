import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
flip = pickle.load(open("flip.pkl", "rb"), encoding = "iso-8859-1")

# loo_v = pickle.load(open("loo.pkl", "rb"), encoding = "iso-8859-1")["loo"]
# loo_i = np.argsort(-loo_v)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(loo_i)):
#     if flip[int(loo_i[i])] == 1:
#         total += 1
# for i in range(len(loo_i)):
#     if flip[int(loo_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(loo_i) + 1)) / len(loo_i) * 100
# x = np.append(x[0:-1:10], x[-1])
# f = np.append(f[0:-1:10], f[-1])
# plt.plot(x, np.array(f) * 100, '^-', color = 'olive', label = "Leave-One-Out", zorder=4, alpha=0.8)

# tmc_v = pickle.load(open("tmc.pkl", "rb"), encoding = "iso-8859-1")
# tmc_i = np.argsort(-tmc_v)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(tmc_i)):
#     if flip[int(tmc_i[i])] == 1:
#         total += 1
# for i in range(len(tmc_i)):
#     if flip[int(tmc_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(tmc_i) + 1)) / len(tmc_i) * 100
# x = np.append(x[0:-1:10], x[-1])
# f = np.append(f[0:-1:10], f[-1])
# plt.plot(x, np.array(f) * 100, 's-', color = 'blue', label = "TMC-Shapley")

# # Only LogisticRegression and NN model have G-Shapley metrics
# g_v = pickle.load(open("g.pkl", "rb"), encoding = "iso-8859-1")
# g_i = np.argsort(-g_v)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(g_i)):
#     if flip[int(g_i[i])] == 1:
#         total += 1
# for i in range(len(g_i)):
#     if flip[int(g_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(g_i) + 1)) / len(g_i) * 100
# x = np.append(x[0:-1:10], x[-1])
# f = np.append(f[0:-1:10], f[-1])
# plt.plot(x, np.array(f) * 100, 's-', color = 'orange', label = "G-Shapley", zorder=5)

# for K in range(10, 11):
#     knn_v = pickle.load(open('looknn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
#     knn_i = np.argsort(-knn_v)[::-1]
#     cnt = 0
#     f = []
#     total = 0
#     cnt = 0
#     for i in range(len(knn_i)):
#         if flip[int(knn_i[i])] == 1:
#             total += 1
#     for i in range(len(knn_i)):
#         if flip[int(knn_i[i])] == 1:
#             cnt += 1
#         f.append(1.0 * cnt / total)
#     x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
#     x = np.append(x[0:-1:10], x[-1])
#     f = np.append(f[0:-1:10], f[-1])
#     plt.plot(x, np.array(f) * 100, 'o-', color='violet', label = 'KNN-LOO-Shapley'.format(K), zorder=6, alpha=0.8)

colors = ["#E6CAFF", "#DCB5FF", "#d3a4ff", "#CA8EFF", "#BE77FF", "#B15BFF", "#9F35FF", "#921AFF"]
for K in range(10, 11):
    knn_v = pickle.load(open('knn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
    knn_v = np.mean(knn_v, axis=1)
    knn_i = np.argsort(-knn_v)[::-1]
    cnt = 0
    f = []
    total = 0
    cnt = 0
    for i in range(len(knn_i)):
        if flip[int(knn_i[i])] == 1:
            total += 1
    for i in range(len(knn_i)):
        if flip[int(knn_i[i])] == 1:
            cnt += 1
        f.append(1.0 * cnt / total)
    x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
    x = np.append(x[0:-1:10], x[-1])
    f = np.append(f[0:-1:10], f[-1])
    plt.plot(x, np.array(f) * 100, 'o-', color='purple', label = 'KNN-Shapley'.format(K), linewidth=3)

ran_v = np.random.rand(len(knn_v))
ran_i = np.argsort(-ran_v)[::-1]
cnt = 0
f = []
total = 0
cnt = 0
for i in range(len(ran_i)):
    if flip[int(ran_i[i])] == 1:
        total += 1
for i in range(len(ran_i)):
    if flip[int(ran_i[i])] == 1:
        cnt += 1
    f.append(1.0 * cnt / total)
x = np.array(range(1, len(ran_i) + 1)) / len(ran_i) * 100
f = x / 100
plt.plot(x, np.array(f) * 100, '--', color='red', label = "Random", zorder=7)



plt.xlabel('Fraction of data inspected (%)')
plt.ylabel('Fraction of incorrect labels (%)')
plt.legend(loc='lower right')
plt.show()