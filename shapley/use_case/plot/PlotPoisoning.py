import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE
import seaborn as sns

sns.set()

watermarked = pickle.load(open("watermarked.pkl", "rb"), encoding = "iso-8859-1")
# tmc_v = pickle.load(open('tmc.pkl', 'rb'), encoding = "iso-8859-1")
# tmc_i = np.argsort(-tmc_v)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(tmc_i)):
#     if watermarked[int(tmc_i[i])] == 1:
#         total += 1
# for i in range(len(tmc_i)):
#     if watermarked[int(tmc_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(tmc_i) + 1)) / len(tmc_i) * 100
# x = np.append(x[0:-1:200], x[-1])
# f = np.append(f[0:-1:200], f[-1])
# plt.plot(x, np.array(f) * 100, 's-', color = 'blue', label = "TMC-Shapley")

# g_v = pickle.load(open('g.pkl', 'rb'), encoding = "iso-8859-1")
# g_i = np.argsort(-g_v)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(g_i)):
#     if watermarked[int(g_i[i])] == 1:
#         total += 1
# for i in range(len(g_i)):
#     if watermarked[int(g_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(g_i) + 1)) / len(g_i) * 100
# x = np.append(x[0:-1:200], x[-1])
# f = np.append(f[0:-1:200], f[-1])
# plt.plot(x, np.array(f) * 100, 's-', color = 'orange', label = "G-Shapley", zorder=5)


# loo_v = pickle.load(open('loo.pkl', 'rb'), encoding = "iso-8859-1")["loo"]
# loo_i = np.argsort(-loo_v)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(loo_i)):
#     if watermarked[int(loo_i[i])] == 1:
#         total += 1
# for i in range(len(loo_i)):
#     if watermarked[int(loo_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(loo_i) + 1)) / len(loo_i) * 100
# x = np.append(x[0:-1:200], x[-1])
# f = np.append(f[0:-1:200], f[-1])
# plt.plot(x, np.array(f) * 100, '^-', color = 'olive', label = "Leave-One-Out", zorder=4, alpha=0.8)

# for K in range(10, 11):
#     knn_v = pickle.load(open('knn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
#     knn1_v = pickle.load(open('knn_layer1_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
#     knn2_v = pickle.load(open('knn_layer2_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
#     knn3_v = pickle.load(open('knn_layer3_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
#     knn_v = (knn1_v + knn2_v + knn3_v + knn_v) / 4
#     knn_i = np.argsort(-knn_v)[::-1]
#    cnt = 0
#     f = []
#     total = 0
#     cnt = 0
#     for i in range(len(knn_i)):
#         if watermarked[int(knn_i[i])] == 1:
#             total += 1
#     for i in range(len(knn_i)):
#         if watermarked[int(knn_i[i])] == 1:
#             cnt += 1
#         f.append(1.0 * cnt / total)
#     x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
#     plt.plot(x, np.array(f) * 100, color = 'violet', label = 'average-KNN-Shapley (k={})'.format(K))

# for K in range(10, 11):
#     knn_v = pickle.load(open('looknn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
#     knn_i = np.argsort(-knn_v)[::-1]
#     cnt = 0
#     f = []
#     total = 0
#     cnt = 0
#     a1 = []
#     a0 = []
#     for i in range(len(knn_i)):
#         if watermarked[int(knn_i[i])] == 1:
#             total += 1
#     for i in range(len(knn_i)):
#         if watermarked[int(knn_i[i])] == 1:
#             cnt += 1
#         f.append(1.0 * cnt / total)
#     x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
#     x = np.append(x[0:-1:200], x[-1])
#     f = np.append(f[0:-1:200], f[-1])
#     plt.plot(x, np.array(f) * 100, 'o-', color='violet', label = 'KNN-LOO-Shapley'.format(K), zorder=6, alpha=0.8)


for K in range(10, 11):
    knn_v = pickle.load(open('knn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
    knn_v = np.mean(knn_v, axis=1)
    knn_i = np.argsort(-knn_v)[::-1]
    cnt = 0
    f = []
    total = 0
    cnt = 0
    for i in range(len(knn_i)):
        if watermarked[int(knn_i[i])] == 1:
            total += 1
    for i in range(len(knn_i)):
        if watermarked[int(knn_i[i])] == 1:
            cnt += 1
        f.append(1.0 * cnt / total)
    x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
    x = np.append(x[0:-1:200], x[-1])
    f = np.append(f[0:-1:200], f[-1])
    plt.plot(x, np.array(f) * 100, 'o-', color='purple', label = 'KNN-Shapley'.format(K), linewidth=3)
    # for i in range(len(knn_i)):
    #     if watermarked[int(knn_i[i])] == 1:
    #         print(knn_v[knn_i[i]])
    #         a1.append(knn_v[knn_i[i]])
    #     else:
    #         a0.append(knn_v[knn_i[i]])
    # plt.hist(a0, bins=30, color='blue', histtype='stepfilled', label = 'benign data')
    # plt.hist(a1, bins=30, color='red', histtype='stepfilled', label = 'poisoned data')
    # 
for K in range(10, 11):
    knn_v = pickle.load(open('knn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
    knn_v = np.max(knn_v, axis=1)
    knn_i = np.argsort(-knn_v)[::-1]
    cnt = 0
    f = []
    total = 0
    cnt = 0
    for i in range(len(knn_i)):
        if watermarked[int(knn_i[i])] == 1:
            total += 1
    for i in range(len(knn_i)):
        if watermarked[int(knn_i[i])] == 1:
            cnt += 1
        f.append(1.0 * cnt / total)
    x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
    x = np.append(x[0:-1:200], x[-1])
    f = np.append(f[0:-1:200], f[-1])
    plt.plot(x, np.array(f) * 100, 'o-', color='green', label = 'max-KNN-Shapley'.format(K), linewidth=3)
#     # a = []
#     # plt.figure(figsize=(12, 6))
#     # plt.subplot(121)
#     # for j in range(len(knn_v)):
#     #     if not watermarked[j]:
#     #         a.append(np.mean(knn_v[j]))
#     # plt.hist(a, bins=100, range=(0, 0.002), color='blue', histtype='stepfilled', label = 'benign (mean value)')
#     # a = []
#     # plt.subplot(122)
#     # for j in range(len(knn_v)):
#     #     if watermarked[j]:
#     #         a.append(np.mean(knn_v[j]))
#     # plt.hist(a, bins=100, range=(0, 0.002), color='red', histtype='stepfilled', label = 'watermarked (mean value)')

# knn_v = pickle.load(open('knn_{}.pkl'.format(K), 'rb'), encoding = "iso-8859-1")
# knn_v = np.sort(knn_v, axis=1)
# pca = PCA(n_components=2)
# pca.fit(knn_v)
# knn_v_pca = pca.fit_transform(knn_v)
# # the bandwidth can be tunable
# kde = KernelDensity(kernel='exponential', bandwidth=0.02).fit(knn_v_pca)
# score = kde.score_samples(knn_v_pca)
# knn_i = np.argsort(-score)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(knn_i)):
#     if watermarked[int(knn_i[i])] == 1:
#         total += 1
# for i in range(len(knn_i)):
#     if watermarked[int(knn_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
# plt.plot(x, np.array(f) * 100, color = 'blue', label = 'KDE-KNN-Shapley (k={})'.format(K))


# tsne = TSNE(n_components=2,perplexity=50)
# knn_v_tsne = tsne.fit_transform(knn_v)
# knn_mean = np.mean(knn_v_tsne, axis=0)
# knn_cov = np.cov(knn_v_tsne, rowvar=0)
# score = multivariate_normal.pdf(knn_v_tsne, mean=knn_mean, cov=knn_cov)
# knn_i = np.argsort(-score)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(knn_i)):
#     if watermarked[int(knn_i[i])] == 1:
#         total += 1
# for i in range(len(knn_i)):
#     if watermarked[int(knn_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
# plt.plot(x, np.array(f) * 100, color = 'darkblue', label = 'TSNE-KNN-Shapley (k={})'.format(K))

# pca = PCA(n_components=2)
# pca.fit(knn_v)
# knn_v_pca = pca.fit_transform(knn_v)
# knn_mean = np.mean(knn_v_pca, axis=0)
# knn_cov = np.cov(knn_v_pca, rowvar=0)
# score = multivariate_normal.pdf(knn_v_pca, mean=knn_mean, cov=knn_cov)
# knn_i = np.argsort(-score)[::-1]
# cnt = 0
# f = []
# total = 0
# cnt = 0
# for i in range(len(knn_i)):
#     if watermarked[int(knn_i[i])] == 1:
#         total += 1
# for i in range(len(knn_i)):
#     if watermarked[int(knn_i[i])] == 1:
#         cnt += 1
#     f.append(1.0 * cnt / total)
# x = np.array(range(1, len(knn_i) + 1)) / len(knn_i) * 100
# plt.plot(x, np.array(f) * 100, color = 'lightblue', label = 'Gaussian-KNN-Shapley (k={})'.format(K))

ran_v = np.random.rand(len(knn_v, ))
ran_i = np.argsort(-ran_v)[::-1]
cnt = 0
f = []
total = 0
cnt = 0
for i in range(len(ran_i)):
    if watermarked[int(ran_i[i])] == 1:
        total += 1
for i in range(len(ran_i)):
    if watermarked[int(ran_i[i])] == 1:
        cnt += 1
    f.append(1.0 * cnt / total)
x = np.array(range(1, len(ran_i) + 1)) / len(ran_i) * 100
f = x / 100
plt.plot(x, np.array(f) * 100, '--', color='red', label = "Random", zorder=7)

plt.xlabel('Fraction of data inspected (%)')
plt.ylabel('Fraction of backdoor images detected (%)')
plt.legend(loc='lower right')
plt.show()