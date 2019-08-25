import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from  sklearn.decomposition import PCA

'''
PCA证明： https://blog.csdn.net/huizxhhui1994/article/details/79125972
'''
x = np.loadtxt('data_wine.txt', dtype=float, delimiter=',', usecols=[0, 1, 2, 3])
y = np.loadtxt('data_wine.txt', dtype=float, delimiter=',', usecols=[6])


'feature scaling'
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x_scale = (x-mean)/std

# ### feature scaling using sklearn
# standard = StandardScaler()
# standard.fit(x)                # cal mean/std
# x_scale2=standard.transform(x) # scaling
#
# # ##OR
# # x_scale3 = standard.fit_transform(x) # fit + transform
# # print(x_scale3)

# print(x_scale)

'cal cov & eigen'
m, n = x_scale.shape
cov = x_scale.T.dot(x_scale)/(m-1)
# # OR
# cov = np.cov(x_scale.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov)

eigen_pairs = {eigen_vals[i]:eigen_vecs[:, i] for i in range(n)}
print(eigen_pairs)

eigen_pairs_sorted = sorted(eigen_pairs.items(), key=lambda x:x[0], reverse=True)
print(eigen_pairs_sorted)

E = np.array([eigen_pairs_sorted[i][1] for i in range(n)]).T

d=3
x_new = x_scale.dot(E[:,:d])
#
linear = LinearRegression(fit_intercept=True)
linear.fit(x_new, y)
# coef = E[:,:d].dot(linear.coef_)
# print(coef)

# predict = x_scale.dot(coef)+linear.intercept_
# print(predict)


pca = PCA(n_components=3)
new=pca.fit_transform(x_scale)
linear = LinearRegression(fit_intercept=True)
linear.fit(new, y)


print(pca.explained_variance_)        # eigen values
print(pca.explained_variance_ratio_)  # Percentage of variance explained by each eigen values/vectors
print(pca.components_)                # eigen vectors
print(pca.singular_values_)           # 对应的每个成分的奇异值
print(pca.n_components_)              # the expected dimensionality after pca
