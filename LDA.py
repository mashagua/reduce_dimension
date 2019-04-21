# * coding:utf-8 *
# @author    :mashagua
# @time      :2019/4/21 14:07
# @File      :LDA.py
# @Software  :PyCharm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
feature_dict = {i: label for i, label in zip(
                range(4),
                ('sepal length in cm',
                 'sepal width in cm',
                 'petal length in cm',
                 'petal width in cm', ))}

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
)
df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True)  # to drop the empty line at file-end
X = df.iloc[:, [0, 1, 2, 3]].values
y = df['class label'].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1, 4):
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl - 1]))

overall_mean = np.mean(X, axis=0)
#Between-class scatter matrix
S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i + 1, :].shape[0]
    print(n)
    mean_vec = mean_vec.reshape(4, 1)  # make column vector
    overall_mean = overall_mean.reshape(4, 1)  # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)
#Within-class scatter matrix
S_W = np.zeros((4, 4))
for cl, mv in zip(range(1, 4), mean_vectors):
    # scatter matrix for every class
    class_sc_mat = np.zeros((4, 4))
    for row in X[y == cl]:
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)  # make column vectors
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)
#step3 Solving the generalized eigenvalue problem for the matrix
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(4, 1)
    print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))

#step4 sort the enginer values
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
#step5:get the matrix
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', W.real)


# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)
