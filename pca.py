# * coding:utf-8 *
# @author    :mashagua
# @time      :2019/4/11 20:52
# @File      :pca.py
# @Software  :PyCharm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn import manifold, datasets, decomposition, discriminant_analysis
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox
# from keras.datasets import mnist
#  import matplotlib
# matplotlib.use('agg')
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# input_shape = x_train.shape[1:]
#
# fig = plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(x_train[0], cmap='gray', interpolation='none')
# plt.title("Digit: {}".format(y_train[0]))
# plt.xticks([])
# plt.yticks([])
# plt.subplot(2, 1, 2)
# plt.hist(x_train[0].reshape(784))
# plt.title("Pixel Value Distribution")
# fig
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#
# # normalizing the data to help with the training
# x_train /= 255
# x_test /= 255
np.set_printoptions(precision=4)
digits = datasets.load_digits()
x = digits.data
y = digits.target
n_samples,n_features=x.shape
##
mean_vectors = []
for cl in range(0,10):
    mean_vectors.append(np.mean(x[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))
   
overall_mean = np.mean(x, axis=0)
S_B = np.zeros((64, 64))
for i, mean_vec in enumerate(mean_vectors):
    print(i)
    n = x[y == i, :].shape[0]
    mean_vec = mean_vec.reshape(64, 1)  # make column vector
    overall_mean = overall_mean.reshape(64, 1)  # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('between-class Scatter Matrix:\n', S_B)

#Within-class scatter matrix
S_W = np.zeros((64, 64))
for cl, mv in zip(range(0, 10), mean_vectors):
    # scatter matrix for every class
    class_sc_mat = np.zeros((64, 64))
    for row in x[y == cl]:
        row, mv = row.reshape(64, 1), mv.reshape(64, 1)  # make column vectors
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)
#step3 Solving the generalized eigenvalue problem for the matrix
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(64, 1)
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



def embedding_plot(x, title):
    # axis=0是每一列
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=y/ 10.)
    shown_images = np.array([[1., 1.]])
    for i in range(x.shape[0]):
        if np.min(np.sum((x[i]-shown_images)**2,axis=1))<1e-2:
            continue
        shown_images = np.r_[shown_images, [x[i]]]
        ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i],cmap=plt.cm.gray_r),x[i]))

    plt.xticks([]),plt.yticks([])
    plt.title(title)

x_pca=decomposition.PCA(n_components=2).fit_transform(x)
embedding_plot(x_pca,"PCA")
plt.show()

# LDA
sklearn_lda = LDA(n_components=2)
x_lda = sklearn_lda.fit_transform(x, y)
embedding_plot(x_lda,"LDA")
plt.show()

# def pca(x, k):
#     n_samples, n_features = x.shape
#
#
# if __name__ == '__main__':
#     pass
