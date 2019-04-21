# * coding:utf-8 *
# @author    :mashagua
# @time      :2019/4/11 20:52
# @File      :pca.py
# @Software  :PyCharm
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import matplotlib
from matplotlib import offsetbox
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
