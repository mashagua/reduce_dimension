# * coding:utf-8 *
# @author    :mashagua
# @time      :2019/4/21 14:07
# @File      :LDA.py
# @Software  :PyCharm
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

S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i + 1, :].shape[0]
    print(n)
    mean_vec = mean_vec.reshape(4, 1)  # make column vector
    overall_mean = overall_mean.reshape(4, 1)  # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)
