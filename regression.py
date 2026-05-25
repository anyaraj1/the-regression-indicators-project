import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import utils.mnist_reader as reader

seed = 1234
np.random.seed(1234)


X_train, y_train = reader.load_mnist('data/fashion', kind = 'train')
X_test, y_test = reader.load_mnist('data/fashion', kind= 't10k')


trained = np.isin(y_train, [8,0])
X_train_from = X_train[trained]
y_train_from = y_train[trained]


tested = np.isin(y_train, [8,0])
X_test2 = X_test[tested]
y_test2 = y_test[tested]




X_train, X_value, y_train, y_value = train_test_split(X_train_from, y_train_from, test_size=0.25, random_state=seed, shuffle=True)

plt.figure(figsize=(20,20))
for q in range(4):
    plt.subplot(1,4,q+1)
    plt.imshow(X_train[q].reshape(28,28))
    if y_train[q] == 0:
        name = "T-Shirt Data"
    else:
        name = 'Bag'
    plt.title(f"{name}")
    plt.axis('off')
plt.show()