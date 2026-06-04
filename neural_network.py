import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import utils.mnist_reader as reader

seed = 1234
np.random.seed(seed)

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', "Bag", "Ankle boot"]

X_tr, y_tr = reader.load_mnist('data/fashion', kind='train')
X_te, y_te = reader.load_mnist('data/fashion', kind='t10k')

X_tr_flat = X_tr / 255.0
X_te_flat = X_te / 255.0

print('X_tr:', X_tr.shape, 'X_te:', X_te.shape)