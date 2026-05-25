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


