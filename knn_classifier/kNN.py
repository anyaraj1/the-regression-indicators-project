import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from utils import mnist_reader

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', "Bag", "Ankle boot"]

print("Loading and prepping data.......")
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind = 'train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train_flat = X_train / 255.0
X_test_flat = X_test / 255.0