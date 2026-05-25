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

fig, axes = plt.subplots(2, 5, figsize = (10,4))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx].reshape(28, 28), cmap = 'gray')
    ax.set_title(labels[i])
    ax.axis('off')
plt.tight_layout()
plt.savefig("01_data_preview_knn.png")
print("Saved the data preview to '01_data_preview_knn.png")

X_train_sub = X_train_flat[:10000]
y_train_sub = y_train[:10000]
X_test_final = X_test_flat[:2000]
y_test_final = y_test[:2000]

X_train_split = X_train_sub[:8000]
y_train_split = y_train_sub[:8000]
X_val_split = X_train_sub[8000:]
y_val_split = y_train_sub[8000:]

print("Hyperparameter Tuning (validation set)")
k_values = [1,3,5,7,9,15]
val_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X+X_train_split, y_train_split)
    score = knn.score(X_val_split, y_val_split)
    val_scores.append(score)
    print(f"Tested k = {k:2d} | Validation accuracy: {score:.4f}")

best_k = k_values[np.argmax(val_scores)]
print(f"-> Chosen optimal k = {best_k}")

plt.figure(figsize=(6,4))
plt.plot(k_values, val_scores, marker = 'o', color = 'b')
plt.title("Hyperparameter Tuning: k V.S Validatoin Accuracy")
plt.xlabel('k (Number of neighbors)')
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("02_tuning_curve.png")
print("Saved hyperparameter tuning to '02_tuning_curve.png'")