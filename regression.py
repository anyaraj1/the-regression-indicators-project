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


tested = np.isin(y_test, [8,0])
X_test = X_test[tested]
y_test = y_test[tested]




X_train, X_value, y_train, y_value = train_test_split(X_train_from, y_train_from, test_size=0.25, random_state=seed, shuffle=True)

plt.figure(figsize=(20,15))
for q in range(4):
    plt.subplot(1,4,q+1)
    plt.imshow(X_train[q].reshape(28,28),cmap='gray')
    if y_train[q] == 0:
        name = "T-Shirt Data"
    else:
        name = 'Bag'
    plt.title(f"{name}")
    plt.axis('off')
plt.show()

min2 = X_train.min(axis=0)
max3 = X_train.max(axis=0)

transform_train = (X_train - min2) / (max3 - min2 + 1e-4)
transform_value = (X_value - min2) / (max3 - min2 + 1e-4)
transform_test = (X_test - min2) / (max3 - min2 + 1e-4)

y_train_check = (y_train ==0).astype(float)
y_value_check = (y_value ==0).astype(float)
y_testcheck = (y_test==0).astype(float)


regularization = [0.02, 0.1, 1, 10 ,50]

plt.figure(figsize=(10,10))

all_results = {}

for p in ['l1','l2']:
    all_accuracies = []
    for z in regularization:
        model =  LogisticRegression(penalty=p, C=z, solver='liblinear')
        model.fit(transform_train, y_train_check)

        prediction = model.predict(transform_value)

        accuracy2 = accuracy_score(y_value_check, prediction)
        all_accuracies.append(accuracy2)

        save_tuple = (p,z)
        all_results[save_tuple] =accuracy2

    plt.plot(regularization, all_accuracies, label = f"{p} Penalty", marker = 'o')

plt.xscale('log')
plt.xlabel ("Regularization Strength")
plt.ylabel ("Accuract of Validaation")
plt.title ("Regularization Strength and Validation Accuracy")
plt.legend()
plt.show()
        

all_scores = list(all_results.values())
bestindex = np.argmax(all_scores)
done = list(all_results.keys())[bestindex]

model2 = LogisticRegression(penalty=done[0], C=done[1], solver = 'liblinear')
model2.fit(transform_train, y_train_check)

testprediction = model2.predict(transform_test)
testaccuracy = accuracy_score(y_testcheck, testprediction)

find_errors = testprediction != y_testcheck
image = X_test[find_errors]
true3 = y_test[find_errors]
predictionerror = testprediction[find_errors]

plt.figure(figsize=(20,15))

for q in range(4):
    plt.subplot(1, 4, q+1)
    plt.imshow(image[q].reshape(28,28),cmap='gray')

    if true3[q] == 0:
        label1 = "T-shirt"
    else:
        label1 = 'Bag'
    
    if predictionerror[q] ==1:
        label2 = "T-shirt"
    else:
        label2 = 'Bag'
    
    plt.title(f"Predicted: {label2}, True: {label1}")    

    plt.axis('off')
plt.show()

