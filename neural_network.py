import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import utils.mnist_reader as reader

seed = 1234
np.random.seed(seed)

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', "Bag", "Ankle boot"]

X_tr, y_tr = reader.load_mnist('data/fashion', kind='train')
X_te, y_te = reader.load_mnist('data/fashion', kind='t10k')

X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=1/6, random_state=seed)

X_tr_flat = X_tr / 255.0
X_val_flat = X_val / 255.0
X_te_flat = X_te / 255.0

tuning_results = []
best_val_acc = -1
final_hyperparameters = {}
best_mlp = None

learning_rates = [0.0005, 0.00075]
alphas = [0.01, 0.015, 0.02]
hidden_layers = [(512, 256, 128), (1024,)]

for lr in learning_rates:
    for a in alphas:
        for h in hidden_layers:
            config = {
                'learning_rate_init': lr,
                'alpha': a,
                'hidden_layer': h
            }

            print(f"Training model {len(tuning_results) + 1}/12 | LR: {lr:<7} | Alpha: {a:<5} ... ", end="", flush=True)

            mlp_model = MLPClassifier(
                hidden_layer_sizes=config['hidden_layer'], 
                alpha=config['alpha'],
                learning_rate_init=config['learning_rate_init'], 
                activation='relu', 
                solver='adam',
                batch_size=128, 
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=seed,
            )

            mlp_model.fit(X_tr_flat, y_tr)

            train_acc = mlp_model.score(X_tr_flat, y_tr)
            val_acc = mlp_model.score(X_val_flat,y_val)

            print(f"Done. Val Acc: {val_acc:.4f}")

            tuning_results.append({
                'config': config,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

            if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_hyperparameters = config
                    best_mlp = mlp_model

final_test_accuracy = best_mlp.score(X_te_flat, y_te)

print('Number of configurations tried:', len(tuning_results))
print('Final hyperparameters:', final_hyperparameters)
print('Final test accuracy:  ', final_test_accuracy)

tuning_table = pd.DataFrame([
    {**r['config'], 'train_acc': r['train_acc'], 'val_acc': r['val_acc']}
    for r in tuning_results
])

plt.figure(figsize=(12, 6))
x = np.arange(len(tuning_table))
width = 0.35

plt.bar(x - width/2, tuning_table['train_acc'], width, label='Train Accuracy', color='blue')
plt.bar(x + width/2, tuning_table['val_acc'], width, label='Val Accuracy', color='orange')

x_labels = [f"LR:{r['learning_rate_init']}\nA:{r['alpha']}\n{r['hidden_layer']}" for _, r in tuning_table.iterrows()]
plt.xticks(x, x_labels, rotation=45, ha='right')

plt.title('Hyperparameter Tuning Comparison (Fashion-MNIST)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.80, 0.95)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()