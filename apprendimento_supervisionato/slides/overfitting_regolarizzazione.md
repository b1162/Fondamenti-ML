---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Apprendimento Supervisionato
## Parte 3: Overfitting e Regolarizzazione

---

# Indice

1. Comprendere l'overfitting
2. Tecniche di regolarizzazione
3. Altre strategie per prevenire l'overfitting
4. Casi di studio e best practices
5. Implementazione pratica

---

# 1. Comprendere l'Overfitting

## Definizione
L'overfitting si verifica quando un modello apprende troppo dai dati di training, incluso il rumore, compromettendo la sua capacità di generalizzare a nuovi dati.

## Sintomi dell'overfitting
- Prestazioni eccellenti sul training set
- Prestazioni scarse sul validation/test set
- Modello complesso che cattura dettagli non rilevanti
- Alta varianza nelle predizioni

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_underfitting_overfitting_001.png)

---

# Bias vs. Varianza

## Bias (Errore sistematico)
- **Definizione**: tendenza del modello a sbagliare sistematicamente
- **Causa**: modello troppo semplice o assunzioni errate
- **Risultato**: underfitting, incapacità di catturare la struttura dei dati
- **Esempio**: approssimare una relazione quadratica con una linea retta

## Varianza (Sensibilità al rumore)
- **Definizione**: variabilità delle predizioni per piccole variazioni nei dati
- **Causa**: modello troppo complesso rispetto ai dati disponibili
- **Risultato**: overfitting, eccessiva sensibilità al rumore
- **Esempio**: polinomio di grado elevato che passa esattamente per tutti i punti

---

# Bias-Variance Tradeoff

## Il dilemma
- Ridurre il bias aumenta la varianza
- Ridurre la varianza aumenta il bias
- L'obiettivo è trovare il punto di equilibrio che minimizza l'errore totale

## Errore totale
$\text{Errore totale} = \text{Bias}^2 + \text{Varianza} + \text{Errore irriducibile}$

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_bias_variance_001.png)

---

# Curve di Apprendimento

## Definizione
Le curve di apprendimento mostrano come variano le performance del modello al variare della quantità di dati di training.

## Interpretazione
- **Underfitting**: sia l'errore di training che quello di validation sono alti e vicini
- **Overfitting**: errore di training basso, errore di validation alto
- **Buona generalizzazione**: entrambi gli errori bassi e vicini

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Calcolare le curve di apprendimento
train_sizes, train_scores, valid_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

# Calcolare media e deviazione standard
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

# Visualizzare le curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.title("Learning Curves")
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Validation Curve

## Definizione
Le validation curve mostrano come variano le performance del modello al variare di un iperparametro.

## Utilizzo
- Identificare il valore ottimale di un iperparametro
- Riconoscere quando un modello inizia a overfittare

```python
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

# Definire il range dell'iperparametro (es. max_depth per un albero)
param_range = np.arange(1, 11)

# Calcolare le validation curve
train_scores, valid_scores = validation_curve(
    model, X, y, param_name="max_depth", param_range=param_range, cv=5)

# Calcolare media e deviazione standard
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

# Visualizzare le curve
plt.figure(figsize=(10, 6))
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(param_range, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color="g")
plt.plot(param_range, train_mean, 'o-', color="r", label="Training score")
plt.plot(param_range, valid_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("max_depth")
plt.ylabel("Score")
plt.legend(loc="best")
plt.title("Validation Curves")
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Underfitting vs. Overfitting

## Underfitting
- **Sintomi**: errore alto sia su training che su validation
- **Cause**: modello troppo semplice, feature insufficienti
- **Soluzioni**:
  - Aumentare la complessità del modello
  - Aggiungere feature più informative
  - Ridurre la regolarizzazione

## Overfitting
- **Sintomi**: errore basso su training, alto su validation
- **Cause**: modello troppo complesso, dati di training insufficienti
- **Soluzioni**:
  - Raccogliere più dati
  - Ridurre la complessità del modello
  - Applicare tecniche di regolarizzazione
  - Utilizzare ensemble methods

---

# 2. Tecniche di Regolarizzazione

## Definizione
La regolarizzazione è un insieme di tecniche che prevengono l'overfitting limitando la complessità del modello.

## Principio generale
Aggiungere un termine di penalità alla funzione obiettivo:
$\text{Funzione obiettivo regolarizzata} = \text{Errore sui dati} + \lambda \cdot \text{Termine di regolarizzazione}$

Dove:
- $\lambda$ è il parametro di regolarizzazione che controlla il trade-off
- Il termine di regolarizzazione penalizza la complessità del modello

---

# Early Stopping

## Definizione
Interrompere l'addestramento quando le performance sul validation set iniziano a peggiorare.

## Vantaggi
- Semplice da implementare
- Efficace per molti algoritmi iterativi
- Non richiede modifiche al modello

## Implementazione
```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

# Suddividere i dati
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Inizializzare il modello
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, tol=1e-8, 
                    early_stopping=True, validation_fraction=0.2)

# Addestrare il modello
model.fit(X_train, y_train)

# Visualizzare la curva di loss
plt.figure(figsize=(10, 6))
plt.plot(model.loss_curve_, label='Training loss')
plt.plot(model.validation_scores_, label='Validation score')
plt.axvline(x=model.best_iteration_, color='r', linestyle='--', 
           label=f'Early stopping iteration: {model.best_iteration_}')
plt.xlabel('Iterations')
plt.ylabel('Loss / Score')
plt.legend()
plt.title('Early Stopping')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Regolarizzazione L1 (Lasso)

## Definizione
Aggiunge una penalità proporzionale alla somma dei valori assoluti dei parametri.
$\text{Penalità L1} = \lambda \sum_{i=1}^{n} |w_i|$

## Caratteristiche
- Promuove la sparsità (molti parametri esattamente zero)
- Utile per selezione automatica delle feature
- Robusta agli outlier

## Implementazione
```python
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np

# Range di valori per alpha
alphas = np.logspace(-4, 1, 50)
coefs = []

# Calcolare i coefficienti per diversi valori di alpha
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

# Visualizzare i coefficienti
plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('Lasso coefficients as a function of alpha')
plt.axis('tight')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Regolarizzazione L2 (Ridge)

## Definizione
Aggiunge una penalità proporzionale alla somma dei quadrati dei parametri.
$\text{Penalità L2} = \lambda \sum_{i=1}^{n} w_i^2$

## Caratteristiche
- Riduce l'impatto di tutte le feature, senza eliminarle completamente
- Efficace contro la multicollinearità
- Soluzione più stabile rispetto a L1

## Implementazione
```python
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np

# Range di valori per alpha
alphas = np.logspace(-4, 1, 50)
coefs = []

# Calcolare i coefficienti per diversi valori di alpha
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

# Visualizzare i coefficienti
plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('Ridge coefficients as a function of alpha')
plt.axis('tight')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Confronto tra Regolarizzazione L1 e L2

| Caratteristica | L1 (Lasso) | L2 (Ridge) |
|----------------|------------|------------|
| Forma geometrica | Rombo (norma L1) | Cerchio (norma L2) |
| Effetto sui coefficienti | Alcuni esattamente zero | Tutti ridotti ma non zero |
| Selezione delle feature | Sì (implicita) | No |
| Stabilità | Meno stabile | Più stabile |
| Multicollinearità | Seleziona una feature tra quelle correlate | Distribuisce il peso tra feature correlate |
| Soluzione | Può essere non unica | Sempre unica |

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_ridge_path_001.png)

---

# Elastic Net

## Definizione
Combina le penalità L1 e L2 per ottenere il meglio di entrambe.
$\text{Penalità Elastic Net} = \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$

## Caratteristiche
- Bilancia sparsità (L1) e stabilità (L2)
- Utile quando il numero di feature è maggiore del numero di osservazioni
- Gestisce efficacemente gruppi di feature correlate

## Implementazione
```python
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import numpy as np

# Definire il modello
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)

# Valutare il modello
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Visualizzare i coefficienti
coef = pd.Series(model.coef_, index=X.columns)
coef = coef[coef != 0].sort_values()

plt.figure(figsize=(10, 6))
coef.plot(kind='bar')
plt.title('Elastic Net Coefficients (non-zero)')
plt.tight_layout()
plt.show()
```

---

# Dropout

## Definizione
Tecnica di regolarizzazione per reti neurali che "spegne" casualmente alcuni neuroni durante l'addestramento.

## Funzionamento
1. Durante ogni iterazione, ogni neurone ha una probabilità p di essere temporaneamente rimosso
2. Durante l'inferenza, tutti i neuroni sono attivi ma i loro output sono scalati di (1-p)

## Vantaggi
- Previene la co-adaptazione dei neuroni
- Simula un ensemble di reti più piccole
- Riduce significativamente l'overfitting

![height:300px](https://miro.medium.com/max/1400/1*iWQzxhVlvadk6VAJjsgXgg.png)

---

# Implementazione del Dropout

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Definire il modello con dropout
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # 30% dei neuroni verranno "spenti" casualmente
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

# Compilare il modello
model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Addestrare il modello
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Visualizzare la curva di loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss with Dropout')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Batch Normalization

## Definizione
Tecnica che normalizza gli input di ogni layer, riducendo lo spostamento della distribuzione dei dati (internal covariate shift).

## Funzionamento
1. Normalizza gli output di un layer prima di passarli al successivo
2. Apprende parametri di scala e shift per mantenere la capacità espressiva

## Vantaggi
- Accelera l'addestramento
- Riduce la sensibilità all'inizializzazione dei pesi
- Ha un effetto regolarizzante
- Permette learning rate più alti

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

# Definire il modello con batch normalization
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dense(64),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dense(1)
])
```

---

# 3. Altre Strategie per Prevenire l'Overfitting

## Data Augmentation

### Definizione
Tecnica che aumenta artificialmente la dimensione del dataset di training generando nuove istanze da quelle esistenti.

### Applicazioni comuni
- **Computer Vision**: rotazioni, traslazioni, zoom, flip, cambiamenti di luminosità
- **Natural Language Processing**: sinonimi, back-translation, word swapping
- **Time Series**: aggiunta di rumore, warping temporale, slicing

### Vantaggi
- Aumenta la dimensione effettiva del dataset
- Migliora la robustezza del modello
- Riduce l'overfitting

---

# Implementazione della Data Augmentation

```python
# Per immagini con Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurare il generatore
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Visualizzare esempi di immagini aumentate
import matplotlib.pyplot as plt

# Assumendo che X_batch sia un batch di immagini
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (X_aug, y_aug) in enumerate(datagen.flow(X_batch, y_batch, batch_size=1)):
    if i >= 5:
        break
    axes[i].imshow(X_aug[0])
    axes[i].set_title(f"Class: {y_aug[0]}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

---

# Feature Selection

## Definizione
Processo di selezione di un sottoinsieme di feature rilevanti per costruire il modello.

## Tecniche
- **Filter methods**: valutano le feature indipendentemente dal modello (correlazione, mutual information)
- **Wrapper methods**: valutano sottoinsiemi di feature usando il modello (recursive feature elimination)
- **Embedded methods**: la selezione è parte del processo di addestramento (Lasso, tree-based importance)

## Vantaggi
- Riduce la dimensionalità
- Migliora le performance
- Riduce l'overfitting
- Aumenta l'interpretabilità

---

# Implementazione della Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

# 1. Filter method: SelectKBest
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)

# Visualizzare i punteggi
scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
})
scores = scores.sort_values('Score', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(scores['Feature'], scores['Score'])
plt.xticks(rotation=45)
plt.title('Feature Importance (f_regression)')
plt.tight_layout()
plt.show()

# 2. Wrapper method: Recursive Feature Elimination
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=5, step=1)
rfe.fit(X, y)

# Visualizzare i risultati
feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': rfe.ranking_,
    'Selected': rfe.support_
})
feature_ranking = feature_ranking.sort_values('Ranking')

print("Selected features:")
print(feature_ranking[feature_ranking['Selected']]['Feature'].tolist())
```

---

# Dimensionality Reduction

## Definizione
Processo di riduzione del numero di variabili random in considerazione, trasformando le feature originali in un nuovo spazio.

## Tecniche
- **Principal Component Analysis (PCA)**: proietta i dati lungo le direzioni di massima varianza
- **t-SNE**: mantiene le relazioni di vicinanza, utile per visualizzazione
- **UMAP**: simile a t-SNE ma più scalabile
- **Autoencoder**: reti neurali che apprendono rappresentazioni compresse

## Vantaggi
- Riduce la complessità del modello
- Elimina la multicollinearità
- Migliora la visualizzazione
- Riduce l'overfitting

---

# Implementazione della Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Standardizzare i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicare PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Visualizzare la varianza spiegata
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance threshold')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.title('PCA: Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# Determinare il numero di componenti per spiegare il 95% della varianza
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed to explain 95% of variance: {n_components}")

# Visualizzare i dati in 2D
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.colorbar(label='Target')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First two components')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# Ensemble Methods

## Definizione
Tecniche che combinano multiple modelli per ottenere performance migliori di quelle ottenibili con un singolo modello.

## Tipi principali
- **Bagging**: addestra modelli su sottoinsiemi bootstrap dei dati (es. Random Forest)
- **Boosting**: addestra modelli sequenzialmente, focalizzandosi sugli errori (es. AdaBoost, Gradient Boosting)
- **Stacking**: combina le predizioni di diversi modelli usando un meta-learner

## Vantaggi
- Riduce la varianza (bagging)
- Riduce il bias (boosting)
- Migliora la robustezza e la generalizzazione
- Spesso produce risultati state-of-the-art

---

# Implementazione di Ensemble Methods

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

# Definire i modelli base
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
model3 = SVR(kernel='rbf')

# Creare un ensemble con voting
ensemble = VotingRegressor([
    ('rf', model1),
    ('gb', model2),
    ('svr', model3)
])

# Addestrare i modelli
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

# Valutare i modelli
models = [model1, model2, model3, ensemble]
names = ['Random Forest', 'Gradient Boosting', 'SVR', 'Ensemble']
mse_scores = []

for model in models:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    
# Visualizzare i risultati
plt.figure(figsize=(10, 6))
plt.bar(names, mse_scores)
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

# Pruning degli Alberi di Decisione

## Definizione
Processo di rimozione di parti dell'albero che forniscono poco potere predittivo, per ridurre la complessità e prevenire l'overfitting.

## Tipi di pruning
- **Pre-pruning**: limita la crescita dell'albero durante la costruzione (max_depth, min_samples_leaf)
- **Post-pruning**: costruisce l'albero completo e poi rimuove i rami meno utili

## Vantaggi
- Riduce la complessità del modello
- Migliora la generalizzazione
- Aumenta l'interpretabilità
- Riduce i costi computazionali

---

# Implementazione del Pruning

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Albero senza pruning
tree_unpruned = DecisionTreeClassifier(random_state=42)
tree_unpruned.fit(X_train, y_train)

# Albero con pre-pruning
tree_pruned = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree_pruned.fit(X_train, y_train)

# Valutare i modelli
models = [tree_unpruned, tree_pruned]
names = ['Unpruned Tree', 'Pruned Tree']

for i, model in enumerate(models):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"{names[i]}:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Number of nodes: {model.tree_.node_count}")
    print()

# Visualizzare gli alberi
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
plot_tree(tree_unpruned, filled=True, feature_names=X.columns, class_names=True, ax=axes[0])
axes[0].set_title('Unpruned Decision Tree')
plot_tree(tree_pruned, filled=True, feature_names=X.columns, class_names=True, ax=axes[1])
axes[1].set_title('Pruned Decision Tree')
plt.tight_layout()
plt.show()
```

---

# 4. Casi di Studio e Best Practices

## Riconoscimento dell'Overfitting in Scenari Reali

### Segnali di overfitting
- Grande divario tra performance di training e validation
- Modello molto complesso rispetto alla quantità di dati
- Coefficienti o pesi con valori estremi
- Predizioni non realistiche o troppo "perfette"
- Scarsa performance su nuovi dati

### Esempio: Modello di credit scoring
- Modello con 100% di accuratezza sul training set
- Solo 65% di accuratezza sul test set
- Decisioni basate su pattern casuali nei dati di training
- Conseguenza: approvazione di prestiti ad alto rischio

---

# Strategie di Regolarizzazione per Diversi Algoritmi

| Algoritmo | Tecniche di regolarizzazione consigliate |
|-----------|------------------------------------------|
| Regressione lineare | Ridge, Lasso, Elastic Net |
| Regressione logistica | L1/L2 regularization, early stopping |
| Alberi di decisione | Pruning, max_depth, min_samples_leaf |
| Random Forest | n_estimators, max_features, bootstrap |
| Gradient Boosting | learning_rate, n_estimators, max_depth, subsample |
| Reti neurali | Dropout, batch normalization, L1/L2, early stopping |
| SVM | Parametro C (inverso della regolarizzazione) |

---

# Workflow per l'Ottimizzazione dei Modelli

## 1. Analisi esplorativa e preparazione dei dati
- Comprendere la distribuzione e le relazioni nei dati
- Gestire valori mancanti e outlier
- Feature engineering e scaling

## 2. Suddivisione dei dati
- Training set (60-70%)
- Validation set (15-20%)
- Test set (15-20%)

## 3. Selezione del modello base
- Iniziare con modelli semplici
- Valutare con cross-validation
- Identificare problemi di bias o varianza

## 4. Ottimizzazione e regolarizzazione
- Applicare tecniche appropriate per il modello scelto
- Ottimizzare gli iperparametri
- Monitorare le performance su validation set

## 5. Valutazione finale
- Testare il modello ottimizzato sul test set (una sola volta)
- Analizzare gli errori
- Interpretare il modello

---

# 5. Implementazione Pratica

## Caso di Studio: Previsione del Prezzo delle Case

```python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Caricamento dati
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Suddivisione train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definire i modelli
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(random_state=42)
}

# Definire gli spazi degli iperparametri
param_grids = {
    'Ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    'RandomForest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
}

# Ottimizzare e valutare i modelli
results = {}

for name, model in models.items():
    print(f"Optimizing {name}...")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Migliori parametri
    best_params = grid_search.best_params_
    print(f"  Best parameters: {best_params}")
    
    # Valutazione
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Training MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print()
    
    # Salvare i risultati
    results[name] = {
        'model': best_model,
        'params': best_params,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred
    }

# Visualizzare i risultati
names = list(results.keys())
train_mse = [results[name]['train_mse'] for name in names]
test_mse = [results[name]['test_mse'] for name in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, train_mse, width, label='Training MSE')
rects2 = ax.bar(x + width/2, test_mse, width, label='Test MSE')

ax.set_ylabel('Mean Squared Error')
ax.set_title('MSE by Model')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()

plt.tight_layout()
plt.show()

# Analizzare il modello migliore
best_model_name = min(results, key=lambda k: results[k]['test_mse'])
print(f"Best model: {best_model_name}")
print(f"Parameters: {results[best_model_name]['params']}")
print(f"Test MSE: {results[best_model_name]['test_mse']:.4f}")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")

# Visualizzare predizioni vs valori reali
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results[best_model_name]['y_test_pred'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valori reali')
plt.ylabel('Predizioni')
plt.title(f'Predizioni vs Valori reali ({best_model_name})')
plt.tight_layout()
plt.show()
```

---

# Caso di Studio: Classificazione con Regolarizzazione

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento dati
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# Suddivisione train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definire i modelli
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'RandomForest': RandomForestClassifier(random_state=42)
}

# Definire gli spazi degli iperparametri
param_grids = {
    'LogisticRegression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'SVM': {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf']
    },
    'RandomForest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
}

# Ottimizzare e valutare i modelli
results = {}

for name, model in models.items():
    print(f"Optimizing {name}...")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Migliori parametri
    best_params = grid_search.best_params_
    print(f"  Best parameters: {best_params}")
    
    # Valutazione
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print()
    
    # Salvare i risultati
    results[name] = {
        'model': best_model,
        'params': best_params,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test_pred': y_test_pred
    }

# Visualizzare i risultati
names = list(results.keys())
train_acc = [results[name]['train_acc'] for name in names]
test_acc = [results[name]['test_acc'] for name in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy')
rects2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()

plt.tight_layout()
plt.show()

# Analizzare il modello migliore
best_model_name = max(results, key=lambda k: results[k]['test_acc'])
print(f"Best model: {best_model_name}")
print(f"Parameters: {results[best_model_name]['params']}")
print(f"Test accuracy: {results[best_model_name]['test_acc']:.4f}")
print(f"F1-score: {results[best_model_name]['f1']:.4f}")

# Visualizzare la matrice di confusione
cm = confusion_matrix(y_test, results[best_model_name]['y_test_pred'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix ({best_model_name})')
plt.tight_layout()
plt.show()
```

---

# Riepilogo

- **Overfitting**: quando un modello impara troppo dai dati di training, incluso il rumore
- **Bias vs. Varianza**: trade-off fondamentale nell'apprendimento automatico
- **Tecniche di regolarizzazione**:
  - Early stopping
  - Regolarizzazione L1 (Lasso)
  - Regolarizzazione L2 (Ridge)
  - Elastic Net
  - Dropout
  - Batch normalization
- **Altre strategie**:
  - Data augmentation
  - Feature selection
  - Dimensionality reduction
  - Ensemble methods
  - Pruning degli alberi
- **Best practices**: workflow completo per ottimizzare i modelli e prevenire l'overfitting

---

# Domande?

![height:500px](https://scikit-learn.org/stable/_images/sphx_glr_plot_underfitting_overfitting_001.png)

---

# Riferimenti

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Scikit-learn documentation: https://scikit-learn.org/stable/
- TensorFlow documentation: https://www.tensorflow.org/
