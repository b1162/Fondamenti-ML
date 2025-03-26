---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Apprendimento Supervisionato
## Parte 1: Algoritmi di Regressione e Classificazione

---

# Indice

1. Introduzione all'apprendimento supervisionato
2. Algoritmi di regressione
3. Algoritmi di classificazione
4. Confronto tra algoritmi
5. Implementazione pratica

---

# 1. Introduzione all'Apprendimento Supervisionato

## Definizione
L'apprendimento supervisionato è un paradigma di machine learning in cui il modello viene addestrato su un dataset etichettato, imparando a mappare input a output conosciuti.

## Caratteristiche principali
- **Dataset etichettato**: ogni esempio ha una risposta corretta (target)
- **Obiettivo**: prevedere l'output per nuovi input non visti
- **Feedback diretto**: l'errore tra previsione e valore reale guida l'apprendimento
- **Generalizzazione**: capacità di performare bene su dati nuovi

---

# Differenze tra Regressione e Classificazione

| Regressione | Classificazione |
|-------------|-----------------|
| Predice valori continui | Predice categorie discrete |
| Output: numeri reali | Output: classi o probabilità |
| Esempio: previsione prezzi | Esempio: spam detection |
| Metriche: MSE, MAE, R² | Metriche: accuracy, precision, recall |
| Funzione di perdita: errore quadratico | Funzione di perdita: cross-entropy |

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

---

# Panoramica del Processo di Apprendimento Supervisionato

![height:500px](https://scikit-learn.org/stable/_images/grid_search_workflow.png)

---

# 2. Algoritmi di Regressione

## Regressione Lineare Semplice

- **Idea**: modellare la relazione tra una variabile dipendente y e una variabile indipendente x
- **Equazione**: $y = \beta_0 + \beta_1 x + \varepsilon$
- **Parametri**: intercetta ($\beta_0$) e coefficiente angolare ($\beta_1$)
- **Ottimizzazione**: minimizzazione dell'errore quadratico medio

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png)

---

# Regressione Lineare Multipla

- **Estensione**: modella la relazione tra y e multiple variabili indipendenti
- **Equazione**: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \varepsilon$
- **Assunzioni**:
  - Linearità
  - Indipendenza degli errori
  - Omoschedasticità (varianza costante)
  - Normalità dei residui
  - Assenza di multicollinearità

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

# Regressione Polinomiale

- **Idea**: catturare relazioni non lineari usando termini polinomiali
- **Equazione**: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n + \varepsilon$
- **Implementazione**: trasformazione delle feature + regressione lineare
- **Vantaggi**: flessibilità per modellare curve complesse
- **Svantaggi**: rischio di overfitting con gradi elevati

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear_regression", LinearRegression())
])

polynomial_regression.fit(X_train, y_train)
```

---

# Regressione Ridge e Lasso

## Ridge Regression (L2)
- **Idea**: aggiunge penalità basata sulla somma dei quadrati dei coefficienti
- **Equazione**: $\min_{\beta} \|y - X\beta\|^2 + \alpha \|\beta\|^2$
- **Effetto**: riduce l'impatto di tutte le feature, senza eliminarle

## Lasso Regression (L1)
- **Idea**: aggiunge penalità basata sulla somma dei valori assoluti dei coefficienti
- **Equazione**: $\min_{\beta} \|y - X\beta\|^2 + \alpha \|\beta\|_1$
- **Effetto**: può portare a zero alcuni coefficienti (selezione delle feature)

```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
```

---

# Support Vector Regression (SVR)

- **Idea**: estensione di SVM per problemi di regressione
- **Obiettivo**: trovare una funzione che abbia al massimo ε di deviazione dai valori target
- **Caratteristiche**:
  - Margine di tolleranza ε
  - Supporta kernel non lineari
  - Robusto agli outlier

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_regression_001.png)

```python
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)
```

---

# Alberi di Decisione per Regressione

- **Idea**: suddividere lo spazio delle feature in regioni e assegnare un valore a ciascuna
- **Criterio di split**: minimizzazione della varianza
- **Vantaggi**:
  - Facile interpretazione
  - Gestisce feature non lineari
  - Non richiede scaling
- **Svantaggi**:
  - Tendenza all'overfitting
  - Instabilità

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png)

```python
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(max_depth=5)
dt_regressor.fit(X_train, y_train)
```

---

# Random Forest per Regressione

- **Idea**: ensemble di alberi di decisione
- **Funzionamento**:
  1. Bootstrap sampling: creazione di sottoinsiemi casuali dei dati
  2. Feature randomization: considerazione di un sottoinsieme casuale di feature per ogni split
  3. Aggregazione: media delle previsioni di tutti gli alberi
- **Vantaggi**:
  - Riduzione dell'overfitting
  - Robustezza
  - Feature importance

```python
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
```

---

# Gradient Boosting per Regressione

- **Idea**: costruire modelli sequenzialmente, ognuno corregge gli errori del precedente
- **Algoritmi**:
  - Gradient Boosting Machines (GBM)
  - XGBoost
  - LightGBM
  - CatBoost
- **Vantaggi**:
  - Alta accuratezza
  - Gestione di dati eterogenei
- **Svantaggi**:
  - Complessità computazionale
  - Rischio di overfitting

```python
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)
```

---

# 3. Algoritmi di Classificazione

## Regressione Logistica

- **Idea**: estensione della regressione lineare per problemi di classificazione
- **Funzione**: trasforma output lineare in probabilità usando la funzione sigmoide
  - $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$
- **Decisione**: classe 1 se P > 0.5, altrimenti classe 0
- **Ottimizzazione**: massimizzazione della log-likelihood

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_001.png)

```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1.0, solver='lbfgs')
log_reg.fit(X_train, y_train)
```

---

# Support Vector Machines (SVM)

- **Idea**: trovare l'iperpiano che massimizza il margine tra le classi
- **Caratteristiche**:
  - Massimizzazione del margine
  - Supporto per kernel non lineari
  - Robustezza in spazi ad alta dimensionalità
- **Kernel comuni**:
  - Lineare: $K(x, y) = x^T y$
  - Polinomiale: $K(x, y) = (x^T y + c)^d$
  - RBF: $K(x, y) = \exp(-\gamma \|x - y\|^2)$

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_margin_001.png)

```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)
```

---

# Naive Bayes

- **Idea**: applicazione del teorema di Bayes con assunzione di indipendenza tra feature
- **Equazione**: $P(y|x_1,...,x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$
- **Varianti**:
  - Gaussian NB: per feature continue con distribuzione normale
  - Multinomial NB: per conteggi (es. text classification)
  - Bernoulli NB: per feature binarie
- **Vantaggi**:
  - Semplice e veloce
  - Efficace con dataset piccoli
  - Buono per dati ad alta dimensionalità

```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
```

---

# K-Nearest Neighbors (KNN)

- **Idea**: classificare in base alla classe più frequente tra i k vicini più prossimi
- **Parametri chiave**:
  - k: numero di vicini da considerare
  - metrica di distanza (euclidea, manhattan, ecc.)
- **Vantaggi**:
  - Semplice da comprendere
  - Non parametrico
  - Adattabile a confini di decisione complessi
- **Svantaggi**:
  - Computazionalmente costoso
  - Sensibile alla scala delle feature
  - Curse of dimensionality

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png)

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)
```

---

# Alberi di Decisione per Classificazione

- **Idea**: suddividere lo spazio delle feature in regioni e assegnare una classe a ciascuna
- **Criteri di split**:
  - Gini impurity
  - Entropy (information gain)
- **Processo di costruzione**:
  1. Selezionare la feature e il valore di split ottimali
  2. Dividere il dataset in base allo split
  3. Ripetere ricorsivamente per ogni sottoinsieme
  4. Fermarsi quando si raggiunge un criterio di stop

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_decision_surface_001.png)

```python
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=5)
dt_classifier.fit(X_train, y_train)
```

---

# Random Forest per Classificazione

- **Idea**: ensemble di alberi di decisione con voto di maggioranza
- **Vantaggi rispetto agli alberi singoli**:
  - Riduzione della varianza
  - Maggiore robustezza
  - Migliore generalizzazione
- **Iperparametri chiave**:
  - n_estimators: numero di alberi
  - max_features: numero di feature considerate per ogni split
  - max_depth: profondità massima degli alberi

```python
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
rf_classifier.fit(X_train, y_train)
```

---

# Gradient Boosting per Classificazione

- **Idea**: costruire modelli sequenzialmente, ognuno focalizzato sugli errori del precedente
- **Algoritmi popolari**:
  - XGBoost
  - LightGBM
  - CatBoost
- **Caratteristiche**:
  - Learning rate: controlla il contributo di ogni albero
  - Regularization: previene l'overfitting
  - Early stopping: interrompe l'addestramento quando le performance non migliorano

```python
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb.fit(X_train, y_train)
```

---

# 4. Confronto tra Algoritmi

## Vantaggi e Svantaggi: Algoritmi Lineari

| Algoritmo | Vantaggi | Svantaggi |
|-----------|----------|-----------|
| Regressione Lineare | Semplice, interpretabile | Limitato a relazioni lineari |
| Regressione Logistica | Probabilità di output, efficiente | Limitato a confini lineari |
| Ridge/Lasso | Gestione della multicollinearità, regolarizzazione | Ancora limitato a relazioni lineari |

## Vantaggi e Svantaggi: Algoritmi Non Lineari

| Algoritmo | Vantaggi | Svantaggi |
|-----------|----------|-----------|
| SVM | Efficace in spazi ad alta dimensionalità | Lento su dataset grandi, difficile da interpretare |
| Alberi di Decisione | Interpretabili, gestiscono feature miste | Tendenza all'overfitting |
| Random Forest | Robusto, accurato | Black box, computazionalmente intensivo |
| Gradient Boosting | Alta accuratezza, flessibile | Complesso da ottimizzare, rischio di overfitting |

---

# Criteri di Scelta dell'Algoritmo Appropriato

## Considerazioni principali
- **Dimensione del dataset**: alcuni algoritmi scalano meglio di altri
- **Interpretabilità**: necessità di spiegare le decisioni del modello
- **Accuratezza vs. velocità**: trade-off tra performance e tempo di addestramento
- **Tipo di feature**: numeriche, categoriche, miste
- **Linearità dei dati**: relazioni lineari o non lineari
- **Rumore nei dati**: robustezza agli outlier
- **Dimensionalità**: numero di feature rispetto al numero di esempi

---

# Casi d'Uso Tipici

| Algoritmo | Casi d'uso ideali |
|-----------|-------------------|
| Regressione Lineare | Previsione vendite, analisi fattoriale, relazioni semplici |
| Regressione Logistica | Classificazione binaria, credit scoring, medical diagnosis |
| SVM | Classificazione di immagini, text classification, bioinformatica |
| Naive Bayes | Spam detection, sentiment analysis, classificazione di documenti |
| KNN | Sistemi di raccomandazione, pattern recognition |
| Alberi di Decisione | Risk assessment, diagnosi medica, casi che richiedono interpretabilità |
| Random Forest | Previsioni finanziarie, computer vision, bioinformatica |
| Gradient Boosting | Competizioni di ML, previsioni accurate, ranking |

---

# 5. Implementazione Pratica

## Workflow di base per problemi di apprendimento supervisionato

1. **Preparazione dei dati**
   - Pulizia e gestione dei valori mancanti
   - Feature engineering
   - Encoding delle variabili categoriche
   - Scaling delle feature numeriche

2. **Suddivisione dei dati**
   - Train-test split
   - Validazione incrociata

3. **Selezione e addestramento del modello**
   - Scelta dell'algoritmo appropriato
   - Ottimizzazione degli iperparametri
   - Addestramento del modello finale

4. **Valutazione e interpretazione**
   - Calcolo delle metriche appropriate
   - Analisi degli errori
   - Interpretazione del modello

---

# Esempio: Regressione con Scikit-learn

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Caricamento dati
data = pd.read_csv('housing.csv')
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Suddivisione train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addestramento
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predizione e valutazione
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")
```

---

# Esempio: Classificazione con Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Caricamento dati
data = pd.read_csv('iris.csv')
X = data.drop('species', axis=1)
y = data['species']

# Suddivisione train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Addestramento
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predizione e valutazione
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Visualizzazione matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

---

# Confronto di Algoritmi su un Dataset

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Lista di modelli da confrontare
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC(probability=True)),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier())
]

# Confronto con cross-validation
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} (±{cv_results.std():.4f})")

# Visualizzazione box plot
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

# Riepilogo

- **Apprendimento supervisionato**: addestramento su dati etichettati per prevedere output
- **Regressione**: predizione di valori continui
  - Regressione lineare, polinomiale, Ridge/Lasso, SVR, alberi, ensemble
- **Classificazione**: predizione di categorie discrete
  - Regressione logistica, SVM, Naive Bayes, KNN, alberi, ensemble
- **Scelta dell'algoritmo**: dipende da dimensione dei dati, interpretabilità, accuratezza, tipo di feature
- **Implementazione pratica**: preparazione dati, suddivisione, addestramento, valutazione

---

# Domande?

![height:500px](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

---

# Riferimenti

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.
- Scikit-learn documentation: https://scikit-learn.org/stable/
- XGBoost documentation: https://xgboost.readthedocs.io/
- LightGBM documentation: https://lightgbm.readthedocs.io/
