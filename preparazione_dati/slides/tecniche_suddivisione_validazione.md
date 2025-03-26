---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Preparazione e Pre-elaborazione dei Dati
## Parte 3: Tecniche di Suddivisione e Validazione dei Dati

---

# Indice

1. Principi di suddivisione dei dati
2. Train-test split
3. Validazione incrociata (cross-validation)
4. Stratificazione
5. Tecniche di validazione per serie temporali
6. Metriche di valutazione
7. Gestione dello sbilanciamento delle classi
8. Best practices per la validazione dei modelli

---

# 1. Principi di Suddivisione dei Dati

## Perché suddividere i dati?
- **Valutazione realistica**: stimare le performance su dati non visti
- **Evitare l'overfitting**: modelli che generalizzano meglio
- **Ottimizzazione degli iperparametri**: selezione dei parametri ottimali
- **Confronto tra modelli**: valutazione oggettiva di diversi approcci

## Obiettivi della suddivisione
- **Rappresentatività**: ogni subset deve rappresentare la popolazione
- **Indipendenza**: evitare data leakage tra training e test
- **Bilanciamento**: mantenere la distribuzione delle classi
- **Dimensione adeguata**: sufficienti dati per training e valutazione

---

# Suddivisioni Tipiche dei Dataset

![height:450px](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

---

# 2. Train-Test Split

## Suddivisione base
- **Training set**: per addestrare il modello (tipicamente 70-80%)
- **Test set**: per valutare il modello finale (tipicamente 20-30%)

```python
from sklearn.model_selection import train_test_split

# Suddivisione base
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Con suddivisione train-validation-test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
```

---

# Considerazioni sul Train-Test Split

## Vantaggi
- **Semplicità**: facile da implementare e comprendere
- **Velocità**: richiede un solo training del modello
- **Adatto per grandi dataset**: quando i dati sono abbondanti

## Limitazioni
- **Variabilità**: risultati possono dipendere dalla specifica suddivisione
- **Spreco di dati**: parte dei dati non viene usata per il training
- **Inadeguato per piccoli dataset**: può portare a stime instabili

## Estensione: Train-Validation-Test
- **Training set**: per addestrare il modello
- **Validation set**: per ottimizzare gli iperparametri
- **Test set**: per valutare il modello finale (mai usato durante lo sviluppo)

---

# 3. Validazione Incrociata (Cross-Validation)

## K-Fold Cross-Validation
- **Procedura**: 
  1. Dividere il dataset in K parti (fold) di uguale dimensione
  2. Per ogni fold i:
     - Usare il fold i come validation set
     - Usare i restanti K-1 fold come training set
  3. Calcolare la media delle performance sui K esperimenti

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Definire il modello
model = LogisticRegression()

# Definire la strategia di cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Calcolare i punteggi
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Punteggi per fold: {scores}")
print(f"Punteggio medio: {scores.mean():.4f} (±{scores.std():.4f})")
```

---

# Varianti della Cross-Validation

## Leave-One-Out Cross-Validation (LOOCV)
- Caso estremo di K-Fold dove K = numero di osservazioni
- Ogni fold contiene una sola osservazione
```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

## Leave-P-Out Cross-Validation
- Simile a LOOCV ma lascia fuori P osservazioni per volta
```python
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=2)
scores = cross_val_score(model, X, y, cv=lpo)
```

## Repeated K-Fold Cross-Validation
- Ripete la K-Fold CV più volte con diverse suddivisioni
```python
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf)
```

---

# Vantaggi e Svantaggi della Cross-Validation

## Vantaggi
- **Utilizzo efficiente dei dati**: ogni osservazione viene usata sia per training che per validation
- **Stima robusta**: riduce la varianza della valutazione
- **Adatta per piccoli dataset**: massimizza l'uso dei dati disponibili

## Svantaggi
- **Costo computazionale**: richiede K training del modello
- **Complessità**: implementazione più elaborata
- **Tempo**: può essere lenta per modelli complessi o grandi dataset

## Quando usare la cross-validation
- **Dataset piccoli o medi**: quando i dati sono limitati
- **Ottimizzazione degli iperparametri**: per una stima robusta
- **Confronto tra modelli**: per una valutazione affidabile

---

# 4. Stratificazione

## Cos'è la stratificazione?
- **Definizione**: mantenere la stessa distribuzione delle classi in tutti i subset
- **Importanza**: cruciale per dataset sbilanciati o con classi rare

## Stratified K-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

## Stratified Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

---

# Visualizzazione della Stratificazione

```python
import matplotlib.pyplot as plt
import numpy as np

# Confronto tra distribuzione originale e split
def plot_class_distribution(y_full, y_train, y_test):
    classes = np.unique(y_full)
    full_dist = np.array([sum(y_full == c) / len(y_full) for c in classes])
    train_dist = np.array([sum(y_train == c) / len(y_train) for c in classes])
    test_dist = np.array([sum(y_test == c) / len(y_test) for c in classes])
    
    width = 0.25
    x = np.arange(len(classes))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, full_dist, width, label='Dataset completo')
    plt.bar(x, train_dist, width, label='Training set')
    plt.bar(x + width, test_dist, width, label='Test set')
    
    plt.xlabel('Classe')
    plt.ylabel('Proporzione')
    plt.title('Distribuzione delle classi')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
```

---

# Stratificazione per Problemi di Regressione

- **Problema**: la stratificazione standard è per classificazione
- **Soluzione**: discretizzare la variabile target continua

```python
from sklearn.model_selection import train_test_split

# Discretizzare la variabile target
y_binned = pd.qcut(y, q=5, labels=False)

# Stratificare in base ai bin
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_binned, random_state=42)
```

---

# 5. Tecniche di Validazione per Serie Temporali

## Problemi con la validazione standard
- **Data leakage temporale**: informazioni future usate per predire il passato
- **Autocorrelazione**: osservazioni vicine nel tempo sono correlate
- **Drift concettuale**: le relazioni nei dati cambiano nel tempo

## Time Series Split
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)

# Visualizzazione
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))
for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ax.plot(np.ones(len(train_index)) * i, train_index, 'o-', c='blue', alpha=0.3, label='Training set' if i == 0 else "")
    ax.plot(np.ones(len(test_index)) * i, test_index, 'o-', c='red', alpha=0.3, label='Test set' if i == 0 else "")
ax.set_xlabel('Split index')
ax.set_ylabel('Sample index')
ax.legend()
plt.tight_layout()
```

---

# Tecniche Avanzate per Serie Temporali

## Expanding Window
- **Procedura**: il training set cresce ad ogni iterazione, il test set è sempre il periodo successivo
```python
def expanding_window_split(X, y, n_splits, test_size):
    n_samples = len(X)
    test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)
    
    for test_start in test_starts:
        train_indices = list(range(0, test_start))
        test_indices = list(range(test_start, test_start + test_size))
        yield train_indices, test_indices
```

## Sliding Window
- **Procedura**: finestra di training di dimensione fissa che si sposta nel tempo
```python
def sliding_window_split(X, y, train_size, test_size, step=1):
    n_samples = len(X)
    for i in range(0, n_samples - train_size - test_size + 1, step):
        train_indices = list(range(i, i + train_size))
        test_indices = list(range(i + train_size, i + train_size + test_size))
        yield train_indices, test_indices
```

---

# Purging e Embargo

## Purging
- **Definizione**: rimuovere osservazioni dal training set che sono temporalmente vicine al test set
- **Scopo**: evitare data leakage in presenza di autocorrelazione

## Embargo
- **Definizione**: escludere osservazioni immediatamente successive al periodo di test
- **Scopo**: evitare che informazioni dal test set influenzino il training futuro

```python
def purged_embargo_split(X, dates, embargo_size, n_splits, test_size):
    n_samples = len(X)
    test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)
    
    for test_start in test_starts:
        test_end = test_start + test_size
        
        # Calcolare le date di inizio e fine del test
        test_start_date = dates[test_start]
        test_end_date = dates[test_end - 1]
        
        # Identificare le osservazioni da escludere (purging)
        purge_indices = [i for i, date in enumerate(dates) 
                         if test_start_date <= date <= test_end_date]
        
        # Identificare le osservazioni da escludere (embargo)
        embargo_indices = []
        if test_end < n_samples:
            embargo_end = min(test_end + embargo_size, n_samples)
            embargo_indices = list(range(test_end, embargo_end))
        
        # Creare gli indici di training escludendo purge e embargo
        exclude_indices = set(purge_indices + embargo_indices)
        train_indices = [i for i in range(test_start) if i not in exclude_indices]
        test_indices = list(range(test_start, test_end))
        
        yield train_indices, test_indices
```

---

# 6. Metriche di Valutazione

## Metriche per problemi di classificazione
- **Accuracy**: proporzione di predizioni corrette
- **Precision**: proporzione di veri positivi tra i predetti positivi
- **Recall**: proporzione di veri positivi identificati
- **F1-score**: media armonica di precision e recall
- **AUC-ROC**: area sotto la curva ROC
- **Log loss**: perdita logaritmica (cross-entropy)

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, log_loss)

# Calcolo delle metriche
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)
loss = log_loss(y_true, y_pred_proba)
```

---

# Metriche per Problemi di Regressione

- **Mean Absolute Error (MAE)**: media degli errori assoluti
- **Mean Squared Error (MSE)**: media degli errori al quadrato
- **Root Mean Squared Error (RMSE)**: radice quadrata di MSE
- **Mean Absolute Percentage Error (MAPE)**: errore percentuale medio
- **R²**: coefficiente di determinazione

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                           r2_score, mean_absolute_percentage_error)
import numpy as np

# Calcolo delle metriche
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

---

# Visualizzazione delle Metriche

## Matrice di confusione
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matrice di confusione')
plt.tight_layout()
```

## Curva ROC
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
```

---

# Curve di Precision-Recall

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
avg_precision = average_precision_score(y_true, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
```

---

# Scelta della Metrica Appropriata

## Fattori da considerare
- **Natura del problema**: classificazione vs regressione
- **Bilanciamento delle classi**: dataset sbilanciati richiedono metriche specifiche
- **Costo degli errori**: falsi positivi vs falsi negativi
- **Interpretabilità**: alcune metriche sono più intuitive di altre

## Esempi di scelta
- **Classificazione bilanciata**: accuracy, F1-score
- **Classificazione sbilanciata**: precision, recall, AUC-ROC
- **Regressione con outlier**: MAE (più robusto di MSE)
- **Regressione con scale diverse**: MAPE, R²
- **Problemi di ranking**: AUC, NDCG

---

# 7. Gestione dello Sbilanciamento delle Classi

## Problemi con dataset sbilanciati
- **Bias verso la classe maggioritaria**: il modello tende a predire la classe più frequente
- **Metriche fuorvianti**: alta accuracy ma bassa capacità predittiva per la classe minoritaria
- **Sottorappresentazione**: la classe minoritaria ha pochi esempi per l'apprendimento

## Tecniche di campionamento
- **Undersampling**: ridurre la classe maggioritaria
- **Oversampling**: aumentare la classe minoritaria
- **Synthetic sampling**: generare nuovi esempi sintetici

---

# Tecniche di Undersampling

- **Random Undersampling**: rimuove casualmente esempi dalla classe maggioritaria
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

- **Tomek Links**: rimuove esempi di confine tra classi
```python
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X, y)
```

- **Edited Nearest Neighbors**: rimuove esempi mal classificati
```python
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X, y)
```

---

# Tecniche di Oversampling

- **Random Oversampling**: duplica casualmente esempi della classe minoritaria
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

- **SMOTE (Synthetic Minority Over-sampling Technique)**: genera esempi sintetici
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

- **ADASYN (Adaptive Synthetic Sampling)**: genera più esempi dove la classificazione è difficile
```python
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

---

# Tecniche Combinate e Avanzate

- **SMOTE + Tomek Links**: oversampling seguito da pulizia
```python
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)
```

- **SMOTE + ENN**: oversampling seguito da pulizia con ENN
```python
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
```

- **Borderline-SMOTE**: genera esempi vicino al confine decisionale
```python
from imblearn.over_sampling import BorderlineSMOTE
b_smote = BorderlineSMOTE(random_state=42)
X_resampled, y_resampled = b_smote.fit_resample(X, y)
```

---

# Approcci Alternativi allo Sbilanciamento

## Pesi delle classi
- **Assegnare pesi maggiori** alla classe minoritaria durante il training
```python
from sklearn.linear_model import LogisticRegression
# Pesi automatici inversamente proporzionali alla frequenza
model = LogisticRegression(class_weight='balanced')

# Pesi personalizzati
weights = {0: 1, 1: 10}  # Classe 1 ha peso 10 volte maggiore
model = LogisticRegression(class_weight=weights)
```

## Soglie di decisione
- **Modificare la soglia** per la classificazione binaria
```python
# Predizione con soglia personalizzata
y_pred_proba = model.predict_proba(X_test)[:, 1]
custom_threshold = 0.3  # Invece del default 0.5
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)
```

---

# 8. Best Practices per la Validazione dei Modelli

## Pipeline di validazione completa
1. **Esplorazione e comprensione dei dati**
2. **Suddivisione iniziale** in training e test set
3. **Preprocessing** dei dati (usando solo informazioni dal training set)
4. **Selezione del modello** con cross-validation sul training set
5. **Ottimizzazione degli iperparametri** con nested cross-validation
6. **Valutazione finale** sul test set (una sola volta)
7. **Interpretazione dei risultati** e iterazione se necessario

---

# Nested Cross-Validation

- **Problema**: ottimizzare gli iperparametri e valutare il modello contemporaneamente
- **Soluzione**: loop interno per ottimizzazione, loop esterno per valutazione

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Definire il modello e lo spazio degli iperparametri
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Loop esterno per la valutazione
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Loop interno per l'ottimizzazione
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Grid search con cross-validation interna
clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

# Nested cross-validation
nested_scores = cross_val_score(clf, X, y, cv=outer_cv)

print(f"Nested CV score: {nested_scores.mean():.4f} (±{nested_scores.std():.4f})")
```

---

# Evitare il Data Leakage

## Tipi di data leakage
- **Target leakage**: informazioni non disponibili al momento della predizione
- **Train-test contamination**: informazioni dal test set usate nel training
- **Temporal leakage**: informazioni future usate per predire il passato

## Prevenzione
- **Preprocessing all'interno della cross-validation**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Pipeline che include preprocessing e modello
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Cross-validation con la pipeline completa
scores = cross_val_score(pipeline, X, y, cv=5)
```

---

# Validazione con Dati Stratificati

## Stratificazione per problemi multiclasse
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

## Stratificazione per problemi multilabel
```python
from sklearn.model_selection import KFold
from itertools import chain
from sklearn.utils import shuffle

def multilabel_stratified_kfold(X, y, n_splits=5):
    # Creare un identificatore unico per ogni combinazione di label
    y_str = y.astype(str)
    label_combinations = [''.join(row) for row in y_str]
    
    # Usare StratifiedKFold sulla combinazione di label
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, label_combinations):
        yield train_idx, test_idx
```

---

# Validazione con Gruppi

## Group K-Fold
- **Uso**: quando le osservazioni appartengono a gruppi (es. pazienti, aziende)
- **Obiettivo**: evitare che osservazioni dello stesso gruppo siano in training e test

```python
from sklearn.model_selection import GroupKFold
# groups: array con l'identificatore del gruppo per ogni osservazione
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf, groups=groups)
```

## Leave-One-Group-Out
- **Uso**: simile a LOOCV ma per gruppi
```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
scores = cross_val_score(model, X, y, cv=logo, groups=groups)
```

---

# Riepilogo: Tecniche di Suddivisione e Validazione

- La **suddivisione dei dati** è fondamentale per valutare correttamente i modelli
- Il **train-test split** è semplice ma può essere instabile
- La **cross-validation** offre stime più robuste delle performance
- La **stratificazione** mantiene la distribuzione delle classi nei subset
- Le **tecniche per serie temporali** evitano il data leakage temporale
- La scelta delle **metriche di valutazione** dipende dal problema specifico
- La gestione dello **sbilanciamento delle classi** è cruciale per molti problemi reali
- Le **best practices** includono nested CV e prevenzione del data leakage

---

# Domande?

![height:450px](https://scikit-learn.org/stable/_images/grid_search_workflow.png)

---

# Riferimenti

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer.
- Raschka, S., & Mirjalili, V. (2019). Python Machine Learning. Packt Publishing.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- Chollet, F. (2021). Deep Learning with Python. Manning Publications.
