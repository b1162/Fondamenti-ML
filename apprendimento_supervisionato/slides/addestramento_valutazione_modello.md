---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Apprendimento Supervisionato
## Parte 2: Addestramento e Valutazione del Modello

---

# Indice

1. Processo di addestramento
2. Tecniche di valutazione
3. Validazione del modello
4. Interpretabilità e spiegabilità
5. Casi pratici

---

# 1. Processo di Addestramento

## Concetti fondamentali
- **Addestramento**: processo di ottimizzazione dei parametri del modello
- **Parametri**: valori interni che il modello apprende dai dati
- **Iperparametri**: configurazioni esterne impostate prima dell'addestramento
- **Funzione obiettivo**: misura della performance da ottimizzare
- **Convergenza**: stato in cui ulteriore addestramento non migliora la performance

---

# Funzioni di Perdita (Loss Functions)

## Definizione
Una funzione di perdita quantifica quanto le previsioni del modello si discostano dai valori reali.

## Funzioni comuni per regressione
- **Mean Squared Error (MSE)**: $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **Mean Absolute Error (MAE)**: $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **Huber Loss**: combina MSE e MAE, più robusta agli outlier

## Funzioni comuni per classificazione
- **Binary Cross-Entropy**: $-\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- **Categorical Cross-Entropy**: $-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m}y_{ij}\log(\hat{y}_{ij})$
- **Hinge Loss**: $\max(0, 1 - y_i \hat{y}_i)$, usata per SVM

---

# Ottimizzazione e Discesa del Gradiente

## Principio base
L'ottimizzazione cerca di trovare i parametri che minimizzano la funzione di perdita.

## Discesa del gradiente
- **Idea**: aggiornare iterativamente i parametri nella direzione opposta al gradiente
- **Formula di aggiornamento**: $\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta)$
- **Learning rate** ($\eta$): controlla la dimensione dei passi di aggiornamento

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_sgd_separation_001.png)

---

# Varianti della Discesa del Gradiente

## Batch Gradient Descent
- Utilizza l'intero dataset per calcolare il gradiente
- Aggiornamento stabile ma computazionalmente costoso
- Convergenza garantita per funzioni convesse

## Stochastic Gradient Descent (SGD)
- Utilizza un singolo esempio per calcolare il gradiente
- Aggiornamenti rapidi ma rumorosi
- Può sfuggire a minimi locali

## Mini-batch Gradient Descent
- Compromesso: utilizza un sottoinsieme di esempi (mini-batch)
- Bilanciamento tra stabilità e velocità
- Dimensione tipica del batch: 32, 64, 128, 256

---

# Algoritmi di Ottimizzazione Avanzati

## Momentum
- Aggiunge "inerzia" agli aggiornamenti
- Aiuta a superare minimi locali e plateau
- $v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$
- $\theta_{t+1} = \theta_t - v_t$

## RMSprop
- Adatta il learning rate per ciascun parametro
- Divide il gradiente per la radice quadrata della media dei gradienti recenti
- Efficace per problemi non stazionari

## Adam
- Combina momentum e adattamento del learning rate
- Mantiene medie mobili del gradiente e del suo quadrato
- Spesso la scelta predefinita per molti problemi

---

# Curve di Apprendimento

## Definizione
Le curve di apprendimento mostrano come la performance del modello evolve durante l'addestramento.

## Interpretazione
- **Underfitting**: alta perdita sia su training che validation
- **Overfitting**: bassa perdita su training, alta su validation
- **Buona generalizzazione**: perdite simili e basse su entrambi i set

![height:350px](https://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png)

---

# Iperparametri e loro Ottimizzazione

## Iperparametri comuni
- **Learning rate**: dimensione dei passi di aggiornamento
- **Numero di epoche**: quante volte il modello vede l'intero dataset
- **Dimensione del batch**: quanti esempi per aggiornamento
- **Architettura del modello**: numero di layer, unità, ecc.
- **Parametri di regolarizzazione**: controllo dell'overfitting

## Tecniche di ottimizzazione
- **Grid Search**: prova tutte le combinazioni in una griglia predefinita
- **Random Search**: campiona casualmente dallo spazio degli iperparametri
- **Bayesian Optimization**: costruisce un modello probabilistico delle performance
- **Gradient-based Hyperparameter Optimization**: ottimizza gli iperparametri con il gradiente

---

# Implementazione dell'Ottimizzazione degli Iperparametri

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Definire il modello
model = RandomForestClassifier(random_state=42)

# Definire lo spazio degli iperparametri
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurare la grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Eseguire la ricerca
grid_search.fit(X_train, y_train)

# Migliori iperparametri e performance
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

---

# 2. Tecniche di Valutazione

## Principi fondamentali
- **Generalizzazione**: capacità del modello di performare su dati non visti
- **Bias-Variance Tradeoff**: equilibrio tra underfitting e overfitting
- **No Free Lunch Theorem**: nessun algoritmo è universalmente migliore di tutti gli altri
- **Valutazione onesta**: mai valutare su dati usati per l'addestramento

---

# Metriche per Problemi di Regressione

## Mean Squared Error (MSE)
- $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- Penalizza fortemente gli errori grandi
- Unità di misura: quadrato dell'unità della variabile target

## Root Mean Squared Error (RMSE)
- $\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
- Stessa unità di misura della variabile target
- Più interpretabile del MSE

## Mean Absolute Error (MAE)
- $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- Meno sensibile agli outlier rispetto a MSE/RMSE
- Unità di misura: stessa della variabile target

---

# Altre Metriche per Regressione

## Coefficient of Determination (R²)
- $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$
- Misura la proporzione di varianza spiegata dal modello
- Range: (-∞, 1], dove 1 indica predizione perfetta
- Limitazione: può aumentare semplicemente aggiungendo variabili

## Mean Absolute Percentage Error (MAPE)
- $\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$
- Esprime l'errore come percentuale del valore reale
- Problematico quando i valori reali sono vicini a zero

## Explained Variance Score
- Misura la proporzione di varianza che il modello cattura
- Simile a R², ma non penalizza il bias sistematico

---

# Implementazione delle Metriche di Regressione

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predizioni del modello
y_pred = model.predict(X_test)

# Calcolo delle metriche
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calcolo manuale del MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Visualizzazione predizioni vs valori reali
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valori reali')
plt.ylabel('Predizioni')
plt.title('Predizioni vs Valori reali')
plt.tight_layout()
plt.show()
```

---

# Metriche per Problemi di Classificazione

## Accuracy
- $\text{Accuracy} = \frac{\text{Predizioni corrette}}{\text{Totale predizioni}}$
- Semplice e intuitiva
- Problematica con classi sbilanciate

## Precision (Precisione)
- $\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$
- Misura quanto sono affidabili le predizioni positive
- Importante quando i falsi positivi sono costosi

## Recall (Sensibilità)
- $\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$
- Misura quanti positivi reali vengono identificati
- Importante quando i falsi negativi sono costosi

---

# F1-Score e Altre Metriche di Classificazione

## F1-Score
- $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- Media armonica di precision e recall
- Bilanciamento tra precision e recall

## Area Under the ROC Curve (AUC-ROC)
- Misura la capacità del modello di distinguere tra classi
- Insensibile allo sbilanciamento delle classi
- Range: [0, 1], dove 0.5 indica un modello casuale

## Log Loss (Cross-Entropy)
- $\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$
- Valuta la qualità delle probabilità predette
- Penalizza fortemente le predizioni confidenti ma errate

---

# Matrice di Confusione

## Definizione
Tabella che mostra le predizioni corrette e incorrette per ciascuna classe.

## Componenti (per classificazione binaria)
- **True Positives (TP)**: esempi positivi classificati correttamente
- **True Negatives (TN)**: esempi negativi classificati correttamente
- **False Positives (FP)**: esempi negativi classificati erroneamente come positivi
- **False Negatives (FN)**: esempi positivi classificati erroneamente come negativi

## Metriche derivate
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **Specificity**: $\frac{TN}{TN + FP}$

---

# Visualizzazione della Matrice di Confusione

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calcolare la matrice di confusione
cm = confusion_matrix(y_test, y_pred)

# Visualizzare la matrice di confusione
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Matrice di Confusione')
plt.tight_layout()
plt.show()

# Per classificazione multiclasse, normalizzare per leggibilità
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap='Blues', values_format='.2f', ax=ax)
plt.title('Matrice di Confusione (normalizzata)')
plt.tight_layout()
plt.show()
```

---

# Curve ROC e PR

## Receiver Operating Characteristic (ROC)
- Grafico di True Positive Rate vs False Positive Rate a diverse soglie
- **True Positive Rate (TPR)**: $\frac{TP}{TP + FN}$ (Recall)
- **False Positive Rate (FPR)**: $\frac{FP}{FP + TN}$ (1 - Specificity)
- AUC-ROC: area sotto la curva ROC, misura la qualità delle predizioni

## Precision-Recall (PR)
- Grafico di Precision vs Recall a diverse soglie
- Più informativo di ROC per dataset sbilanciati
- Average Precision (AP): area sotto la curva PR

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png)

---

# Implementazione delle Curve ROC e PR

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt

# Ottenere le probabilità per la classe positiva
y_scores = model.predict_proba(X_test)[:, 1]

# Calcolare la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Visualizzare la curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calcolare la curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
ap = average_precision_score(y_test, y_scores)

# Visualizzare la curva Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'Precision-Recall curve (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

# Classification Report

```python
from sklearn.metrics import classification_report

# Generare il report di classificazione
report = classification_report(y_test, y_pred, target_names=model.classes_)
print("Classification Report:")
print(report)

# Output:
# Classification Report:
#               precision    recall  f1-score   support
#
#      class 0       0.85      0.87      0.86       150
#      class 1       0.88      0.86      0.87       165
#
#    accuracy                           0.87       315
#   macro avg       0.87      0.87      0.87       315
#weighted avg       0.87      0.87      0.87       315
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
- **Classificazione sbilanciata**: precision, recall, AUC-ROC, F1
- **Regressione con outlier**: MAE (più robusto di MSE)
- **Regressione con scale diverse**: MAPE, R²
- **Problemi di ranking**: AUC, NDCG

---

# 3. Validazione del Modello

## Holdout Validation
- **Procedura**: suddivisione in training e test set
- **Vantaggi**: semplice, veloce
- **Svantaggi**: alta varianza, spreco di dati

## Cross-Validation
- **Procedura**: suddivisione in k fold, training su k-1 e test su 1
- **Vantaggi**: utilizzo efficiente dei dati, stima robusta
- **Svantaggi**: computazionalmente costoso

## Stratified Cross-Validation
- **Procedura**: mantiene la distribuzione delle classi in ogni fold
- **Vantaggi**: più affidabile per dataset sbilanciati
- **Svantaggi**: applicabile solo a problemi di classificazione

---

# Implementazione della Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import numpy as np

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Stratified K-Fold Cross-Validation (per classificazione)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_strat = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"Stratified CV Scores: {cv_scores_strat}")
print(f"Mean Stratified CV Score: {np.mean(cv_scores_strat):.4f} (±{np.std(cv_scores_strat):.4f})")

# Cross-validation con multiple metriche
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
```

---

# Nested Cross-Validation

## Problema
- Ottimizzare gli iperparametri e valutare il modello contemporaneamente può portare a sovrastime delle performance

## Soluzione: Nested CV
- **Loop esterno**: valuta le performance del modello
- **Loop interno**: ottimizza gli iperparametri

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

# Definire il modello e lo spazio degli iperparametri
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
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

# Validazione Temporale per Serie Storiche

## Problema
- La cross-validation standard può causare data leakage temporale

## Time Series Split
- **Procedura**: training su dati passati, test su dati futuri
- **Vantaggi**: rispetta l'ordine temporale, evita il data leakage
- **Svantaggi**: meno efficiente nell'uso dei dati

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

# Visualizzazione
fig, ax = plt.subplots(figsize=(10, 5))
for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ax.plot(np.ones(len(train_index)) * i, train_index, 'o-', c='blue', alpha=0.3, 
            label='Training set' if i == 0 else "")
    ax.plot(np.ones(len(test_index)) * i, test_index, 'o-', c='red', alpha=0.3, 
            label='Test set' if i == 0 else "")
ax.set_xlabel('Split index')
ax.set_ylabel('Sample index')
ax.legend()
plt.tight_layout()
plt.show()
```

---

# 4. Interpretabilità e Spiegabilità

## Importanza dell'interpretabilità
- **Fiducia**: comprendere perché il modello prende certe decisioni
- **Debugging**: identificare e correggere errori sistematici
- **Conformità**: rispettare requisiti normativi (es. GDPR)
- **Scoperta di conoscenza**: estrarre insight dai modelli

## Tipi di interpretabilità
- **Interpretabilità intrinseca**: modelli naturalmente interpretabili (es. alberi di decisione)
- **Interpretabilità post-hoc**: tecniche applicate dopo l'addestramento

---

# Feature Importance

## Importanza basata su permutazione
- **Idea**: misurare quanto peggiorano le performance quando una feature viene permutata
- **Vantaggi**: applicabile a qualsiasi modello, basata sulle performance effettive
- **Svantaggi**: computazionalmente costosa, può essere instabile

## Importanza basata sul modello
- **Idea**: estrarre l'importanza direttamente dal modello (es. coefficienti, impurità)
- **Vantaggi**: veloce, specifica per il modello
- **Svantaggi**: può non riflettere l'impatto reale sulle performance

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd

# Importanza basata sul modello (per Random Forest)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualizzazione
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Importanza basata su permutazione
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = pd.Series(result.importances_mean, index=X.columns)
perm_importances = perm_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
perm_importances.plot.bar()
plt.title("Permutation Importances")
plt.tight_layout()
plt.show()
```

---

# SHAP Values

## Definizione
SHAP (SHapley Additive exPlanations) values quantificano il contributo di ciascuna feature alla predizione per un'istanza specifica.

## Caratteristiche
- **Basati sulla teoria dei giochi**: distribuzione equa del "credito" tra le feature
- **Consistenza**: cambiamenti nel modello si riflettono nei valori SHAP
- **Additività**: i valori SHAP sommano alla differenza tra predizione e media

## Visualizzazioni
- **Summary plot**: panoramica dell'impatto delle feature
- **Dependence plot**: relazione tra feature e target
- **Force plot**: contributo di ciascuna feature per una singola predizione
- **Waterfall plot**: costruzione step-by-step della predizione

---

# Implementazione di SHAP

```python
import shap

# Creare un explainer SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Dependence plot per una feature specifica
shap.dependence_plot("feature_name", shap_values.values, X_test)

# Force plot per una singola predizione
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Waterfall plot per una singola predizione
shap.waterfall_plot(shap.Explanation(values=shap_values[0].values, 
                                    base_values=explainer.expected_value, 
                                    data=X_test.iloc[0]))
```

---

# Partial Dependence Plots

## Definizione
I Partial Dependence Plots (PDP) mostrano la relazione marginale tra le feature selezionate e l'output previsto.

## Caratteristiche
- **Marginalizzazione**: media su tutte le altre feature
- **Interpretazione**: mostra l'effetto di una feature indipendentemente dalle altre
- **Limitazioni**: assume indipendenza tra le feature

```python
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt

# Calcolare e visualizzare PDP per feature selezionate
features = [0, 1]  # Indici delle feature di interesse
fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(model, X_train, features, ax=ax)
plt.tight_layout()
plt.show()

# PDP per interazioni tra feature
fig, ax = plt.subplots(figsize=(10, 8))
plot_partial_dependence(model, X_train, [(0, 1)], kind='both', ax=ax)
plt.tight_layout()
plt.show()
```

---

# Interpretazione dei Coefficienti

## Per modelli lineari
- **Coefficienti standardizzati**: confronto diretto dell'importanza delle feature
- **Intervalli di confidenza**: incertezza associata ai coefficienti

```python
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardizzare le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Addestrare un modello lineare
linear_model = LogisticRegression(C=1.0)
linear_model.fit(X_scaled, y)

# Visualizzare i coefficienti
coef = pd.Series(linear_model.coef_[0], index=X.columns)
coef = coef.sort_values()

plt.figure(figsize=(10, 6))
coef.plot(kind='barh')
plt.title('Coefficienti del modello')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Bootstrap per intervalli di confidenza
n_bootstraps = 1000
bootstrap_coefs = np.zeros((n_bootstraps, X.shape[1]))

for i in range(n_bootstraps):
    # Campionamento bootstrap
    indices = np.random.choice(range(len(X_scaled)), len(X_scaled), replace=True)
    X_boot, y_boot = X_scaled[indices], y[indices]
    
    # Addestramento modello
    model_boot = LogisticRegression(C=1.0)
    model_boot.fit(X_boot, y_boot)
    
    # Salvataggio coefficienti
    bootstrap_coefs[i, :] = model_boot.coef_[0]

# Calcolo intervalli di confidenza (95%)
conf_intervals = np.percentile(bootstrap_coefs, [2.5, 97.5], axis=0)
```

---

# 5. Casi Pratici

## Workflow completo per l'addestramento e la valutazione

1. **Preparazione dei dati**
   - Pulizia e gestione dei valori mancanti
   - Feature engineering
   - Encoding delle variabili categoriche
   - Scaling delle feature numeriche

2. **Suddivisione dei dati**
   - Train-test split o cross-validation
   - Stratificazione se necessario

3. **Selezione e addestramento del modello**
   - Scelta dell'algoritmo appropriato
   - Ottimizzazione degli iperparametri
   - Addestramento del modello finale

4. **Valutazione e interpretazione**
   - Calcolo delle metriche appropriate
   - Analisi degli errori
   - Interpretazione del modello

---

# Esempio: Classificazione di Iris

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento dati
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Suddivisione train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ottimizzazione iperparametri
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Addestramento modello finale
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Valutazione
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

---

# Esempio: Regressione su Boston Housing

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
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

# Cross-validation per scegliere alpha
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
cv_scores = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train_scaled, y_train, 
                            cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())  # Negativo perché MSE è una loss

# Visualizzare i risultati della CV
plt.figure(figsize=(10, 6))
plt.plot(alphas, cv_scores, marker='o')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Results')
plt.grid(True)
plt.tight_layout()
plt.show()

# Scegliere il miglior alpha
best_alpha = alphas[np.argmin(cv_scores)]
print(f"Best alpha: {best_alpha}")

# Addestrare il modello finale
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

# Valutazione
y_pred = final_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Visualizzare predizioni vs valori reali
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valori reali')
plt.ylabel('Predizioni')
plt.title('Predizioni vs Valori reali')
plt.tight_layout()
plt.show()

# Coefficienti del modello
coef = pd.Series(final_model.coef_, index=X.columns)
coef = coef.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
coef.plot(kind='bar')
plt.title('Coefficienti del modello')
plt.tight_layout()
plt.show()
```

---

# Riepilogo

- **Processo di addestramento**: funzioni di perdita, ottimizzazione, discesa del gradiente, iperparametri
- **Tecniche di valutazione**: metriche per regressione e classificazione, matrice di confusione, curve ROC e PR
- **Validazione del modello**: holdout, cross-validation, nested CV, validazione temporale
- **Interpretabilità e spiegabilità**: feature importance, SHAP values, partial dependence plots, coefficienti
- **Casi pratici**: workflow completo per classificazione e regressione

---

# Domande?

![height:500px](https://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png)

---

# Riferimenti

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.
- Molnar, C. (2020). Interpretable Machine Learning. https://christophm.github.io/interpretable-ml-book/
- Scikit-learn documentation: https://scikit-learn.org/stable/
- SHAP documentation: https://shap.readthedocs.io/
