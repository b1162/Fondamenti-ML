# Corso di Apprendimento Supervisionato

Questo repository contiene i materiali didattici per il corso di Apprendimento Supervisionato, che copre i seguenti temi:

1. Algoritmi di regressione e classificazione
2. Addestramento e valutazione del modello
3. Overfitting e regolarizzazione

## Struttura del Repository

- **slides/**: Contiene le presentazioni in formato Markdown compatibile con Marp
  - `algoritmi_regressione_classificazione.md`: Slide sugli algoritmi di regressione e classificazione
  - `addestramento_valutazione_modello.md`: Slide sull'addestramento e valutazione del modello
  - `overfitting_regolarizzazione.md`: Slide sull'overfitting e regolarizzazione

- **notebooks/**: Contiene i notebook Colab per le esercitazioni pratiche
  - `apprendimento_supervisionato_notebook.ipynb`: Notebook completo con esercitazioni su tutti i temi

## Come Utilizzare i Materiali

### Slide

Le slide sono in formato Markdown compatibile con [Marp](https://marp.app/), che permette di convertirle facilmente in presentazioni. Per utilizzarle:

1. Installa Marp CLI o utilizza l'estensione Marp per VS Code
2. Converti le slide in formato PDF o HTML:
   ```
   marp --pdf slides/algoritmi_regressione_classificazione.md
   ```

### Notebook Colab

Il notebook è progettato per essere utilizzato con Google Colab:

1. Carica il file `apprendimento_supervisionato_notebook.ipynb` su Google Colab
2. Esegui le celle in sequenza per seguire le esercitazioni
3. Completa gli esercizi proposti alla fine del notebook

## Requisiti

Per eseguire il notebook sono necessarie le seguenti librerie Python:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Tutte queste librerie sono già disponibili nell'ambiente Google Colab.

## Contenuti del Corso

### 1. Algoritmi di Regressione e Classificazione
- Regressione lineare, polinomiale, Ridge, Lasso
- Regressione logistica, SVM, alberi decisionali, Random Forest
- Implementazione e confronto di diversi algoritmi
- Applicazioni reali

### 2. Addestramento e Valutazione del Modello
- Divisione dei dati (train/test/validation)
- Metriche di valutazione per regressione e classificazione
- Cross-validation
- Curve di apprendimento
- Ottimizzazione degli iperparametri
- Interpretabilità del modello

### 3. Overfitting e Regolarizzazione
- Definizione di overfitting e underfitting
- Bias-variance tradeoff
- Tecniche di regolarizzazione (L1, L2, Elastic Net)
- Regolarizzazione nei modelli ad albero
- Early stopping
- Altre tecniche per prevenire l'overfitting

## Licenza

Questi materiali sono forniti per uso didattico e possono essere liberamente utilizzati e modificati citando la fonte originale.
