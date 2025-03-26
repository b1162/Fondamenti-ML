# Corso di Preparazione e Pre-elaborazione dei Dati

Questo repository contiene tutti i materiali didattici per il corso sulla "Preparazione e Pre-elaborazione dei Dati", che copre i seguenti temi:

1. **Raccolta e pulizia dei dati**
2. **Feature engineering**
3. **Tecniche di suddivisione e validazione dei dati**

## Struttura del Repository

- **slides/**: Contiene le presentazioni in formato Markdown (compatibile con Marp)
  - `raccolta_pulizia_dati.md`: Slide sulla raccolta e pulizia dei dati
  - `feature_engineering.md`: Slide sul feature engineering
  - `tecniche_suddivisione_validazione.md`: Slide sulle tecniche di suddivisione e validazione dei dati

- **notebooks/**: Contiene i notebook Jupyter/Colab per le esercitazioni pratiche
  - `preparazione_dati_notebook.ipynb`: Notebook completo con esercitazioni su tutti i temi del corso

- **struttura_corso.md**: Descrizione dettagliata della struttura e degli obiettivi del corso
- **verifica_qualita.md**: Documento di verifica della qualità e coerenza dei materiali

## Come Utilizzare i Materiali

### Slide
Le slide sono in formato Markdown compatibile con [Marp](https://marp.app/), che permette di convertirle facilmente in presentazioni. Per visualizzarle:

1. Installa Marp CLI o utilizza l'estensione Marp per VS Code
2. Converti le slide in formato PDF o HTML:
   ```
   marp slides/raccolta_pulizia_dati.md --pdf
   ```

### Notebook
Il notebook è progettato per essere eseguito in [Google Colab](https://colab.research.google.com/):

1. Carica il file `notebooks/preparazione_dati_notebook.ipynb` su Google Colab
2. Esegui le celle in sequenza per seguire le esercitazioni pratiche
3. Il notebook include esempi completi e interattivi su tutti i temi del corso

## Temi Trattati

### 1. Raccolta e Pulizia dei Dati
- Fonti di dati e metodi di raccolta
- Identificazione e gestione dei valori mancanti
- Rilevamento e trattamento degli outlier
- Tecniche di pulizia dei dati
- Normalizzazione e standardizzazione
- Gestione dei dati duplicati

### 2. Feature Engineering
- Concetti fondamentali
- Trasformazione delle variabili
- Creazione di nuove feature
- Selezione delle feature
- Riduzione della dimensionalità
- Encoding di variabili categoriche
- Scaling e normalizzazione
- Tecniche avanzate di feature engineering

### 3. Tecniche di Suddivisione e Validazione dei Dati
- Principi di suddivisione dei dati
- Train-test split
- Validazione incrociata (cross-validation)
- Stratificazione
- Tecniche di validazione per serie temporali
- Metriche di valutazione
- Gestione dello sbilanciamento delle classi
- Best practices per la validazione dei modelli

## Requisiti

Per utilizzare questi materiali è necessario:

- Python 3.6+
- Jupyter Notebook o Google Colab
- Librerie Python: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn


## Licenza

Questi materiali sono forniti per uso educativo.
