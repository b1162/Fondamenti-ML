---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Preparazione e Pre-elaborazione dei Dati
## Parte 1: Raccolta e Pulizia dei Dati

---

# Indice

1. Introduzione alla raccolta dati
2. Fonti di dati e metodi di acquisizione
3. Identificazione e gestione dei valori mancanti
4. Rilevamento e trattamento degli outlier
5. Tecniche di pulizia dei dati
6. Normalizzazione e standardizzazione
7. Gestione dei dati duplicati
8. Casi studio pratici

---

# 1. Introduzione alla Raccolta Dati

## Importanza della preparazione dei dati
- **Garbage in, garbage out**: la qualità dei dati determina la qualità dei risultati
- **Tempo dedicato**: 60-80% del tempo di un data scientist è dedicato alla preparazione dei dati
- **Impatto sulle performance**: dati ben preparati migliorano significativamente le performance dei modelli

## Fasi principali del processo
1. Raccolta dei dati
2. Esplorazione e comprensione
3. Pulizia e pre-elaborazione
4. Trasformazione e feature engineering
5. Suddivisione per training e testing

---

# 2. Fonti di Dati e Metodi di Acquisizione

## Tipologie di fonti dati
- **Database relazionali**: SQL Server, MySQL, PostgreSQL
- **Database NoSQL**: MongoDB, Cassandra, Redis
- **API e web services**: REST API, SOAP, GraphQL
- **Web scraping**: estrazione dati da pagine web
- **File**: CSV, JSON, XML, Excel, parquet
- **Streaming**: dati in tempo reale (Kafka, Kinesis)
- **Sensori e IoT**: dispositivi connessi
- **Open data**: dataset pubblici (Kaggle, data.gov)

---

# Metodi di Acquisizione dei Dati

## Tecniche di acquisizione
- **Query SQL**: estrazione da database relazionali
```sql
SELECT customer_id, purchase_date, amount 
FROM transactions 
WHERE purchase_date > '2023-01-01'
```

- **API calls**: richieste a servizi web
```python
import requests
response = requests.get('https://api.example.com/data', 
                       headers={'Authorization': 'Bearer token'})
data = response.json()
```

- **Web scraping**: estrazione da pagine web
```python
from bs4 import BeautifulSoup
import requests
page = requests.get('https://example.com')
soup = BeautifulSoup(page.content, 'html.parser')
```

---

# Considerazioni Etiche e Legali

- **GDPR e privacy**: conformità con le normative sulla protezione dei dati
- **Consenso informato**: ottenere il permesso per l'utilizzo dei dati
- **Anonimizzazione**: rimozione di informazioni personali identificabili
- **Proprietà intellettuale**: rispetto dei diritti d'autore e dei termini di servizio
- **Bias nei dati**: consapevolezza e mitigazione dei pregiudizi nei dataset

---

# 3. Identificazione e Gestione dei Valori Mancanti

## Tipi di valori mancanti
- **MCAR (Missing Completely At Random)**: assenza casuale indipendente dalle variabili
- **MAR (Missing At Random)**: assenza dipendente da altre variabili osservate
- **MNAR (Missing Not At Random)**: assenza dipendente dal valore mancante stesso

## Identificazione
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizzare i valori mancanti
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Mappa dei valori mancanti')
plt.tight_layout()

# Conteggio dei valori mancanti
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
```

---

# Strategie per Gestire i Valori Mancanti

## Eliminazione
- **Eliminazione per righe**: rimuovere le osservazioni con valori mancanti
```python
df_cleaned = df.dropna()  # Elimina righe con qualsiasi valore mancante
df_cleaned = df.dropna(thresh=5)  # Elimina righe con meno di 5 valori non-NA
```

- **Eliminazione per colonne**: rimuovere le variabili con troppi valori mancanti
```python
df_cleaned = df.dropna(axis=1, thresh=0.7*len(df))  # Elimina colonne con >30% di valori mancanti
```

## Imputazione
- **Imputazione statistica**: sostituire con media, mediana, moda
```python
df['age'].fillna(df['age'].mean(), inplace=True)  # Media
df['age'].fillna(df['age'].median(), inplace=True)  # Mediana
df['category'].fillna(df['category'].mode()[0], inplace=True)  # Moda
```

---

# Tecniche Avanzate di Imputazione

- **Imputazione basata su modelli**: KNN, regressione, algoritmi di machine learning
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

- **Imputazione multipla**: generare più valori plausibili per ogni valore mancante
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

- **Indicatori di missingness**: aggiungere variabili che indicano la presenza di valori mancanti
```python
df['age_missing'] = df['age'].isnull().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)
```

---

# 4. Rilevamento e Trattamento degli Outlier

## Definizione di outlier
- **Outlier**: osservazioni che deviano significativamente dal resto dei dati
- **Cause**: errori di misurazione, variabilità naturale, frodi, eventi rari

## Metodi di rilevamento
- **Metodi statistici**: Z-score, IQR (Interquartile Range)
```python
# Z-score
from scipy import stats
z_scores = stats.zscore(df['column'])
outliers = df[abs(z_scores) > 3]  # Soglia comune: 3 deviazioni standard

# IQR
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['column'] < Q1 - 1.5 * IQR) | (df['column'] > Q3 + 1.5 * IQR)]
```

---

# Visualizzazione degli Outlier

- **Box plot**: visualizzazione della distribuzione e degli outlier
```python
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['column'])
plt.title('Box Plot con Outlier')
plt.tight_layout()
```

- **Scatter plot**: visualizzazione in 2D per identificare pattern anomali
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='feature1', y='feature2', data=df)
plt.title('Scatter Plot per Identificazione Outlier')
plt.tight_layout()
```

---

# Strategie per Gestire gli Outlier

## Trattamento degli outlier
- **Eliminazione**: rimuovere le osservazioni outlier
```python
# Rimozione basata su Z-score
df_cleaned = df[abs(stats.zscore(df['column'])) <= 3]
```

- **Trasformazione**: applicare trasformazioni per ridurre l'impatto degli outlier
```python
# Log transformation
df['column_log'] = np.log1p(df['column'])  # log(1+x) per gestire valori zero

# Winsorization (capping)
lower_bound = df['column'].quantile(0.05)
upper_bound = df['column'].quantile(0.95)
df['column_winsorized'] = df['column'].clip(lower_bound, upper_bound)
```

- **Modelli robusti**: utilizzare algoritmi meno sensibili agli outlier

---

# 5. Tecniche di Pulizia dei Dati

## Correzione degli errori
- **Errori di battitura**: correzione di valori errati
```python
# Mappatura di correzione
corrections = {'Maile': 'Male', 'Femael': 'Female'}
df['gender'] = df['gender'].replace(corrections)
```

- **Errori di formato**: standardizzazione dei formati
```python
# Standardizzazione date
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Standardizzazione numeri di telefono
import re
df['phone'] = df['phone'].apply(lambda x: re.sub(r'\D', '', str(x)))
```

---

# Gestione dei Tipi di Dati

- **Conversione dei tipi**: assicurarsi che i dati abbiano il tipo corretto
```python
# Conversione a tipi appropriati
df['age'] = df['age'].astype('int64')
df['price'] = df['price'].astype('float64')
df['is_customer'] = df['is_customer'].astype('bool')
```

- **Parsing di stringhe**: estrarre informazioni da stringhe
```python
# Estrarre componenti da una stringa
df['first_name'] = df['full_name'].str.split(' ').str[0]
df['last_name'] = df['full_name'].str.split(' ').str[1]

# Estrarre con regex
df['zip_code'] = df['address'].str.extract(r'(\d{5})')
```

---

# 6. Normalizzazione e Standardizzazione

## Perché normalizzare?
- **Convergenza più rapida** per algoritmi di gradient descent
- **Equa importanza** tra feature con scale diverse
- **Requisito** per alcuni algoritmi (es. k-means, SVM)

## Tecniche di scaling
- **Min-Max Scaling**: trasforma i dati nell'intervallo [0,1]
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

- **Standardizzazione (Z-score)**: media 0 e deviazione standard 1
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

---

# Altre Tecniche di Scaling

- **Robust Scaling**: utilizza statistiche robuste (mediana e IQR)
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

- **Normalizzazione L1/L2**: normalizzazione per riga
```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer(norm='l2')  # Norma euclidea
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

- **Log Transformation**: utile per dati con distribuzione asimmetrica
```python
df['log_income'] = np.log1p(df['income'])
```

---

# 7. Gestione dei Dati Duplicati

## Identificazione dei duplicati
- **Duplicati esatti**: righe completamente identiche
```python
# Trovare duplicati
duplicates = df.duplicated()
print(f"Numero di righe duplicate: {duplicates.sum()}")

# Visualizzare i duplicati
df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
```

- **Duplicati parziali**: righe simili ma non identiche
```python
# Duplicati basati su sottoinsiemi di colonne
duplicates = df.duplicated(subset=['name', 'email'], keep=False)
```

---

# Strategie per Gestire i Duplicati

- **Rimozione dei duplicati esatti**
```python
# Rimuovere tutte le righe duplicate tranne la prima occorrenza
df_cleaned = df.drop_duplicates()

# Rimuovere tutte le righe duplicate tranne l'ultima occorrenza
df_cleaned = df.drop_duplicates(keep='last')

# Rimuovere tutte le occorrenze di righe duplicate
df_cleaned = df.drop_duplicates(keep=False)
```

- **Gestione dei duplicati parziali**
```python
# Rimuovere duplicati basati su specifiche colonne
df_cleaned = df.drop_duplicates(subset=['id', 'email'])

# Aggregare duplicati
df_grouped = df.groupby(['id', 'email']).agg({
    'age': 'mean',
    'purchases': 'sum',
    'last_visit': 'max'
}).reset_index()
```

---

# 8. Casi Studio Pratici

## Caso 1: Pulizia di un dataset di vendite al dettaglio
- Identificazione e gestione di valori mancanti nelle date di acquisto
- Correzione di errori nei codici prodotto
- Rilevamento di transazioni anomale (outlier nei prezzi)
- Standardizzazione dei nomi dei negozi
- Aggregazione delle vendite per categoria e periodo

## Caso 2: Preparazione di dati sanitari
- Anonimizzazione dei dati personali
- Gestione di valori mancanti nei risultati dei test
- Normalizzazione di misurazioni da diversi dispositivi
- Rilevamento di errori di registrazione
- Creazione di feature temporali (giorni tra visite)

---

# Best Practices per la Pulizia dei Dati

1. **Documentare ogni trasformazione** applicata ai dati
2. **Automatizzare il processo** con pipeline di pulizia
3. **Validare i risultati** dopo ogni fase di pulizia
4. **Mantenere versioni** dei dati in diverse fasi di pulizia
5. **Bilanciare accuratezza e completezza** (non eliminare troppi dati)
6. **Considerare il contesto del dominio** nelle decisioni di pulizia
7. **Utilizzare visualizzazioni** per comprendere l'impatto delle trasformazioni
8. **Iterare il processo** di pulizia in base ai risultati dei modelli

---

# Strumenti per la Pulizia dei Dati

- **Pandas**: manipolazione e analisi dei dati
- **NumPy**: calcolo numerico
- **Scikit-learn**: preprocessing e scaling
- **Missingno**: visualizzazione di valori mancanti
- **Dedupe**: identificazione di duplicati fuzzy
- **Great Expectations**: validazione dei dati
- **Dask/Spark**: pulizia di grandi dataset
- **OpenRefine**: pulizia interattiva dei dati

---

# Riepilogo: Raccolta e Pulizia dei Dati

- La **qualità dei dati** è fondamentale per il successo dei progetti di ML
- **Valori mancanti** richiedono strategie appropriate (eliminazione o imputazione)
- **Outlier** possono essere rilevati con metodi statistici e trattati in vari modi
- **Normalizzazione e standardizzazione** sono essenziali per molti algoritmi
- **Dati duplicati** devono essere identificati e gestiti correttamente
- La pulizia dei dati è un **processo iterativo** che richiede comprensione del dominio

---

# Domande?

![height:450px](https://media.istockphoto.com/id/1299982879/vector/data-cleaning-flat-concept-vector-illustration.jpg?s=612x612&w=0&k=20&c=3YOMJUqL0VC4KZQZjxLmPUy7qNBJxYzm_klcDjxEbPI=)

---

# Riferimenti

- Müller, A. C., & Guido, S. (2016). Introduction to Machine Learning with Python: A Guide for Data Scientists. O'Reilly Media.
- VanderPlas, J. (2016). Python Data Science Handbook. O'Reilly Media.
- McKinney, W. (2017). Python for Data Analysis. O'Reilly Media.
- Grus, J. (2019). Data Science from Scratch. O'Reilly Media.
- Kuhn, M., & Johnson, K. (2019). Feature Engineering and Selection: A Practical Approach for Predictive Models. CRC Press.
