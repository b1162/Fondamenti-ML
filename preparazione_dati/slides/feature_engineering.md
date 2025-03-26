---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Preparazione e Pre-elaborazione dei Dati
## Parte 2: Feature Engineering

---

# Indice

1. Concetti fondamentali di feature engineering
2. Trasformazione delle variabili
3. Creazione di nuove feature
4. Selezione delle feature
5. Riduzione della dimensionalità
6. Encoding di variabili categoriche
7. Scaling e normalizzazione
8. Tecniche avanzate di feature engineering

---

# 1. Concetti Fondamentali di Feature Engineering

## Cos'è il Feature Engineering?
- **Definizione**: processo di creazione, trasformazione e selezione delle variabili (feature) per migliorare le performance dei modelli di machine learning
- **Obiettivo**: estrarre il massimo valore informativo dai dati grezzi
- **Importanza**: spesso ha un impatto maggiore sulla performance rispetto alla scelta dell'algoritmo

## Principi chiave
- **Conoscenza del dominio**: comprendere il contesto del problema
- **Creatività**: pensare a nuove rappresentazioni dei dati
- **Iterazione**: processo ciclico di creazione e valutazione
- **Validazione**: misurare l'impatto delle nuove feature sui modelli

---

# Ciclo del Feature Engineering

![height:450px](https://miro.medium.com/max/1400/1*_E_HACcY0h2-YBVvF-9qrQ.png)

---

# 2. Trasformazione delle Variabili

## Trasformazioni matematiche
- **Logaritmica**: riduce l'asimmetria, gestisce distribuzioni con coda lunga
```python
df['log_income'] = np.log1p(df['income'])  # log(1+x) per gestire valori zero
```

- **Radice quadrata**: meno aggressiva della logaritmica
```python
df['sqrt_distance'] = np.sqrt(df['distance'])
```

- **Box-Cox**: famiglia di trasformazioni parametriche
```python
from scipy import stats
df['boxcox_sales'], lambda_value = stats.boxcox(df['sales'])
```

- **Yeo-Johnson**: estensione di Box-Cox che gestisce valori negativi
```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['yj_feature'] = pt.fit_transform(df[['feature']])
```

---

# Visualizzazione dell'Effetto delle Trasformazioni

![height:450px](https://miro.medium.com/max/1400/1*ef38EH1v1YHVN3ACj_9Iuw.png)

---

# Trasformazioni Trigonometriche e Polinomiali

- **Trigonometriche**: utili per dati ciclici (ore, giorni, stagioni)
```python
# Trasformare l'ora del giorno (0-24) in coordinate circolari
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Trasformare il giorno della settimana (0-6) in coordinate circolari
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

- **Polinomiali**: catturano relazioni non lineari
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['x1', 'x2']])
poly_df = pd.DataFrame(poly_features, 
                      columns=poly.get_feature_names_out(['x1', 'x2']))
```

---

# 3. Creazione di Nuove Feature

## Feature basate sul dominio
- **Rapporti e differenze**: relazioni tra variabili
```python
# Rapporto prezzo/superficie per immobili
df['price_per_sqm'] = df['price'] / df['area']

# Body Mass Index (BMI)
df['bmi'] = df['weight'] / (df['height'] ** 2)
```

- **Aggregazioni**: statistiche su gruppi
```python
# Media di acquisti per cliente
avg_purchase = df.groupby('customer_id')['purchase_amount'].mean()
df = df.merge(avg_purchase.rename('avg_purchase_per_customer'), 
             left_on='customer_id', right_index=True)
```

---

# Feature Temporali

- **Estrazione di componenti temporali**
```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
```

- **Lag e differenze**
```python
# Feature di lag (valori precedenti)
df['sales_lag1'] = df.groupby('store_id')['sales'].shift(1)
df['sales_lag7'] = df.groupby('store_id')['sales'].shift(7)  # Settimana precedente

# Differenze (cambiamenti)
df['sales_diff1'] = df['sales'] - df['sales_lag1']
df['sales_pct_change'] = df['sales'].pct_change()
```

---

# Feature di Testo

- **Conteggi e lunghezze**
```python
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
df['char_count'] = df['text'].apply(lambda x: len(x.replace(" ", "")))
```

- **Presenza di pattern**
```python
df['contains_url'] = df['text'].str.contains('http|www').astype(int)
df['exclamation_count'] = df['text'].str.count('!')
df['question_count'] = df['text'].str.count('\?')
df['uppercase_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
```

---

# Feature di Interazione

- **Prodotti incrociati**: catturano interazioni tra variabili
```python
df['age_income'] = df['age'] * df['income']
```

- **Feature binarie combinate**
```python
df['high_income_college'] = ((df['income'] > 50000) & 
                            (df['education'] == 'college')).astype(int)
```

- **Clustering come feature**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
df['cluster'] = kmeans.fit_predict(df[['feature1', 'feature2']])
```

---

# 4. Selezione delle Feature

## Perché selezionare le feature?
- **Riduzione dell'overfitting**: meno feature = modello più semplice
- **Miglioramento delle performance**: eliminazione di feature irrilevanti
- **Riduzione dei tempi di training**: meno dati = calcoli più veloci
- **Interpretabilità**: modelli più semplici sono più facili da spiegare

## Approcci alla selezione
- **Filter methods**: basati su statistiche, indipendenti dal modello
- **Wrapper methods**: utilizzano il modello per valutare sottoinsiemi di feature
- **Embedded methods**: la selezione è parte del processo di training

---

# Filter Methods

- **Correlazione**: rimuovere feature altamente correlate
```python
# Matrice di correlazione
corr_matrix = df.corr().abs()

# Identificare coppie di feature con alta correlazione
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_reduced = df.drop(to_drop, axis=1)
```

- **Test statistici**: ANOVA, chi-quadro
```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2
# ANOVA F-value per feature numeriche
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
```

---

# Wrapper Methods

- **Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
```

- **Forward/Backward Selection**
```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(RandomForestClassifier(), 
         k_features=10, 
         forward=True,  # True: forward, False: backward
         scoring='accuracy',
         cv=5)
sfs.fit(X, y)
selected_features = X.columns[list(sfs.k_feature_idx_)]
```

---

# Embedded Methods

- **Lasso Regression**: penalità L1 che può portare coefficienti a zero
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]
```

- **Random Forest Feature Importance**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features = X.columns[indices[:10]]  # Top 10 feature
```

---

# Visualizzazione dell'Importanza delle Feature

```python
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
```

![height:350px](https://scikit-learn.org/stable/_images/sphx_glr_plot_forest_importances_001.png)

---

# 5. Riduzione della Dimensionalità

## Tecniche di riduzione
- **PCA (Principal Component Analysis)**: proiezione lineare che massimizza la varianza
- **t-SNE**: visualizzazione di dati ad alta dimensionalità preservando la struttura locale
- **UMAP**: alternativa a t-SNE con migliore preservazione della struttura globale
- **Autoencoder**: reti neurali per compressione non lineare

## Vantaggi
- **Visualizzazione**: riduzione a 2-3 dimensioni per visualizzazione
- **Rimozione del rumore**: componenti minori spesso contengono rumore
- **Gestione della multicollinearità**: rimuove correlazioni tra feature
- **Miglioramento delle performance**: riduce l'overfitting

---

# Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardizzazione (importante per PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicazione PCA
pca = PCA(n_components=2)  # Riduzione a 2 dimensioni
X_pca = pca.fit_transform(X_scaled)

# Varianza spiegata
explained_variance = pca.explained_variance_ratio_
print(f"Varianza spiegata: {explained_variance}")
print(f"Varianza cumulativa: {np.sum(explained_variance)}")

# Visualizzazione
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.xlabel('Prima componente principale')
plt.ylabel('Seconda componente principale')
plt.colorbar(label='Classe')
plt.title('PCA a 2 dimensioni')
```

---

# t-SNE e UMAP

```python
from sklearn.manifold import TSNE
from umap import UMAP

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
umap_model = UMAP(n_components=2, random_state=0)
X_umap = umap_model.fit_transform(X_scaled)

# Visualizzazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8)
ax1.set_title('t-SNE')

ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.8)
ax2.set_title('UMAP')

plt.tight_layout()
```

---

# Confronto tra Tecniche di Riduzione della Dimensionalità

![height:450px](https://miro.medium.com/max/1400/1*QS7rFiCMOWGSdQpOYOzprQ.png)

---

# 6. Encoding di Variabili Categoriche

## Perché è necessario l'encoding?
- La maggior parte degli algoritmi di ML richiede input numerici
- Le variabili categoriche contengono informazioni importanti
- Diversi tipi di encoding catturano diverse relazioni

## Tipi di variabili categoriche
- **Nominali**: categorie senza ordine intrinseco (es. colori, paesi)
- **Ordinali**: categorie con un ordine naturale (es. basso/medio/alto)
- **Binarie**: solo due categorie (es. sì/no, vero/falso)
- **Cicliche**: categorie che si ripetono (es. giorni della settimana)

---

# One-Hot Encoding

- **Uso**: variabili categoriche nominali con poche categorie
- **Funzionamento**: crea una colonna binaria per ogni categoria

```python
# Pandas
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=False)

# Scikit-learn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop=None)
encoded_features = encoder.fit_transform(df[['color']])
encoded_df = pd.DataFrame(encoded_features, 
                         columns=encoder.get_feature_names_out(['color']))
```

- **Vantaggi**: semplice, preserva tutte le informazioni
- **Svantaggi**: crea molte colonne con dataset ad alta cardinalità

---

# Label Encoding e Ordinal Encoding

- **Label Encoding**: converte ogni categoria in un numero intero
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['color_encoded'] = encoder.fit_transform(df['color'])
```

- **Ordinal Encoding**: simile al label encoding ma con ordine specificato
```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['basso', 'medio', 'alto']])
df['level_encoded'] = encoder.fit_transform(df[['level']])
```

- **Vantaggi**: compatto, mantiene l'ordine (per ordinal encoding)
- **Svantaggi**: introduce relazioni ordinali che potrebbero non esistere

---

# Target Encoding e Mean Encoding

- **Funzionamento**: sostituisce la categoria con la media della variabile target
```python
# Per ogni categoria, calcola la media della variabile target
target_means = df.groupby('category')['target'].mean()

# Applica l'encoding
df['category_encoded'] = df['category'].map(target_means)

# Con validazione incrociata per evitare data leakage
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'], df['target'])
```

- **Vantaggi**: efficace per categorie ad alta cardinalità, cattura relazioni con il target
- **Svantaggi**: rischio di overfitting, richiede gestione del data leakage

---

# Frequency Encoding e Weight of Evidence

- **Frequency Encoding**: sostituisce la categoria con la sua frequenza
```python
frequency = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(frequency)
```

- **Weight of Evidence**: log del rapporto tra probabilità di eventi positivi e negativi
```python
def calculate_woe(df, feature, target):
    df_woe = pd.DataFrame()
    df_woe['total'] = df.groupby([feature])[target].count()
    df_woe['positives'] = df.groupby([feature])[target].sum()
    df_woe['negatives'] = df_woe['total'] - df_woe['positives']
    df_woe['p_pos'] = df_woe['positives'] / df_woe['positives'].sum()
    df_woe['p_neg'] = df_woe['negatives'] / df_woe['negatives'].sum()
    df_woe['woe'] = np.log(df_woe['p_pos'] / df_woe['p_neg'])
    return df_woe['woe'].to_dict()

woe_dict = calculate_woe(df, 'category', 'target')
df['category_woe'] = df['category'].map(woe_dict)
```

---

# Encoding per Variabili ad Alta Cardinalità

- **Problema**: troppe categorie uniche (es. codici postali, ID prodotto)
- **Soluzioni**:
  - **Raggruppamento**: combinare categorie rare
  ```python
  # Sostituire categorie rare con "Other"
  counts = df['category'].value_counts()
  mask = df['category'].isin(counts[counts < 10].index)
  df.loc[mask, 'category'] = 'Other'
  ```
  
  - **Hashing**: mappare categorie a un numero fisso di feature
  ```python
  from sklearn.feature_extraction import FeatureHasher
  hasher = FeatureHasher(n_features=10, input_type='string')
  hashed_features = hasher.transform(df['category'].astype(str))
  ```

---

# 7. Scaling e Normalizzazione

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

- **Robust Scaling**: utilizza statistiche robuste (mediana e IQR)
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

---

# Quando Usare Ciascuna Tecnica di Scaling

| Tecnica | Quando usarla | Vantaggi | Svantaggi |
|---------|---------------|----------|-----------|
| **Min-Max Scaling** | Quando si desidera un range fisso [0,1] | Preserva la distribuzione | Sensibile agli outlier |
| **Standardizzazione** | Algoritmi che assumono distribuzione normale | Gestisce outlier meglio di Min-Max | Non garantisce un range fisso |
| **Robust Scaling** | Dati con molti outlier | Resistente agli outlier | Può comprimere troppo i dati normali |
| **Normalizzazione L1/L2** | Quando la scala relativa è importante | Utile per dati sparsi | Può distorcere le relazioni tra feature |

---

# 8. Tecniche Avanzate di Feature Engineering

## Automated Feature Engineering
- **Featuretools**: libreria Python per automated feature engineering
```python
import featuretools as ft
# Definire le entità
entities = {
    "customers": (customers_df, "customer_id"),
    "transactions": (transactions_df, "transaction_id")
}
# Definire le relazioni
relationships = [("customers", "customer_id", "transactions", "customer_id")]
# Creare il feature store
feature_matrix, feature_defs = ft.dfs(
    entityset=ft.EntitySet(entities=entities, relationships=relationships),
    target_entity="customers",
    agg_primitives=["sum", "mean", "count", "std"],
    trans_primitives=["month", "year", "day"]
)
```

---

# Deep Feature Synthesis

- **Transfer Learning**: utilizzare feature estratte da modelli pre-addestrati
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Caricare modello pre-addestrato
model = VGG16(weights='imagenet', include_top=False)

# Funzione per estrarre feature
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Applicare a tutte le immagini
image_features = [extract_features(img) for img in image_paths]
```

---

# Feature Learning con Autoencoder

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Definire l'architettura dell'autoencoder
input_dim = X.shape[1]
encoding_dim = 10  # Dimensione dello spazio latente

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Autoencoder completo
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Training
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True,
               validation_data=(X_test, X_test))

# Estrarre l'encoder per generare le feature
encoder_model = Model(inputs=input_layer, outputs=encoder)
encoded_features = encoder_model.predict(X)
```

---

# Best Practices nel Feature Engineering

1. **Iniziare semplice**: prima le trasformazioni base, poi quelle avanzate
2. **Conoscere i dati**: esplorare e visualizzare prima di ingegnerizzare
3. **Validare l'impatto**: misurare l'effetto di ogni nuova feature
4. **Evitare il data leakage**: attenzione alle informazioni future
5. **Documentare**: tenere traccia delle trasformazioni e del loro scopo
6. **Automatizzare**: creare pipeline riutilizzabili
7. **Bilanciare complessità e interpretabilità**: feature più complesse non sempre migliorano il modello
8. **Considerare il costo computazionale**: alcune feature possono essere costose da calcolare

---

# Pipeline di Feature Engineering

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Definire i preprocessori per diversi tipi di colonne
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'category', 'region']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinare i preprocessori
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creare la pipeline completa con un modello
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit e predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

# Riepilogo: Feature Engineering

- Il **feature engineering** è spesso il fattore più importante per il successo di un modello
- La **trasformazione delle variabili** può migliorare la distribuzione e catturare relazioni non lineari
- La **creazione di nuove feature** richiede creatività e conoscenza del dominio
- La **selezione delle feature** riduce l'overfitting e migliora l'interpretabilità
- La **riduzione della dimensionalità** aiuta a visualizzare e comprimere i dati
- L'**encoding di variabili categoriche** trasforma informazioni qualitative in quantitative
- Lo **scaling** è essenziale per molti algoritmi di machine learning
- Le **tecniche avanzate** come l'automated feature engineering possono scoprire pattern complessi

---

# Domande?

![height:450px](https://miro.medium.com/max/1400/1*2T5rbjOBGVFdSvtlhCqlNg.png)

---

# Riferimenti

- Zheng, A., & Casari, A. (2018). Feature Engineering for Machine Learning. O'Reilly Media.
- Kuhn, M., & Johnson, K. (2019). Feature Engineering and Selection: A Practical Approach for Predictive Models. CRC Press.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- Kanter, J. M., & Veeramachaneni, K. (2015). Deep feature synthesis: Towards automating data science endeavors. IEEE DSAA.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of machine learning research.
