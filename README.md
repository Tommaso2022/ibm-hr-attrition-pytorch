#  IBM HR Analytics: Previsione Dimissioni con Rete Neurale (PyTorch)

Questo progetto implementa un modello di Deep Learning per prevedere l'attrition (dimissioni) dei dipendenti usando il dataset IBM HR Analytics. Include addestramento con early stopping, valutazione completa e tecnice di Explainable AI con SHAP e LIME.

Il modello è una **Rete Neurale Artificiale (ANN)** in PyTorch per una classificazione binaria (0 = Rimasto, 1 = Licenziato).

##  Caratteristiche Tecniche del Progetto

* **Download Automatico:** Il dataset viene scaricato direttamente da Kaggle tramite la libreria `kagglehub` alla prima esecuzione, senza bisogno di file CSV locali.
* **Pre-processing:** * Trasformazione delle variabili categoriche in numeriche tramite One-Hot Encoding (`pd.get_dummies`).
  * Normalizzazione dei dati (`StandardScaler`).

* **Data Splitting Rigoroso (Prevenzione Data Leakage):** * I dati sono divisi in **Training (70%)**, **Validation (15%)** e **Test (15%)**. 
  * Lo scaler viene calcolato ("fittato") *esclusivamente* sui dati di Training per evitare fughe di informazioni, e poi applicato al resto.

* **Gestione dello Sbilanciamento delle Classi:** Dato che i dipendenti dimissionari sono una minoranza, la funzione di costo (`BCEWithLogitsLoss`) include un **peso proporzionale (`pos_weight`)**. Questo costringe la rete a penalizzare maggiormente i falsi negativi (i dimissionari non individuati).

* **Early Stopping**: Ferma l'addestramento quando il modello smette di migliorare per evitare l'overfitting.

* **Explainable AI**:
    - **SHAP**: Visualizzazione dell'impatto delle feature a livello globale (Beeswarm, Bar Plot) e locale (Waterfall Scatter plot).

    - **LIME**: Spiegazione locale della singola predizione con esportazione in report HTML.

##  Architettura del Modello

La rete neurale è composta da tre strati lineari (Fully Connected):
1. **Input Layer -> Hidden Layer 1:** 32 neuroni con attivazione ReLU.
2. **Hidden Layer 1 -> Hidden Layer 2:** 16 neuroni con attivazione ReLU.
3. **Hidden Layer 2 -> Output Layer:** 1 neurone (Output Logit elaborato internamente dalla Loss function).

L'ottimizzazione è gestita tramite **Adam** con un Learning Rate di `0.001`.

## Metriche di valutazione

Lo script produrrà in output le metriche di valutazione sul Test Set (dati mai visti dal modello):

* **Accuratezza:** Percentuale di predizioni corrette.

* **Precisione:** Capacità di non classificare come "licenziato" un dipendente che resta.

* **Recall (Richiamo):** Capacità di scovare effettivamente chi se ne andrà (fondamentale in HR).

* **F1-Score:** Media armonica tra Precisione e Recall.

##  Come eseguire il progetto

1. Clona questo repository sul tuo computer.
2. Attiva un ambiente virtuale
3. Lancia da terminale:
    pip install -r requirements.txt
3. Esegui lo script principale:
    python modello_ibm.py

