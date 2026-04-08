#  IBM HR Analytics: Previsione Dimissioni con Rete Neurale (PyTorch)

Questo progetto utilizza il Deep Learning per affrontare un problema classico di Risorse Umane: prevedere se un dipendente lascerà l'azienda (Attrition) basandosi sui suoi dati lavorativi e anagrafici. Il progetto utilizza il celebre dataset "IBM HR Analytics Employee Attrition & Performance".

Il modello è una **Rete Neurale Artificiale (ANN)** in PyTorch per una classificazione binaria (0 = Rimasto, 1 = Licenziato).

##  Caratteristiche Tecniche del Progetto

* **Download Automatico:** Il dataset viene scaricato direttamente da Kaggle tramite la libreria `kagglehub` alla prima esecuzione, senza bisogno di file CSV locali.
* **Pre-processing:** * Trasformazione delle variabili categoriche in numeriche tramite One-Hot Encoding (`pd.get_dummies`).
  * Normalizzazione dei dati (`StandardScaler`).

* **Data Splitting Rigoroso (Prevenzione Data Leakage):** * I dati sono divisi in **Training (70%)**, **Validation (15%)** e **Test (15%)**. 
  * Lo scaler viene calcolato ("fittato") *esclusivamente* sui dati di Training per evitare fughe di informazioni, e poi applicato al resto.

* **Gestione dello Sbilanciamento delle Classi:** Dato che i dipendenti dimissionari sono una minoranza, la funzione di costo (`BCEWithLogitsLoss`) include un **peso proporzionale (`pos_weight`)**. Questo costringe la rete a penalizzare maggiormente i falsi negativi (i dimissionari non individuati).

##  Architettura del Modello

La rete neurale è composta da tre strati lineari (Fully Connected):
1. **Input Layer -> Hidden Layer 1:** 32 neuroni con attivazione ReLU.
2. **Hidden Layer 1 -> Hidden Layer 2:** 16 neuroni con attivazione ReLU.
3. **Hidden Layer 2 -> Output Layer:** 1 neurone (Output Logit elaborato internamente dalla Loss function).

L'ottimizzazione è gestita tramite **Adam** con un Learning Rate di `0.001`.

##  Come eseguire il progetto

1. Clona questo repository sul tuo computer.
2. Attiva un ambiente virtuale
3. Lancia da terminale:
    pip install -r requirements.txt
3. Esegui lo script principale:
    python modello_ibm.py