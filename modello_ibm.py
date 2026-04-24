import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import kagglehub
import os
import numpy as np

import shap
import lime.lime_tabular

# ==========================================
# 1. DOWNLOAD E LETTURA DEL DATASET

print("Scaricamento del dataset da Kagglehub in corso...")
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
file_csv = [f for f in os.listdir(path) if f.endswith('.csv')][0]
percorso_completo = os.path.join(path, file_csv)
dataset = pd.read_csv(percorso_completo)

# ==========================================
# 2. PULIZIA E SPLIT DEI DATI

# La colonna da prevedere è 'Attrition' (Sì/No). Trasformare in 1 (Sì) e 0 (No)
dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0})
y_numpy = dataset['Attrition'].values
dataset = dataset.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

#ENCODING FEATURES get_dummies trasforma tutte le colonne di testo (es. Genere, Ruolo) in numeri (0 e 1)
dataset = pd.get_dummies(dataset, drop_first=True)
nomi_features = dataset.columns.tolist()    # Utile a calcolare i valori di SHAP e LIME 
X_numpy = dataset.values

# --- DIVISIONE IN TRAIN, VAL E TEST ---
# 1. Separare il Training Set (70%) dal resto (30%)
# Usare stratify=y_numpy per assicurarsi che la percentuale di dimissioni sia uguale in tutti i gruppi
X_train_np, X_temp, y_train_np, y_temp = train_test_split(X_numpy, y_numpy, test_size=0.30, random_state=42, stratify=y_numpy)

# 2. Dividere il "temp" (30%) a metà, ottenendo Validation (15%) e Test (15%)
X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# SCALING FEATURES Normalizzare (Calcolare lo scaler SOLO sul Train, per non intaccare accuratezza, precisone...)
scaler = StandardScaler()
X_train_scalati = scaler.fit_transform(X_train_np)
X_val_scalati = scaler.transform(X_val_np)
X_test_scalati = scaler.transform(X_test_np)

# Convertire in tensori PyTorch
X_train = torch.tensor(X_train_scalati, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)

X_val = torch.tensor(X_val_scalati, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test_scalati, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

# ==========================================
# 3. IL MODELLO E IL PESO

numero_di_features = X_train.shape[1]

class ReteAziendale(nn.Module):
    def __init__(self):
        super(ReteAziendale, self).__init__()
        self.strato_1 = nn.Linear(numero_di_features, 32)
        self.relu_1 = nn.ReLU()           
        self.strato_2 = nn.Linear(32, 16) 
        self.relu_2 = nn.ReLU()
        self.strato_3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu_1(self.strato_1(x))
        x = self.relu_2(self.strato_2(x))
        x = self.strato_3(x)
        return x


modello = ReteAziendale()

# Ricalcolare i pesi solo sul set di Training
moltiplicatore = (y_train_np == 0).sum() / (y_train_np == 1).sum()  # dipendenti rimasti(0) / dipendenti licenziati(1) 
peso_positivi = torch.tensor([moltiplicatore], dtype=torch.float32) # il peso ottenuto trasformato in tensore 
criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivi) # peso passato alla loss function 
ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)

# ==========================================
# 4. ADDESTRAMENTO CON VALIDATION 
epoche = 300
loss_train_lista = []
loss_val_lista = []

# --- VARIABILI PER L'EARLY STOPPING ---
limite_max = 20                           # Quante epoche aspettare senza miglioramenti prima di fermarsi
miglior_loss_val = float('inf')         # Inizializziamo il record all'infinito
epoche_senza_miglioramenti = 0          # Contatore
migliori_pesi = None                    # "Fotografia" del modello perfetto

print("Inizio dell'addestramento")
for epoca in range(epoche):
    # --- FASE DI TRAINING ---
    modello.train() # Dire a PyTorch che si sta addestrando
    ottimizzatore.zero_grad() 
    previsioni_train = modello(X_train)
    loss_train = criterio(previsioni_train, y_train)
    loss_train.backward()
    ottimizzatore.step()
    
    # --- FASE DI VALIDATION ---
    modello.eval() # Dire a PyTorch che si sta testando
    with torch.no_grad(): # Spegnere il calcolo dei gradienti per risparmiare memoria
        previsioni_val = modello(X_val)
        loss_val = criterio(previsioni_val, y_val)
    
    loss_train_corrente = loss_train.item()
    loss_val_corrente = loss_val.item()
    loss_train_lista.append(loss_train_corrente)
    loss_val_lista.append(loss_val_corrente)    
    
    # --- LOGICA DELL'EARLY STOPPING ---
    # 1. Se la loss attuale è un nuovo record assoluto...
    if loss_val_corrente < miglior_loss_val:
        miglior_loss_val = loss_val_corrente       # Aggiorna il record
        epoche_senza_miglioramenti = 0             # Azzera il contatore
        migliori_pesi = modello.state_dict()       # Fotografa e salva i "cervelli" della rete
    
    else:
        epoche_senza_miglioramenti += 1            
        
        # 3. Se la pazienza è finita, interrompere il ciclo
        if epoche_senza_miglioramenti >= limite_max:
            print(f"\nEARLY STOPPING ATTIVATO! L'addestramento si è fermato all'epoca {epoca+1}.")
            print(f"La Validation Loss non migliorava da {limite_max} epoche.")
            break 
    
    if (epoca + 1) % 20 == 0:
        print(f'Epoca [{epoca+1}/{epoche}], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}')
        
    # --- RIPRISTINO DEL MODELLO MIGLIORE ---
    #Il modello deve usare i pesi salvati quando la loss era al suo minimo storico, altrimenti testerà l'ultima epoca (quella peggiorata).
modello.load_state_dict(migliori_pesi)
print(f"\n Modello ripristinato ai pesi ottimali (Miglior Val Loss: {miglior_loss_val:.4f}). Pronto per il Test!")

# ==========================================
# 5. VALUTAZIONE FINALE SUL TEST SET 

modello.eval()
with torch.no_grad():
    logits_test = modello(X_test)
    #trasformare i logits in intervallo 0-1 (sigmoid)
    probabilita_test = torch.sigmoid(logits_test).detach().numpy()
    
valori_reali_test = y_test.numpy()
# Impostare una soglia di classificazione (0.5)
predizioni_finali_test = (probabilita_test > 0.5).astype(int)

#ACCURATEZZA [classificazioni corrette / totale classificazoni]
accuracy_test = accuracy_score(valori_reali_test, predizioni_finali_test)
#PRECISIONE [veri positivi / veri positivi + falsi positivi]
precisione = precision_score(valori_reali_test, predizioni_finali_test, zero_division=0)
#RECALL [veri positivi / veri positivi + falsi negativi] [percentuale veri positivi]
richiamo = recall_score(valori_reali_test, predizioni_finali_test, zero_division=0)
#F1-SCORE [media armonica di precisione e richiamo] [2 * ((precision * recall) / (precison + recall))]
f1 = f1_score(valori_reali_test, predizioni_finali_test, zero_division=0)
print(f"\nPUNTEGGI METRICHE DI VALUTAZIONE SUL TEST SET")
print(f"\n--- Accuratezza: {accuracy_test*100:.2f}% ")
print(f"\n--- Precisione:  {precisione*100:.2f}%")
print(f"\n--- Recall:  {richiamo*100:.2f}%")
print(f"\n--- F1-Score:  {f1*100:.2f}%")

# ==========================================
# 6. GRAFICI
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grafico Loss: Confrontiamo Train e Val
ax1.plot(loss_train_lista, color='blue', label='Train Loss')
ax1.plot(loss_val_lista, color='red', linestyle='--', label='Validation Loss')
ax1.set_title("Andamento dell'Errore")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Confusion Matrix sul TEST SET
matrice = confusion_matrix(valori_reali_test, predizioni_finali_test)
grafico_matrice = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=['Rimasto (0)', 'Licenziato (1)'])
grafico_matrice.plot(cmap=plt.cm.Blues, ax=ax2)
ax2.set_title("Matrice di Confusione (Test Set)")

plt.tight_layout()
plt.show()

# ==========================================
# 7. EXPLAINABLE AI: SHAP

print("\nCalcolo dei valori SHAP in corso...")

# Creare la funzione wrapper per SHAP
def predici_probabilita_shap(x_numpy):
    modello.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        logits = modello(x_tensor)
        # Trasformare in probabilità e converitre in array numpy 1D
        probabilita = torch.sigmoid(logits).numpy().flatten()
    return probabilita

# Usare un campione del set di addestramento (es. 100 istanze) come "background" 
# Per definire i valori di base (baseline) e velocizzare il calcolo.
background = X_train_scalati[:100]

# Inizializzare l'Explainer di SHAP usando il wrapper model-agnostic
explainer_shap = shap.Explainer(predici_probabilita_shap, background, feature_names=nomi_features)

# Calcolare i valori SHAP per il set di test (limitare a 100 campioni per velocità, X_test_scalati intero se si ha tempo di calcolo)
shap_values = explainer_shap(X_test_scalati[:100])

# --- VISUALIZZAZIONE GLOBALE SHAP ---
print("\nGenerazione del grafico SHAP Globale (Beeswarm)...")
plt.figure(figsize=(10, 6))
# Il BEESWARM plot mostra l'impatto di ogni feature su tutte le predizioni del test set.
# I colori rosso/blu indicano se il valore originale della feature era alto o basso.
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP: Importanza Globale delle Features")
plt.tight_layout()
plt.show()

# ---  BAR PLOT GLOBALE ---
#print("\nGenerazione del Bar Plot Globale SHAP...")
#plt.figure()
#shap.plots.bar(shap_values, show=False)
#plt.title("SHAP: Importanza Media Assoluta")
#plt.tight_layout()
#plt.show()

# ---  SCATTER / DEPENDENCE PLOT LOCALE ---
# Scegliere una colonna specifica usando il suo indice o il nome. 
# Mettiamo '0' per prendere la prima feature (es. 'Age') come esempio.
#print("\nGenerazione dello Scatter Plot per la prima variabile...")
#plt.figure()
#shap.plots.scatter(shap_values[:, 0], show=False)
#plt.title(f"SHAP: Impatto della variabile '{nomi_features[0]}'")
#plt.tight_layout()
#plt.show()

# --- VISUALIZZAZIONE LOCALE SHAP (WATERFALL PLOT) ---
# Analisi di un singolo dipendente (il primo del test set, indice 0)
print("\nGenerazione del grafico SHAP Locale (Waterfall) per la prima istanza...")
plt.figure()
# Il Waterfall plot parte dal valore di base (media) e mostra come ogni variabile 
# "Spinge" la probabilità in alto (rosso) o in basso (blu) per questo specifico utente.
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP: Spiegazione Locale (Istanza 0)")
plt.tight_layout()
plt.show()


# ==========================================
# 8. EXPLAINABLE AI: LIME

print("\nCalcolo delle spiegazioni LIME in corso...")

# Crere la funzione wrapper per LIME
# LIME per la classificazione binaria richiede che l'output sia una matrice 
# Con 2 colonne: [probabilità_classe_0, probabilità_classe_1]
def predici_probabilita_lime(x_numpy):
    modello.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        logits = modello(x_tensor)
        prob_classe_1 = torch.sigmoid(logits).numpy()
        prob_classe_0 = 1.0 - prob_classe_1
        # Unire le due probabilità affiancandole
        return np.hstack((prob_classe_0, prob_classe_1))

# Inizializzare l'explainer di LIME addestrandolo sui dati di training (scalati)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scalati,
    feature_names=nomi_features,
    class_names=['Rimasto', 'Licenziato'],
    mode='classification',
    random_state=42
)

# --- VISUALIZZAZIONE LOCALE LIME ---
# Generare la spiegazione per lo stesso dipendente di prima (indice 0 del Test Set)
indice_da_spiegare = 0

spiegazione_lime = explainer_lime.explain_instance(
    data_row=X_test_scalati[indice_da_spiegare], 
    predict_fn=predici_probabilita_lime,
    num_features=10 # Mostrare solo le 10 features più rilevanti per questo dipendente
)

# Grafico LIME
fig_lime = spiegazione_lime.as_pyplot_figure()
plt.title(f"LIME: Spiegazione Locale (Istanza {indice_da_spiegare})")
plt.tight_layout()
plt.show()

# Stampa testuale di LIME
print(f"\nProbabilità predetta per l'istanza {indice_da_spiegare}: {predici_probabilita_lime(X_test_scalati[[indice_da_spiegare]])[0][1]*100:.2f}% (Licenziato)")
print(f"Classe Reale: {int(y_test_np[indice_da_spiegare].item())}")

# --- ESPORTAZIONE LIME IN HTML ---
nome_file_html = f"spiegazione_lime_dipendente_{indice_da_spiegare}.html"
spiegazione_lime.save_to_file(nome_file_html)
print(f"\nReport LIME interattivo salvato con successo nel file: {nome_file_html}")