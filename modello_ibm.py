import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # <-- NUOVO IMPORT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import kagglehub
import os
import numpy as np

# ==========================================
# 1. DOWNLOAD E LETTURA DEL DATASET

print("Scaricamento del dataset da Kaggle in corso...")
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
file_csv = [f for f in os.listdir(path) if f.endswith('.csv')][0]
percorso_completo = os.path.join(path, file_csv)
dataset = pd.read_csv(percorso_completo)

# ==========================================
# 2. PULIZIA E SPLIT DEI DATI (NUOVO)

# La colonna da prevedere è 'Attrition' (Sì/No). La trasformiamo in 1 (Sì) e 0 (No)
dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0})
y_numpy = dataset['Attrition'].values
dataset = dataset.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

#ENCODING FEATURES get_dummies trasforma tutte le colonne di testo (es. Genere, Ruolo) in numeri (0 e 1)
dataset = pd.get_dummies(dataset, drop_first=True)
X_numpy = dataset.values

# --- DIVISIONE IN TRAIN, VAL E TEST ---
# 1. Separiamo il Training Set (70%) dal resto (30%)
# Usiamo stratify=y_numpy per assicurarci che la percentuale di dimissioni sia uguale in tutti i gruppi
X_train_np, X_temp, y_train_np, y_temp = train_test_split(X_numpy, y_numpy, test_size=0.30, random_state=42, stratify=y_numpy)

# 2. Dividiamo il "temp" (30%) a metà, ottenendo Validation (15%) e Test (15%)
X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# SCALING FEATURES Normalizziamo (Calcoliamo lo scaler SOLO sul Train!)
scaler = StandardScaler()
X_train_scalati = scaler.fit_transform(X_train_np)
X_val_scalati = scaler.transform(X_val_np)
X_test_scalati = scaler.transform(X_test_np)

# Convertiamo in tensori PyTorch
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

# Ricalcoliamo i pesi solo sul set di Training
moltiplicatore = (y_train_np == 0).sum() / (y_train_np == 1).sum()
peso_positivi = torch.tensor([moltiplicatore], dtype=torch.float32)
criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivi) 
ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)

# ==========================================
# 4. ADDESTRAMENTO CON VALIDATION (AGGIORNATO)
epoche = 150
loss_train_lista = []
loss_val_lista = []

print("Inizio dell'addestramento...")
for epoca in range(epoche):
    # --- FASE DI TRAINING ---
    modello.train() # Diciamo a PyTorch che stiamo addestrando
    ottimizzatore.zero_grad() 
    previsioni_train = modello(X_train)
    loss_train = criterio(previsioni_train, y_train)
    loss_train.backward()
    ottimizzatore.step()
    
    # --- FASE DI VALIDATION ---
    modello.eval() # Diciamo a PyTorch che stiamo solo testando
    with torch.no_grad(): # Spegniamo il calcolo dei gradienti per risparmiare memoria
        previsioni_val = modello(X_val)
        loss_val = criterio(previsioni_val, y_val)
    
    loss_train_lista.append(loss_train.item())
    loss_val_lista.append(loss_val.item())
    
    if (epoca + 1) % 20 == 0:
        print(f'Epoca [{epoca+1}/{epoche}], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}')

# ==========================================
# 5. VALUTAZIONE FINALE SUL TEST SET 

modello.eval()
with torch.no_grad():
    logits_test = modello(X_test)
    probabilita_test = torch.sigmoid(logits_test).detach().numpy()
    
valori_reali_test = y_test.numpy()
predizioni_finali_test = (probabilita_test > 0.5).astype(int)

#ACCURATEZZA 
accuracy_test = (predizioni_finali_test == valori_reali_test).sum() / len(valori_reali_test)
print(f"\n--- Accuratezza sul TEST SET (dati mai visti): {accuracy_test*100:.2f}% ---")

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

# Matrice di Confusione sul TEST SET
matrice = confusion_matrix(valori_reali_test, predizioni_finali_test)
grafico_matrice = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=['Rimasto (0)', 'Licenziato (1)'])
grafico_matrice.plot(cmap=plt.cm.Blues, ax=ax2)
ax2.set_title("Matrice di Confusione (Test Set)")

plt.tight_layout()
plt.show()