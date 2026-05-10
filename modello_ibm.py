import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report,accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import kagglehub
import numpy as np
import copy
import seaborn as sns


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
    # Se la loss attuale è un nuovo record assoluto...
    if loss_val_corrente < miglior_loss_val:
        miglior_loss_val = loss_val_corrente       # Aggiorna il record
        epoche_senza_miglioramenti = 0             # Azzera il contatore
        migliori_pesi = modello.state_dict()       # Fotografa e salva i "cervelli" della rete
    
    else:
        epoche_senza_miglioramenti += 1            
        
        # Se il limite è suprato, interrompere il ciclo
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
risultati_modelli = []
modello.eval()
with torch.no_grad():
    logits_test = modello(X_test)
    #trasformare i logits in intervallo 0-1 (sigmoid)
    probabilita_test = torch.sigmoid(logits_test).detach().numpy()
    
valori_reali_test = y_test.numpy()
# Impostare una soglia di classificazione (0.5)
predizioni_finali_test = (probabilita_test > 0.5).astype(int)

# Salvare le metriche della Rete Neurale per il grafico
risultati_modelli.append({
    'Modello': 'Rete Neurale',
    'Accuracy': accuracy_score(valori_reali_test, predizioni_finali_test),
    'Precision': precision_score(valori_reali_test, predizioni_finali_test, zero_division=0),
    'Recall': recall_score(valori_reali_test, predizioni_finali_test, zero_division=0),
    'F1-Score': f1_score(valori_reali_test, predizioni_finali_test, zero_division=0)
})

print("\n=== REPORT RETE NEURALE  ===")
print(classification_report(valori_reali_test, predizioni_finali_test, target_names=['Rimasto (0)', 'Licenziato (1)']))
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

print("\nAvvio il confronto tra modelli in corso...")

#  Definire i nuovi modelli da testare
modelli_tradizionali = {
    'Regressione Logistica': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=1),
    'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=moltiplicatore.item(), random_state=42, eval_metric='logloss', n_jobs=1)
}

#  Addestramento e valutazione automatica nel ciclo
# usare y_train_np.flatten() perché sklearn preferisce array 1D (es. [0, 1, 0]) invece di array 2D colonna (es. [[0], [1], [0]]) che si usa per PyTorch.

for nome_modello, modello_test in modelli_tradizionali.items(): # Usare modello_test invece di modello
    # Nella variabile modello c'è ancora la rete neurale, da cui si calcoleranno i valori shap e lime 
    
    # Fase di addestramento
    modello_test.fit(X_train_scalati, y_train_np.flatten())
    
    # Fase di predizione
    predizioni = modello_test.predict(X_test_scalati)
    
    # Salvare le metriche durante i report per poter generare il grafico
    risultati_modelli.append({
        'Modello': nome_modello,
        'Accuracy': accuracy_score(y_test_np, predizioni),
        'Precision': precision_score(y_test_np, predizioni, zero_division=0),
        'Recall': recall_score(y_test_np, predizioni, zero_division=0),
        'F1-Score': f1_score(y_test_np, predizioni, zero_division=0)
    })
    
    print(f"\n=== REPORT {nome_modello.upper()} ===")
    print(classification_report(y_test_np, predizioni, target_names=['Rimasto (0)', 'Licenziato (1)']))
    # --- Generazione del Grafico a Barre Comparativo ---

tabella_confronto = pd.DataFrame(risultati_modelli)

# Creazione del grafico
tabella_confronto.set_index('Modello').plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Confronto Metriche: Rete Neurale vs XGB, Random Forest, Log. Regression (Focus Classe 1)')
plt.ylabel('Punteggio (0 - 1)')
plt.xticks(rotation=15)
plt.ylim(0, 1.1) # il limite y poco sopra l'1 per far spazio alla legenda
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 7. EXPLAINABLE AI: SHAP GLOBALE PER TUTTI I MODELLI

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

#SHAP A LIVELLO GLOBALE E LOCALE SOLO RETE NEURALE 
# --- VISUALIZZAZIONE GLOBALE SHAP ---
#print("\nGenerazione del grafico SHAP Globale (Beeswarm)...")
#plt.figure(figsize=(10, 6))
# Il BEESWARM plot mostra l'impatto di ogni feature su tutte le predizioni del test set.
# I colori rosso/blu indicano se il valore originale della feature era alto o basso.
#shap.plots.beeswarm(shap_values, show=False)
#plt.title("SHAP: Importanza Globale delle Features")
#plt.tight_layout()
#plt.show()

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
#print("\nGenerazione del grafico SHAP Locale (Waterfall) per la prima istanza...")
#plt.figure()
# Il Waterfall plot parte dal valore di base (media) e mostra come ogni variabile 
# "Spinge" la probabilità in alto (rosso) o in basso (blu) per questo specifico utente.
#shap.plots.waterfall(shap_values[0], show=False)
#plt.title("SHAP: Spiegazione Locale (Istanza 0)")
#plt.tight_layout()
#plt.show()

# FEATURE ABLATION (tecnica dell'oscuramento)
# mettere a 0 le feature più importanti rilevate da shap

# 1. Identificazione Top 3 Feature da SHAP
indici_top = np.argsort(np.abs(shap_values.values).mean(0))[::-1][:3]
metriche = {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score}

# Calcolo Base
base_perf = {n: f(y_test_np, predizioni_finali_test, **({'zero_division':0} if n!='Accuracy' else {})) for n, f in metriche.items()}

# 2. Test di Oscuramento
for idx in indici_top:
    X_osc = copy.deepcopy(X_test_scalati)
    X_osc[:, idx] = 0
    with torch.no_grad():
        p_osc = (torch.sigmoid(modello(torch.tensor(X_osc, dtype=torch.float32))).numpy() > 0.5).astype(int)
    
    res = {n: f(y_test_np, p_osc, **({'zero_division':0} if n!='Accuracy' else {})) for n, f in metriche.items()}
# ========================================================================
# FUNZIONE PER NORMALIZZARE L’OUTPUT DI SHAP 


def normalizza_shap_output(shap_vals):
    
    # Caso 1: lista → significa modello binario → USARE la classe 1
    if isinstance(shap_vals, list):
        return np.array(shap_vals[1])

    shap_vals = np.array(shap_vals)

    # Caso 2: array 3D (n, f, 2) → prendere solo la colonna della classe 1
    if shap_vals.ndim == 3:
        return shap_vals[:, :, 1]

    # Caso 3: array 2D → perfetto
    if shap_vals.ndim == 2:
        return shap_vals

    raise ValueError(f"Formato SHAP non riconosciuto: {shap_vals.shape}")


# ========================================================================
# 8. SHAP PER TUTTI I MODELLI

print("\n=== Calcolo SHAP per tutti i modelli ===")

modelli_addestrati = {
    'Rete Neurale': modello,
    'Regressione Logistica': modelli_tradizionali['Regressione Logistica'],
    'Random Forest': modelli_tradizionali['Random Forest'],
    'XGBoost': modelli_tradizionali['XGBoost']
}

shap_values_modelli = {}

for nome, model_corr in modelli_addestrati.items():
    print(f"\nCalcolo SHAP per: {nome}")

    # Rete neurale → wrapper personalizzato
    if nome == 'Rete Neurale':
        explainer = shap.Explainer(predici_probabilita_shap, background, feature_names=nomi_features)
        raw_vals = explainer(X_test_scalati).values

    # Modelli ad albero
    elif nome in ['Random Forest', 'XGBoost']:
        explainer = shap.TreeExplainer(model_corr)
        raw_vals = explainer.shap_values(X_test_scalati)

    # Regressione logistica
    elif nome == 'Regressione Logistica':
        explainer = shap.LinearExplainer(model_corr, X_train_scalati)
        raw_vals = explainer.shap_values(X_test_scalati)

    # Normalizzazione shap_values per renderlo sempre (n_sample, n_feat)
    shap_vals_matrix = normalizza_shap_output(raw_vals)
    shap_values_modelli[nome] = shap_vals_matrix

    # Beeswarm globale
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_matrix, X_test_scalati, feature_names=nomi_features, show=False)
    plt.title(f"SHAP Beeswarm - {nome}")
    plt.tight_layout()
    plt.show()


# ========================================================================
# 9. CONFRONTO TOP-FEATURE TRA I MODELLI

print("\n=== Confronto delle TOP 10 features tra modelli ===\n")

top_features_modelli = {}

for nome, sv in shap_values_modelli.items():
    importance = np.abs(sv).mean(axis=0)
    indici = np.argsort(importance)[::-1]
    top10 = [nomi_features[i] for i in indici[:10]]
    top_features_modelli[nome] = top10
    print(f"{nome}: {top10}\n")


# ========================================================================
# 10. FEATURE ABLATION 

print("\n=== Feature Ablation  ===")

# Top 3 comuni
top3_comuni = list(set.intersection(*(set(v[:3]) for v in top_features_modelli.values())))
print("\nTop 3 comuni tra tutti i modelli:", top3_comuni)

# Top 5 comuni
top5_comuni = list(set.intersection(*(set(v[:5]) for v in top_features_modelli.values())))
print("\nTop 5 comuni tra tutti i modelli:", top5_comuni)

# Dataset come DataFrame
df_train = pd.DataFrame(X_train_scalati, columns=nomi_features)
df_test = pd.DataFrame(X_test_scalati, columns=nomi_features)

# Dataset con rimozione top3
X_train_rimozione = df_train.drop(columns=top3_comuni).values
X_test_rimozione = df_test.drop(columns=top3_comuni).values

# Dataset solo top5
X_train_top = df_train[top5_comuni].values
X_test_top = df_test[top5_comuni].values

def riaddestra(Xtr, Xts):
    risultati = {}
    for nome in ['Regressione Logistica', 'Random Forest', 'XGBoost']:
        model_new = copy.deepcopy(modelli_tradizionali[nome])
        model_new.fit(Xtr, y_train_np)
        pred = model_new.predict(Xts)
        risultati[nome] = {
            "Accuracy": accuracy_score(y_test_np, pred),
            "Precision": precision_score(y_test_np, pred, zero_division=0),
            "Recall": recall_score(y_test_np, pred, zero_division=0),
            "F1": f1_score(y_test_np, pred, zero_division=0)
        }
    return risultati

print("\n=== CONFRONTO METRICHE TRA I MODELLI (3 configurazioni) ===\n")
print("\n=== Creazione dataset completi e ridotti con feature masking ===")
# Dataset completo (nessuna modifica)
X_train_completo = X_train_scalati.copy()
X_test_completo = X_test_scalati.copy()
# Dataset con top 3 features AZZERATE
X_train_no3 = X_train_scalati.copy()
X_test_no3 = X_test_scalati.copy()
for feat in top3_comuni:
    idx = nomi_features.index(feat)
    X_train_no3[:, idx] = 0
    X_test_no3[:, idx] = 0
# Dataset con SOLO top 5 features (tutte le altre a 0)
X_train_top5 = np.zeros_like(X_train_scalati)
X_test_top5 = np.zeros_like(X_test_scalati)
for feat in top5_comuni:
    idx = nomi_features.index(feat)
    X_train_top5[:, idx] = X_train_scalati[:, idx]
    X_test_top5[:, idx] = X_test_scalati[:, idx]
    
# ========================================================================
# 11. FUNZIONE PER VALUTARE NN + MODELLI SKLEARN

def valuta_modelli_tutti(Xtr, Xts, ytr=y_train_np, yts=y_test_np):
    risultati = []

    # 1) RETE NEURALE 

    modello.eval()
    with torch.no_grad():
        logits = modello(torch.tensor(Xts, dtype=torch.float32))
        preds_nn = (torch.sigmoid(logits).numpy() > 0.5).astype(int)
    risultati.append({
        "Modello": "Rete Neurale",
        "Accuracy": accuracy_score(yts, preds_nn),
        "Precision": precision_score(yts, preds_nn, zero_division=0),
        "Recall": recall_score(yts, preds_nn, zero_division=0),
        "F1": f1_score(yts, preds_nn, zero_division=0)
    })

    # 2) MODELLI SKLEARN

    for nome, modello_base in modelli_tradizionali.items():
        modello_new = copy.deepcopy(modello_base)
        modello_new.fit(Xtr, ytr)
        preds = modello_new.predict(Xts)
        risultati.append({
            "Modello": nome,
            "Accuracy": accuracy_score(yts, preds),
            "Precision": precision_score(yts, preds, zero_division=0),
            "Recall": recall_score(yts, preds, zero_division=0),
            "F1": f1_score(yts, preds, zero_division=0)
        })
    return pd.DataFrame(risultati)
# ========================================================================
# 12. VALUTAZIONE SU TUTTI E 3 I DATASET

print("\n=== Dataset COMPLETO ===")
ris_completo = valuta_modelli_tutti(X_train_completo, X_test_completo)
print(ris_completo)
print("\n=== Dataset SENZA TOP 3 (feature azzerate) ===")
ris_no3 = valuta_modelli_tutti(X_train_no3, X_test_no3)
print(ris_no3)
print("\n=== Dataset SOLO TOP 5 (altre feature = 0) ===")
ris_top5 = valuta_modelli_tutti(X_train_top5, X_test_top5)
print(ris_top5)

# ========================================================================
# 13. TABELLA FINALE DI CONFRONTO

tabella_finale = pd.concat([
    ris_completo.assign(Dataset="Completo"),
    ris_no3.assign(Dataset="Senza Top 3"),
    ris_top5.assign(Dataset="Solo Top 5")
], ignore_index=True)
print("\n=== TABELLA COMPLETA DI CONFRONTO (NN + 3 MODELLI) ===")
print(tabella_finale)

# ========================================================================
# 14. GRAFICO COMPARATIVO (solo F1)

plt.figure(figsize=(12,6))
unique_models = tabella_finale["Modello"].unique()
x = np.arange(len(unique_models))
width = 0.25
for i, dataset in enumerate(tabella_finale["Dataset"].unique()):
    valori = tabella_finale[tabella_finale["Dataset"] == dataset]["F1"]
    plt.bar(x + i*width, valori, width, label=dataset)
plt.xticks(x + width, unique_models)
plt.ylabel("F1")
plt.title("Confronto F1 dei modelli nelle 3 configurazioni di feature")
plt.legend()
plt.tight_layout()
plt.show()

# ========================================================================
# 15. ANALISI COMPLETA DI UN DIPENDENTE CON SHAP + LIME + NATURAL LANGUAGE

indice = int(input(f"\nInserisci l'indice del dipendente da analizzare (0 - {len(X_test_np)-1}): "))

x = X_test_scalati[indice]
x2d = X_test_scalati[indice:indice+1]
y_true = int(y_test_np[indice])

print("\n============================================================")
print(f" ANALISI COMPLETA ISTANZA {indice}  (Classe reale: {'Licenziato' if y_true else 'Non licenziato'})")
print("============================================================\n")


# ------------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ------------------------------------------------------------------------

def interpretazione(predizione, feature, valore, modello):
    stato = "Licenziato" if predizione == 1 else "Non licenziato"
    direzione = "aumenta" if valore > 0 else "riduce"
    return (
        f"Il modello **{modello}** classifica il dipendente come **{stato}** "
        f"e la variabile più influente è **{feature}**, che {direzione} la probabilità."
    )


def pred_fn_sklearn(model):
    return lambda x: np.column_stack([
        1 - model.predict_proba(x)[:, 1],
        model.predict_proba(x)[:, 1]
    ])


def normalizza_shap_local(ex):
    vals = ex.values

    # (1, n_features, 2)
    if vals.ndim == 3 and vals.shape[-1] == 2:
        vals = vals[:, :, 1][0]

    # (n_features, 2)
    elif vals.ndim == 2 and vals.shape[-1] == 2:
        vals = vals[:, 1]

    # (1, n_features)
    elif vals.ndim == 2 and vals.shape[0] == 1:
        vals = vals[0]

    # (n_features,)
    elif vals.ndim == 1:
        pass

    else:
        raise ValueError(f"Formato SHAP locale non gestito: {vals.shape}")

    # Base values sempre scalare
    base = ex.base_values
    if isinstance(base, np.ndarray):
        base = float(np.mean(base))

    return vals, base


def shap_local_one_model(explainer, x2d, x, name):
    ex = explainer(x2d)
    vals, base = normalizza_shap_local(ex)
    idx = np.argmax(np.abs(vals))
    feat = nomi_features[idx]
    val = vals[idx]

    explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=x,
        feature_names=nomi_features
    )

    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP - {name} (Istanza {indice})")
    plt.tight_layout()
    plt.show()

    return feat, val

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scalati,
    feature_names=nomi_features,
    class_names=['Rimasto', 'Licenziato'],
    mode='classification',
    random_state=42
)

def predici_probabilita_lime(x_numpy):
    modello.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        logits = modello(x_tensor)
        prob1 = torch.sigmoid(logits).numpy()
        prob0 = 1 - prob1
        return np.hstack((prob0, prob1))
    
def frase_shap(feature, valore):
    direzione = "aumenta" if valore > 0 else "riduce"
    return f"Feature più influente(SHAP) **{feature}** è quella con l’impatto maggiore e {direzione} la probabilità di licenziamento."




# ------------------------------------------------------------------------
# REPORT FINALE
# 1) RETE NEURALE
print("\n================= RETE NEURALE =================")

modello.eval()
with torch.no_grad():
    prob = torch.sigmoid(modello(torch.tensor(x2d, dtype=torch.float32))).item()
pred = int(prob > 0.5)

print(f"Predizione: {'Licenziato' if pred else 'Non licenziato'}  (prob={prob:.4f})")

# LIME
esp_nn = explainer_lime.explain_instance(x, predici_probabilita_lime, num_features=10)
feat_lime_nn, peso_lime_nn = esp_nn.as_list()[0]
print(f"Feature più influente (LIME): {feat_lime_nn}")

plt.figure()
esp_nn.as_pyplot_figure()
plt.title(f"LIME - NN (Istanza {indice})")
plt.tight_layout()
plt.show()

# SHAP NN
explainer_nn = shap.Explainer(predici_probabilita_shap, background, feature_names=nomi_features)
feat_shap_nn, val_shap_nn = shap_local_one_model(explainer_nn, x2d, x, "Rete Neurale")

print(frase_shap(feat_shap_nn, val_shap_nn))
print(interpretazione(pred, feat_shap_nn, val_shap_nn, "Rete Neurale"))



# ------------------------------------------------------------------------
# REPORT FINALE
# 2) ALTRI MODELLI (LogReg, RF, XGB)
for nome, model in modelli_tradizionali.items():
    print(f"\n================= {nome.upper()} =================")

    # Predizione
    prob = model.predict_proba(x2d)[0][1]
    pred = int(prob > 0.5)
    print(f"Predizione: {'Licenziato' if pred else 'Non licenziato'}  (prob={prob:.4f})")

    # LIME
    esp = explainer_lime.explain_instance(x, pred_fn_sklearn(model), num_features=10)
    feat_lime, peso_lime = esp.as_list()[0]
    print(f"Feature più influente (LIME): {feat_lime}")

    plt.figure()
    esp.as_pyplot_figure()
    plt.title(f"LIME - {nome} (Istanza {indice})")
    plt.tight_layout()
    plt.show()

    # SHAP
    if nome in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        feat_shap, val_shap = shap_local_one_model(explainer, x2d, x, nome)
        print(frase_shap(feat_shap, val_shap))


    elif nome == "Regressione Logistica":
        coef = model.coef_[0]
        idx = np.argmax(np.abs(coef))
        feat_shap = nomi_features[idx]
        val_shap = coef[idx]
        
        print(frase_shap(feat_shap, val_shap))

        print(f"Feature più influente (LogReg): {feat_shap} (coef={val_shap:.4f})")

        # REPORT INTERPRETABILITÀ LOGISTICA
        odds = np.exp(coef[idx])
        effetto = "aumenta" if coef[idx] > 0 else "riduce"
        print(f"Interpretazione LogReg: la feature '{feat_shap}' {effetto} la probabilità di licenziamento (odds ratio={odds:.3f}).")

    # Frase naturale
    print(interpretazione(pred, feat_shap, val_shap, nome))
