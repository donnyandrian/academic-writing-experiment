# ==============================================================================
# FASE 1: PERSIAPAN LINGKUNGAN KOMPUTASI & AUDIT SISTEM (Subbab 3.4.1)
# ==============================================================================
print("=== VERIFIKASI LINGKUNGAN KOMPUTASI ===")
!nvidia-smi # Memastikan driver GPU aktif (Target: Driver 580.82.07 / CUDA 13.0)
!lsb_release -a # Verifikasi OS Ubuntu 22.04.5 LTS
!python --version # Verifikasi Python 3.12.13

import os
import time
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Impor SMOTE secara aman
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    !pip install -q imbalanced-learn
    from imblearn.over_sampling import SMOTE

# Menetapkan Random Seed untuk Replikasi Konsisten (Tabel 3.2)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("\nPustaka berhasil dimuat dengan aman.")

# ==============================================================================
# FASE 2: SIMULASI AKUISISI DATA & PRA-PEMROSESAN (Subbab 3.4.2)
# ==============================================================================
def generate_mock_datasets():
    """Membuat data tiruan dengan struktur fitur yang identik dengan Bab 3.2.1"""
    # 1. Dataset Skala Kecil: PIMA Indians Diabetes (768 sampel, 8 fitur)
    pima_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_pima = np.random.randn(768, 8)
    y_pima = np.random.choice([0, 1], size=768, p=[0.65, 0.35]) # Mengandung Imbalance awal
    df_pima = pd.DataFrame(X_pima, columns=pima_cols)
    df_pima['Outcome'] = y_pima
    
    # 2. Dataset Skala Besar: Diabetes Risk Dataset (100.000 sampel, 6 fitur)
    risk_cols = ['Age', 'BMI', 'GlucoseLevel', 'PhysicalActivityLevel', 'FamilyHistory', 'Smoker']
    X_risk = np.random.randn(100000, 6)
    y_risk = np.random.choice([0, 1], size=100000, p=[0.70, 0.30])
    df_risk = pd.DataFrame(X_risk, columns=risk_cols)
    df_risk['At Risk Diabetes'] = y_risk
    
    return df_pima, df_risk

print("\n=== MEMPROSES PRA-PEMROSESAN DATA ===")
df_pima, df_risk = generate_mock_datasets()

def preprocess_pipeline(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Analisis Deskriptif Sebelum SMOTE
    print(col_name := f"Distribusi kelas awal untuk {target_column}:", np.bincount(y))
    
    # 1. Normalisasi Fitur menggunakan Z-score (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Penanganan Class Imbalance menggunakan SMOTE
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"Distribusi kelas setelah SMOTE:", np.bincount(y_resampled))
    
    # 3. Data Splitting (80% Train, 20% Test)
    return train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=RANDOM_SEED)

print("\n--- Memproses Dataset PIMA (Skala Kecil) ---")
X_train_p, X_test_p, y_train_p, y_test_p = preprocess_pipeline(df_pima, 'Outcome')

print("\n--- Memproses Dataset Diabetes Risk (Skala Besar) ---")
X_train_r, X_test_r, y_train_r, y_test_r = preprocess_pipeline(df_risk, 'At Risk Diabetes')


# ==============================================================================
# FASE 3: FUNGSI FITNESS & MAPPING HIPERPARAMETER (Tabel 3.1 & Tabel 3.2)
# ==============================================================================
# Batas Rentang Hiperparameter XGBoost [Min, Max] sesuai Tabel 3.1
BOUNDS = {
    'learning_rate': [0.01, 0.3],
    'max_depth': [3, 15],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'n_estimators': [50, 300]
}

def map_vector_to_hyperparameters(vector):
    """Mengubah posisi koordinat kontinu algoritma ke parameter XGBoost asli"""
    params = {
        'learning_rate': float(np.clip(vector[0], BOUNDS['learning_rate'][0], BOUNDS['learning_rate'][1])),
        'max_depth': int(np.clip(round(vector[1]), BOUNDS['max_depth'][0], BOUNDS['max_depth'][1])),
        'subsample': float(np.clip(vector[2], BOUNDS['subsample'][0], BOUNDS['subsample'][1])),
        'colsample_bytree': float(np.clip(vector[3], BOUNDS['colsample_bytree'][0], BOUNDS['colsample_bytree'][1])),
        'n_estimators': int(np.clip(round(vector[4]), BOUNDS['n_estimators'][0], BOUNDS['n_estimators'][1]))
    }
    return params

def fitness_function(vector, X_train, y_train):
    """Fungsi Kriteria Kebalikan F1-Score: 1 - F1_Score (Tabel 3.2)"""
    config = map_vector_to_hyperparameters(vector)
    
    # Inisialisasi model XGBoost dengan akselerasi GPU jika tersedia
    model = xgb.XGBClassifier(
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        subsample=config['subsample'],
        colsample_bytree=config['colsample_bytree'],
        n_estimators=config['n_estimators'],
        random_state=RANDOM_SEED,
        tree_method='hist', 
        device='cuda' # Memaksa pemrosesan pada GPU NVIDIA T4
    )
    
    # Menggunakan porsi validasi internal sederhana untuk menghitung fitness data latih
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    
    f1 = f1_score(y_val, preds, average='binary')
    return 1.0 - f1 # Menghasilkan nilai kesalahan (Makin kecil makin optimal)


# ==============================================================================
# FASE 4: IMPLEMENTASI METODE OPTIMASI (Subbab 3.4.3)
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.1. ALGORITMA GREY WOLF OPTIMIZATION (GWO) - MODEL USULAN
# ------------------------------------------------------------------------------
def grey_wolf_optimizer(X_train, y_train, num_agents=20, max_iter=50):
    dim = 5
    # Inisialisasi populasi acak serigala sesuai batas rentang parameter
    X = np.zeros((num_agents, dim))
    for d, key in enumerate(['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'n_estimators']):
        X[:, d] = np.random.uniform(BOUNDS[key][0], BOUNDS[key][1], num_agents)
        
    # Inisialisasi posisi pemimpin alpha, beta, delta
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    
    delta_pos = np.zeros(dim)
    delta_score = float("inf")
    
    loss_history = []
    
    for t in range(max_iter):
        for i in range(num_agents):
            # Clamping posisi agar tetap dalam rentang pencarian (Boundary Handling)
            for d, key in enumerate(['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'n_estimators']):
                X[i, d] = np.clip(X[i, d], BOUNDS[key][0], BOUNDS[key][1])
                
            # Hitung nilai Fitness
            fitness = fitness_function(X[i], X_train, y_train)
            
            # Perbarui Pemimpin Posisi GWO
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = X[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = X[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = X[i].copy()
                
        # Penurunan linear komponen konvergensi 'a' dari 2 ke 0
        a = 2.0 - (2.0 * t / max_iter)
        
        # Perbarui posisi seluruh agen serigala
        for i in range(num_agents):
            for j in range(dim):
                # Perhitungan interaksi terhadap Alpha
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - X[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # Perhitungan interaksi terhadap Beta
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - X[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # Perhitungan interaksi terhadap Delta
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - X[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # Posisi akhir rata-rata kombinasi tiga pemimpin
                X[i, j] = (X1 + X2 + X3) / 3.0
                
        loss_history.append(alpha_score)
        
    return alpha_pos, loss_history

# ------------------------------------------------------------------------------
# 4.2. ALGORITMA WHALE OPTIMIZATION ALGORITHM (WOA) - MODEL PEMBANDING
# ------------------------------------------------------------------------------
def whale_optimization_algorithm(X_train, y_train, num_agents=20, max_iter=50):
    dim = 5
    b = 1 # Konstanta Spiral
    X = np.zeros((num_agents, dim))
    for d, key in enumerate(['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'n_estimators']):
        X[:, d] = np.random.uniform(BOUNDS[key][0], BOUNDS[key][1], num_agents)
        
    leader_pos = np.zeros(dim)
    leader_score = float("inf")
    loss_history = []
    
    for t in range(max_iter):
        # Evaluasi Fitness
        for i in range(num_agents):
            for d, key in enumerate(['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'n_estimators']):
                X[i, d] = np.clip(X[i, d], BOUNDS[key][0], BOUNDS[key][1])
                
            fitness = fitness_function(X[i], X_train, y_train)
            
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = X[i].copy()
                
        # Penurunan linear parameter internal komponen konvergensi
        a = 2.0 - (2.0 * t / max_iter)
        
        for i in range(num_agents):
            p = random.random()
            r = random.random()
            A = 2 * a * r - a
            C = 2 * random.random()
            l = np.random.uniform(-1, 1)
            
            if p < 0.5:
                if abs(A) < 1:
                    # Mekanisme Encircling Prey
                    D = abs(C * leader_pos - X[i])
                    X[i] = leader_pos - A * D
                else:
                    # Mekanisme Search for Prey (Eksplorasi Global)
                    rand_leader = X[random.randint(0, num_agents - 1)]
                    D = abs(C * rand_leader - X[i])
                    X[i] = rand_leader - A * D
            else:
                # Mekanisme Perbaruan Model Geometris Spiral
                D_prime = abs(leader_pos - X[i])
                X[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
                
        loss_history.append(leader_score)
        
    return leader_pos, loss_history


# ==============================================================================
# FASE 5: STRATEGI EKSEKUSI PENGUJIAN DAN EVALUASI (Subbab 3.4.4 & 3.5)
# ==============================================================================
def run_full_experiment(X_train, X_test, y_train, y_test, dataset_name="PIMA"):
    print(f"\n==========================================")
    print(f" JALAN EKSPERIMEN UNTUK DATASET: {dataset_name} ")
    print(f"==========================================")
    
    # 1. Eksekusi Optimasi Menggunakan GWO (Usulan)
    print("-> Memulai Pencarian Hiperparameter GWO...")
    start_time = time.time()
    gwo_best_vector, gwo_loss = grey_wolf_optimizer(X_train, y_train)
    gwo_duration = time.time() - start_time
    gwo_params = map_vector_to_hyperparameters(gwo_best_vector)
    
    # 2. Eksekusi Optimasi Menggunakan WOA (Re-running Baseline)
    print("-> Memulai Pencarian Hiperparameter WOA...")
    start_time = time.time()
    woa_best_vector, woa_loss = whale_optimization_algorithm(X_train, y_train)
    woa_duration = time.time() - start_time
    woa_params = map_vector_to_hyperparameters(woa_best_vector)
    
    # --- EVALUASI MODEL FINAL GWO-XGBOOST ---
    model_gwo = xgb.XGBClassifier(**gwo_params, random_state=RANDOM_SEED, tree_method='hist', device='cuda')
    model_gwo.fit(X_train, y_train)
    preds_gwo = model_gwo.predict(X_test)
    
    # --- EVALUASI MODEL FINAL WOA-XGBOOST ---
    model_woa = xgb.XGBClassifier(**woa_params, random_state=RANDOM_SEED, tree_method='hist', device='cuda')
    model_woa.fit(X_train, y_train)
    preds_woa = model_woa.predict(X_test)
    
    # --- CETAK LAPORAN HASIL PERBANDINGAN (Subbab 3.5.2 & 3.5.3) ---
    print(f"\n### HASIL EFISIENSI KOMPUTASI & PARAMETER ({dataset_name}) ###")
    print(f"{'Metrik':<25} | {'GWO-XGBoost (Usulan)':<25} | {'WOA-XGBoost (Pembanding)':<25}")
    print("-" * 83)
    print(f"{'Waktu Komputasi (s)':<25} | {gwo_duration:<25.4f} | {woa_duration:<25.4f}")
    print(f"{'Best Fitness (Min Error)':<25} | {gwo_loss[-1]:<25.4f} | {woa_loss[-1]:<25.4f}")
    print(f"{'Learning Rate':<25} | {gwo_params['learning_rate']:<25.4f} | {woa_params['learning_rate']:<25.4f}")
    print(f"{'Max Depth':<25} | {gwo_params['max_depth']:<25} | {woa_params['max_depth']:<25}")
    print(f"{'N Estimators':<25} | {gwo_params['n_estimators']:<25} | {woa_params['n_estimators']:<25}")
    
    print(f"\n### METRIK KLASIFIKASI CONFUSION MATRIX ({dataset_name}) ###")
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_test, preds_gwo).ravel()
    tn_w, fp_w, fn_w, tp_w = confusion_matrix(y_test, preds_woa).ravel()
    
    # Penghitungan manual formula representatif sesuai deskripsi instrumen Bab 3
    acc_g = (tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g)
    acc_w = (tp_w + tn_w) / (tp_w + tn_w + fp_w + fn_w)
    
    prec_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0
    prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
    
    rec_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    rec_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
    
    f1_g = f1_score(y_test, preds_gwo, average='binary')
    f1_w = f1_score(y_test, preds_woa, average='binary')
    
    print(f"{'Indikator Performa':<25} | {'GWO-XGBoost':<25} | {'WOA-XGBoost':<25}")
    print("-" * 83)
    print(f"{'Akurasi (Accuracy)':<25} | {acc_g:<25.4%} | {acc_w:<25.4%}")
    print(f"{'Presisi (Precision)':<25} | {prec_g:<25.4%} | {prec_w:<25.4%}")
    print(f"{'Sensitivitas (Recall)*':<25} | {rec_g:<25.4%} | {rec_w:<25.4%}")
    print(f"{'F1-Score':<25} | {f1_g:<25.4%} | {f1_w:<25.4%}")
    print("\n* Catatan: Recall adalah metrik paling krusial untuk diagnosis medis (Bab 3.2.2)")

# Menjalankan Pengujian secara berturut-turut untuk kedua skala data
run_full_experiment(X_train_p, X_test_p, y_train_p, y_test_p, dataset_name="PIMA Indians Diabetes (Skala Kecil)")
run_full_experiment(X_train_r, X_test_r, y_train_r, y_test_r, dataset_name="Diabetes Risk Dataset (Skala Besar)")