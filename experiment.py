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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import cupy as cp

# Impor SMOTE secara aman
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    !pip install -q imbalanced-learn
    from imblearn.over_sampling import SMOTE

print(f"numpy\t\t\t: {np.__version__}")
print(f"pandas\t\t\t: {pd.__version__}")
print(f"xgboost\t\t\t: {xgb.__version__}")
print(f"cupy\t\t\t: {cp.__version__}")
!pip list | awk '/^(scikit-learn)/ {print $1 "\t\t: " $2}'
!pip list | awk '/^(imbalanced-learn)/ {print $1 "\t: " $2}'

# Menetapkan Random Seed untuk Replikasi Konsisten (Tabel 3.2)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("\nPustaka dan visualisasi grafik berhasil dimuat.")

# ==============================================================================
# FASE 2: AKUISISI DATA & PROSEDUR PEMBERSIHAN DATA (Subbab 3.4.2)
# ==============================================================================
print("\n=== MEMPROSES DATA & PEMBERSIHAN DATA ===")

# Membaca Dataset dari Direktori Kerja
try:
    df_pima_raw = pd.read_csv('pima_diabetes.csv')
    df_risk_raw = pd.read_csv('diabetes_risk_dataset.csv')
    print("Berhasil memuat berkas pima_diabetes.csv dan diabetes_risk_dataset.csv.")
except FileNotFoundError as e:
    print(f"Error: Pastikan file CSV sudah diunggah di root folder Colab. Detail: {e}")

# --- PEMBERSIHAN DATASET PIMA (SKALA KECIL) ---
print("\n--- Membersihkan Dataset PIMA (Skala Kecil) ---")
# Menangani invalid zeros pada indikator klinis (diubah ke NaN lalu diimputasi median)
clinical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("Jumlah nilai 0 tidak valid sebelum dibersihkan:\n", (df_pima_raw[clinical_cols] == 0).sum())

df_pima_clean = df_pima_raw.copy()
df_pima_clean[clinical_cols] = df_pima_clean[clinical_cols].replace(0, np.nan)
# Imputasi menggunakan nilai median masing-masing kolom klinis
df_pima_clean[clinical_cols] = df_pima_clean[clinical_cols].fillna(df_pima_clean[clinical_cols].median())
print("Jumlah nilai 0 tidak valid setelah penanganan missing values:", (df_pima_clean[clinical_cols] == 0).sum().sum())

# --- PEMBERSIHAN DATASET DIABETES RISK (SKALA BESAR) ---
print("\n--- Membersihkan Dataset Diabetes Risk (Skala Besar) ---")
df_risk_clean = df_risk_raw.copy()

# Hapus baris yang kolom targetnya ('at_risk_diabetes') bernilai NaN/Kosong
print("Jumlah nilai NaN/Kosong sebelum dibersihkan:", df_risk_clean['at_risk_diabetes'].isna().sum())
df_risk_clean = df_risk_clean.dropna(subset=['at_risk_diabetes'])
print("Jumlah nilai NaN/Kosong setelah dibersihkan:", df_risk_clean['at_risk_diabetes'].isna().sum())
# Ubah tipe data target secara eksplisit menjadi integer
df_risk_clean['at_risk_diabetes'] = df_risk_clean['at_risk_diabetes'].astype(int)
# Melakukan encoding pada fitur text/object 'physical_activity_level' (low, moderate, high)
activity_mapping = {'low': 0, 'moderate': 1, 'high': 2}
df_risk_clean['physical_activity_level'] = df_risk_clean['physical_activity_level'].map(activity_mapping)
# Tangani fitur-fitur integer (Hanya memproses sisa baris yang sudah valid)
int_features = ['age', 'physical_activity_level', 'family_history', 'smoker']
for col in int_features:
    df_risk_clean[col] = df_risk_clean[col].fillna(df_risk_clean[col].median()).astype(int)
# Tangani fitur murni desimal/float (bmi & glucose_level)
float_features = ['bmi', 'glucose_level']
df_risk_clean[float_features] = df_risk_clean[float_features].fillna(df_risk_clean[float_features].median())

print("Verifikasi tipe data Diabetes Risk setelah encoding:\n", df_risk_clean.dtypes)


# --- PIPELINE PRA-PEMROSESAN LANJUTAN (Z-SCORE & SMOTE) ---
def preprocess_pipeline(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Analisis Deskriptif Sebelum SMOTE
    print(f"Distribusi kelas awal untuk target '{target_column}':", np.bincount(y))
    
    # Normalisasi Fitur menggunakan Z-score (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Penanganan Class Imbalance menggunakan SMOTE
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"Distribusi kelas setelah penerapan SMOTE:", np.bincount(y_resampled))
    
    # Data Splitting (80% Train, 20% Test)
    return train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=RANDOM_SEED)

print("\n--- Eksekusi Pipeline Pra-pemrosesan Akhir ---")
X_train_p, X_test_p, y_train_p, y_test_p = preprocess_pipeline(df_pima_clean, 'Outcome')
X_train_r, X_test_r, y_train_r, y_test_r = preprocess_pipeline(df_risk_clean, 'at_risk_diabetes')


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
# Daftar Hiperparameter XGBoost
PARAMETERS = [*BOUNDS]

def map_vector_to_hyperparameters(vector):
    """Mengubah posisi koordinat kontinu algoritma ke parameter XGBoost asli"""
    return {
        'learning_rate': float(np.clip(vector[0], BOUNDS['learning_rate'][0], BOUNDS['learning_rate'][1])),
        'max_depth': int(np.clip(round(vector[1]), BOUNDS['max_depth'][0], BOUNDS['max_depth'][1])),
        'subsample': float(np.clip(vector[2], BOUNDS['subsample'][0], BOUNDS['subsample'][1])),
        'colsample_bytree': float(np.clip(vector[3], BOUNDS['colsample_bytree'][0], BOUNDS['colsample_bytree'][1])),
        'n_estimators': int(np.clip(round(vector[4]), BOUNDS['n_estimators'][0], BOUNDS['n_estimators'][1]))
    }

def fitness_function(vector, X_tr_gpu, y_tr_gpu, X_val_gpu, y_val_cpu):
    """Fungsi Kriteria Kebalikan F1-Score: 1 - F1_Score (Tabel 3.2)"""
    config = map_vector_to_hyperparameters(vector)
    
    # Inisialisasi model XGBoost dengan akselerasi GPU
    model = xgb.XGBClassifier(
        **config,
        random_state=RANDOM_SEED,
        tree_method='hist', 
        device='cuda' # Memaksa pemrosesan pada GPU NVIDIA T4
    )
    
    # Latih model dan prediksi
    model.fit(X_tr_gpu, y_tr_gpu)
    preds_gpu = model.predict(X_val_gpu)
    preds_cpu = cp.asnumpy(preds_gpu)
    return 1.0 - f1_score(y_val_cpu, preds_cpu, average='binary') # Menghasilkan nilai kesalahan (Makin kecil makin optimal)


# ==============================================================================
# FASE 4: IMPLEMENTASI ALGORITMA METODE OPTIMASI (GWO & WOA) (Subbab 3.4.3)
# ==============================================================================

def initialize_agents(num_agents, dim):
    X = np.zeros((num_agents, dim))
    for d, key in enumerate(PARAMETERS):
        X[:, d] = np.random.uniform(BOUNDS[key][0], BOUNDS[key][1], num_agents)
    return X

# ------------------------------------------------------------------------------
# 4.1. ALGORITMA GREY WOLF OPTIMIZATION (GWO) - MODEL USULAN
# ------------------------------------------------------------------------------
def grey_wolf_optimizer(X_tr_gpu, y_tr_gpu, X_val_gpu, y_val_cpu, num_agents=20, max_iter=50):
    d = 5
    # Inisialisasi populasi acak serigala sesuai batas rentang parameter (No. 1)
    X = initialize_agents(num_agents, d)
        
    # Inisialisasi posisi pemimpin alpha, beta, delta (No. 3 & No. #4 - Pembuatan variabel, nilai belum bisa digunakan)
    alpha_pos, alpha_score = np.zeros(d), float("inf")
    beta_pos, beta_score = np.zeros(d), float("inf")
    delta_pos, delta_score = np.zeros(d), float("inf")
    loss_history = []
    
    for t in range(max_iter):
        # Penurunan linear komponen konvergensi 'a' dari 2 ke 0
        # Untuk iterasi pertama, ini bertindak sebagai pemberian nilai (No. 2)
        # Selanjutnya, ini bertindak sebagai perbaruan nilai
        a = 2.0 - (2.0 * t / max_iter)
        
        # Persiapan data untuk perhitungan agen serigala berikutnya
        # Untuk iterasi pertama, ini bertindak sebagai pemberian nilai (No. 3 & No. 4)
        # Selanjutnya, ini bertindak sebagai perbaruan nilai
        for i in range(num_agents):
            # Hitung nilai Fitness
            fitness = fitness_function(X[i], X_tr_gpu, y_tr_gpu, X_val_gpu, y_val_cpu)
            
            # Perbarui Pemimpin Posisi GWO
            if fitness < alpha_score:
                alpha_score = fitness; alpha_pos = X[i].copy()
            elif fitness < beta_score:
                beta_score = fitness; beta_pos = X[i].copy()
            elif fitness < delta_score:
                delta_score = fitness; delta_pos = X[i].copy()
                
        leaders_pos = [alpha_pos, beta_pos, delta_pos]
        # Perbarui posisi seluruh agen serigala
        for i in range(num_agents):
            XK = []
            # Perhitungan interaksi terhadap Alpha, Beta, dan Delta
            for k, P in leaders_pos:
                r1 = np.random.rand(d), r2 = np.random.rand(d)
                A = 2 * a * r1 - a
                C = 2 * r2
                XK.append(P - A * np.abs(C * P - X[i]))
            
            # Posisi akhir rata-rata kombinasi tiga pemimpin untuk iterasi selanjutnya (t + 1)
            new_position = np.mean(XK, axis=0)
            
            # Clamping posisi agar tetap dalam rentang pencarian (Boundary Handling)
            for j, key in enumerate(PARAMETERS):
                X[i, j] = np.clip(new_position[j], BOUNDS[key][0], BOUNDS[key][1])
        loss_history.append(alpha_score)
    return alpha_pos, loss_history

# ------------------------------------------------------------------------------
# 4.2. ALGORITMA WHALE OPTIMIZATION ALGORITHM (WOA) - MODEL PEMBANDING
# ------------------------------------------------------------------------------
def whale_optimization_algorithm(X_tr_gpu, y_tr_gpu, X_val_gpu, y_val_cpu, num_agents=20, max_iter=50):
    d = 5; b = 1
    # Inisialisasi populasi acak paus sesuai batas rentang parameter (No. 2)
    X = initialize_agents(num_agents, d)
    
    # Inisialisasi posisi paus pemimpin (X*) (No. 3)
    leader_pos, leader_score = np.zeros(d), float("inf")
    loss_history = []
    
    # Iterasi t (No. 4)
    for t in range(max_iter):
        # Penurunan linear komponen konvergensi 'a' dari 2 ke 0
        a = 2.0 - (2.0 * t / max_iter)
        
        # Evaluasi Fitness dan Penetapan Posisi Pemimpin 
        for i in range(num_agents):
            fitness = fitness_function(X[i], X_tr_gpu, y_tr_gpu, X_val_gpu, y_val_cpu)
            if fitness < leader_score:
                leader_score = fitness; leader_pos = X[i].copy()
                
        for i in range(num_agents):
            p = random.random(); l = np.random.uniform(-1, 1)
            r1, r2 = np.random.rand(d), np.random.rand(d)
            A = 2 * a * r1 - a; C = 2 * r2
            
            if p < 0.5:
                # Kandidat Aksi 1: Encircling Prey
                D_leader = np.abs(C * leader_pos - X[i])
                X_encircling = leader_pos - A * D_leader
                
                # Kandidat Aksi 2: Search for Prey
                rand_leader = X[random.randint(0, num_agents - 1)]
                D_rand = np.abs(C * rand_leader - X[i])
                X_search = rand_leader - A * D_rand
                
                # np.where untuk saklar keputusan element-wise otomatis
                X[i] = np.where(np.abs(A) < 1, X_encircling, X_search)
            else:
                # Mekanisme Perbaruan Model Geometris Spiral
                D_prime = np.abs(leader_pos - X[i])
                X[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
            
            # Clamping posisi agar tetap dalam rentang pencarian (Boundary Handling)
            for j, key in enumerate(PARAMETERS):
                X[i, j] = np.clip(X[i, j], BOUNDS[key][0], BOUNDS[key][1])
        loss_history.append(leader_score)
    return leader_pos, loss_history


# ==============================================================================
# FASE 5: STRATEGI EKSEKUSI, EVALUASI & PLOT GRAFIK KONVERGENSI (Subbab 3.4.4 & 3.5)
# ==============================================================================
def run_full_experiment(X_train, X_test, y_train, y_test, dataset_name="PIMA"):
    header = f" JALAN EKSPERIMEN UNTUK DATASET: {dataset_name} "
    print(f"\n{'=' * len(header)}")
    print(header)
    print("=" * len(header))
    
    # Validasi internal split 75:25 dari data latih global
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_SEED)
    
    # Pindahkan data internal split ke GPU menggunakan CuPy
    print("-> Memigrasikan data latih ke VRAM GPU...")
    X_tr_gpu = cp.array(X_tr)
    y_tr_gpu = cp.array(y_tr)
    X_val_gpu = cp.array(X_val)
    # y_val tetap di CPU karena dibutuhkan scikit-learn di fungsi fitness
    
    # Eksekusi Optimasi Menggunakan GWO (Usulan)
    print("-> Memulai Pencarian Hiperparameter GWO...")
    start = time.perf_counter_ns(); gwo_best_vector, gwo_loss = grey_wolf_optimizer(X_tr_gpu, y_tr_gpu, X_val_gpu, y_val); gwo_time = time.perf_counter_ns() - start
    gwo_params = map_vector_to_hyperparameters(gwo_best_vector)
    
    # Eksekusi Optimasi Menggunakan WOA (Re-running Baseline)
    print("-> Memulai Pencarian Hiperparameter WOA...")
    start = time.perf_counter_ns(); woa_best_vector, woa_loss = whale_optimization_algorithm(X_tr_gpu, y_tr_gpu, X_val_gpu, y_val); woa_time = time.perf_counter_ns() - start
    woa_params = map_vector_to_hyperparameters(woa_best_vector)
    
    # Evaluasi Model Akhir pada Test Set (Pindahkan data test global ke GPU)
    X_train_gpu = cp.array(X_train)
    y_train_gpu = cp.array(y_train)
    X_test_gpu = cp.array(X_test)
    
    # Latih Model Akhir Berdasarkan Parameter Terbaik
    model_gwo = xgb.XGBClassifier(**gwo_params, random_state=RANDOM_SEED, tree_method='hist', device='cuda').fit(X_train_gpu, y_train_gpu)
    preds_gwo = cp.asnumpy(model_gwo.predict(X_test_gpu))
    model_woa = xgb.XGBClassifier(**woa_params, random_state=RANDOM_SEED, tree_method='hist', device='cuda').fit(X_train_gpu, y_train_gpu)
    preds_woa = cp.asnumpy(model_woa.predict(X_test_gpu))
    
    # Hitung Metrik Kebingungan (Confusion Matrix)
    cm_gwo = confusion_matrix(y_test, preds_gwo)
    cm_woa = confusion_matrix(y_test, preds_woa)
    
    tn_g, fp_g, fn_g, tp_g = cm_gwo.ravel()
    tn_w, fp_w, fn_w, tp_w = cm_woa.ravel()
    
    # Penghitungan manual formula representatif sesuai deskripsi instrumen Bab 3
    acc_g = (tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g)
    acc_w = (tp_w + tn_w) / (tp_w + tn_w + fp_w + fn_w)
    
    prec_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0
    prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
    
    rec_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    rec_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
    
    f1_g = f1_score(y_test, preds_gwo, average='binary')
    f1_w = f1_score(y_test, preds_woa, average='binary')
    
    # Cetak Hasil Akhir Evaluasi Eksperimen
    print(f"\n### METRIK PERFORMA AKHIR ({dataset_name}) ###")
    cols = ["GWO-XGBoost (Usulan)", "WOA-XGBoost (Pembanding)"]
    df = pd.DataFrame({
        "Metrik Indikator": ["Waktu Eksekusi", "Akurasi Akhir", "Presisi (Precision)", "Sensitivitas (Recall)", "F1-Score"],
        cols[0]: [gwo_time, acc_g, prec_g, rec_g, f1_g],
        cols[1]: [woa_time, acc_w, prec_w, rec_w, f1_w]
    })
    df[cols] = df[cols].apply(lambda x: x.index.map(lambda i: f"{x[i] / 1e9:.3f}s" if i == 0 else f"{x[i]:.4%}"), axis=0)
    print(df.to_markdown(tablefmt="github", index=False))

    # Visualisasi Confusion Matrix Berdampingan
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Plot GWO-XGBoost
    disp_gwo = ConfusionMatrixDisplay(confusion_matrix=cm_gwo)
    disp_gwo.plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title(f'Confusion Matrix: GWO-XGBoost\n({dataset_name})', fontsize=11, fontweight='bold')
    ax[0].set_xlabel('Kelas Prediksi')
    ax[0].set_ylabel('Kelas Sebenarnya')

    # Plot WOA-XGBoost
    disp_woa = ConfusionMatrixDisplay(confusion_matrix=cm_woa)
    disp_woa.plot(ax=ax[1], cmap='Oranges', values_format='d')
    ax[1].set_title(f'Confusion Matrix: WOA-XGBoost\n({dataset_name})', fontsize=11, fontweight='bold')
    ax[1].set_xlabel('Kelas Prediksi')
    ax[1].set_ylabel('Kelas Sebenarnya')
    
    plt.tight_layout()
    plt.show()

    # Visualisasi Grafik Loss Curve
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(range(1, len(gwo_loss) + 1), gwo_loss, label='GWO-XGBoost (Usulan)', color='#1f77b4', linewidth=2.5, marker='o', markevery=5)
    plt.plot(range(1, len(woa_loss) + 1), woa_loss, label='WOA-XGBoost (Pembanding)', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s', markevery=5)
    plt.title(f'Grafik Loss Curve & Stabilitas Konvergensi\n({dataset_name})', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Jumlah Iterasi Komputasi', fontsize=10)
    plt.ylabel('Nilai Loss (1 - F1-Score)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.show()

# Menjalankan pengujian secara berturut-turut untuk kedua skala data
run_full_experiment(X_train_p, X_test_p, y_train_p, y_test_p, dataset_name="PIMA Indians Diabetes Dataset")
run_full_experiment(X_train_r, X_test_r, y_train_r, y_test_r, dataset_name="Diabetes Risk Dataset")