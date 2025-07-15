# app.py
from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io # Untuk menyimpan plot ke memori
import base64 # Untuk mengkodekan plot ke base64 agar bisa ditampilkan di HTML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib # Untuk menyimpan dan memuat model

app = Flask(__name__)

# --- Inisialisasi Data dan Model (Variabel Global) ---
# Deklarasikan di sini agar jelas bahwa ini adalah variabel global
df = None
model = None
features = []

def load_and_prepare_data():
    """Memuat, membersihkan, dan menyiapkan dataset anggur."""
    global df, model, features # Pastikan df, model, dan features dideklarasikan global
    print("Memuat dan menyiapkan data...") # Tambahkan print untuk debugging

    try:
        # Delimiter for winequality-red.csv is usually ';'
        df = pd.read_csv('winequality-red.csv', delimiter=';')
    except FileNotFoundError:
        print("Dataset winequality-red.csv tidak ditemukan. Membuat data dummy.")
        # Data dummy jika file tidak ditemukan
        data = {
            'fixed acidity': np.random.uniform(4, 15, 1600), 'volatile acidity': np.random.uniform(0.1, 1.5, 1600),
            'citric acid': np.random.uniform(0, 1, 1600), 'residual sugar': np.random.uniform(0.5, 15, 1600),
            'chlorides': np.random.uniform(0.01, 0.6, 1600), 'free sulfur dioxide': np.random.uniform(1, 70, 1600),
            'total sulfur dioxide': np.random.uniform(6, 300, 1600), 'density': np.random.uniform(0.99, 1.01, 1600),
            'pH': np.random.uniform(2.7, 4, 1600), 'sulphates': np.random.uniform(0.3, 2, 1600),
            'alcohol': np.random.uniform(8, 15, 1600), 'quality': np.random.randint(3, 9, 1600)
        }
        df = pd.DataFrame(data)

    df.dropna(inplace=True)
    df.columns = df.columns.str.strip() # Bersihkan spasi di nama kolom

    # Latih model dan simpan
    train_model()

def train_model():
    """Melatih model regresi dan menyimpannya."""
    global df, model, features # Pastikan df, model, dan features dideklarasikan global
    print("Melatih model...") # Tambahkan print untuk debugging
    features = [col for col in df.columns if col != 'quality']
    target = 'quality'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Simpan model agar bisa digunakan nanti tanpa melatih ulang
    joblib.dump(model, 'wine_quality_model.pkl')
    print("Model berhasil dilatih dan disimpan sebagai 'wine_quality_model.pkl'")

# --- Fungsi Pembantu untuk Plot ---
def get_plot_as_base64(plot_func, *args, **kwargs):
    """Menjalankan fungsi plot Matplotlib dan mengembalikan gambar sebagai string base64."""
    plt.figure(figsize=(kwargs.pop('figsize', (8, 5)))) # Ambil figsize dari kwargs atau default
    plot_func(*args, **kwargs)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close() # Tutup figure untuk menghemat memori
    return base64.b64encode(img.getvalue()).decode('utf8')

# --- Rute Flask ---

@app.route('/')
def index():
    """Halaman utama (landing page)."""
    global df # Akses df global
    # Perbaikan: Muat data dan model jika belum dimuat
    if df is None:
        load_and_prepare_data()

    # Hanya data minimal untuk halaman index, atau tautan ke bagian lain
    # Untuk contoh ini, index akan menjadi overview singkat dan link ke halaman lain
    return render_template('index.html')


@app.route('/statistics')
def show_statistics():
    """Halaman statistik ringkasan."""
    global df
    if df is None:
        load_and_prepare_data()

    # Statistik Umum
    avg_quality = df['quality'].mean()
    quality_dist = df['quality'].value_counts().sort_index().to_dict()
    avg_props = df[['alcohol', 'pH', 'fixed acidity', 'volatile acidity', 'sulphates']].mean().to_dict()

    return render_template('statistics.html',
                           avg_quality=f"{avg_quality:.2f}",
                           quality_dist=quality_dist,
                           avg_props=avg_props)


@app.route('/visualizations')
def show_visualizations():
    """Halaman visualisasi data."""
    global df
    if df is None:
        load_and_prepare_data()

    # Plot Distribusi Kualitas Anggur
    quality_dist_plot = get_plot_as_base64(sns.countplot, x='quality', data=df, palette='viridis', figsize=(8,5))

    # Plot Matriks Korelasi
    correlation_matrix = df.corr()
    corr_plot = get_plot_as_base64(sns.heatmap, data=correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, figsize=(12,10))

    # Plot Distribusi Sifat Fisikokimia Utama (Contoh beberapa saja)
    alcohol_dist_plot = get_plot_as_base64(sns.histplot, x='alcohol', data=df, bins=30, kde=True, color='lightcoral', figsize=(8,5))
    volatile_acidity_dist_plot = get_plot_as_base64(sns.histplot, x='volatile acidity', data=df, bins=30, kde=True, color='lightcoral', figsize=(8,5))

    # Plot Box Plot Sifat Fisikokimia Berdasarkan Kualitas (Contoh beberapa saja)
    alcohol_boxplot = get_plot_as_base64(sns.boxplot, x='quality', y='alcohol', data=df, palette='coolwarm', figsize=(10,6))
    volatile_acidity_boxplot = get_plot_as_base64(sns.boxplot, x='quality', y='volatile acidity', data=df, palette='coolwarm', figsize=(10,6))

    return render_template('visualizations.html',
                           quality_dist_plot=quality_dist_plot,
                           corr_plot=corr_plot,
                           alcohol_dist_plot=alcohol_dist_plot,
                           volatile_acidity_dist_plot=volatile_acidity_dist_plot,
                           alcohol_boxplot=alcohol_boxplot,
                           volatile_acidity_boxplot=volatile_acidity_boxplot)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model # Akses variabel global 'model'
    global features # Akses variabel global 'features'
    global df # Akses variabel global 'df'

    prediction_result = None

    # Perbaikan: Muat data dan model jika belum dimuat
    if df is None: # Cek df untuk memastikan data sudah dimuat
        load_and_prepare_data()

    if request.method == 'POST':
        try:
            # Ambil input dari form
            input_data = {}
            for feature in features:
                input_data[feature] = float(request.form[feature])

            # Konversi input ke DataFrame yang sesuai untuk model
            new_data_df = pd.DataFrame([input_data])

            # Pastikan model sudah dimuat dari file jika belum ada di memori
            if model is None:
                try:
                    model = joblib.load('wine_quality_model.pkl')
                    print("Model berhasil dimuat dari file (di dalam /predict).")
                except FileNotFoundError:
                    print("Model file 'wine_quality_model.pkl' tidak ditemukan. Pastikan sudah dilatih.")
                    raise Exception("Model tidak tersedia untuk prediksi.")

            # Lakukan prediksi
            predicted_quality = model.predict(new_data_df)[0]
            prediction_result = f"Kualitas Anggur yang Diprediksi: {predicted_quality:.2f}"
        except Exception as e:
            prediction_result = f"Error: Masukkan tidak valid atau terjadi masalah. {e}"

    # Buat daftar fitur dengan nilai rata-rata sebagai placeholder untuk form GET request
    # Pastikan df ada sebelum mencoba mengambil rata-rata untuk placeholder
    feature_placeholders = {f: f"{df[f].mean():.2f}" for f in features} if df is not None else {f: "0.0" for f in features}


    return render_template('predict.html', prediction_result=prediction_result, features=features, feature_placeholders=feature_placeholders)

# Untuk menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
