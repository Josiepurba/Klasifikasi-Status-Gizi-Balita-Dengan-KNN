from flask import Flask, render_template, request
from fungsi import min_max_scaling, knn_predict
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

app = Flask(__name__)
app.debug = True
current_page = "home"

BASE_URL = "http://127.0.0.1:5000/"

# Baca dataset dari file CSV
data = pd.read_csv('dataset.csv')

# Memisahkan fitur dan label
X = data[['Usia (bulan)', 'Berat Badan (kg)', 'Tinggi Badan (cm)', 'Lingkar Lengan', 'Jenis Kelamin']]
y = data['Status Gizi']
X['Jenis Kelamin'] = X['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})
X_normalized = X.drop(columns=['Jenis Kelamin']).apply(min_max_scaling)

# Memisahkan data uji dari data latih secara manual
test_size = int(0.2 * len(X))
X_train = X_normalized.iloc[:-test_size]
y_train = y.iloc[:-test_size]
X_test = X_normalized.iloc[-test_size:]
y_test = y.iloc[-test_size:]

k_folds = None
k = None

@app.route('/')
def index():
    dr = {'BASE_URL': BASE_URL}
    return render_template('home.html', dRes=dr)


@app.route('/klasifikasi', methods=['GET', 'POST'])
def predict():
    global current_page
    current_page = "klasifikasi"

    if request.method == 'POST':
        # Ambil data uji dari pengguna
        nama_bayi = request.form.get('nama_bayi')
        usia = float(request.form.get('usia'))
        berat_badan = float(request.form.get('berat_badan'))
        tinggi_badan = float(request.form.get('tinggi_badan'))
        lingkar_lengan = float(request.form.get('lingkar_lengan'))

        # Validasi input sesuai dengan batas
        if usia < 1 or usia > 60:
            return render_template('error.html', message='Usia harus antara 1 dan 60 bulan')

        if berat_badan < 1 or berat_badan > 30:
            return render_template('error.html', message='Berat badan harus antara 1 dan 30 kg')

        if tinggi_badan < 20 or tinggi_badan > 100:
            return render_template('error.html', message='Tinggi badan harus antara 20 dan 100 cm')

        if lingkar_lengan < 1 or lingkar_lengan > 25:
            return render_template('error.html', message='Lingkar lengan harus antara 1 dan 25 cm')

        # Normalisasi data uji
        X_normalized_test = min_max_scaling(pd.Series([usia, berat_badan, tinggi_badan, lingkar_lengan]))

        k = 5  # Ganti jumlah tetangga (k) sesuai kebutuhan
        y_pred = []
        for i in range(len(X_normalized_test)):
            x_test = X_normalized_test[i]
            predicted_labels = [knn_predict(X_train, y_train, x_test, k) for _ in range(k)]  # Gunakan data latih untuk prediksi
            predicted_label = mode(predicted_labels)
            y_pred.append(predicted_label)

        # Hitung mode dari hasil prediksi
        predicted_gizi_mode = mode(y_pred)
        

        # Tampilkan hasil prediksi status gizi, akurasi, dan nama bayi
        return render_template('result.html', predicted_gizi=predicted_gizi_mode, nama_bayi=nama_bayi)

    return render_template('klasifikasi.html')  # Tambahkan ini untuk menampilkan halaman 'Predict'

@app.route('/evaluasi', methods=['GET', 'POST'])
def k_fold_cross_validation():
    global k  # Mendeklarasikan variabel global k
    global k_folds  # Mendeklarasikan variabel global k_folds

    if request.method == 'POST':
        # Mengambil nilai k dan k_folds dari formulir POST
        k = int(request.form.get('k'))
        k_folds = int(request.form.get('k_folds'))

        accuracies = []
        precisions = []  
     
        recalls = []
        f1_scores = []

        all_fold_accuracies = []
        all_fold_precisions = []  

        all_actual_statuses = []  # untuk menyimpan status aktual
        all_predicted_statuses = []  # untuk menyimpan status yang diprediksi

        # Melakukan K-Fold Cross Validation
        for fold in range(k_folds):
            test_start = fold * len(X_normalized) // k_folds
            test_end = (fold + 1) * len(X_normalized) // k_folds
            X_test_fold = X_normalized.iloc[test_start:test_end]
            y_test_fold = y.iloc[test_start:test_end]
            X_train_fold = pd.concat([X_normalized.iloc[:test_start], X_normalized.iloc[test_end:]])
            y_train_fold = pd.concat([y.iloc[:test_start], y.iloc[test_end:]])

            y_pred = []
            for i in range(len(X_test_fold)):
                x_test = X_test_fold.iloc[i]
                predicted_label = knn_predict(X_train_fold, y_train_fold, x_test, k)
                y_pred.append(predicted_label)

             
        
            # Menghitung akurasi
            accuracy = accuracy_score(y_test_fold, y_pred)
            accuracies.append(accuracy)

            # Menghitung presisi
            precision = precision_score(y_test_fold, y_pred, average='weighted')
            precisions.append(precision)

            # Menghitung recall
            recall = recall_score(y_test_fold, y_pred, average='weighted')
            recalls.append(recall)

            # Menghitung nilai F1
            f1 = f1_score(y_test_fold, y_pred, average='weighted')
            f1_scores.append(f1)

            all_fold_accuracies.append(accuracy)
            all_fold_precisions.append(precision)  

            # Menyimpan status aktual dan yang diprediksi
            all_actual_statuses.append(y_test_fold.tolist())
            all_predicted_statuses.append(y_pred)
            
            
        # Menghitung matriks kebingungan (confusion matrix) setelah seluruh loop selesai
        total_confusion_matrix = confusion_matrix(np.array(all_actual_statuses).flatten(), np.array(all_predicted_statuses).flatten())

        # Menambahkan total_confusion_matrix_all_folds ke dalam konteks template
        total_confusion_matrix_all_folds = {
                    'data': total_confusion_matrix.tolist()
            }

        # Menghitung nilai rata-rata akurasi, presisi, dan matriks kebingungan
        average_accuracy = np.mean(all_fold_accuracies)
        average_precision = np.mean(all_fold_precisions)
        

        # Membuat grafik menggunakan Matplotlib
        fold_names = [f'Fold {i + 1}' for i in range(k_folds)]
        plt.figure(figsize=(10, 6))
        plt.plot(fold_names, accuracies, color='blue', alpha=0.7, label='Akurasi')
        plt.xlabel('Lipatan K-Fold')
        plt.ylim(0.94, 1.0)
        plt.ylabel('Nilai')
        plt.title('Grafik Akurasi K-Fold Cross-Validation')
        plt.legend()

        # Menggantinya dengan menyimpan gambar sebagai file di folder "static"
        image_filename = "kfold_result.png"
        image_path = os.path.join("static", image_filename)
        plt.savefig(image_path, format='png')
        plt.close()
        
        # Menyimpan nama file gambar dalam konteks template
        image_filename = "kfold_result.png"

       


        # Mengembalikan hasil evaluasi performa model dalam bentuk HTML menggunakan template 'kfold_result.html'
        # dengan menyertakan berbagai metrik seperti akurasi, presisi, matriks kebingungan, recall, F1-score, dan lainnya.
        return render_template('kfold_result.html', accuracies=accuracies, average_accuracy=average_accuracy,
                       precisions=precisions, average_precision=average_precision,
                       confusion_matrix=total_confusion_matrix.tolist(), recall=recalls[-1],
                       f1_score=f1_scores[-1], k=k, k_folds=k_folds, image_filename=image_filename,
                       actual_statuses=all_actual_statuses, predicted_statuses=all_predicted_statuses,
                       total_confusion_matrix_all_folds=total_confusion_matrix_all_folds)


    # Menampilkan formulir pengaturan K-Fold jika metode request bukan POST
    return render_template('kfold_settings.html', k=k, k_folds=k_folds)



if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
