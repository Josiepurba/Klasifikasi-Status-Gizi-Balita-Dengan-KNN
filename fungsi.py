import numpy as np

# Fungsi normalisasi min-max
def min_max_scaling(data):
    min_val = data.min()
    max_val = data.max()
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# Fungsi prediksi dengan KNN
def knn_predict(X_train, y_train, x_test, k):
    # Menghitung jarak Euclidean antara x_test dan semua sampel latih
    distances = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
    
    # Menemukan indeks k tetangga terdekat
    nearest_indices = np.argsort(distances)[:k]
    
    # Mengambil label dari tetangga terdekat
    nearest_labels = y_train.iloc[nearest_indices]
    
    # Menghitung label yang paling sering muncul
    predicted_label = nearest_labels.value_counts().idxmax()
    
    return predicted_label

