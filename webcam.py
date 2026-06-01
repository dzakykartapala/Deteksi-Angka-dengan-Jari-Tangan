import cv2
import numpy as np
import tensorflow as tf

print("Sedang memuat model AI... (mohon tunggu sebentar)")
# Load model yang sudah dilatih dari Google Colab
model = tf.keras.models.load_model('model_jari.h5')

# Daftar nama kelas (angka 0 sampai 5)
class_names = ['0', '1', '2', '3', '4', '5']

# Buka Webcam (angka 0 untuk kamera bawaan laptop)
cap = cv2.VideoCapture(0)

print("Kamera siap! Tekan 'q' pada keyboard untuk keluar.")

while True:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca kamera.")
        break
        
    # Balik gambar seperti cermin (opsional, agar lebih nyaman dilihat)
    frame = cv2.flip(frame, 1)
        
    # Buat kotak biru untuk area deteksi jari (ROI - Region of Interest)
    # Titik awal (x1,y1) dan titik akhir (x2,y2)
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
    
    # Potong gambar hanya pada area di dalam kotak biru tersebut
    roi = frame[100:400, 100:400]
    
    # Preprocessing ROI agar formatnya persis dengan saat AI dilatih
    img = cv2.resize(roi, (128, 128)) # Ukuran harus 128x128
    img_array = np.array(img, dtype='float32') / 255.0 # Normalisasi 0-1
    img_expand = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
    
    # Lakukan prediksi
    prediction = model.predict(img_expand, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Tampilkan teks hasil prediksi di atas kotak biru jika AI cukup yakin (> 70%)
    if confidence > 70:
        teks = f"Jari: {class_names[predicted_class]} ({confidence:.2f}%)"
        cv2.putText(frame, teks, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tidak Terdeteksi", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan layar
    cv2.imshow("Deteksi Angka Jari AI", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan memori
cap.release()
cv2.destroyAllWindows()