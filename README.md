# Rice Varieties Image Classification (Proyek Klasifikasi Gambar)

Proyek ini merupakan implementasi dari model *Machine Learning* berbasis *Deep Learning* menggunakan arsitektur **Convolutional Neural Network (CNN)** berbasis `Sequential` untuk mengklasifikasikan 5 varietas beras yang berbeda. Model dikembangkan menggunakan framework TensorFlow/Keras dan dioptimalkan agar dapat dideploy ke berbagai platform dengan mengonversinya ke format **SavedModel**, **TF-Lite**, dan **TensorFlow.js (TFJS)**.

## Deskripsi Proyek & Spesifikasi Dataset

Proyek ini memenuhi seluruh kriteria submission evaluasi sebagai berikut:
* **Sumber Dataset:** [Rice Image Dataset (Kaggle)](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) diunduh via `kagglehub`.
* **Total Data & Kelas:** 75.000 gambar resolusi asli $250 \times 250$ piksel (memenuhi kriteria $>1000$ gambar).
* **Varietas Beras (5 Kelas):** `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, dan `Karacadag`.
* **Pembagian Dataset:** * **Training Set (80%):** 60.000 gambar
    * **Validation Set (10%):** 7.500 gambar
    * **Test Set (10%):** 7.500 gambar

---

## Arsitektur Model CNN

Model dibangun menggunakan API `tf.keras.models.Sequential` dengan susunan layer sebagai berikut:

1.  **Rescaling Layer:** Mengubah rentang nilai piksel dari $[0, 255]$ menjadi $[0, 1]$.
2.  **4 Blok Konvolusi & Pooling:**
    * `Conv2D` (32 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
    * `Conv2D` (64 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
    * `Conv2D` (128 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
    * `Conv2D` (256 filter, kernel $3 \times 3$, fungsi aktivasi ReLU) + `MaxPooling2D` ($2 \times 2$)
3.  **Flatten Layer:** Mengubah representasi fitur 2D menjadi vektor 1D.
4.  **Dense Layers (Fully Connected) dengan Regularisasi:**
    * `Dense` (512 unit, ReLU) + `Dropout` (50% untuk mencegah overfitting)
    * `Dense` (256 unit, ReLU) + `Dropout` (30% untuk mencegah overfitting)
5.  **Output Layer:** `Dense` (5 unit sesuai jumlah kelas dengan aktivasi Softmax).

---

## Performa & Hasil Pelatihan

Pelatihan menggunakan optimizer **Adam** ($learning\_rate = 0.0001$) dan loss function `sparse_categorical_crossentropy`. Menggunakan *custom callback* untuk menghentikan pelatihan (*early stop*) saat akurasi *training* dan *validation* sekaligus menyentuh target $\ge 96\%$.

* **Lama Pelatihan:** Berhasil mencapai target dalam **2 Epoch** (berkat arsitektur yang optimal dan data pipeline `prefetch`).
* **Akurasi Training:** 97.33%
* **Akurasi Validation:** 96.74%
* **Akurasi Test Set:** **96.06%** *(Melampaui kriteria batas minimal 85%)*

---

## Struktur Direktori Hasil Ekspor Model

Model disimpan ke dalam struktur folder `submission/` sesuai regulasi standardisasi deployment:

```text
submission/
├── saved_model/
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
├── tflite/
│   ├── label.txt
│   └── model.tflite
└── tfjs_model/
