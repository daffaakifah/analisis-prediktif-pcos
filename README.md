# ANALISIS PREDIKTIF PCOS (Polycystic Ovarian syndrome) - LAPORAN PROYEK MACHINE LEARNING TERAPAN 
Oleh Daffa Akifah Balqis
## Domain Proyek 
PCOS atau atau Sindrom Ovarium Polikistik merupakan kelainan hormonal yang dialami oleh sekitar 13% wanita dalam usia reproduksi. Gejalanya mencakup masalah metabolik, psikologis, dan reproduksi, seperti kadar hormon androgen yang tinggi, siklus haid tidak teratur, gangguan ovulasi, serta kesulitan memiliki anak [[1]](https://jurnal.upertis.ac.id/index.php/JKP/article/view/971/436). Deteksi dini PCOS sangat penting karena dapat membantu wanita untuk mendapatkan pengobatan yang tepat dan mencegah komplikasi jangka panjang. Analisis ini bertujuan untuk mengetahui faktor-faktor yang berpengaruh terhadap PCOS dan mengetahui algoritma machine learning yang dapat membantu dalam memprediksi PCOS. 

## Business Understanding

### Problem Statements
Diagnosis PCOS saat ini masih bergantung pada tes medis yang kompleks, seperti USG panggul dan pemeriksaan hormon, yang membutuhkan waktu lama dan biaya tinggi. Selain itu, keterbatasan akses ke fasilitas medis di daerah terpencil membuat banyak pasien tidak mendapatkan diagnosis tepat waktu. Meningkatkan dan membantu deteksi dini dan pengobatan tepat waktu untuk Sindrom Ovarium Polikistik (PCOS) sebagai upaya untuk mengatasi PCOS bagi wanita dapat dilakukan dengan membangun sistem machine learning yang mampu memprediksi dengan akurat hasil diagnosis PCOS. Berdasarkan latar belakang di atas, berikut ini merupakan rumusan masalah yang dapat diselesaikan pada proyek ini:
1. Bagaimana cara memprediksi atau mendiagnosis PCOS dengan akurat menggunakan model machine learning?
2. Model apa yang memiliki akurasi yang paling baik dalam memprediksi PCOS sehingga dengan akurat mendiagnosis PCOS berdasarkan faktor-faktor yang diberikan?

### Goals
Tujuan dari proyek ini adalah:
1.	Mengembangkan model machine learning yang akurat untuk mendiagnosis PCOS.
2. Membandingkan beberapa algoritma model sehingga ditemukan akurasi yang paling baik untuk memprediksikan PCOS.

### Solution Statements
Upaya yang dilakukan untuk mencapai tujuan tersebut antara lain:
1. Menggunakan beberapa algoritma machine learning yang untuk kemudian dibandingkan hasilnya. Berikut adalah model machine learning yang digunakan:
   - Logistic Regression
   - Decision Tree 
   - Random Forest 
   - Support Vector Machine 
2. Evaluasi model menggunakan metrik akurasi, precision, recall, dan F1-score dengan tujuan untuk mengetahui:
   - Akurasi: Seberapa baik model memprediksi PCOS secara keseluruhan.
   - Precision: Seberapa banyak prediksi positif yang benar (positif nyata).
   - Recall: Seberapa banyak kasus PCOS yang berhasil dideteksi oleh model.
   - F1-Score: Keseimbangan rata-rata dari Precision dan Recall, yang memberikan gambaran umum tentang kinerja model.

## Data Understanding
#### Penjelasan singkat mengenai dataset
Dataset yang digunakan digunakan dalam laporan ini bersumber dari kaggle serta memiliki 1000 baris dan 6 kolom, 5 kolom sebagai fitur dan 1 kolom sebagai variabel target. Berikut penjelasan kolom dataset dari sumber yang dapat dilihat di [dataset pcos](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset):
| Column                        | Non-Null Count | Dtype   |
|-------------------------------|----------------|---------|
| Age                           | 1000 non-null  | int64   |
| BMI                           | 1000 non-null  | float64 |
| Menstrual_Irregularity        | 1000 non-null  | int64   |
| Testosterone_Level (ng/dL)    | 1000 non-null  | float64 |
| Antral_Follicle_Count         | 1000 non-null  | int64   |
| PCOS_Diagnosis                | 1000 non-null  | int64   |
- Fitur:
  1. Usia (tahun): Usia pasien, berkisar antara 18 hingga 45 tahun. Tipe data integer.
  2. BMI (kg/m²): Indeks Massa Tubuh, pengukur lemak tubuh berdasarkan tinggi dan berat badan, dengan rentang 18 hingga 35. Tipe data float.
  3. Ketidakteraturan Menstruasi (biner): Indikator biner yang menunjukkan apakah pasien memiliki siklus haid tidak teratur (0 = Tidak, 1 = Ya). Tipe data integer.
  4. Kadar Testosteron (ng/dL): Tingkat testosteron dalam darah pasien, indikator hormonal penting PCOS, dengan rentang 20 hingga 100 ng/dL. Tipe data float.
  5. Jumlah Folikel Antral: Jumlah folikel antral yang terdeteksi melalui USG, berkisar antara 5 hingga 30, membantu menilai cadangan ovarium dan keberadaan PCOS. Tipe data integer.
- Variabel Target:
  1. Diagnosis PCOS (biner): Indikator biner apakah pasien terdiagnosis PCOS (0 = Tidak, 1 = Ya), berdasarkan kombinasi faktor risiko seperti IMT tinggi, kadar testosteron, ketidakteraturan menstruasi, dan jumlah folikel antral. Tipe data integer.
<br>

#### Mengecek missing value dan duplikat pada dataset
Dataset tidak memiliki fitur yang terdapat misssing value: <br>
![image](https://github.com/user-attachments/assets/92caacc9-474c-42d3-89e5-c6ce8153b5e3)

Dataset tidak memiliki fitur yang terdapat duplikat data:
![image](https://github.com/user-attachments/assets/11a863ab-79d5-4910-8b58-4a9574727e90)

Dataset tersebut tidak memiliki baik missing value maupun data yang duplikat.
<br>
#### Exploratory Data Analysis (EDA)
- Melihat persebaran data diagnosis: <br>
  ![image](https://github.com/user-attachments/assets/3a7fe4cf-f823-4bc3-b50d-59c08b577c9e)
  <br>
  Persebaran diagnosis data didominasi oleh pasien yang tidak mengidap PCOS yaitu total 80.1% dari total keseluruhan data yang ada pada dataset.
  
- Melihat distribusi fitur dengan histogram:
  ![image](https://github.com/user-attachments/assets/85392997-cd50-47d6-b7f1-dabf80e2f2e9)
  Interpretasi distribusi variabel feature terhadap target:
  - Rentang usia tersebar dari usian 20an sampai 40an untuk pengidap PCOS
  - Distribusi BMI untuk pengidap PCOS cenderung tinggi
  - Distribusi testosteron level untuk pengidap PCOS cenderung tinggi
  - Distribusi Antral Follicle untuk pengidap PCOS cenderung tinggi
  ![image](https://github.com/user-attachments/assets/384636bc-a57b-4016-a334-a2ee9e1dad68)
  Sebanyak 53% pengidap PCOS dari dataset mengalami ketidakteraturan dalam menstruasi.
- Melihat distribusi fitur dengan boxplot:
  ![image](https://github.com/user-attachments/assets/36aa0266-5da5-459a-a5a2-14cc0513118b)
  Berdasarkan boxplot di atas ditemukan bahwa:
  - Tidak adanya outlier dalam data
  - Perbedaan rentang usia antara pengidap PCOS dan tidak adalah hampir seragam
  - Perbedaan rentang BMI, testosteron level, Antral Follicle cukup terlihat berbeda dengan pengidap PCOS memiliki rentang BMI, testosteron level, Antral Follicle yang lebih tinggi dibanding yang tidak.

- Melihat korelasi antar fitur dengan heatmap:
  ![image](https://github.com/user-attachments/assets/ee15ccaf-2a3a-4bab-8c0e-092d29225265)
  Berdasarkan matriks korelasi dan visualisasi yang diberikan, berikut adalah deskripsi hubungan antara fitur-fitur yang terkait dengan PCOS (Polycystic Ovary Syndrome):
  1. Age (Usia): <br>
     Memiliki korelasi -0.065 dengan PCOS_Diagnosis. Usia memiliki korelasi lemah terhadap faktor penyebab PCOS.
  2. BMI (Indeks Massa Tubuh): <br>
     Memiliki korelasi positif dengan PCOS_Diagnosis (0.38). Ini menunjukkan bahwa peningkatan BMI mungkin berkontribusi terhadap 
     diagnosis PCOS [[2]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727).
  3. Menstrual_Irregularity (Ketidakteraturan Menstruasi):<br>
     Ketidakteraturan menstruasi memiliki korelasi positif dengan PCOS_Diagnosis (0.47). Ini menunjukkan bahwa ketidakteraturan 
     menstruasi adalah salah satu indikator PCOS. Siklus  menstruasi  yang  tidak teratur  sering  menunjukkan  hiperandrogenisme yang  
     dapat berisiko  terjadinya  PCOS [[2]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727/920).
  4. Testosterone_level(ng/dL) (Kadar Testosteron): <br>
     Memiliki korelasi positif dengan PCOS_Diagnosis (0.2). Kadar testosteron tinggi merupakan faktor resiko penting dalam PCOS [[3]](https://www.frontiersin.org/journals/reproductive-health/articles/10.3389/frph.2024.1475132/full).
  5. Antral Follicle Count (Jumlah Folikel Antral): <br>
     Memiliki korelasi positif dengan PCOS_Diagnosis (0.19). Jumlah folikel antral yang tinggi juga merupakan ciri khas faktor PCOS [[4]](https://oaj.jurnalhst.com/index.php/jim/article/view/4513).
<br>
  PCOS_Diagnosis: <br>
  Korelasi tertinggi dengan Menstrual_Irregularity (0.47) dan BMI (0.38), diikuti oleh Testosterone_level (0.2) dan Antral_Follicle_Count 
  (0.19). Ini menunjukkan bahwa ketidakteraturan menstruasi dan obesitas (BMI tinggi) adalah faktor dominan yang terkait dengan PCOS 
  dalam dataset ini.

## Data Preparation
Teknik yang digunakan dalam penyiapan data *(Data Preparation)* yaitu:
- **Split Data** merupakan proses untuk membagi antara data latih dan data uji. Split data adalah salah satu dari beberapa aspek yang  mempengaruhi seberapa baik kinerja model klasifikasi pada algoritma [[5]](https://jsi.politala.ac.id/index.php/JSI/article/view/622).
Pada dataset ini secara acak dibagi menjadi dua subset yaitu data latih (80%) dan data uji (20%) dengan kode sebagai berikut: <br>
```sh
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Dengan rincian pemabagian data:
Total sampel di keseluruhan dataset: 1000
Total sampel di dalam data latih: 800
Total sampel di dalam data uji : 200
- **Scaling Fitur** adalah teknik standarisasi yang menyamakan rentang nilai data numerik dalam dataset, sehingga semua variabel memiliki skala yang seimbang tanpa dominasi satu variabel terhadap lainnya. [[6]](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/14710). Scaling dilakukan dalam tahap data preparation dan dilakukan setelah pembagian train-test split adalah untuk menghindari data leakage.

## Modeling
#### Penjelasan model algoritma:
Pada tahap modeling ini dibuat beberapa model dengan algoritma yang berbeda-beda. Pada proyek ini dataset dilatih dengan 4 model, yaitu menggunakan Logistic Regression, Decision Tree, Random Forest, dan Support Vector Machine. Berikut masing-masing penjelasannya:
   - Logistic Regression
     Logistic Regression adalah algoritma klasifikasi machine learning yang digunakan untuk memprediksi probabilitas variabel dependen kategoris [[7]](https://locus.rivierapublishing.id/index.php/jl/article/view/135). Kelebihan logistic regression yaitu cepat, sederhana, dan mudah diinterpretasi (koefisien menunjukkan pengaruh fitur) serta cocok untuk data linear yang terpisah jelas. Sedangkan kekurangannya adalah tidak efektif untuk hubungan non-linear serta rentan terhadap outliers dan multikolinearitas.
   - Decision Tree adalah algoritma yang bisa menghasilkan keputusan dengan cara membentuk pohon keputusan [[8]](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/173). Kelebihan Decision Tree yaitu mudah divisualisasikan dan diinterpretasi (seperti flowchart) serta Tidak memerlukan scaling data dan handal untuk data non-linear. Sedangkan kekurangannya cenderung overfitting jika terlalu dalam dan sensitif terhadap perubahan kecil pada data.
   - Random Forest adalah kombinasi dari masing masing teknik pohon keputusan yang ada, lalu kemudian digabung dan dikombinasikan kedalam suatu model [[7]](https://locus.rivierapublishing.id/index.php/jl/article/view/135). Kelebihan Random Forest yaitu mengurangi overfitting dengan agregasi banyak pohon (ensemble) dan robust terhadap noise dan outliers. Sedangkan kekurangannya adalah kompleks dan sulit diinterpretasi dibanding Decision Tree tunggal serta lebih lambat dalam pelatihan karena banyak pohon.
   - Support Vector Machine merupakan sekumpulan metode supervised learning yang membuat hyperlane atau sekumpulan hyperlane pada proses klasifikasi, regresi, dan outlier detection [[7]](https://locus.rivierapublishing.id/index.php/jl/article/view/135). Kelebihan Support Vector Machine adalah efektif untuk data berdimensi tinggi dan non-linear (dengan kernel trick) serta tahan terhadap overfitting jika parameter optimal. Sedangkan kekurangannya adalah komputasi berat untuk dataset besar dan sulit diinterpretasi dan membutuhkan tuning kernel.
#### Inisialisasi model:
```sh
# Inisialisasi model
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}
```
Keseluruhan algoritma yang dipakai menggunakan paramater default:
- Parameter default untuk Logistic Regression: <br>
  penalty='l2': Jenis regularisasi. <br>
  C=1.0: Tingkat inversi regularisasi (semakin kecil, semakin kuat regularisasinya). <br>
  solver='lbfgs': Metode optimizer yang digunakan. <br>
- Parameter default untuk Decision Tree: <br>
  criterion='gini': Kriteria pemisahan. <br>
  max_depth=None: Batas kedalaman maksimum pohon. <br>
  min_samples_split=2: Jumlah sampel minimum untuk membagi node. <br>
- Parameter default untuk Random Forest: <br>
  n_estimators=100: Jumlah pohon dalam ensemble. <br>
  criterion='gini': Kriteria pemisahan node. <br>
  max_depth=None: Batas kedalaman maksimum pohon. <br>
- Parameter default untuk SVC: <br>
  kernel='rbf': Jenis kernel yang digunakan (Radial Basis Function). <br>
  C=1.0: Tingkat inversi regularisasi. <br>
  gamma='scale': Parameter kernel yang otomatis dihitung berdasarkan data. <br>

#### Training dan evaluasi model:
```sh
# Training dan evaluasi model
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    }
```
Setiap model dilatih menggunakan data latih yang telah dinormalisasi (X_train_scaled) dan label target (y_train). Model juga melakukan prediksi pada data uji yang juga telah dinormalisasi (X_test_scaled).

## Evaluation
#### Evaluasi model dengan metrik:
Evaluasi performa model dilakukan menggunakan berbagai metrik evaluasi yang umum digunakan dalam klasifikasi, seperti akurasi, presisi, recall, dan skor F1. Akurasi mengukur sejauh mana model dapat mengklasifikasikan dengan benar, sedangkan presisi mengukur sejauh mana model memberikan prediksi yang benar untuk kelas positif. Recall mengukur sejauh mana model dapat mendeteksi dengan benar kelas positif, sedangkan skor F1 adalah penggabungan antara presisi dan recall [[9]](https://jurnal.umt.ac.id/index.php/jt/article/viewFile/9099/4575).
<br>
Perumusannya antara lain [[7]](https://locus.rivierapublishing.id/index.php/jl/article/view/135): 
- A. Accuracy/Akurasi : persentase  nilai prediksi yang benar dari keseluruhan pengamatan. Rumus → (TP+TN)/(TP+FP+TN+FN);
- B. Precision/Presisi : persentase kasus yang diprediksi AI dan memang terjadi berdasarkan realitanya. Rumus → (TP)/(TP+FP)
- C. Recall : mengukur pecahan kasus yang terjadi dan diprediksi tepat oleh AI. Rumus → (TP)/(TP+FN)
  
<br>
Keterangan [8]:
<br>
- TP : True Positive (TP), yaitu data  positif  yang terprediksi benar. <br>
- TN : True Negative (TN), yaitu data negatif yang terprediksi dengan benar <br>
- FP : False Positive (FP),  yaitu data  negatif tapi terprediksi sebagai data positif. <br>
- FN : False Negative (FN), yaitu data positif yang terprediksi sebagai data negatif.
<br>

#### Hasil evaluasi metrik:
Berikut adalah metrik evaluasi untuk 2 model dengan akurasi tertinggi:
```sh
Decision Tree:
Accuracy: 0.995
Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      1.00       161
           1       1.00      0.97      0.99        39

    accuracy                           0.99       200
   macro avg       1.00      0.99      0.99       200
weighted avg       1.00      0.99      0.99       200
```
<br>

```sh
Random Forest:
Accuracy: 0.99
Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       161
           1       1.00      0.95      0.97        39

    accuracy                           0.99       200
   macro avg       0.99      0.97      0.98       200
weighted avg       0.99      0.99      0.99       200
```

<br>
Interpretasi hasil: <br>
- Recall Lebih Baik untuk Kelas Minoritas (Kelas 1):
  DT memiliki recall 0.97 (hanya 3% false negatives) vs RF 0.95 (5% false negatives). Ini penting jika tujuan utama adalah mendeteksi kasus positif (pasien PCOS). <br>
- DT lebih konsisten dalam menangkap kasus positif (recall lebih tinggi). <br>
- Kesederhanaan Model: DT lebih mudah diinterpretasi (visualisasi pohon) dan lebih cepat dalam prediksi dibanding RF yang kompleks.
<br>

<br>
Berikut merupakan hasil metriks akurasi dalam bentuk diagram batang:

![image](https://github.com/user-attachments/assets/0b0fff64-292f-4dea-a4a3-329f1998f8ba)

<br>
Hasil akurasi Random Forest (RF) dan Decision Tree (DT) yang tidak jauh berbeda dapat terjadi karena beberapa alasan: <br>
1. Jika dataset memiliki sedikit fitur atau pola yang mudah dipelajari, DT tunggal mungkin sudah mencapai performa maksimal. RF (yang terdiri dari banyak DT) tidak selalu meningkatkan akurasi dalam kasus ini [10]. <br>
2. Ketidakseimbangan Kelas (Class Imbalance).
Jika dataset sangat tidak seimbang, metrik akurasi bisa menyesatkan. RF dan DT mungkin sama-sama memprediksi kelas mayoritas dengan baik tetapi gagal menangkap minoritas [11] <br>
<br>
#### Model pilihan
Dari paparan di atas dapat diketahui bahwa model dengan algoritma Decision Tree memiliki kinerja yang terbaik. Untuk itu model tersebut yang akan dipilih untuk digunakan.

#### Evaluasi Terhadap Business Understanding
Menjawab Problem Statement: Model yang dibuat berhasil menjawab problem statement dengan memprediksi harga sewa apartemen berdasarkan fitur-fitur yang ada dan mengidentifikasi fitur-fitur yang paling berpengaruh.
Mencapai Goals: Model Random Forest dengan hyperparameter yang dioptimalkan berhasil mencapai tujuan untuk memberikan prediksi harga sewa yang akurat dan mengidentifikasi fitur penting.
Dampak dari Solution Statement: Penggunaan beberapa algoritma dan hyperparameter tuning memberikan dampak positif dengan meningkatkan akurasi prediksi dan memungkinkan pemilihan model terbaik. Solusi yang direncanakan memberikan hasil yang signifikan dalam mencapai tujuan proyek.

#### Saran:
- Menambahkan lagi dataset dengan rentang usia beragam agar mengetahui pengaruh usia terhadap PCOS [[12]](https://etd.umy.ac.id/id/eprint/115/).
- Menambahkan data faktor genetik agar melihatt pengaruhnya dalam faktor prediksi PCOS [[12]](https://etd.umy.ac.id/id/eprint/115/).

## Referensi:
[[1]](https://jurnal.upertis.ac.id/index.php/JKP/article/view/971/436) Kurniawati, E., Hutabarat, N., & Noviasari, E. (2023). Status Gizi dan Gaya Hidup Wanita dengan Sindrom Ovarium Polikistik (PCOS) di Yogyakarta. JURNAL KESEHATAN PERINTIS, 10(1), 74-82. https://doi.org/10.33653/jkp.v10i1.971 <br>
[[2]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727) Noviyanti, N., Johan, R., & Ruqaiyah, R. (2024). The Effect of Menstrual Cycle and Body Mass Index on The Risk of Polycystic Ovarian Syndrome (PCOS) in Adolescent Females in Tarakan City. ancasakti ournal f ublic ealth cience nd esearch, 4(3), 89-96. https://doi.org/10.47650/pjphsr.v4i3.1727 <br>
[[3]](https://www.frontiersin.org/journals/reproductive-health/articles/10.3389/frph.2024.1475132/full) Bushell, A., & Crespi, B. J. (2024). The evolutionary basis of elevated testosterone in women with polycystic ovary syndrome: An overview of systematic reviews of the evidence. Frontiers in Reproductive Health, 6, Article 1475132. https://doi.org/10.3389/frph.2024.1475132 <br>
[[4]](https://oaj.jurnalhst.com/index.php/jim/article/view/4513) Alamsyah, F., Halim, B., & Tanjung, T. (2024). Kadar serum anti-mullerian hormon sebagai alternatif pemeriksaan jumlah folikel antral dalam menegakkan diagnosa polycystic ovarian syndrome. Jurnal Ilmiah Multidisipliner (JIM), 8(9), 174.<br>
[[5]](https://jsi.politala.ac.id/index.php/JSI/article/view/622) Oktafiani, R., Hermawan, A., & Avianto, D. (2023). Pengaruh komposisi split data terhadap performa klasifikasi penyakit kanker payudara menggunakan algoritma machine learning. Jurnal Sains dan Informatika, 9(1), 19. https://doi.org/10.34128/jsi.v9i1.622 <br>
[[6]](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/14710) Aryuni, A. F., Putrada, A. G., & Abdurohman, M. (2021). Klasifikasi penumpang naik dan turun dengan sensor load cell menggunakan ekstraksi fitur dan support vector machine. e-Proceeding of Engineering, 8(2), 3197. <br>
[[7]](https://locus.rivierapublishing.id/index.php/jl/article/view/135) F. Azimah dan K. R. N. Wardani, "Klasifikasi deteksi gejala awal COVID-19 dengan metodologi logistic regression, random forest classifier dan support vector machine," Locus: Penelitian & Pengabdian, vol. 1, no. 6, Sep. 2022. [Online]. Tersedia: https://locus.rivierapublishing.id/index.php/jl doi: 10.36418/locus.v1i6.135405. <br>
[[8]](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/173) Hana, F. M. (2020). Klasifikasi penderita penyakit diabetes menggunakan algoritma Decision Tree C4.5. Jurnal Sistem Komputer dan Kecerdasan Buatan, 4(1). <br>
[[9]](https://jurnal.umt.ac.id/index.php/jt/article/viewFile/9099/4575) Fadli, M., & Saputra, R. A. (2023). Klasifikasi dan evaluasi performa model Random Forest untuk prediksi stroke [Classification and evaluation of performance models Random Forest for stroke prediction]. Jurnal Teknik, 15(2), 45–60. http://jurnal.umt.ac.id/index.php/jt/index <br>
[[10]](https://www.researchgate.net/publication/284219299_A_Random_Forest_Guided_Tour) Biau, Gérard & Scornet, Erwan. (2015). A Random Forest Guided Tour. TEST. 25. 10.1007/s11749-016-0481-7. <br>
[[11]](https://www.jmlr.org/papers/volume15/delgado14a/delgado14a.pdf) Fernández-Delgado, M., et al. (2014). Do we need hundreds of classifiers to solve real world classification problems? JMLR, 15(1), 3133-3181 <br>
[[12]](https://etd.umy.ac.id/id/eprint/115/) Kamila Sedah Kirana. (2020). Hubungan Antara Faktor Resiko Usia, Riwayat Keluarga, dan Usia Menarkhe Terhadap Kejadian Polycystic Ovarian Syndrome (PCOS). S1 thesism, Universitas Muhammadiyah Yogyakarta.
