# ANALISIS PREDIKTIF PCOS (Polycystic Ovarian syndrome) - LAPORAN PROYEK MACHINE LEARNING TERAPAN 

## Domain Proyek 
PCOS atau atau Sindrom Ovarium Polikistik merupakan kelainan hormonal yang dialami oleh sekitar 13% wanita dalam usia reproduksi. Gejalanya mencakup masalah metabolik, psikologis, dan reproduksi, seperti kadar hormon androgen yang tinggi, siklus haid tidak teratur, gangguan ovulasi, serta kesulitan memiliki anak [[1]](https://jurnal.upertis.ac.id/index.php/JKP/article/view/971/436). Deteksi dini PCOS sangat penting karena dapat membantu wanita untuk mendapatkan pengobatan yang tepat dan mencegah komplikasi jangka panjang. Analisis ini bertujuan untuk mengetahui faktor-faktor yang berpengaruh terhadap PCOS dan mengetahui algoritma machine learning yang dapat membantu dalam memprediksi PCOS. 

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
1. Bagaimana membuat model machine learning yang dapat memprediksi atau mendiagnosis PCOS?
2. Model apa yang memiliki akurasi yang paling baik dalam memprediksi PCOS berdasarkan faktor-faktor yang diberikan?

### Goals
Tujuan dari proyek ini adalah:
1.	Mengembangkan model machine learning yang akurat untuk mendiagnosis PCOS.
2. Membandingkan beberapa algoritma model sehingga ditemukan akurasi yang paling baik untuk memprediksikan PCOS.

### Solution Statements
Upaya yang dilakukan untuk mencapai tujuan tersebut antara lain:
1. Menggunakan beberapa algoritma machine learning yang untuk kemudian dibandingkan hasilnya. Berikut adalah model machine learning yang digunakan:
   - Logistic Regression
     Logistic Regression adalah algoritma klasifikasi machine learning yang digunakan untuk memprediksi probabilitas variabel dependen kategoris [[2]](https://locus.rivierapublishing.id/index.php/jl/article/view/135). Kelebihan logistic regression yaitu cepat, sederhana, dan mudah diinterpretasi (koefisien menunjukkan pengaruh fitur) serta cocok untuk data linear yang terpisah jelas. Sedangkan kekurangannya adalah tidak efektif untuk hubungan non-linear serta rentan terhadap outliers dan multikolinearitas.
   - Decision Tree adalah algoritma yang bisa menghasilkan keputusan dengan cara membentuk pohon keputusan [[3]](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/173). Kelebihan Decision Tree yaitu mudah divisualisasikan dan diinterpretasi (seperti flowchart) serta Tidak memerlukan scaling data dan handal untuk data non-linear. Sedangkan kekurangannya cenderung overfitting jika terlalu dalam dan sensitif terhadap perubahan kecil pada data.
   - Random Forest adalah kombinasi dari masing masing teknik pohon keputusan yang ada, lalu kemudian digabung dan dikombinasikan kedalam suatu model [[2]](https://locus.rivierapublishing.id/index.php/jl/article/view/135). Kelebihan Random Forest yaitu mengurangi overfitting dengan agregasi banyak pohon (ensemble) dan robust terhadap noise dan outliers. Sedangkan kekurangannya adalah kompleks dan sulit diinterpretasi dibanding Decision Tree tunggal serta lebih lambat dalam pelatihan karena banyak pohon.
   - Support Vector Machine merupakan sekumpulan metode supervised learning yang membuat hyperlane atau sekumpulan hyperlane pada proses klasifikasi, regresi, dan outlier detection [[2]](https://locus.rivierapublishing.id/index.php/jl/article/view/135). Kelebihan Support Vector Machine adalah efektif untuk data berdimensi tinggi dan non-linear (dengan kernel trick) serta tahan terhadap overfitting jika parameter optimal. Sedangkan kekurangannya adalah komputasi berat untuk dataset besar dan sulit diinterpretasi dan membutuhkan tuning kernel.
2. Evaluasi model menggunakan metrik akurasi, precision, recall, dan F1-score

## Data Understanding
Dataset yang digunakan digunakan dalam laporan ini bersumber dari kaggle serta memiliki 1000 baris dan 6 kolom, 5 kolom sebagai fitur dan 1 kolom sebagai variabel target. Berikut penjelasan kolom dataset dari sumber yang dapat dilihat di [dataset pcos](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset):
- Fitur:
  1. Usia (tahun): Usia pasien, berkisar antara 18 hingga 45 tahun.
  2. IMT (kg/m²): Indeks Massa Tubuh, pengukur lemak tubuh berdasarkan tinggi dan berat badan, dengan rentang 18 hingga 35.
  3. Ketidakteraturan Menstruasi (biner): Indikator biner yang menunjukkan apakah pasien memiliki siklus haid tidak teratur (0 = Tidak, 1 = Ya).
  4. Kadar Testosteron (ng/dL): Tingkat testosteron dalam darah pasien, indikator hormonal penting PCOS, dengan rentang 20 hingga 100 ng/dL.
  5. Jumlah Folikel Antral: Jumlah folikel antral yang terdeteksi melalui USG, berkisar antara 5 hingga 30, membantu menilai cadangan ovarium dan keberadaan PCOS.
- Variabel Target:
  1. Diagnosis PCOS (biner): Indikator biner apakah pasien terdiagnosis PCOS (0 = Tidak, 1 = Ya), berdasarkan kombinasi faktor risiko seperti IMT tinggi, kadar testosteron, ketidakteraturan menstruasi, dan jumlah folikel antral.
<br>
Dataset tersebut tidak memiliki baik missing value maupun data yang duplikat.
<br>
<br>
Berikut merupakan matriks korelasi antar fitur:

![matriks_korelasi_pcos](https://github.com/daffaakifah/analisis-prediktif-pcos/blob/main/matriks%20korelasi%20pcos.png)

Berdasarkan matriks korelasi dan visualisasi yang diberikan, berikut adalah deskripsi hubungan antara fitur-fitur yang terkait dengan PCOS (Polycystic Ovary Syndrome):
1. Age (Usia): <br>
Memiliki korelasi -0.065 dengan PCOS_Diagnosis. Usia memiliki korelasi lemah terhadap faktor penyebab PCOS.
2. BMI (Indeks Massa Tubuh): <br>
Memiliki korelasi positif dengan PCOS_Diagnosis (0.38). Ini menunjukkan bahwa peningkatan BMI mungkin berkontribusi terhadap diagnosis PCOS [[4]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727).
3. Menstrual_Irregularity (Ketidakteraturan Menstruasi):<br>
Ketidakteraturan menstruasi memiliki korelasi positif dengan PCOS_Diagnosis (0.47). Ini menunjukkan bahwa ketidakteraturan menstruasi adalah salah satu indikator PCOS. Siklus  menstruasi  yang  tidak teratur  sering  menunjukkan  hiperandrogenisme yang  dapat berisiko  terjadinya  PCOS [[4]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727/920).
4. Testosterone_level(ng/dL) (Kadar Testosteron): <br>
Memiliki korelasi positif dengan PCOS_Diagnosis (0.2). Kadar testosteron tinggi merupakan faktor resiko penting dalam PCOS [[5]](https://www.frontiersin.org/journals/reproductive-health/articles/10.3389/frph.2024.1475132/full).
5. Antral Follicle Count (Jumlah Folikel Antral): <br>
Memiliki korelasi positif dengan PCOS_Diagnosis (0.19). Jumlah folikel antral yang tinggi juga merupakan ciri khas faktor PCOS [[6]](https://oaj.jurnalhst.com/index.php/jim/article/view/4513).
<br>
PCOS_Diagnosis: <br>
Korelasi tertinggi dengan Menstrual_Irregularity (0.47) dan BMI (0.38), diikuti oleh Testosterone_level (0.2) dan Antral_Follicle_Count (0.19). Ini menunjukkan bahwa ketidakteraturan menstruasi dan obesitas (BMI tinggi) adalah faktor dominan yang terkait dengan PCOS dalam dataset ini.

## Data Preparation
Teknik yang digunakan dalam penyiapan data *(Data Preparation)* yaitu:
- **Split Data** merupakan proses untuk membagi antara data latih dan data uji. Split data adalah salah satu dari beberapa aspek yang  mempengaruhi seberapa baik kinerja model klasifikasi pada algoritma [[7]](https://jsi.politala.ac.id/index.php/JSI/article/view/622).
Pada dataset ini secara acak dibagi menjadi dua subset: latih (80%) dan uji (20%) dengan kode berikut: <br>
```sh
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
- **Scaling Fitur** merupakan  cara  untuk  membuat numberical  datapada  dataset  memiliki rentang  nilai  yang  sama.  Dengan  artian  tidak  ada  satu  pun  variabel  data yang  mendominasi variabel data lainnya [[8]](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/14710). 

## Modeling
Pada tahap modeling ini dibuat beberapa model dengan algoritma yang berbeda-beda. Pada proyek ini dataset dilatih dengan 4 model, yaitu menggunakan Logistic Regression, Decision Tree, Random Forest, dan Support Vector Machine.

## Evaluation
Evaluasi performa model dilakukan menggunakan berbagai metrik evaluasi yang umum digunakan dalam klasifikasi, seperti akurasi, presisi, recall, dan skor F1. Akurasi mengukur sejauh mana model dapat mengklasifikasikan dengan benar, sedangkan presisi mengukur sejauh mana model memberikan prediksi yang benar untuk kelas positif. Recall mengukur sejauh mana model dapat mendeteksi dengan benar kelas positif, sedangkan skor F1 adalah penggabungan antara presisi dan recall [[9]](https://jurnal.umt.ac.id/index.php/jt/article/viewFile/9099/4575).
<br>
Perumusannya antara lain [[2]](https://locus.rivierapublishing.id/index.php/jl/article/view/135): 
- A. Accuracy/Akurasi : persentase  nilai prediksi yang benar dari keseluruhan pengamatan. Rumus → (TP+TN)/(TP+FP+TN+FN);
- B. Precision/Presisi : persentase kasus yang diprediksi AI dan memang terjadi berdasarkan realitanya. Rumus → (TP)/(TP+FP)
- C. Recall : mengukur pecahan kasus yang terjadi dan diprediksi tepat oleh AI. Rumus → (TP)/(TP+FN)
  
<br>
Keterangan [[3]](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/173):
- TP : True Positive (TP), yaitu data  positif  yang terprediksi benar.
- TN : True Negative (TN), yaitu data negatif yang terprediksi dengan benar
- FP : False Positive (FP),  yaitu data  negatif tapi terprediksi sebagai data positif.
- FN : False Negative (FN), yaitu data positif yang terprediksi sebagai data negatif.
<br>

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

Berikut merupakan hasil metriks akurasi dalam bentuk diagram batang:

![Perbandingan Akurasi Model ML](https://github.com/daffaakifah/analisis-prediktif-pcos/blob/main/perbandingan%20akurasi%20model%20ml.png)

<br>
Hasil akurasi Random Forest (RF) dan Decision Tree (DT) yang tidak jauh berbeda dapat terjadi karena beberapa alasan:
1. Jika dataset memiliki sedikit fitur atau pola yang mudah dipelajari, DT tunggal mungkin sudah mencapai performa maksimal. RF (yang terdiri dari banyak DT) tidak selalu meningkatkan akurasi dalam kasus ini [[10]](https://www.researchgate.net/publication/284219299_A_Random_Forest_Guided_Tour).
2. Ketidakseimbangan Kelas (Class Imbalance).
Jika dataset sangat tidak seimbang, metrik akurasi bisa menyesatkan. RF dan DT mungkin sama-sama memprediksi kelas mayoritas dengan baik tetapi gagal menangkap minoritas [[11]](https://www.jmlr.org/papers/volume15/delgado14a/delgado14a.pdf).

Dari paparan sebelumnya dapat diketahui bahwa model dengan algoritma Decision Tree memiliki kinerja yang lebih baik. Untuk itu model tersebut yang akan dipilih untuk digunakan.

## Saran:
- Menambahkan lagi dataset dengan rentang usia beragam agar mengetahui pengaruh usia terhadap PCOS [[12]](https://etd.umy.ac.id/id/eprint/115/).
- Menambahkan data faktor genetik agar melihatt pengaruhnya dalam faktor prediksi PCOS [[12]](https://etd.umy.ac.id/id/eprint/115/).

## Referensi:
[[1]](https://jurnal.upertis.ac.id/index.php/JKP/article/view/971/436) Kurniawati, E., Hutabarat, N., & Noviasari, E. (2023). Status Gizi dan Gaya Hidup Wanita dengan Sindrom Ovarium Polikistik (PCOS) di Yogyakarta. JURNAL KESEHATAN PERINTIS, 10(1), 74-82. https://doi.org/10.33653/jkp.v10i1.971 <br>
[[2]](https://locus.rivierapublishing.id/index.php/jl/article/view/135) F. Azimah dan K. R. N. Wardani, "Klasifikasi deteksi gejala awal COVID-19 dengan metodologi logistic regression, random forest classifier dan support vector machine," Locus: Penelitian & Pengabdian, vol. 1, no. 6, Sep. 2022. [Online]. Tersedia: https://locus.rivierapublishing.id/index.php/jl doi: 10.36418/locus.v1i6.135405.
[[3]](https://jurnal.tau.ac.id/index.php/siskom-kb/article/view/173) Hana, F. M. (2020). Klasifikasi penderita penyakit diabetes menggunakan algoritma Decision Tree C4.5. Jurnal Sistem Komputer dan Kecerdasan Buatan, 4(1).
[[4]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727) Noviyanti, N., Johan, R., & Ruqaiyah, R. (2024). The Effect of Menstrual Cycle and Body Mass Index on The Risk of Polycystic Ovarian Syndrome (PCOS) in Adolescent Females in Tarakan City. ancasakti ournal f ublic ealth cience nd esearch, 4(3), 89-96. https://doi.org/10.47650/pjphsr.v4i3.1727 <br>
[[5]](https://www.frontiersin.org/journals/reproductive-health/articles/10.3389/frph.2024.1475132/full) Bushell, A., & Crespi, B. J. (2024). The evolutionary basis of elevated testosterone in women with polycystic ovary syndrome: An overview of systematic reviews of the evidence. Frontiers in Reproductive Health, 6, Article 1475132. https://doi.org/10.3389/frph.2024.1475132 <br>
[[6]](https://oaj.jurnalhst.com/index.php/jim/article/view/4513) Alamsyah, F., Halim, B., & Tanjung, T. (2024). Kadar serum anti-mullerian hormon sebagai alternatif pemeriksaan jumlah folikel antral dalam menegakkan diagnosa polycystic ovarian syndrome. Jurnal Ilmiah Multidisipliner (JIM), 8(9), 174.<br>
[[7]](https://jsi.politala.ac.id/index.php/JSI/article/view/622) Oktafiani, R., Hermawan, A., & Avianto, D. (2023). Pengaruh komposisi split data terhadap performa klasifikasi penyakit kanker payudara menggunakan algoritma machine learning. Jurnal Sains dan Informatika, 9(1), 19. https://doi.org/10.34128/jsi.v9i1.622
[[8]](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/14710) Aryuni, A. F., Putrada, A. G., & Abdurohman, M. (2021). Klasifikasi penumpang naik dan turun dengan sensor load cell menggunakan ekstraksi fitur dan support vector machine. e-Proceeding of Engineering, 8(2), 3197.
[[9]](https://jurnal.umt.ac.id/index.php/jt/article/viewFile/9099/4575) Fadli, M., & Saputra, R. A. (2023). Klasifikasi dan evaluasi performa model Random Forest untuk prediksi stroke [Classification and evaluation of performance models Random Forest for stroke prediction]. Jurnal Teknik, 15(2), 45–60. http://jurnal.umt.ac.id/index.php/jt/index
[[10]](https://www.researchgate.net/publication/284219299_A_Random_Forest_Guided_Tour) Biau, Gérard & Scornet, Erwan. (2015). A Random Forest Guided Tour. TEST. 25. 10.1007/s11749-016-0481-7. <br>
[[11]](https://www.jmlr.org/papers/volume15/delgado14a/delgado14a.pdf) Fernández-Delgado, M., et al. (2014). Do we need hundreds of classifiers to solve real world classification problems? JMLR, 15(1), 3133-3181 <br>
[[12]](https://etd.umy.ac.id/id/eprint/115/) Kamila Sedah Kirana. (2020). Hubungan Antara Faktor Resiko Usia, Riwayat Keluarga, dan Usia Menarkhe Terhadap Kejadian Polycystic Ovarian Syndrome (PCOS). S1 thesism, Universitas Muhammadiyah Yogyakarta.
