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
2.  Membandingkan beberapa algoritma model sehingga ditemukan akurasi yang paling baik untuk memprediksikan PCOS.

### Solution Statements
Upaya yang dilakukan untuk mencapai tujuan tersebut antara lain:
1. Menggunakan beberapa algoritma machine learning yang untuk kemudian dibandingkan hasilnya. Berikut adalah model machine learning yang digunakan:
   •	Logistic Regression
   •	Decision Tree
   •	Random Forest
   •	Support Vector Machine
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
Berikut merupakan matriks korelasi antar fitur:

![matriks_korelasi_pcos](https://github.com/daffaakifah/analisis-prediktif-pcos/blob/main/matriks%20korelasi%20pcos.png)

Berdasarkan matriks korelasi dan visualisasi yang diberikan, berikut adalah deskripsi hubungan antara fitur-fitur yang terkait dengan PCOS (Polycystic Ovary Syndrome):
1. Age (Usia): <br>
Memiliki korelasi -0.065 dengan PCOS_Diagnosis. Usia memiliki korelasi lemah terhadap faktor penyebab PCOS.
2. BMI (Indeks Massa Tubuh): <br>
Memiliki korelasi positif dengan PCOS_Diagnosis (0.38). Ini menunjukkan bahwa peningkatan BMI mungkin berkontribusi terhadap diagnosis PCOS [[2]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727).
3. Menstrual_Irregularity (Ketidakteraturan Menstruasi):<br>
Ketidakteraturan menstruasi memiliki korelasi positif dengan PCOS_Diagnosis (0.47). Ini menunjukkan bahwa ketidakteraturan menstruasi adalah salah satu indikator PCOS. Siklus  menstruasi  yang  tidak teratur  sering  menunjukkan  hiperandrogenisme yang  dapat berisiko  terjadinya  PCOS [[2]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727/920).
4. Testosterone_level(ng/dL) (Kadar Testosteron): <br>
Memiliki korelasi positif dengan PCOS_Diagnosis (0.2). Kadar testosteron tinggi merupakan faktor resiko penting dalam PCOS [[3]](https://www.frontiersin.org/journals/reproductive-health/articles/10.3389/frph.2024.1475132/full).
5. Antral Follicle Count (Jumlah Folikel Antral): <br>
Memiliki korelasi positif dengan PCOS_Diagnosis (0.19). Jumlah folikel antral yang tinggi juga merupakan ciri khas faktor PCOS [[4]](https://oaj.jurnalhst.com/index.php/jim/article/view/4513).
<br>
PCOS_Diagnosis: <br>
Korelasi tertinggi dengan Menstrual_Irregularity (0.47) dan BMI (0.38), diikuti oleh Testosterone_level (0.2) dan Antral_Follicle_Count (0.19). Ini menunjukkan bahwa ketidakteraturan menstruasi dan obesitas (BMI tinggi) adalah faktor dominan yang terkait dengan PCOS dalam dataset ini.

## Data Preparation
Teknik yang digunakan dalam penyiapan data *(Data Preparation)* yaitu:
- **Split Data** atau pembagian dataset menjadi data latih dan data uji. Proses ini membantu mencegah overfitting dan underfitting dengan menjaga data pengujian terpisah dari data pelatihan, sehingga kinerja prediktif model dapat dinilai secara akurat.
- **Scaling Fitur**

## Modeling
Pada tahap modeling ini dibuat beberapa model dengan algoritma yang berbeda-beda. Pada proyek ini akan dibuat 4 model, diantaranya yaitu menggunakan Logistic Regression, Decision Tree, Random Forest, dan Support Vector Machine.
Setelah melatih keempat model tersebut, didapatkan metriks akurasi sebagai berikut seperti pada diagram di bawah ini.

![Perbandingan Akurasi Model](https://i.postimg.cc/ZnwYHYdV/Screenshot-5.png)

Dari hasil tersebut dapat diketahui bahwa model dengan algoritma Random Forest memiliki kinerja yang lebih baik. Untuk itu model tersebut yang akan dipilih untuk digunakan.

## Evaluation
Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi.

Akurasi merupakan kalkulasi presentase jumlah ketepatan prediksi dari jumlah seluruh data yang diprediksi. Nilai akurasi dapat dihitung dengan rumus berikut.

![accuracy](https://i.postimg.cc/TwSPSscb/Screenshot-15.png)

## Saran:
- Menambahkan lagi dataset dengan rentang usia beragam agar mengetahui pengaruh usia terhadap PCOS [[9]](https://etd.umy.ac.id/id/eprint/115/).
- Menambahkan data faktor genetik agar melihqat pengaruhnya dalam faktor prediksi PCOS [[9]](https://etd.umy.ac.id/id/eprint/115/).

## Referensi:
[[1]](https://jurnal.upertis.ac.id/index.php/JKP/article/view/971/436)Kurniawati, E., Hutabarat, N., & Noviasari, E. (2023). Status Gizi dan Gaya Hidup Wanita dengan Sindrom Ovarium Polikistik (PCOS) di Yogyakarta. JURNAL KESEHATAN PERINTIS, 10(1), 74-82. https://doi.org/10.33653/jkp.v10i1.971 <br>
[[2]](https://journal.unpacti.ac.id/index.php/pjphsr/article/view/1727)Noviyanti, N., Johan, R., & Ruqaiyah, R. (2024). The Effect of Menstrual Cycle and Body Mass Index on The Risk of Polycystic Ovarian Syndrome (PCOS) in Adolescent Females in Tarakan City. ancasakti ournal f ublic ealth cience nd esearch, 4(3), 89-96. https://doi.org/10.47650/pjphsr.v4i3.1727 <br>
[[3]](https://www.frontiersin.org/journals/reproductive-health/articles/10.3389/frph.2024.1475132/full)Bushell, A., & Crespi, B. J. (2024). The evolutionary basis of elevated testosterone in women with polycystic ovary syndrome: An overview of systematic reviews of the evidence. Frontiers in Reproductive Health, 6, Article 1475132. https://doi.org/10.3389/frph.2024.1475132 <br>
[[4]](https://oaj.jurnalhst.com/index.php/jim/article/view/4513)Alamsyah, F., Halim, B., & Tanjung, T. (2024). Kadar serum anti-mullerian hormon sebagai alternatif pemeriksaan jumlah folikel antral dalam menegakkan diagnosa polycystic ovarian syndrome. Jurnal Ilmiah Multidisipliner (JIM), 8(9), 174.<br>
[[5]](https://www.researchgate.net/publication/284219299_A_Random_Forest_Guided_Tour)Biau, Gérard & Scornet, Erwan. (2015). A Random Forest Guided Tour. TEST. 25. 10.1007/s11749-016-0481-7. <br>
[[6]](https://www.jmlr.org/papers/volume15/delgado14a/delgado14a.pdf)Fernández-Delgado, M., et al. (2014). Do we need hundreds of classifiers to solve real world classification problems? JMLR, 15(1), 3133-3181 <br>
[[7]](https://link.springer.com/article/10.1023/A:1010933404324)Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. <br>
[[8]](https://www.sciencedirect.com/science/article/abs/pii/S0167865510000954?via%3Dihub)Genuer, R., et al. (2010). Variable selection using random forests. Pattern Recognition Letters, 31(14), 2225-2236. <br>
[[9]](https://etd.umy.ac.id/id/eprint/115/)Kamila Sedah Kirana. (2020). Hubungan Antara Faktor Resiko Usia, Riwayat Keluarga, dan Usia Menarkhe Terhadap Kejadian Polycystic Ovarian Syndrome (PCOS). S1 thesism, Universitas Muhammadiyah Yogyakarta.
