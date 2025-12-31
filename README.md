# Data-Driven Traffic Signal Control using RL 🚦🤖

Proyek ini mengimplementasikan **Data-Driven Control System (DDCS)** menggunakan pendekatan **Deep Reinforcement Learning (DRL)** untuk mengoptimalkan pengaturan lampu lalu lintas pada persimpangan.

## 📝 Deskripsi Proyek
Dibuat sebagai **Laporan Akhir Mata Kuliah Data-Driven Control System** di Fakultas Ilmu Komputer, Universitas Brawijaya.
Sistem ini bertujuan untuk meminimalkan waktu tunggu kendaraan (*waiting time*) dan panjang antrean (*queue length*) dengan membiarkan agen AI belajar mengatur durasi lampu hijau secara adaptif berdasarkan kondisi lalu lintas real-time.

## ⚙️ Metodologi & Fitur
1.  **Environment Simulation:** Menggunakan **SUMO (Simulation of Urban MObility)** untuk memodelkan persimpangan jalan dan arus kendaraan.
2.  **State Representation:** Mengambil data posisi kendaraan, kecepatan, dan kepadatan antrean sebagai input.
3.  **Reinforcement Learning (DQN):** Agen cerdas menggunakan algoritma **Deep Q-Network** untuk memilih aksi (fase lampu hijau) terbaik.
4.  **Reward Function:** Memberikan *reward* negatif (hukuman) berdasarkan total waktu tunggu akumulatif kendaraan.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Simulation:** SUMO (TraCI Interface)
* **ML Framework:** TensorFlow / Keras / TF-Agents
* **Libraries:** Scikit-learn, NumPy, Pandas, Gym (OpenAI)

## 📊 Hasil Eksperimen
Sistem mampu beradaptasi dengan fluktuasi trafik dan mengurangi rata-rata waktu tunggu kendaraan dibandingkan dengan kontroler *fixed-time* konvensional.

## 👥 Tim Pengembang (Kelompok 2)
1.  Achmad Nadhif Ma'ruf
2.  Dhio Rahmansyah
3.  Muhammad Irsyaddhia F.
4.  **Muhfi Fawwaz Rizqullah**

---
*Fakultas Ilmu Komputer - Universitas Brawijaya (2025)*
