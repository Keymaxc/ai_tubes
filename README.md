# AI-Based Business Intelligence UMKM F&B

Aplikasi Streamlit untuk dashboard AI-Based Business Intelligence UMKM F&B multi-outlet.

Dashboard berisi:
- ringkasan penjualan,
- prediksi demand,
- evaluasi score model,
- rekomendasi stok/produksi,
- insight otomatis,
- chatbot analisis berbasis Gemini dengan fallback rule-based.

## Struktur file untuk GitHub

Upload semua file/folder ini ke root repository GitHub:

```text
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
data/
```

File `app.py` harus berada langsung di root repo, bukan di dalam folder lain.

## File data yang wajib ada di folder data/

Dashboard membaca hasil output notebook dari folder `data/`:

```text
data/cleaned_transactions.csv
data/daily_item_sales.csv
data/prediction_results.csv
data/prediction_results_detail.csv
data/stock_recommendations.csv
data/model_comparison.csv
data/metadata.json
data/llm_insight.txt
```

Jika file data belum ada, jalankan notebook sampai bagian export output dashboard, lalu salin semua file output ke folder `data/`.

## Cara deploy ke Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Upload isi folder ini ke repository tersebut.
3. Buka Streamlit Community Cloud.
4. Login memakai GitHub.
5. Pilih **Create app**.
6. Pilih repository, branch `main`, dan main file `app.py`.
7. Klik **Deploy**.

## Cara menambahkan Gemini API Key

Jangan menulis API key langsung di `app.py` atau repository GitHub.

Di Streamlit Community Cloud:

1. Buka app yang sudah dibuat.
2. Masuk ke **Settings**.
3. Pilih **Secrets**.
4. Isi dengan format TOML berikut:

```toml
GEMINI_API_KEY = "isi_api_key_gemini_kamu"
GEMINI_MODEL_NAME = "models/gemini-2.0-flash"
```

`GEMINI_MODEL_NAME` boleh dikosongkan. Jika kosong, aplikasi akan memilih model Gemini yang tersedia secara otomatis.

Jika API Gemini tidak diisi, dashboard tetap berjalan. Fitur chatbot akan memakai fallback jawaban otomatis berbasis aturan.

## Catatan keamanan

File `.streamlit/secrets.toml` tidak boleh di-upload ke GitHub. File tersebut sudah dimasukkan ke `.gitignore`.
