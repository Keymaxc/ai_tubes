
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="AI-Based BI UMKM F&B",
    page_icon="BI",
    layout="wide"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "data"

REQUIRED_FILES = [
    "cleaned_transactions.csv",
    "daily_item_sales.csv",
    "prediction_results.csv",
    "prediction_results_detail.csv",
    "stock_recommendations.csv",
    "model_comparison.csv",
    "metadata.json",
]

missing_files = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).exists()]
if missing_files:
    st.error("File data dashboard belum lengkap. Pastikan folder data/ ikut diunggah bersama app.py.")
    st.write("File yang belum ditemukan:", missing_files)
    st.stop()


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    cleaned = pd.read_csv(OUTPUT_DIR / "cleaned_transactions.csv")
    daily_items = pd.read_csv(OUTPUT_DIR / "daily_item_sales.csv")
    pred = pd.read_csv(OUTPUT_DIR / "prediction_results.csv")
    pred_detail = pd.read_csv(OUTPUT_DIR / "prediction_results_detail.csv")
    rec = pd.read_csv(OUTPUT_DIR / "stock_recommendations.csv")
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

    with open(OUTPUT_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    insight_path = OUTPUT_DIR / "llm_insight.txt"
    insight = insight_path.read_text(encoding="utf-8") if insight_path.exists() else "Insight belum tersedia."

    for d in [cleaned, daily_items, pred, pred_detail, rec]:
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")

    return cleaned, daily_items, pred, pred_detail, rec, comparison, metadata, insight


# =========================
# FORMAT TABEL REKOMENDASI USER-FRIENDLY
# =========================

def format_recommendation_table(df):
    if df.empty:
        return df

    table = df.copy()

    rename_cols = {
        "date": "Tanggal Rekomendasi",
        "outlet": "Outlet",
        "category": "Kategori",
        "product": "Produk/Menu",
        "variant": "Varian",
        "predicted_demand": "Perkiraan Terjual Besok (Item)",
        "estimated_current_stock": "Estimasi Stok Saat Ini",
        "safety_stock": "Stok Cadangan",
        "recommended_stock": "Stok yang Disarankan",
        "restock_qty": "Jumlah yang Perlu Ditambah",
        "status": "Status Stok",
        "recommended_action": "Saran Tindakan"
    }

    table = table.rename(columns=rename_cols)

    numeric_round_cols = [
        "Perkiraan Terjual Besok (Item)",
        "Estimasi Stok Saat Ini",
        "Stok Cadangan",
        "Stok yang Disarankan",
        "Jumlah yang Perlu Ditambah"
    ]

    for col in numeric_round_cols:
        if col in table.columns:
            table[col] = (
                pd.to_numeric(table[col], errors="coerce")
                .fillna(0)
                .round(0)
                .astype(int)
            )

    if "Tanggal Rekomendasi" in table.columns:
        table["Tanggal Rekomendasi"] = pd.to_datetime(
            table["Tanggal Rekomendasi"],
            errors="coerce"
        ).dt.strftime("%d %b %Y")

    wanted_cols = [
        "Tanggal Rekomendasi",
        "Outlet",
        "Kategori",
        "Produk/Menu",
        "Varian",
        "Perkiraan Terjual Besok (Item)",
        "Estimasi Stok Saat Ini",
        "Stok Cadangan",
        "Stok yang Disarankan",
        "Jumlah yang Perlu Ditambah",
        "Status Stok",
        "Saran Tindakan"
    ]

    existing_cols = [c for c in wanted_cols if c in table.columns]
    return table[existing_cols]


# =========================
# FORMAT DETAIL PREDIKSI USER-FRIENDLY
# =========================

def format_prediction_detail_table(df):
    if df.empty:
        return df

    table = df.copy()

    table = table.rename(columns={
        "date": "Tanggal",
        "outlet": "Outlet",
        "product": "Produk/Menu",
        "variant": "Varian",
        "category": "Kategori",
        "actual_demand": "Aktual Terjual",
        "predicted_demand": "Prediksi Terjual",
        "abs_error": "Selisih Prediksi"
    })

    for col in ["Aktual Terjual", "Prediksi Terjual", "Selisih Prediksi"]:
        if col in table.columns:
            table[col] = (
                pd.to_numeric(table[col], errors="coerce")
                .fillna(0)
                .round(0)
                .astype(int)
            )

    if "Tanggal" in table.columns:
        table["Tanggal"] = pd.to_datetime(
            table["Tanggal"],
            errors="coerce"
        ).dt.strftime("%d %b %Y")

    wanted_cols = [
        "Tanggal",
        "Outlet",
        "Produk/Menu",
        "Varian",
        "Kategori",
        "Aktual Terjual",
        "Prediksi Terjual",
        "Selisih Prediksi"
    ]

    existing_cols = [c for c in wanted_cols if c in table.columns]
    return table[existing_cols]


# =========================
# FORMAT PERBANDINGAN MODEL USER-FRIENDLY
# =========================

def format_model_comparison_table(df):
    if df.empty:
        return df

    table = df.copy()

    table = table.rename(columns={
        "model_name": "Nama Model",
        "item_mae": "MAE Item-Level",
        "item_rmse": "RMSE Item-Level",
        "item_r2": "R2 Item-Level",
        "item_mae_mean_ratio_pct": "Error Item-Level (%)",
        "daily_mae": "MAE Harian",
        "daily_rmse": "RMSE Harian",
        "daily_r2": "R2 Harian",
        "daily_mae_mean_ratio_pct": "Error Harian (%)"
    })

    numeric_cols = [
        "MAE Item-Level",
        "RMSE Item-Level",
        "R2 Item-Level",
        "Error Item-Level (%)",
        "MAE Harian",
        "RMSE Harian",
        "R2 Harian",
        "Error Harian (%)"
    ]

    for col in numeric_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(4)

    wanted_cols = [
        "Nama Model",
        "MAE Item-Level",
        "RMSE Item-Level",
        "R2 Item-Level",
        "Error Item-Level (%)",
        "MAE Harian",
        "RMSE Harian",
        "R2 Harian",
        "Error Harian (%)"
    ]

    existing_cols = [c for c in wanted_cols if c in table.columns]
    return table[existing_cols]




# =========================
# EVALUASI SCORE MODEL
# =========================

def compute_regression_metrics(df, actual_col="actual_demand", pred_col="predicted_demand"):
    """Hitung MAE, RMSE, R2, dan MAE/Mean Ratio tanpa dependency sklearn."""
    if df is None or df.empty or actual_col not in df.columns or pred_col not in df.columns:
        return {
            "Jumlah Data": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAE/Mean (%)": np.nan,
            "Rata-rata Aktual": np.nan,
            "Rata-rata Prediksi": np.nan,
        }

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    valid = actual.notna() & pred.notna()
    actual = actual[valid]
    pred = pred[valid]

    if len(actual) == 0:
        return {
            "Jumlah Data": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAE/Mean (%)": np.nan,
            "Rata-rata Aktual": np.nan,
            "Rata-rata Prediksi": np.nan,
        }

    error = actual - pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_actual = float(np.mean(actual))
    mean_pred = float(np.mean(pred))

    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - mean_actual) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    ratio = (mae / mean_actual * 100) if mean_actual != 0 else np.nan

    return {
        "Jumlah Data": int(len(actual)),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAE/Mean (%)": ratio,
        "Rata-rata Aktual": mean_actual,
        "Rata-rata Prediksi": mean_pred,
    }


def metrics_to_dataframe(metrics_dict):
    rows = []
    for level, metrics in metrics_dict.items():
        row = {"Level Evaluasi": level}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric_cols = ["MAE", "RMSE", "R2", "MAE/Mean (%)", "Rata-rata Aktual", "Rata-rata Prediksi"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


def interpret_mae_ratio(ratio):
    if pd.isna(ratio):
        return "Belum dapat dihitung karena data aktual/prediksi tidak lengkap."
    if ratio <= 10:
        return "Sangat baik: rata-rata error relatif kecil dibandingkan demand aktual."
    if ratio <= 20:
        return "Baik: error masih cukup terkendali untuk kebutuhan dashboard prototype."
    if ratio <= 40:
        return "Cukup: model masih bisa memberi gambaran, tetapi perlu peningkatan untuk keputusan operasional."
    return "Perlu ditingkatkan: error relatif besar, sehingga prediksi sebaiknya hanya dipakai sebagai estimasi awal."


def build_filtered_score_by_group(df, group_cols):
    if df is None or df.empty:
        return pd.DataFrame()
    existing_groups = [c for c in group_cols if c in df.columns]
    if not existing_groups:
        return pd.DataFrame()

    rows = []
    for keys, part in df.groupby(existing_groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(existing_groups, keys))
        row.update(compute_regression_metrics(part))
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for col in ["MAE", "RMSE", "R2", "MAE/Mean (%)", "Rata-rata Aktual", "Rata-rata Prediksi"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(4)
    return result.sort_values("MAE/Mean (%)", ascending=True, na_position="last")


# =========================
# RAG-LITE CONTEXT RETRIEVER
# =========================

def build_rag_context(question, filtered_items, filtered_rec, filtered_pred, metadata):
    """
    RAG-lite:
    Mengambil konteks relevan dari dataframe dashboard.
    Ini bukan vector database, tetapi retrieval berbasis keyword + ringkasan tabel.
    """

    q = question.lower()
    context_parts = []

    total_qty = filtered_items["quantity_sold"].sum() if len(filtered_items) else 0
    total_revenue = filtered_items["revenue"].sum() if len(filtered_items) else 0
    total_outlet = filtered_items["outlet"].nunique() if len(filtered_items) else 0
    total_product = filtered_items["product"].nunique() if len(filtered_items) else 0

    context_parts.append(f"""
Ringkasan KPI berdasarkan filter aktif:
- Total item terjual: {total_qty:,.0f}
- Total penjualan: Rp {total_revenue:,.0f}
- Jumlah outlet aktif: {total_outlet}
- Jumlah produk/menu: {total_product}
""")

    context_parts.append(f"""
Metrik model prediksi:
- Model: {metadata.get("best_model", "-")}
- MAE harian: {metadata.get("daily_mae", 0):.2f}
- RMSE harian: {metadata.get("daily_rmse", 0):.2f}
- R2 harian: {metadata.get("daily_r2", 0):.4f}
- MAE/Mean harian: {metadata.get("daily_mae_mean_ratio_pct", 0):.2f}%
""")

    if any(k in q for k in ["stok", "restock", "produksi", "tambah", "aman", "overstock", "rekomendasi"]):
        if len(filtered_rec):
            status_summary = (
                filtered_rec
                .groupby("status", as_index=False)
                .agg(jumlah_produk=("product", "count"))
                .sort_values("jumlah_produk", ascending=False)
            )

            top_restock = (
                filtered_rec[filtered_rec["status"] == "Restock"]
                .sort_values("predicted_demand", ascending=False)
                .head(10)
            )

            context_parts.append("Ringkasan status rekomendasi stok:")
            context_parts.append(status_summary.to_string(index=False))

            if len(top_restock):
                context_parts.append("Top produk/menu prioritas Restock:")
                context_parts.append(
                    top_restock[
                        [
                            "outlet",
                            "category",
                            "product",
                            "variant",
                            "predicted_demand",
                            "recommended_stock",
                            "restock_qty",
                            "status",
                            "recommended_action"
                        ]
                    ].to_string(index=False)
                )
            else:
                context_parts.append("Tidak ada produk/menu berstatus Restock pada filter aktif.")
        else:
            context_parts.append("Tidak ada data rekomendasi stok pada filter aktif.")

    if any(k in q for k in ["produk", "menu", "terlaris", "paling laris", "laris", "kategori"]):
        if len(filtered_items):
            top_products = (
                filtered_items
                .groupby(["product", "variant", "category"], as_index=False)
                .agg(
                    total_terjual=("quantity_sold", "sum"),
                    total_penjualan=("revenue", "sum")
                )
                .sort_values("total_terjual", ascending=False)
                .head(10)
            )

            top_categories = (
                filtered_items
                .groupby("category", as_index=False)
                .agg(
                    total_terjual=("quantity_sold", "sum"),
                    total_penjualan=("revenue", "sum")
                )
                .sort_values("total_penjualan", ascending=False)
                .head(10)
            )

            context_parts.append("Top produk/menu berdasarkan jumlah terjual:")
            context_parts.append(top_products.to_string(index=False))

            context_parts.append("Top kategori berdasarkan penjualan:")
            context_parts.append(top_categories.to_string(index=False))
        else:
            context_parts.append("Tidak ada data produk/menu pada filter aktif.")

    if any(k in q for k in ["outlet", "cabang", "toko", "performa"]):
        if len(filtered_items):
            by_outlet = (
                filtered_items
                .groupby("outlet", as_index=False)
                .agg(
                    total_terjual=("quantity_sold", "sum"),
                    total_penjualan=("revenue", "sum")
                )
                .sort_values("total_penjualan", ascending=False)
                .head(10)
            )

            context_parts.append("Performa outlet berdasarkan penjualan:")
            context_parts.append(by_outlet.to_string(index=False))
        else:
            context_parts.append("Tidak ada data outlet pada filter aktif.")

    if any(k in q for k in ["prediksi", "forecast", "forecasting", "akurasi", "model", "error", "mae", "rmse", "r2"]):
        if len(filtered_pred):
            pred_summary = filtered_pred.copy()
            pred_summary["abs_error"] = (
                pred_summary["actual_demand"] - pred_summary["predicted_demand"]
            ).abs()

            pred_context = {
                "rata_actual": float(pred_summary["actual_demand"].mean()),
                "rata_prediksi": float(pred_summary["predicted_demand"].mean()),
                "rata_abs_error": float(pred_summary["abs_error"].mean()),
                "max_actual": float(pred_summary["actual_demand"].max()),
                "max_prediksi": float(pred_summary["predicted_demand"].max()),
                "periode_awal": str(pred_summary["date"].min().date()),
                "periode_akhir": str(pred_summary["date"].max().date())
            }

            context_parts.append("Ringkasan prediksi pada filter aktif:")
            context_parts.append(json.dumps(pred_context, indent=2, ensure_ascii=False))

            context_parts.append("Contoh data aktual vs prediksi terbaru:")
            context_parts.append(pred_summary.tail(10).to_string(index=False))
        else:
            context_parts.append("Tidak ada data prediksi pada filter aktif.")

    if len(context_parts) <= 2:
        if len(filtered_items):
            top_products_general = (
                filtered_items
                .groupby(["product", "category"], as_index=False)
                .agg(
                    total_terjual=("quantity_sold", "sum"),
                    total_penjualan=("revenue", "sum")
                )
                .sort_values("total_terjual", ascending=False)
                .head(5)
            )
            context_parts.append("Top produk/menu umum:")
            context_parts.append(top_products_general.to_string(index=False))

        if len(filtered_rec):
            status_summary_general = (
                filtered_rec
                .groupby("status", as_index=False)
                .agg(jumlah_produk=("product", "count"))
            )
            context_parts.append("Ringkasan rekomendasi umum:")
            context_parts.append(status_summary_general.to_string(index=False))

    return "\n\n".join(context_parts)


# =========================
# FALLBACK RULE-BASED CHATBOT
# =========================

def generate_rule_based_answer(question, filtered_items, filtered_rec, filtered_pred, metadata):
    q = question.lower()

    total_qty = filtered_items["quantity_sold"].sum() if len(filtered_items) else 0
    total_revenue = filtered_items["revenue"].sum() if len(filtered_items) else 0

    restock_count = (filtered_rec["status"] == "Restock").sum() if len(filtered_rec) else 0
    aman_count = (filtered_rec["status"] == "Aman").sum() if len(filtered_rec) else 0
    overstock_count = (filtered_rec["status"] == "Overstock").sum() if len(filtered_rec) else 0

    if "restock" in q or "stok" in q or "produksi" in q:
        if len(filtered_rec) == 0:
            return "Belum ada data rekomendasi untuk filter yang dipilih."

        top_restock = (
            filtered_rec[filtered_rec["status"] == "Restock"]
            .sort_values("predicted_demand", ascending=False)
            .head(5)
        )

        if len(top_restock) == 0:
            return (
                f"Untuk filter saat ini, tidak ada item yang perlu Restock. "
                f"Ada {aman_count} item berstatus Aman dan {overstock_count} item berstatus Overstock."
            )

        items = []
        for _, row in top_restock.iterrows():
            items.append(
                f"- {row['product']} di {row['outlet']}: perkiraan terjual {row['predicted_demand']:.1f} item, "
                f"stok disarankan {row['recommended_stock']:.0f}, tambah sekitar {row['restock_qty']:.0f} item."
            )

        return (
            f"Berdasarkan filter saat ini, ada {restock_count} item yang perlu Restock. "
            f"Prioritas tertinggi:\n\n" + "\n".join(items)
        )

    if "model" in q or "prediksi" in q or "akurasi" in q or "error" in q:
        return (
            f"Model yang digunakan adalah {metadata.get('best_model', '-')}. "
            f"Pada evaluasi agregasi harian, MAE sebesar {metadata.get('daily_mae', 0):.2f}, "
            f"RMSE sebesar {metadata.get('daily_rmse', 0):.2f}, "
            f"R2 sebesar {metadata.get('daily_r2', 0):.4f}, dan "
            f"MAE/Mean sebesar {metadata.get('daily_mae_mean_ratio_pct', 0):.2f}%. "
            f"Artinya, rata-rata error prediksi harian sekitar "
            f"{metadata.get('daily_mae_mean_ratio_pct', 0):.2f}% dari rata-rata demand harian."
        )

    if "produk" in q or "menu" in q or "terlaris" in q or "laris" in q:
        if len(filtered_items) == 0:
            return "Belum ada data penjualan untuk filter yang dipilih."

        top_products = (
            filtered_items
            .groupby(["product", "category"], as_index=False)
            .agg(total_terjual=("quantity_sold", "sum"), total_penjualan=("revenue", "sum"))
            .sort_values("total_terjual", ascending=False)
            .head(5)
        )

        lines = []
        for _, row in top_products.iterrows():
            lines.append(f"- {row['product']} ({row['category']}): terjual {row['total_terjual']:.0f} item.")

        return "Produk/menu terlaris berdasarkan filter saat ini:\n\n" + "\n".join(lines)

    if "outlet" in q or "cabang" in q:
        if len(filtered_items) == 0:
            return "Belum ada data outlet untuk filter yang dipilih."

        by_outlet = (
            filtered_items
            .groupby("outlet", as_index=False)
            .agg(total_terjual=("quantity_sold", "sum"), total_penjualan=("revenue", "sum"))
            .sort_values("total_penjualan", ascending=False)
            .head(5)
        )

        lines = []
        for _, row in by_outlet.iterrows():
            lines.append(
                f"- {row['outlet']}: penjualan Rp {row['total_penjualan']:,.0f}, total item {row['total_terjual']:.0f}."
            )

        return "Performa outlet berdasarkan filter saat ini:\n\n" + "\n".join(lines)

    return (
        f"Ringkasan filter saat ini: total item terjual {total_qty:,.0f}, "
        f"total penjualan Rp {total_revenue:,.0f}. "
        f"Status rekomendasi: {restock_count} Restock, {aman_count} Aman, dan {overstock_count} Overstock. "
        f"Kamu bisa tanya: 'produk apa yang perlu restock?', 'bagaimana akurasi model?', "
        f"'menu apa yang paling laris?', atau 'outlet mana paling bagus?'."
    )


# =========================
# GEMINI LLM ANSWER
# =========================

def generate_gemini_answer(question, rag_context):
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        raise ValueError("GEMINI_API_KEY belum tersedia di environment.")

    genai.configure(api_key=api_key)

    selected_model = os.environ.get("GEMINI_MODEL_NAME", "").strip()

    if not selected_model:
        try:
            selected_model = st.secrets.get("GEMINI_MODEL_NAME", "").strip()
        except Exception:
            selected_model = ""

    if not selected_model:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        preferred_models = [
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-1.5-flash"
        ]

        selected_model = next((m for m in preferred_models if m in available_models), None)

    if selected_model is None:
        raise ValueError("Tidak ada model Gemini yang tersedia untuk generateContent.")

    model = genai.GenerativeModel(selected_model)

    prompt = f"""
Anda adalah chatbot analisis Business Intelligence untuk dashboard UMKM F&B multi-outlet.

Gunakan konteks RAG-lite di bawah ini untuk menjawab pertanyaan user.
Jawab dengan bahasa Indonesia yang mudah dipahami pemilik UMKM.
Jangan terlalu teknis.
Kalau konteks tidak cukup, katakan data tidak tersedia pada filter aktif.
Jangan mengarang angka di luar konteks.

Konteks RAG-lite:
{rag_context}

Pertanyaan user:
{question}

Jawaban:
"""

    response = model.generate_content(prompt)
    return response.text


# =========================
# LOAD OUTPUT
# =========================

try:
    cleaned, daily_items, pred, pred_detail, rec, comparison, metadata, insight = load_data()
except Exception as e:
    st.error(f"File output belum lengkap. Jalankan notebook sampai cell export. Error: {e}")
    st.stop()


# =========================
# HEADER
# =========================

st.title("AI-Based Business Intelligence UMKM F&B Multi-Outlet")
st.write(
    "Dashboard untuk analisis penjualan, prediksi demand, rekomendasi stok/produksi, "
    "insight otomatis, dan chatbot analisis berbasis RAG-lite."
)


# =========================
# SIDEBAR FILTER
# =========================

st.sidebar.header("Filter Dashboard")

outlet_options = sorted(daily_items["outlet"].dropna().unique().tolist())
category_options = sorted(daily_items["category"].dropna().unique().tolist())
product_options = sorted(daily_items["product"].dropna().unique().tolist())
status_options = sorted(rec["status"].dropna().unique().tolist())

selected_outlet = st.sidebar.multiselect("Outlet", outlet_options, default=outlet_options)
selected_category = st.sidebar.multiselect("Kategori", category_options, default=category_options)

top_default_products = (
    daily_items
    .groupby("product")["quantity_sold"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index
    .tolist()
)

selected_product = st.sidebar.multiselect(
    "Produk/Menu",
    product_options,
    default=[p for p in top_default_products if p in product_options]
)

selected_status = st.sidebar.multiselect("Status Rekomendasi", status_options, default=status_options)

filtered_items = daily_items[
    daily_items["outlet"].isin(selected_outlet)
    & daily_items["category"].isin(selected_category)
    & daily_items["product"].isin(selected_product)
].copy()

filtered_pred_detail = pred_detail[
    pred_detail["outlet"].isin(selected_outlet)
    & pred_detail["category"].isin(selected_category)
    & pred_detail["product"].isin(selected_product)
].copy()

filtered_rec = rec[
    rec["outlet"].isin(selected_outlet)
    & rec["category"].isin(selected_category)
    & rec["product"].isin(selected_product)
    & rec["status"].isin(selected_status)
].copy()

if len(filtered_pred_detail) > 0:
    filtered_pred = (
        filtered_pred_detail
        .groupby("date", as_index=False)
        .agg(
            actual_demand=("actual_demand", "sum"),
            predicted_demand=("predicted_demand", "sum")
        )
    )
else:
    filtered_pred = pred.copy()


# =========================
# KPI
# =========================

total_qty = filtered_items["quantity_sold"].sum() if len(filtered_items) else 0
total_revenue = filtered_items["revenue"].sum() if len(filtered_items) else 0
total_products = filtered_items["product"].nunique() if len(filtered_items) else 0
total_outlets = filtered_items["outlet"].nunique() if len(filtered_items) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Item Terjual", f"{total_qty:,.0f}")
c2.metric("Total Penjualan", f"Rp {total_revenue:,.0f}")
c3.metric("Outlet Aktif", total_outlets)
c4.metric("Produk/Menu", total_products)


# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Ringkasan Penjualan",
    "Prediksi Demand",
    "Evaluasi Score",
    "Rekomendasi Stok",
    "Insight Otomatis",
    "Chatbot Analisis",
    "Catatan Data"
])


# =========================
# TAB 1 RINGKASAN PENJUALAN
# =========================

with tab1:
    st.subheader("Ringkasan Penjualan")

    if len(filtered_items) == 0:
        st.warning("Tidak ada data untuk filter yang dipilih.")
    else:
        daily_sales = (
            filtered_items
            .groupby("date", as_index=False)
            .agg(quantity_sold=("quantity_sold", "sum"), revenue=("revenue", "sum"))
        )

        fig = px.line(
            daily_sales,
            x="date",
            y="quantity_sold",
            title="Tren Item Terjual Harian"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            by_outlet = (
                filtered_items
                .groupby("outlet", as_index=False)
                .agg(quantity_sold=("quantity_sold", "sum"), revenue=("revenue", "sum"))
                .sort_values("revenue", ascending=False)
            )
            fig_outlet = px.bar(
                by_outlet,
                x="outlet",
                y="revenue",
                title="Penjualan per Outlet"
            )
            st.plotly_chart(fig_outlet, use_container_width=True)

        with col2:
            by_category = (
                filtered_items
                .groupby("category", as_index=False)
                .agg(quantity_sold=("quantity_sold", "sum"), revenue=("revenue", "sum"))
                .sort_values("revenue", ascending=False)
            )
            fig_category = px.bar(
                by_category,
                x="category",
                y="revenue",
                title="Penjualan per Kategori"
            )
            st.plotly_chart(fig_category, use_container_width=True)

        top_product = (
            filtered_items
            .groupby(["product", "variant", "category"], as_index=False)
            .agg(quantity_sold=("quantity_sold", "sum"), revenue=("revenue", "sum"))
            .sort_values("quantity_sold", ascending=False)
            .head(20)
        )

        top_product = top_product.rename(columns={
            "product": "Produk/Menu",
            "variant": "Varian",
            "category": "Kategori",
            "quantity_sold": "Total Terjual",
            "revenue": "Total Penjualan"
        })

        st.subheader("Produk/Menu Terlaris")
        st.dataframe(top_product, use_container_width=True, hide_index=True)


# =========================
# TAB 2 PREDIKSI DEMAND
# =========================

with tab2:
    st.subheader("Prediksi Demand")

    st.info(
        "Bagian ini menunjukkan perbandingan jumlah item yang benar-benar terjual dengan hasil prediksi model. "
        "Semakin dekat garis prediksi dengan aktual, semakin baik model membantu memperkirakan demand."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", metadata.get("best_model", "-"))
    m2.metric("Rata-rata Selisih Harian", f"{metadata.get('daily_mae', 0):.2f} item")
    m3.metric("RMSE Harian", f"{metadata.get('daily_rmse', 0):.2f}")
    m4.metric("Error Rata-rata", f"{metadata.get('daily_mae_mean_ratio_pct', 0):.2f}%")

    if len(filtered_pred) == 0:
        st.warning("Tidak ada data prediksi untuk filter yang dipilih.")
    else:
        pred_plot = filtered_pred.melt(
            id_vars="date",
            value_vars=["actual_demand", "predicted_demand"],
            var_name="Tipe",
            value_name="Demand"
        )

        pred_plot["Tipe"] = pred_plot["Tipe"].replace({
            "actual_demand": "Aktual",
            "predicted_demand": "Prediksi"
        })

        fig_pred = px.line(
            pred_plot,
            x="date",
            y="Demand",
            color="Tipe",
            title="Aktual vs Prediksi Demand"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.caption(
            f"Interpretasi: rata-rata error prediksi harian sekitar "
            f"{metadata.get('daily_mae_mean_ratio_pct', 0):.2f}% dari rata-rata demand harian."
        )

        with st.expander("Lihat perbandingan model"):
            st.caption("Bagian ini ditujukan untuk evaluasi teknis model.")
            comparison_user = format_model_comparison_table(comparison)
            st.dataframe(comparison_user, use_container_width=True, hide_index=True)

        with st.expander("Lihat detail prediksi sederhana"):
            st.caption("Tabel ini hanya menampilkan kolom utama agar mudah dipahami pengguna awam.")
            detail_user = format_prediction_detail_table(filtered_pred_detail)
            st.dataframe(detail_user, use_container_width=True, hide_index=True)

        with st.expander("Lihat detail prediksi teknis asli"):
            st.caption("Bagian ini ditujukan untuk dokumentasi teknis.")
            st.dataframe(filtered_pred_detail, use_container_width=True, hide_index=True)



# =========================
# TAB 3 EVALUASI SCORE
# =========================

with tab3:
    st.subheader("Evaluasi Score Model")

    st.info(
        "Bagian ini menampilkan skor evaluasi model agar performa prediksi mudah dinilai. "
        "MAE menunjukkan rata-rata selisih prediksi, RMSE memberi penalti lebih besar pada error besar, "
        "R2 menunjukkan kemampuan model menjelaskan variasi data, dan MAE/Mean Ratio menunjukkan persentase error terhadap rata-rata demand."
    )

    daily_metrics_filtered = compute_regression_metrics(filtered_pred)
    item_metrics_filtered = compute_regression_metrics(filtered_pred_detail)

    overall_daily_metrics = {
        "Jumlah Data": len(pred),
        "MAE": metadata.get("daily_mae", np.nan),
        "RMSE": metadata.get("daily_rmse", np.nan),
        "R2": metadata.get("daily_r2", np.nan),
        "MAE/Mean (%)": metadata.get("daily_mae_mean_ratio_pct", np.nan),
        "Rata-rata Aktual": metadata.get("daily_actual_mean", np.nan),
        "Rata-rata Prediksi": metadata.get("daily_predicted_mean", np.nan),
    }

    score_table = metrics_to_dataframe({
        "Overall Harian": overall_daily_metrics,
        "Filter Aktif Harian": daily_metrics_filtered,
        "Filter Aktif Item-Level": item_metrics_filtered,
    })

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE Harian", f"{daily_metrics_filtered['MAE']:.2f}" if pd.notna(daily_metrics_filtered['MAE']) else "-")
    k2.metric("RMSE Harian", f"{daily_metrics_filtered['RMSE']:.2f}" if pd.notna(daily_metrics_filtered['RMSE']) else "-")
    k3.metric("R2 Harian", f"{daily_metrics_filtered['R2']:.4f}" if pd.notna(daily_metrics_filtered['R2']) else "-")
    k4.metric("MAE/Mean Harian", f"{daily_metrics_filtered['MAE/Mean (%)']:.2f}%" if pd.notna(daily_metrics_filtered['MAE/Mean (%)']) else "-")

    st.markdown("**Interpretasi filter aktif:**")
    st.write(interpret_mae_ratio(daily_metrics_filtered["MAE/Mean (%)"]))

    st.subheader("Tabel Ringkasan Score")
    st.dataframe(score_table, use_container_width=True, hide_index=True)

    plot_score = score_table.melt(
        id_vars="Level Evaluasi",
        value_vars=[c for c in ["MAE", "RMSE", "MAE/Mean (%)"] if c in score_table.columns],
        var_name="Metrik",
        value_name="Nilai"
    ).dropna()

    if len(plot_score):
        fig_score = px.bar(
            plot_score,
            x="Metrik",
            y="Nilai",
            color="Level Evaluasi",
            barmode="group",
            title="Perbandingan MAE, RMSE, dan MAE/Mean Ratio"
        )
        st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Evaluasi per Outlet")
    outlet_score = build_filtered_score_by_group(filtered_pred_detail, ["outlet"])
    if len(outlet_score):
        st.dataframe(outlet_score, use_container_width=True, hide_index=True)
    else:
        st.warning("Evaluasi per outlet belum tersedia untuk filter ini.")

    st.subheader("Evaluasi per Kategori")
    category_score = build_filtered_score_by_group(filtered_pred_detail, ["category"])
    if len(category_score):
        st.dataframe(category_score, use_container_width=True, hide_index=True)
    else:
        st.warning("Evaluasi per kategori belum tersedia untuk filter ini.")

    with st.expander("Lihat perbandingan model dari notebook"):
        comparison_user = format_model_comparison_table(comparison)
        st.dataframe(comparison_user, use_container_width=True, hide_index=True)

    with st.expander("Catatan cara membaca score"):
        st.markdown("""
        - **MAE**: rata-rata selisih absolut antara aktual dan prediksi. Semakin kecil semakin baik.
        - **RMSE**: mirip MAE, tetapi lebih sensitif terhadap error besar. Semakin kecil semakin baik.
        - **R2**: semakin mendekati 1 semakin baik. Nilai negatif berarti model belum lebih baik dari prediksi rata-rata.
        - **MAE/Mean Ratio**: persentase error terhadap rata-rata demand. Untuk dashboard prototype, nilai di bawah 20% biasanya lebih mudah diterima.
        """)


# =========================
# TAB 4 REKOMENDASI STOK
# =========================

with tab4:
    st.subheader("Rekomendasi Stok/Produksi")

    st.info(
        "Tabel ini membantu menentukan menu mana yang perlu ditambah stok/produksinya. "
        "Karena dataset tidak memiliki stok aktual, nilai stok di sini adalah estimasi dari pola penjualan historis."
    )

    restock_count = int((filtered_rec["status"] == "Restock").sum()) if len(filtered_rec) else 0
    aman_count = int((filtered_rec["status"] == "Aman").sum()) if len(filtered_rec) else 0
    overstock_count = int((filtered_rec["status"] == "Overstock").sum()) if len(filtered_rec) else 0

    r1, r2, r3 = st.columns(3)
    r1.metric("Perlu Ditambah", restock_count)
    r2.metric("Stok Aman", aman_count)
    r3.metric("Potensi Berlebih", overstock_count)

    st.markdown("""
    **Cara membaca status:**

    - **Restock**: stok/produksi perlu ditambah.
    - **Aman**: stok relatif cukup.
    - **Overstock**: produksi/pembelian sebaiknya dikurangi agar tidak berlebih.
    """)

    if len(filtered_rec) == 0:
        st.warning("Tidak ada rekomendasi untuk filter yang dipilih.")
    else:
        status_chart = (
            filtered_rec
            .groupby("status", as_index=False)
            .agg(total_item=("product", "count"))
        )

        status_chart = status_chart.rename(columns={
            "status": "Status Stok",
            "total_item": "Jumlah Produk/Menu"
        })

        fig_status = px.bar(
            status_chart,
            x="Status Stok",
            y="Jumlah Produk/Menu",
            title="Jumlah Produk/Menu per Status Stok"
        )
        st.plotly_chart(fig_status, use_container_width=True)

        user_friendly_rec = format_recommendation_table(filtered_rec)

        st.subheader("Daftar Rekomendasi yang Mudah Dibaca")
        st.dataframe(user_friendly_rec, use_container_width=True, hide_index=True)

        with st.expander("Lihat tabel teknis asli"):
            st.caption("Bagian ini ditujukan untuk dokumentasi teknis.")
            st.dataframe(filtered_rec, use_container_width=True, hide_index=True)


# =========================
# TAB 4 INSIGHT OTOMATIS
# =========================

with tab5:
    st.subheader("Insight Otomatis")
    st.markdown(insight)


# =========================
# TAB 5 CHATBOT RAG-LITE
# =========================

with tab6:
    st.subheader("Chatbot Analisis Cepat Berbasis RAG-lite")

    st.write(
        "Chatbot ini membaca konteks dari data dashboard yang sedang difilter, "
        "lalu menjawab pertanyaan menggunakan Gemini. Jika Gemini error atau quota habis, "
        "sistem akan memakai jawaban otomatis berbasis aturan."
    )

    st.markdown("""
    Contoh pertanyaan:
    - Produk apa yang perlu restock?
    - Bagaimana akurasi model prediksi?
    - Menu apa yang paling laris?
    - Outlet mana yang performanya paling bagus?
    - Apa rekomendasi keputusan untuk pemilik UMKM?
    """)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Halo, saya bisa bantu analisis dashboard ini. Coba tanya: produk apa yang perlu restock?"
            }
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Tulis pertanyaan analisis di sini...")

    if user_question:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Mengambil konteks RAG-lite dan menganalisis data..."):
                rag_context = build_rag_context(
                    user_question,
                    filtered_items,
                    filtered_rec,
                    filtered_pred,
                    metadata
                )

                try:
                    answer = generate_gemini_answer(user_question, rag_context)
                except Exception:
                    answer = generate_rule_based_answer(
                        user_question,
                        filtered_items,
                        filtered_rec,
                        filtered_pred,
                        metadata
                    )
                    answer += "\n\nCatatan: Gemini tidak digunakan saat ini, sistem memakai analisis otomatis berbasis aturan."

                st.markdown(answer)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer
        })

    with st.expander("Lihat konteks RAG-lite terakhir berdasarkan filter"):
        sample_context = build_rag_context(
            "ringkasan umum",
            filtered_items,
            filtered_rec,
            filtered_pred,
            metadata
        )
        st.text(sample_context[:5000])


# =========================
# TAB 6 CATATAN DATA
# =========================

with tab7:
    st.subheader("Catatan Data dan Keterbatasan")

    st.markdown("""
    Dashboard ini sesuai dengan konsep AI-Based Business Intelligence karena mencakup:

    - **Descriptive BI**: ringkasan penjualan historis.
    - **Predictive Analytics**: prediksi demand menggunakan model machine learning.
    - **Prescriptive Analytics**: rekomendasi stok/produksi berdasarkan hasil prediksi.
    - **RAG-lite**: pengambilan konteks relevan dari data dashboard untuk chatbot.
    - **LLM Insight**: insight otomatis dan chatbot analisis cepat.

    ### Keterbatasan Dataset

    Dataset hanya memiliki transaksi penjualan dan metadata produk. Dataset belum memiliki:

    - stok aktual,
    - promo,
    - hari libur atau event,
    - biaya bahan baku,
    - data cuaca,
    - kapasitas produksi.

    Karena itu, rekomendasi stok pada dashboard ini bersifat **estimasi prototype**, bukan keputusan operasional final.

    ### Interpretasi untuk UMKM

    Dashboard ini dapat membantu pemilik usaha melihat pola penjualan, memperkirakan demand, dan menentukan prioritas produksi. Namun, keputusan akhir tetap perlu mempertimbangkan kondisi lapangan seperti promo, event, bahan baku, dan kapasitas outlet.
    """)

