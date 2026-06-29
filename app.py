import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import requests
import warnings
import os
import json
import geopandas as gpd
from openai import OpenAI

warnings.filterwarnings('ignore')

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="AI Prediksi Harga Cabai Nasional", layout="wide", page_icon="ðŸŒ¶ï¸")
st.title("Sistem Prediksi Harga Cabai Mingguan & Konsultan AI")

# --- 1A. LOAD GEOJSON DENGAN GEOPANDAS + PERBAIKAN TOPOLOGI ---
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    gdf = gpd.read_file(url)

    def bersihkan_nama(nama):
        if not nama: return "UNKNOWN"
        nama = str(nama).upper().strip()
        kamus = {
            'JAKARTA RAYA': 'DKI JAKARTA',
            'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            'BANGKA BELITUNG': 'KEP. BANGKA BELITUNG',
            'KEPULAUAN BANGKA BELITUNG': 'KEP. BANGKA BELITUNG',
            'KEPULAUAN RIAU': 'KEP. RIAU',
            'SUMATRA UTARA': 'SUMATERA UTARA',
            'SUMATRA BARAT': 'SUMATERA BARAT',
            'SUMATRA SELATAN': 'SUMATERA SELATAN',
            'DI. ACEH': 'ACEH',
            'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
            'NUSATENGGARA TIMUR': 'NUSA TENGGARA TIMUR'
        }
        return kamus.get(nama, nama)

    gdf['Propinsi'] = gdf['Propinsi'].apply(bersihkan_nama)
    gdf['geometry'] = gdf['geometry'].make_valid()
    gdf = gdf.dissolve(by='Propinsi').reset_index()
    return json.loads(gdf.to_json())

# --- 1B. LOAD DATA & IMPUTASI GRID WAKTU ---
@st.cache_data
def load_and_preprocess_data():
    prefix = 'sample_data/' if os.path.exists('sample_data/Cabai Rawit Merah_Konsumen_Harian.csv') else ''

    df_cabai = pd.read_csv(prefix + 'Cabai Rawit Merah_Konsumen_Harian.csv')
    df_hujan = pd.read_csv(prefix + 'Curah_Hujan_Bulanan.csv')
    df_hbkn = pd.read_csv(prefix + 'HBKN_Bulanan.csv')

    df_cabai['harga'] = df_cabai['harga'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df_cabai['harga'] = pd.to_numeric(df_cabai['harga'], errors='coerce')
    df_cabai = df_cabai.dropna(subset=['harga'])
    df_cabai['tanggal'] = pd.to_datetime(df_cabai['tanggal'])

    koreksi_papua = {
        'PAPUA BARAT DAYA': 'PAPUA BARAT',
        'PAPUA SELATAN': 'PAPUA',
        'PAPUA TENGAH': 'PAPUA',
        'PAPUA PEGUNUNGAN': 'PAPUA'
    }
    df_cabai['provinsi'] = df_cabai['provinsi'].astype(str).str.upper().str.strip().replace(koreksi_papua)

    df_cabai_weekly = df_cabai.groupby(['provinsi', pd.Grouper(key='tanggal', freq='W-MON')]).agg({'harga': 'mean'}).reset_index()

    all_provs = df_cabai_weekly['provinsi'].unique()
    all_dates = df_cabai_weekly['tanggal'].unique()
    idx = pd.MultiIndex.from_product([all_provs, all_dates], names=['provinsi', 'tanggal'])
    df_cabai_weekly = df_cabai_weekly.set_index(['provinsi', 'tanggal']).reindex(idx).reset_index()

    df_cabai_weekly = df_cabai_weekly.sort_values(['provinsi', 'tanggal'])
    df_cabai_weekly['harga'] = df_cabai_weekly.groupby('provinsi')['harga'].transform(lambda x: x.ffill().bfill())

    df_hbkn['tanggal'] = pd.to_datetime(df_hbkn['tanggal'])
    df_hbkn_weekly = df_hbkn.groupby(pd.Grouper(key='tanggal', freq='W-MON')).agg({
        'hbkn': 'max',
        'keterangan': lambda x: 'Normal' if all(x.astype(str).str.lower() == 'tidak ada') else x[x.astype(str).str.lower() != 'tidak ada'].iloc[0]
    }).reset_index()

    df_weekly = pd.merge(df_cabai_weekly, df_hbkn_weekly, on='tanggal', how='left')
    df_weekly['hbkn'] = df_weekly['hbkn'].fillna(0)
    df_weekly['keterangan'] = df_weekly['keterangan'].fillna('Normal')

    bulan_map = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
    df_hujan['Bulan'] = df_hujan['Bulan'].map(bulan_map)
    df_hujan['Nama Provinsi'] = df_hujan['Nama Provinsi'].astype(str).str.upper().str.strip().replace(koreksi_papua)

    df_hujan['Curah Hujan'] = df_hujan['Curah Hujan'].astype(str).str.replace(',', '.', regex=False)
    df_hujan['Curah Hujan'] = pd.to_numeric(df_hujan['Curah Hujan'], errors='coerce')

    df_hujan['Tahun'] = df_hujan['Tahun'].astype(int)
    df_hujan['Hari'] = 15
    df_hujan['tanggal'] = pd.to_datetime(df_hujan[['Tahun', 'Bulan', 'Hari']].rename(columns={'Tahun':'year', 'Bulan':'month', 'Hari':'day'}))
    df_hujan_proxy = df_hujan[['Nama Provinsi', 'tanggal', 'Curah Hujan']].rename(columns={'Nama Provinsi': 'provinsi'})

    df_combined = pd.concat([df_weekly, df_hujan_proxy], ignore_index=True)
    df_combined = df_combined.sort_values(['provinsi', 'tanggal']).reset_index(drop=True)

    df_combined['Curah_Hujan'] = df_combined.groupby('provinsi')['Curah Hujan'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

    df_final = df_combined.dropna(subset=['harga']).reset_index(drop=True)
    df_final = df_final.rename(columns={'harga': 'Harga_Riil', 'keterangan': 'Momen', 'provinsi': 'Provinsi', 'tanggal': 'Tanggal'})
    df_final = df_final.sort_values(['Provinsi', 'Tanggal']).reset_index(drop=True)

    df_final['Plafon_HET'] = 40000
    df_final['Deviasi_HET'] = df_final['Harga_Riil'] - df_final['Plafon_HET']
    df_final['Status_HET'] = np.where(df_final['Deviasi_HET'] > 0, 'Melanggar', 'Aman')

    df_final['Harga_Minggu_Lalu'] = df_final.groupby('Provinsi')['Harga_Riil'].shift(1).fillna(df_final['Harga_Riil'])
    df_final['Deviasi_Tren_Mingguan'] = df_final['Harga_Riil'] - df_final['Harga_Minggu_Lalu']

    df_final['Rata_Nasional'] = df_final.groupby('Tanggal')['Harga_Riil'].transform('mean')
    df_final['Deviasi_Nasional'] = df_final['Harga_Riil'] - df_final['Rata_Nasional']

    df_final['Tanggal_Str'] = df_final['Tanggal'].dt.strftime('%Y-%m-%d')
    return df_final

with st.spinner("Memproses penyatuan pulau spasial & waktu..."):
    df_all = load_and_preprocess_data()
    geojson_indo = load_geojson()

# --- 2. SIDEBAR FILTER ---
with st.sidebar:
    st.header("Konfigurasi Utama")
    api_key = st.text_input("OpenAI API Key (Opsional):", type="password")

    provinsi_list = sorted(df_all['Provinsi'].unique())
    default_prov = 'DKI JAKARTA' if 'DKI JAKARTA' in provinsi_list else provinsi_list[0]
    prov_terpilih = st.selectbox("Pilih Provinsi Fokus:", provinsi_list, index=provinsi_list.index(default_prov))
    status_filter = st.multiselect("Status HET:", ["Aman", "Melanggar"], default=["Aman", "Melanggar"])

    st.markdown("---")
    st.header("Filter Peta Nasional")
    list_minggu = sorted(df_all['Tanggal_Str'].unique(), reverse=True)
    minggu_peta = st.selectbox("Pilih Minggu Peta:", list_minggu)

    jenis_peta = st.selectbox("Tampilkan Metrik Peta Berdasarkan:", [
        "Deviasi thd HET Resmi",
        "Deviasi thd Rata-rata Nasional",
        "Deviasi thd Harga Minggu Lalu"
    ])

    df_filtered = df_all[(df_all['Provinsi'] == prov_terpilih) & (df_all['Status_HET'].isin(status_filter))]

if df_filtered.empty:
    st.warning("Data tidak ditemukan berdasarkan filter yang dipilih.")
    st.stop()

# --- 3. PIPELINE MACHINE LEARNING CYCLICAL ENCODING & FORECASTING ---
@st.cache_resource(show_spinner="Melatih AI Prediktif dengan Cyclical Encoding...")
def train_and_evaluate_models(df_prov):
    df_prov = df_prov.sort_values('Tanggal').reset_index(drop=True)
    df_ml = pd.get_dummies(df_prov[['Curah_Hujan', 'Momen']], drop_first=True)

    bulan_series = df_prov['Tanggal'].dt.month
    df_ml['Bulan_sin'] = np.sin(2 * np.pi * bulan_series / 12)
    df_ml['Bulan_cos'] = np.cos(2 * np.pi * bulan_series / 12)

    feat_names = df_ml.columns.tolist()
    X, y = df_ml.values, df_prov['Harga_Riil'].values

    if len(X) < 10: return None, None, None, df_prov, None, None, None
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)

    feat_imps = rf.feature_importances_
    df_feat_raw = pd.DataFrame({'Faktor': feat_names, 'Kepentingan': feat_imps})

    bobot_bulan = df_feat_raw[df_feat_raw['Faktor'].isin(['Bulan_sin', 'Bulan_cos'])]['Kepentingan'].sum()
    df_feat = df_feat_raw[~df_feat_raw['Faktor'].isin(['Bulan_sin', 'Bulan_cos'])].copy()
    df_feat = pd.concat([df_feat, pd.DataFrame({'Faktor': ['Siklus Musiman (Bulan)'], 'Kepentingan': [bobot_bulan]})])

    df_feat['Faktor'] = df_feat['Faktor'].str.replace('Momen_', 'Momen: ')
    df_feat = df_feat.sort_values('Kepentingan', ascending=False)

    rincian_features = ", ".join([f"{r['Faktor']} ({r['Kepentingan']*100:.2f}%)" for _, r in df_feat.iterrows()])

    gb = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    rf_pred, gb_pred = rf.predict(X_test), gb.predict(X_test)
    rf_mae, gb_mae = mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, gb_pred)

    future_steps = 8
    future_dates = [df_prov['Tanggal'].iloc[-1] + pd.Timedelta(days=7*i) for i in range(1, future_steps+1)]

    future_X_df = pd.DataFrame(index=range(future_steps), columns=feat_names)
    for col in feat_names:
        if col not in ['Bulan_sin', 'Bulan_cos']:
            future_X_df[col] = df_ml[col].iloc[-1]

    future_months = np.array([d.month for d in future_dates])
    future_X_df['Bulan_sin'] = np.sin(2 * np.pi * future_months / 12)
    future_X_df['Bulan_cos'] = np.cos(2 * np.pi * future_months / 12)

    rf_future = rf.predict(future_X_df.values).tolist()
    gb_future = gb.predict(future_X_df.values).tolist()

    scaler = MinMaxScaler()
    harga_scaled = scaler.fit_transform(y.reshape(-1, 1))
    lookback = 4
    X_seq, y_seq = [], []
    for i in range(len(harga_scaled) - lookback):
        X_seq.append(harga_scaled[i:(i + lookback)])
        y_seq.append(harga_scaled[i + lookback])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    split_idx_lstm = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_idx_lstm], X_seq[split_idx_lstm:]
    y_train_seq, y_test_seq = y_seq[:split_idx_lstm], y_seq[split_idx_lstm:]

    lstm_model = Sequential([LSTM(32, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2), GRU(16), Dense(1)])
    lstm_model.compile(optimizer='adam', loss='mse')

    lstm_future = []
    if len(X_train_seq) > 0:
        lstm_model.fit(X_train_seq, y_train_seq, epochs=15, batch_size=8, verbose=0)
        lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
        y_test_lstm = scaler.inverse_transform(y_test_seq).flatten()
        lstm_mae = mean_absolute_error(y_test_lstm, lstm_pred)

        if len(harga_scaled) >= lookback:
            curr_seq = harga_scaled[-lookback:].reshape(1, lookback, 1)
            for _ in range(future_steps):
                p = lstm_model.predict(curr_seq, verbose=0)
                lstm_future.append(p[0,0])
                curr_seq = np.append(curr_seq[:, 1:, :], [p], axis=1)
            lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1,1)).flatten().tolist()
    else:
        lstm_mae, lstm_pred, y_test_lstm = float('inf'), [], []
        lstm_future = [0]*future_steps

    models_eval = {
        'Random Forest': {'mae': rf_mae, 'pred': rf_pred, 'y_test': y_test, 'forecast': rf_future},
        'Gradient Boosting': {'mae': gb_mae, 'pred': gb_pred, 'y_test': y_test, 'forecast': gb_future}
    }
    if lstm_mae != float('inf'):
        models_eval['LSTM & GRU'] = {'mae': lstm_mae, 'pred': lstm_pred, 'y_test': y_test_lstm, 'forecast': lstm_future}

    best_model_name = min(models_eval, key=lambda k: models_eval[k]['mae'])
    return best_model_name, models_eval, split_idx, df_prov, rincian_features, df_feat, future_dates

# Run ML Pipeline
best_model_name, models_eval, split_idx, df_prov, rincian_features, df_feat, future_dates = train_and_evaluate_models(df_filtered)

if not best_model_name:
    st.error("Data terlalu sedikit untuk melakukan permodelan AI.")
    st.stop()

best_mae = models_eval[best_model_name]['mae']
best_pred = models_eval[best_model_name]['pred']
best_forecast = models_eval[best_model_name]['forecast']
test_dates = df_prov['Tanggal'].iloc[-len(best_pred):]

# --- 4. LAYOUT UTAMA DASHBOARD ---
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader(f"Analisis Spasio-Temporal: {prov_terpilih}")

    st.markdown(f"##### Visualisasi Peta ({minggu_peta})")
    df_map = df_all[df_all['Tanggal_Str'] == minggu_peta]

    if jenis_peta == "Deviasi thd HET Resmi":
        color_col, title_map = "Deviasi_HET", "Gradasi Merah = Melebihi HET 40rb"
    elif jenis_peta == "Deviasi thd Rata-rata Nasional":
        color_col, title_map = "Deviasi_Nasional", "Gradasi Merah = Harga Di Atas Rata-rata Nasional"
    else:
        color_col, title_map = "Deviasi_Tren_Mingguan", "Gradasi Merah = Harga Naik Tajam dari Minggu Lalu"

    if df_map.empty:
        st.info("Data spasial tidak tersedia untuk minggu ini.")
    else:
        fig_map = px.choropleth(df_map, geojson=geojson_indo, locations="Provinsi", featureidkey="properties.Propinsi", color=color_col, hover_name="Provinsi",
                                hover_data={"Harga_Riil": True, "Deviasi_HET": True, "Deviasi_Nasional": True, "Deviasi_Tren_Mingguan": True, "Provinsi": False},
                                color_continuous_scale="RdYlGn_r", title=title_map)
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown(f"##### Proyeksi & Forecasting 2 Bulan ({best_model_name})")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_prov['Tanggal'], y=df_prov['Harga_Riil'], mode='lines', name='Harga Riil', line=dict(color='blue')))
    fig_line.add_trace(go.Scatter(x=test_dates, y=best_pred, mode='lines', name='Validasi Model', line=dict(color='orange', dash='dash')))
    fig_line.add_trace(go.Scatter(x=[df_prov['Tanggal'].iloc[-1]] + future_dates, y=[df_prov['Harga_Riil'].iloc[-1]] + list(best_forecast),
                                  mode='lines+markers', name='Forecast 8 Minggu', line=dict(color='purple', width=3, dash='dot'), marker=dict(size=6)))
    fig_line.add_trace(go.Scatter(x=df_prov['Tanggal'], y=df_prov['Plafon_HET'], mode='lines', name='HET', line=dict(color='red', width=1)))
    fig_line.add_trace(go.Scatter(x=df_prov['Tanggal'], y=df_prov['Rata_Nasional'], mode='lines', name='Rata-Rata Nasional', line=dict(color='grey', width=1, dash='dot')))

    split_date = df_prov['Tanggal'].iloc[split_idx]
    fig_line.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="green")
    fig_line.add_annotation(x=split_date, y=0.95, yref="paper", text="Validasi AI", showarrow=False, xanchor="left", font=dict(color="green"))
    fig_line.update_layout(margin=dict(t=10, b=10), height=350)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("###### Tabel Komparasi Validasi Ilmiah Performa Model")
    metrics_list = []
    for model_name, model_data in models_eval.items():
        metrics_list.append({
            "Model Algoritma": model_name,
            "Mean Absolute Error (MAE)": model_data['mae'],
            "Margin Kesalahan (Error)": f"± Rp {model_data['mae']:,.0f}",
            "Status Seleksi": "ðŸ† TERBAIK (Dipilih Otomatis)" if model_name == best_model_name else "Alternatif"
        })
    df_metrics = pd.DataFrame(metrics_list).sort_values(by="Mean Absolute Error (MAE)").reset_index(drop=True)
    df_metrics["Mean Absolute Error (MAE)"] = df_metrics["Mean Absolute Error (MAE)"].map(lambda x: f"Rp {x:,.2f}")
    st.dataframe(df_metrics, use_container_width=True)

    col_ext1, col_ext2 = st.columns(2)
    with col_ext1:
        st.markdown(f"##### Curah Hujan & Momen")
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(x=df_prov['Tanggal'], y=df_prov['Curah_Hujan'], name='Curah Hujan (mm)', marker_color='lightblue'))
        df_hbkn = df_prov[df_prov['hbkn'] == 1]
        if not df_hbkn.empty:
            y_pos = df_prov['Curah_Hujan'].max() * 1.1
            fig_ext.add_trace(go.Scatter(x=df_hbkn['Tanggal'], y=[y_pos]*len(df_hbkn), mode='markers+text',
                                         name='Momen HBKN', marker=dict(color='red', size=10, symbol='star'),
                                         text=df_hbkn['Momen'], textposition='top center', textfont=dict(size=10)))
        fig_ext.update_layout(margin=dict(t=10, b=0), height=250, barmode='overlay')
        st.plotly_chart(fig_ext, use_container_width=True)

    with col_ext2:
        st.markdown(f"##### Tingkat Pengaruh Variabel (Feature Importance)")
        fig_feat = px.bar(df_feat, x='Kepentingan', y='Faktor', orientation='h', color='Kepentingan', color_continuous_scale='Blues')
        fig_feat.update_layout(margin=dict(l=0, r=20, t=10, b=0), height=250, coloraxis_showscale=False, xaxis_title="Bobot Pengaruh", yaxis_title="")
        st.plotly_chart(fig_feat, use_container_width=True)

# --- 5. PANEL KONSULTAN AI (CHAT INTERACTION) ---
with col2:
    col_judul, col_btn = st.columns([6, 4])
    with col_judul:
        st.subheader("ðŸ¤– Chat AI")
    with col_btn:
        if st.button("ðŸ—‘ï¸ Bersihkan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state: 
        st.session_state.messages = []

    # Tampilkan Histori Pesan
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # Tangkap Input Baru dari chat_input bawaan Streamlit
    prompt = st.chat_input("Tanya AI Konsultan...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if not api_key: 
                st.warning("Masukkan OpenAI API Key di Sidebar untuk berinteraksi.")
            else:
                try:
                    client = OpenAI(api_key=api_key)

                    deviasi_terakhir = df_prov['Deviasi_HET'].iloc[-1]
                    harga_terakhir = df_prov['Harga_Riil'].iloc[-1]
                    harga_minggu_lalu = df_prov['Harga_Minggu_Lalu'].iloc[-1]
                    rata_nasional = df_prov['Rata_Nasional'].iloc[-1]
                    deviasi_nasional = df_prov['Deviasi_Nasional'].iloc[-1]
                    curah_hujan_terakhir = df_prov['Curah_Hujan'].iloc[-1]
                    momen_terakhir = df_prov['Momen'].iloc[-1]

                    if harga_terakhir > harga_minggu_lalu: 
                        status_tren = "NAIK ðŸ“ˆ (Memburuk)"
                    elif harga_terakhir < harga_minggu_lalu: 
                        status_tren = "TURUN ðŸ“‰ (Membaik)"
                    else: 
                        status_tren = "STABIL –"
                    selisih_tren = abs(harga_terakhir - harga_minggu_lalu)

                    teks_forecast = ", ".join([f"Rp {p:,.0f}" for p in best_forecast])

                    system_prompt = f"""
                    Anda adalah Konsultan AI spesialis Ketahanan Pangan Nasional.
                    Konteks Mikro & Spasial:
                    1. Wilayah: {prov_terpilih}
                    2. Harga Aktual: Rp {harga_terakhir:,.0f} (Sedang {status_tren} sebesar Rp {selisih_tren:,.0f} dari minggu lalu).
                    3. Kepatuhan HET: {'Melanggar' if deviasi_terakhir > 0 else 'Aman'}, selisih Rp {abs(deviasi_terakhir):,.0f}
                    4. Deviasi Nasional: Harga berjarak Rp {deviasi_nasional:,.0f} dari tren rata-rata nasional.
                    5. Cuaca: Curah Hujan {curah_hujan_terakhir:.1f} mm. Momentum: "{momen_terakhir}".

                    Konteks Makro & FORECASTING JANGKA MENENGAH:
                    6. Prediksi AI ({best_model_name}): Memproyeksikan harga 2 BULAN (8 MINGGU) KE DEPAN secara berturut-turut akan menjadi: {teks_forecast}.
                    7. Ekstraksi Fitur Lengkap (Feature Importance): Rincian kontribusi pengaruh setiap variabel secara statistik di daerah ini adalah: {rincian_features}.

                    TUGAS: Analisis keterkaitan rincian bobot pengaruh statistik tersebut dengan tren proyeksi 2 bulan ke depan. Berikan rekomendasi mitigasi rantai pasok struktural yang spesifik dan taktis.
                    """
                    messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages

                    stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, stream=True)
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "â–Œ")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e: 
                    st.error(f"Error API: {str(e)}")
