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

st.set_page_config(page_title="AI Prediksi Harga Cabai Nasional", layout="wide", page_icon="🌶️")
st.title("🌶️ Sistem Prediksi Harga Cabai Mingguan & Konsultan AI")

# --- 1A. LOAD GEOJSON DENGAN GEOPANDAS ---
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    try:
        gdf = gpd.read_file(url)
    except Exception as e:
        st.error(f"Gagal memuat peta GeoJSON dari GitHub: {e}")
        st.stop()

    def bersihkan_nama(nama):
        if not nama: return "UNKNOWN"
        nama = str(nama).upper().strip()
        kamus = {
            'JAKARTA RAYA': 'DKI JAKARTA', 'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            'BANGKA BELITUNG': 'KEP. BANGKA BELITUNG', 'KEPULAUAN BANGKA BELITUNG': 'KEP. BANGKA BELITUNG',
            'KEPULAUAN RIAU': 'KEP. RIAU', 'SUMATRA UTARA': 'SUMATERA UTARA',
            'SUMATRA BARAT': 'SUMATERA BARAT', 'SUMATRA SELATAN': 'SUMATERA SELATAN',
            'DI. ACEH': 'ACEH', 'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
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
    # Menangani variasi path di server Streamlit Cloud
    base_dir = 'sample_data'
    
    file_cabai = os.path.join(base_dir, 'Cabai Rawit Merah_Konsumen_Harian.csv')
    file_hujan = os.path.join(base_dir, 'Curah_Hujan_Bulanan.csv')
    file_hbkn = os.path.join(base_dir, 'HBKN_Bulanan.csv')

    # Proteksi jika file tidak ditemukan
    if not os.path.exists(file_cabai) or not os.path.exists(file_hujan) or not os.path.exists(file_hbkn):
        st.error(f"Kritis: File data tidak ditemukan di folder 'sample_data/'. Silakan periksa kembali berkas Anda di GitHub.")
        st.stop()

    df_cabai = pd.read_csv(file_cabai)
    df_hujan = pd.read_csv(file_hujan)
    df_hbkn = pd.read_csv(file_hbkn)

    # Standardisasi nama kolom menjadi lowercase untuk menghindari KeyError
    df_cabai.columns = df_cabai.columns.str.lower().str.strip()
    df_hujan.columns = df_hujan.columns.str.strip() # Tetap jaga camel case untuk hujan
    df_hbkn.columns = df_hbkn.columns.str.lower().str.strip()

    # Cek keberadaan kolom utama cabai
    if 'harga' not in df_cabai.columns or 'tanggal' not in df_cabai.columns or 'provinsi' not in df_cabai.columns:
        st.error(f"Kolom pada file Cabai tidak sesuai. Ditemukan: {list(df_cabai.columns)}. Harus ada: 'provinsi', 'tanggal', 'harga'")
        st.stop()

    df_cabai['harga'] = df_cabai['harga'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df_cabai['harga'] = pd.to_numeric(df_cabai['harga'], errors='coerce')
    df_cabai = df_cabai.dropna(subset=['harga'])
    df_cabai['tanggal'] = pd.to_datetime(df_cabai['tanggal'], errors='coerce')
    df_cabai = df_cabai.dropna(subset=['tanggal'])

    koreksi_papua = {
        'PAPUA BARAT DAYA': 'PAPUA BARAT', 'PAPUA SELATAN': 'PAPUA',
        'PAPUA TENGAH': 'PAPUA', 'PAPUA PEGUNUNGAN': 'PAPUA'
    }
    df_cabai['provinsi'] = df_cabai['provinsi'].astype(str).str.upper().str.strip().replace(koreksi_papua)

    df_cabai_weekly = df_cabai.groupby(['provinsi', pd.Grouper(key='tanggal', freq='W-MON')]).agg({'harga': 'mean'}).reset_index()

    all_provs = df_cabai_weekly['provinsi'].unique()
    all_dates = df_cabai_weekly['tanggal'].unique()
    idx = pd.MultiIndex.from_product([all_provs, all_dates], names=['provinsi', 'tanggal'])
    df_cabai_weekly = df_cabai_weekly.set_index(['provinsi', 'tanggal']).reindex(idx).reset_index()
    df_cabai_weekly = df_cabai_weekly.sort_values(['provinsi', 'tanggal'])
    df_cabai_weekly['harga'] = df_cabai_weekly.groupby('provinsi')['harga'].transform(lambda x: x.ffill().bfill())

    df_hbkn['tanggal'] = pd.to_datetime(df_hbkn['tanggal'], errors='coerce')
    df_hbkn = df_hbkn.dropna(subset=['tanggal'])
    
    # Cek kolom HBKN
    if 'hbkn' not in df_hbkn.columns: 
        df_hbkn['hbkn'] = 0
    if 'keterangan' not in df_hbkn.columns:
        df_hbkn['keterangan'] = 'Normal'

    df_hbkn_weekly = df_hbkn.groupby(pd.Grouper(key='tanggal', freq='W-MON')).agg({
        'hbkn': 'max',
        'keterangan': lambda x: 'Normal' if all(x.astype(str).str.lower() == 'tidak ada') else x[x.astype(str).str.lower() != 'tidak ada'].iloc[0]
    }).reset_index()

    df_weekly = pd.merge(df_cabai_weekly, df_hbkn_weekly, on='tanggal', how='left')
    df_weekly['hbkn'] = df_weekly['hbkn'].fillna(0)
    df_weekly['keterangan'] = df_weekly['keterangan'].fillna('Normal')

    bulan_map = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
    
    # Cek kolom Hujan
    if 'Bulan' in df_hujan.columns:
        df_hujan['Bulan'] = df_hujan['Bulan'].map(bulan_map).fillna(1)
    else:
        df_hujan['Bulan'] = 1

    prov_col_hujan = 'Nama Provinsi' if 'Nama Provinsi' in df_hujan.columns else df_hujan.columns[0]
    df_hujan['Nama Provinsi'] = df_hujan[prov_col_hujan].astype(str).str.upper().str.strip().replace(koreksi_papua)

    rain_col = 'Curah Hujan' if 'Curah Hujan' in df_hujan.columns else df_hujan.columns[-1]
    df_hujan['Curah Hujan'] = df_hujan[rain_col].astype(str).str.replace(',', '.', regex=False)
    df_hujan['Curah Hujan'] = pd.to_numeric(df_hujan['Curah Hujan'], errors='coerce').fillna(0)

    df_hujan['Tahun'] = pd.to_numeric(df_hujan['Tahun'], errors='coerce').fillna(2024).astype(int)
    df_hujan['Hari'] = 15
    df_hujan['tanggal'] = pd.to_datetime(df_hujan[['Tahun', 'Bulan', 'Hari']].rename(columns={'Tahun':'year', 'Bulan':'month', 'Hari':'day'}), errors='coerce')
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
    st.header("⚙️ Konfigurasi Utama")
    api_key = st.text_input("OpenAI API Key (Opsional):", type="password")

    provinsi_list = sorted(df_all['Provinsi'].unique())
    default_prov = 'DKI JAKARTA' if 'DKI JAKARTA' in provinsi_list else provinsi_list[0]
    prov_terpilih = st.selectbox("Pilih Provinsi Fokus:", provinsi_list, index=provinsi_list.index(default_prov))
    status_filter = st.multiselect("Status HET:", ["Aman", "Melanggar"], default=["Aman", "Melanggar"])

    st.markdown("---")
    st.header("🗺️ Filter Peta Nasional")
    list_minggu = sorted(df_all['Tanggal_Str'].unique(), reverse=True)
    minggu_peta = st.selectbox("Pilih Minggu Peta:", list_minggu)

    jenis_peta = st.selectbox("Tampilkan Metrik Peta Berdasarkan:", [
        "Deviasi thd HET Resmi", "Deviasi thd Rata-rata Nasional", "Deviasi thd Harga Minggu Lalu"
    ])

    df_filtered = df_all[(df_all['Provinsi'] == prov_terpilih) & (df_all['Status_HET'].isin(status_filter))]

if df_filtered.empty:
    st.warning("Data tidak ditemukan berdasarkan filter.")
    st.stop()

# --- 3. PIPELINE MACHINE LEARNING ---
@st.cache_resource(show_spinner="Melatih AI Prediktif...")
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
    if len(X_train_seq := X_seq[:split_idx_lstm]) > 0:
        y_train_seq, X_test_seq, y_test_seq = y_seq[:split_idx_lstm], X_seq[split_idx_lstm:], y_seq[split_idx_lstm:]
        lstm_model = Sequential([LSTM(32, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2), GRU(16), Dense(1)])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train_seq, y_train_seq, epochs=15, batch_size=8, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test_seq, verbose=0)).flatten()
        lstm_mae = mean_absolute_error(scaler.inverse_transform(y_test_seq).flatten(), lstm_pred)

        curr_seq = harga_scaled[-lookback:].reshape(1, lookback, 1)
        lstm_future = []
        for _ in range(future_steps):
            p = lstm_model.predict(curr_seq, verbose=0)
            lstm_future.append(p[0,0])
            curr_seq = np.append(curr_seq[:, 1:, :], [p], axis=1)
        lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1,1)).flatten().tolist()
    else:
        lstm_mae, lstm_pred, lstm_future = float('inf'), [], [0]*future_steps

    models_eval = {
        'Random Forest': {'mae': rf_mae, 'pred': rf_pred, 'y_test': y_test, 'forecast': rf_future},
        'Gradient Boosting': {'mae': gb_mae, 'pred': gb_pred, 'y_test': y_test, 'forecast': gb_future}
    }
    if lstm_mae != float('inf'):
        models_eval['LSTM & GRU'] = {'mae': lstm_mae, 'pred': lstm_pred, 'y_test': y_test, 'forecast': lstm_future}

    best_model_name = min(models_eval, key=lambda k: models_eval[k]['mae'])
    return best_model_name, models_eval, split_idx, df_prov, rincian_features, df_feat, future_dates

best_model_name, models_eval, split_idx, df_prov, rincian_features, df_feat, future_dates = train_and_evaluate_models(df_filtered)

best_mae = models_eval[best_model_name]['mae']
best_pred = models_eval[best_model_name]['pred']
best_forecast = models_eval[best_model_name]['forecast']
test_dates = df_prov['Tanggal'].iloc[-len(best_pred):]

# --- 4. LAYOUT UTAMA DASHBOARD ---
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader(f"📊 Analisis Spasio-Temporal: {prov_terpilih}")
    st.markdown(f"##### 🗺️ Visualisasi Peta ({minggu_peta})")
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

    st.markdown(f"##### 📈 Proyeksi & Forecasting 2 Bulan ({best_model_name})")
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

    st.markdown("###### 📑 Tabel Komparasi Validasi Ilmiah Performa Model")
    metrics_list = [{"Model Algoritma": k, "Mean Absolute Error (MAE)": f"Rp {v['mae']:,.2f}", "Margin Kesalahan (Error)": f"± Rp {v['mae']:,.0f}", "Status Seleksi": "🏆 TERBAIK" if k == best_model_name else "Alternatif"} for k, v in models_eval.items()]
    st.dataframe(pd.DataFrame(metrics_list), use_container_width=True)

    col_ext1, col_ext2 = st.columns(2)
    with col_ext1:
        st.markdown(f"##### 🌧️ Curah Hujan & Momen")
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(x=df_prov['Tanggal'], y=df_prov['Curah_Hujan'], name='Curah Hujan (mm)', marker_color='lightblue'))
        st.plotly_chart(fig_ext, use_container_width=True)
    with col_ext2:
        st.markdown(f"##### Tingkat Pengaruh Variabel (Feature Importance)")
        fig_feat = px.bar(df_feat, x='Kepentingan', y='Faktor', orientation='h', color='Kepentingan', color_continuous_scale='Blues')
        fig_feat.update_layout(margin=dict(l=0, r=20, t=10, b=0), height=250, coloraxis_showscale=False, xaxis_title="Bobot Pengaruh", yaxis_title="")
        st.plotly_chart(fig_feat, use_container_width=True)

# --- 5. PANEL KONSULTAN AI ---
with col2:
    st.subheader("🤖 Chat AI")
    if st.button("🗑️ Bersihkan"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    prompt = st.chat_input("Tanya AI Konsultan...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if not api_key:
                st.warning("⚠️ Masukkan OpenAI API Key di Sidebar.")
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    system_prompt = f"Anda adalah Konsultan AI spesialis Ketahanan Pangan Nasional untuk wilayah {prov_terpilih}. Tren fitur: {rincian_features}."
                    messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                    stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, stream=True)
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e: st.error(f"Error API: {e}")
