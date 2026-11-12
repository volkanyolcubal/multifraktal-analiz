import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# Sayfa Ayarları
st.set_page_config(
    page_title="Multifraktal Finansal Analiz Paneli",
    page_icon="📈",
    layout="wide"
)

# --- TEMEL MULTİFRAKTAL ANALİZ (MFDFA) MOTORU ---
def calculate_mfdfa(series, q_list, scales):
    """
    Finansal zaman serisi için temel MFDFA (Hurst üssü h(q), Mass exponent tau(q) 
    ve Singularity spectrum f(alpha)) hesaplaması yapar.
    """
    # Logaritmik getiri (Log-returns) hesaplama
    returns = np.diff(np.log(series))
    profile = np.cumsum(returns - np.mean(returns))
    N = len(profile)
    
    fluctuations = np.zeros((len(q_list), len(scales)))
    
    for idx_s, s in enumerate(scales):
        n_segments = N // s
        if n_segments < 2:
            continue
        
        rms = []
        # İleri ve geri yönde segmentasyon
        for v in range(n_segments):
            segment = profile[v * s : (v + 1) * s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1) # Lineer trend arındırma
            fit = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((segment - fit) ** 2)))
            
        rms = np.array(rms)
        rms = rms[rms > 0] # Sıfıra bölünme hatasını önle
        
        for idx_q, q in enumerate(q_list):
            if q == 0:
                fluctuations[idx_q, idx_s] = np.exp(0.5 * np.mean(np.log(rms ** 2)))
            else:
                fluctuations[idx_q, idx_s] = (np.mean(rms ** q)) ** (1.0 / q)
                
    # Genelleştirilmiş Hurst Üssü h(q) hesaplama
    hq = []
    for idx_q in range(len(q_list)):
        valid = fluctuations[idx_q] > 0
        if np.sum(valid) > 1:
            slope, _ = np.polyfit(np.log(scales[valid]), np.log(fluctuations[idx_q, valid]), 1)
            hq.append(slope)
        else:
            hq.append(np.nan)
            
    hq = np.array(hq)
    tau_q = q_list * hq - 1
    alpha = np.gradient(tau_q, q_list)
    f_alpha = q_list * alpha - tau_q
    
    return pd.DataFrame({
        'q': q_list,
        'h(q) [Hurst]': np.round(hq, 4),
        'tau(q)': np.round(tau_q, 4),
        'alpha': np.round(alpha, 4),
        'f(alpha)': np.round(f_alpha, 4)
    })

# --- WEB ARAYÜZÜ TASARIMI ---
st.title("📈 Multifraktal Finans Piyasaları Analiz Platformu")
st.markdown("Doktora Tez Çalışması | **MFDFA (Multifractal Detrended Fluctuation Analysis)** Modülü")
st.divider()

# Sol Menü (Parametre Kontrolü)
with st.sidebar:
    st.header("⚙️ Analiz Parametreleri")
    uploaded_file = st.file_uploader("Veri Seti Yükle (CSV veya Excel)", type=["csv", "xlsx"])
    
    st.subheader("q-Moment Aralığı")
    q_min = st.number_input("Minimum q", value=-5.0, step=1.0)
    q_max = st.number_input("Maksimum q", value=5.0, step=1.0)
    q_step = st.number_input("q Adım Aralığı", value=0.5, step=0.1)
    
    st.subheader("Ölçek (Scale) Ayarları")
    min_scale = st.number_input("Minimum Pencere (Gün/Bar)", value=10, min_value=5)
    max_scale = st.number_input("Maksimum Pencere (Gün/Bar)", value=100, min_value=20)
    
    run_btn = st.button("🚀 Analizi Başlat", type="primary", use_container_width=True)

# Veri Hazırlığı (Dosya yüklenmezse test için otomatik sentetik finansal veri üretir)
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"Dosya başarıyla yüklendi: **{uploaded_file.name}** ({len(df)} satır)")
    target_col = st.selectbox("Analiz edilecek fiyat sütununu seçin:", df.select_dtypes(include=[np.number]).columns)
    price_series = df[target_col].dropna().values
else:
    st.info("💡 Şu an önizleme modundasınız. Sol menüden kendi veri setinizi yükleyebilir veya mevcut örnek seri üzerinden testi başlatabilirsiniz.")
    np.random.seed(42)
    # Örnek fiyat serisi simülasyonu
    price_series = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 1000)))
    df = pd.DataFrame({"Ornek_Kapanis_Fiyati": price_series})

# Ana Ekran Düzeni
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Veri Önizleme")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("📉 Zaman Serisi Grafiği")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(y=price_series, mode='lines', name='Fiyat', line=dict(color='#1f77b4')))
    fig_price.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
    st.plotly_chart(fig_price, use_container_width=True)

# Analiz Çalıştırma ve Sonuç Ekranı
if run_btn:
    with st.spinner("Multifraktal spektrum ve Hurst üsleri hesaplanıyor..."):
        q_list = np.arange(q_min, q_max + q_step, q_step)
        q_list = q_list[np.abs(q_list) > 1e-5] # 0 noktasını temizle veya ayrı işle
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 15).astype(int))
        
        results_df = calculate_mfdfa(price_series, q_list, scales)
        
        st.divider()
        st.header("📊 Analiz Çıktıları ve Tez Grafikleri")
        
        # Özet Metrikler
        delta_h = results_df['h(q) [Hurst]'].max() - results_df['h(q) [Hurst]'].min()
        delta_alpha = results_df['alpha'].max() - results_df['alpha'].min()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Multifraktallik Derecesi (Δh)", f"{delta_h:.4f}")
        m2.metric("Spektrum Genişliği (Δα)", f"{delta_alpha:.4f}")
        m3.metric("Klasik Hurst Üssü h(2)", f"{results_df.iloc[(results_df['q']-2).abs().argsort()[:1]]['h(q) [Hurst]'].values[0]:.4f}")
        
        # Grafikler (Yan Yana 2 Grafik)
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            fig_hq = go.Figure()
            fig_hq.add_trace(go.Scatter(x=results_df['q'], y=results_df['h(q) [Hurst]'], mode='lines+markers', name='h(q)', line=dict(color='#d62728', width=3)))
            fig_hq.update_layout(title="Genelleştirilmiş Hurst Üssü: h(q)", xaxis_title="q", yaxis_title="h(q)", template="plotly_white")
            st.plotly_chart(fig_hq, use_container_width=True)
            
        with g_col2:
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(x=results_df['alpha'], y=results_df['f(alpha)'], mode='lines+markers', name='f(α)', line=dict(color='#2ca02c', width=3)))
            fig_spec.update_layout(title="Multifraktal Singülarite Spektrumu: f(α) - α", xaxis_title="α (Singülarite)", yaxis_title="f(α)", template="plotly_white")
            st.plotly_chart(fig_spec, use_container_width=True)
            
        # Sonuç Tablosu ve İndirme Butonu
        st.subheader("📑 Sayısal Sonuç Tablosu")
        st.dataframe(results_df, use_container_width=True)
        
        # Excel olarak indirme tamponu
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='MFDFA_Sonuclari')
            
        st.download_button(
            label="📥 Sonuçları Excel (.xlsx) Olarak İndir",
            data=buffer.getvalue(),
            file_name="multifraktal_analiz_sonuclari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
