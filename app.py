import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="PPG Аналізатор",
    layout="wide"
)

st.title("🫀 Система аналізу PPG сигналів")
st.markdown("Аналіз фотоплетизмографічних (PPG) сигналів")

RESULTS_FILE = "results.csv"
PLOT_FILE = "plot.png"

# ================= GLOBAL VARIABLES =================
bpm = None
hrv = None
stress = None

signal_global = None
filtered_global = None
peaks_global = None

# ================= FILTER =================
def bandpass_filter(signal, low=0.5, high=8, fs=250, order=3):

    nyq = 0.5 * fs

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype='band'
    )

    return filtfilt(b, a, signal)

# ================= SAVE HISTORY =================
def save_result(filename, bpm, hrv, stress, fs):

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "bpm": bpm,
        "hrv": hrv,
        "stress": stress,
        "fs": fs
    }

    if os.path.exists(RESULTS_FILE):

        df = pd.read_csv(RESULTS_FILE)

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True
        )

    else:
        df = pd.DataFrame([row])

    df.to_csv(RESULTS_FILE, index=False)

# ================= ANALYSIS =================
def analyze(signal, fs):

    filtered = bandpass_filter(signal, fs=fs)

    peaks, _ = find_peaks(
        filtered,
        distance=int(fs * 0.4),
        prominence=np.std(filtered) * 0.3
    )

    bpm, hrv, stress = None, None, None

    if len(peaks) > 2:

        rr = np.diff(peaks) / fs

        bpm = 60 / np.mean(rr)

        hrv = np.std(rr) * 1000

        stress = 100 / (hrv + 1)

    return filtered, peaks, bpm, hrv, stress

# ================= SAVE PLOT =================
def save_plot(signal, filtered, peaks):

    plt.figure(figsize=(10, 4))

    plt.plot(signal, label="Raw Signal")

    plt.plot(filtered, label="Filtered Signal")

    plt.scatter(
        peaks,
        filtered[peaks],
        color="red",
        label="Detected Peaks"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(PLOT_FILE)

    plt.close()

# ================= PDF =================
def create_pdf(bpm, hrv, stress, peaks_count):

    save_plot(
        signal_global,
        filtered_global,
        peaks_global
    )

    file_name = "report.pdf"

    doc = SimpleDocTemplate(file_name)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "PPG Analysis Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"BPM: {bpm:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"HRV: {hrv:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Stress Index: {stress:.2f}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Detected Peaks: {peaks_count}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Image(
            PLOT_FILE,
            width=450,
            height=220
        )
    )

    doc.build(story)

    return file_name

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "📊 Аналіз",
    "🔄 Порівняння",
    "💾 Історія"
])

# ================= TAB 1 =================
with tab1:

    st.subheader("📂 Завантаження сигналів")

    files = st.file_uploader(
        "Завантаж один або декілька PPG сигналів",
        type=["csv", "txt"],
        accept_multiple_files=True
    )

    fs = st.slider(
        "Частота дискретизації (Hz)",
        50,
        500,
        250
    )

    if files:

        for file in files:

            st.divider()

            st.header(f"📄 Файл: {file.name}")

            data = pd.read_csv(
                file,
                header=None
            )

            signal = data.iloc[:, -1].values

            filtered, peaks, bpm, hrv, stress = analyze(signal, fs)

            signal_global = signal
            filtered_global = filtered
            peaks_global = peaks

            # ================= METRICS =================
            st.subheader("📈 Основні показники")

            col1, col2, col3 = st.columns(3)

            if bpm is not None:

                col1.metric(
                    "BPM",
                    f"{bpm:.2f}"
                )

                col2.metric(
                    "HRV",
                    f"{hrv:.2f}"
                )

                col3.metric(
                    "Stress Index",
                    f"{stress:.2f}"
                )

            # ================= RAW GRAPH =================
            st.subheader("📉 Сирий PPG сигнал (Raw Signal)")

            fig1 = go.Figure()

            fig1.add_trace(
                go.Scatter(
                    y=signal,
                    name="Raw Signal"
                )
            )

            fig1.update_layout(
                height=350,
                xaxis_title="Samples",
                yaxis_title="Amplitude"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            # ================= FILTERED GRAPH =================
            st.subheader("🧹 Відфільтрований сигнал (Filtered Signal)")

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    y=filtered,
                    name="Filtered Signal"
                )
            )

            fig2.update_layout(
                height=350,
                xaxis_title="Samples",
                yaxis_title="Amplitude"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            # ================= PEAKS GRAPH =================
            st.subheader("❤️ Детекція піків серцевих скорочень")

            fig3 = go.Figure()

            fig3.add_trace(
                go.Scatter(
                    y=filtered,
                    name="Filtered Signal"
                )
            )

            fig3.add_trace(
                go.Scatter(
                    x=peaks,
                    y=filtered[peaks],
                    mode="markers",
                    name="Detected Peaks"
                )
            )

            fig3.update_layout(
                height=350,
                xaxis_title="Samples",
                yaxis_title="Amplitude"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

            # ================= INTERPRETATION =================
            st.subheader("🧠 Інтерпретація результатів")

            if bpm is not None:

                # ===== BPM STATUS =====
                if bpm < 60:

                    st.info(
                        f"💙 Пульс: {bpm:.2f} BPM\n\n"
                        "Понижений діапазон пульсу.\n"
                        "Може спостерігатись у стані спокою або під час сну."
                    )

                elif bpm <= 100:

                    st.success(
                        f"💚 Пульс: {bpm:.2f} BPM\n\n"
                        "Пульс знаходиться в межах норми."
                    )

                elif bpm <= 120:

                    st.warning(
                        f"💛 Пульс: {bpm:.2f} BPM\n\n"
                        "Підвищений пульс.\n"
                        "Можливий стрес або фізичне навантаження."
                    )

                else:

                    st.error(
                        f"❤️‍🔥 Пульс: {bpm:.2f} BPM\n\n"
                        "Дуже високий пульс.\n"
                        "Рекомендується відпочинок або додатковий контроль стану."
                    )

                # ===== STRESS STATUS =====
                if stress < 1:

                    st.success(
                        f"🟢 Stress Index: {stress:.2f}\n\n"
                        "Рівень стресу в нормі."
                    )

                elif stress < 2.5:

                    st.warning(
                        f"🟡 Stress Index: {stress:.2f}\n\n"
                        "Помірний рівень стресу."
                    )

                elif stress < 4:

                    st.error(
                        f"🟠 Stress Index: {stress:.2f}\n\n"
                        "Високий рівень стресу."
                    )

                else:

                    st.error(
                        f"🔴 Stress Index: {stress:.2f}\n\n"
                        "Рівень стресу зашкалює.\n"
                        "Система рекомендує відпочинок."
                    )

                # ===== HRV STATUS =====
                if hrv < 20:

                    st.warning(
                        f"📉 HRV: {hrv:.2f} ms\n\n"
                        "Низька варіабельність серцевого ритму."
                    )

                elif hrv < 80:

                    st.success(
                        f"📈 HRV: {hrv:.2f} ms\n\n"
                        "HRV у нормальному діапазоні."
                    )

                else:

                    st.info(
                        f"📊 HRV: {hrv:.2f} ms\n\n"
                        "Висока варіабельність серцевого ритму."
                    )

            # ================= SAVE =================
            if bpm is not None:

                colA, colB = st.columns(2)

                with colA:

                    if st.button(
                        f"💾 Зберегти {file.name}",
                        key=f"save_{file.name}"
                    ):

                        save_result(
                            file.name,
                            bpm,
                            hrv,
                            stress,
                            fs
                        )

                        st.success("Результат збережено")

                with colB:

                    if st.button(
                        f"📄 PDF звіт {file.name}",
                        key=f"pdf_{file.name}"
                    ):

                        pdf = create_pdf(
                            bpm,
                            hrv,
                            stress,
                            len(peaks)
                        )

                        st.download_button(
                            "⬇️ Завантажити PDF",
                            open(pdf, "rb"),
                            file_name=pdf,
                            key=f"download_{file.name}"
                        )

# ================= TAB 2 =================
with tab2:

    st.subheader("🔄 Порівняння двох сигналів")

    f1 = st.file_uploader(
        "Signal A",
        type=["csv", "txt"],
        key="a"
    )

    f2 = st.file_uploader(
        "Signal B",
        type=["csv", "txt"],
        key="b"
    )

    fs2 = st.slider(
        "Частота дискретизації для порівняння",
        50,
        500,
        250,
        key="fs2"
    )

    if f1 and f2:

        s1 = pd.read_csv(f1, header=None).iloc[:, -1].values
        s2 = pd.read_csv(f2, header=None).iloc[:, -1].values

        _, _, bpm1, hrv1, stress1 = analyze(s1, fs2)
        _, _, bpm2, hrv2, stress2 = analyze(s2, fs2)

        st.subheader("📊 Результати порівняння")

        comparison_df = pd.DataFrame({
            "Signal": ["A", "B"],
            "BPM": [round(bpm1, 2), round(bpm2, 2)],
            "HRV": [round(hrv1, 2), round(hrv2, 2)],
            "Stress": [round(stress1, 2), round(stress2, 2)]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

# ================= TAB 3 =================
with tab3:

    st.subheader("📁 Історія вимірювань")

    if os.path.exists(RESULTS_FILE):

        df = pd.read_csv(RESULTS_FILE)

        st.dataframe(
            df,
            use_container_width=True
        )

        cols = [
            c for c in ["bpm", "hrv", "stress"]
            if c in df.columns
        ]

        if len(cols) > 0:

            st.subheader("📈 Динаміка показників")

            st.line_chart(df[cols])

    else:
        st.info("Ще немає збережених вимірювань")