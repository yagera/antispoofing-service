import streamlit as st
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from inference import AntiSpoofingModel
import pandas as pd

st.set_page_config(
    page_title="Audio Anti-Spoofing Service",
    layout="wide"
)

st.title("Audio Anti-Spoofing Detection Service")
st.markdown("""
Этот сервис определяет является ли голос в аудиозаписи **настоящим** или **фейковым**.
- **Класс 0**: FAKE
- **Класс 1**: REAL
""")

@st.cache_resource
def load_model():
    model_path = "weights/torch_script_weigths.pt"
    if not Path(model_path).exists():
        st.error(f"Модель не найдена по пути: {model_path}")
        st.stop()
    return AntiSpoofingModel(model_path)


try:
    model = load_model()
    st.success("Модель успешно загружена!")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.exception(e)
    st.stop()

st.markdown("---")

uploaded_file = st.file_uploader(
    "Загрузите аудиофайл",
    type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
    help="Поддерживаемые форматы: WAV, MP3, FLAC, OGG, M4A"
)

if uploaded_file is not None:
    st.subheader("📁 Информация о файле")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Имя файла:** {uploaded_file.name}")
    with col2:
        st.write(f"**Размер:** {uploaded_file.size / 1024:.2f} KB")

    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

    if st.button("🔍 Анализировать аудио", type="primary"):
        with st.spinner("Обработка аудио..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                results = model.predict(tmp_path)
                st.markdown("---")
                st.subheader("Результаты анализа")
                st.info(f"Обнаружено каналов: **{len(results)}**")

                for idx, result in enumerate(results):
                    with st.expander(f"Канал {result['channel']} - {result['prediction']}", expanded=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            prediction_color = "🟢" if result['prediction'] == 'REAL' else "🔴"
                            st.metric(
                                label="Предсказание",
                                value=f"{prediction_color} {result['prediction']}"
                            )

                        with col2:
                            st.metric(
                                label="Уверенность",
                                value=f"{result['confidence'] * 100:.2f}%"
                            )

                        with col3:
                            threshold_status = "✅" if result['real_prob'] > 0.5 else "⚠️"
                            st.metric(
                                label="Статус",
                                value=f"{threshold_status}"
                            )

                        st.markdown("#### Вероятности классов")
                        prob_col1, prob_col2 = st.columns(2)

                        with prob_col1:
                            st.metric(
                                label="🔴 FAKE (класс 0)",
                                value=f"{result['fake_prob'] * 100:.2f}%",
                                delta=None
                            )

                        with prob_col2:
                            st.metric(
                                label="🟢 REAL (класс 1)",
                                value=f"{result['real_prob'] * 100:.2f}%",
                                delta=None
                            )

                        fig = go.Figure(data=[
                            go.Bar(
                                x=['FAKE', 'REAL'],
                                y=[result['fake_prob'], result['real_prob']],
                                marker_color=['#ff4b4b', '#00cc00'],
                                text=[f"{result['fake_prob'] * 100:.2f}%", f"{result['real_prob'] * 100:.2f}%"],
                                textposition='auto',
                            )
                        ])

                        fig.update_layout(
                            title=f"Распределение вероятностей - Канал {result['channel']}",
                            yaxis_title="Вероятность",
                            xaxis_title="Класс",
                            yaxis=dict(range=[0, 1]),
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Детальная информация"):
                            st.json({
                                'channel': result['channel'],
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'probabilities': {
                                    'fake': result['fake_prob'],
                                    'real': result['real_prob']
                                },
                                'logits': result['logits']
                            })

                if len(results) > 1:
                    st.markdown("---")
                    st.subheader("📋 Сводная таблица по всем каналам")

                    df = pd.DataFrame([
                        {
                            'Канал': r['channel'],
                            'Предсказание': r['prediction'],
                            'Уверенность': f"{r['confidence'] * 100:.2f}%",
                            'FAKE вероятность': f"{r['fake_prob'] * 100:.2f}%",
                            'REAL вероятность': f"{r['real_prob'] * 100:.2f}%"
                        }
                        for r in results
                    ])

                    st.dataframe(df, use_container_width=True)
                    fig_summary = go.Figure()

                    fig_summary.add_trace(go.Bar(
                        name='FAKE',
                        x=[f"Канал {r['channel']}" for r in results],
                        y=[r['fake_prob'] for r in results],
                        marker_color='#ff4b4b'
                    ))

                    fig_summary.add_trace(go.Bar(
                        name='REAL',
                        x=[f"Канал {r['channel']}" for r in results],
                        y=[r['real_prob'] for r in results],
                        marker_color='#00cc00'
                    ))

                    fig_summary.update_layout(
                        title='Сравнение вероятностей по всем каналам',
                        yaxis_title='Вероятность',
                        barmode='group',
                        height=500
                    )

                    st.plotly_chart(fig_summary, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка при обработке аудио: {e}")
                st.exception(e)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Audio Anti-Spoofing Detection Service</p>
</div>
""", unsafe_allow_html=True)
