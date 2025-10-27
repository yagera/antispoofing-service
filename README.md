# Audio Anti-Spoofing Service

Простой Streamlit-сервис для детекции синтетической речи (антиспуфинг).

---

## Установка и запуск

1. Создай новое окружение:
```
   conda create -n antispoofing python=3.10
```
```
   conda activate antispoofing
```

2. Установи зависимости:
```
   uv pip install -r requirements.txt
```

4. Установи FFmpeg (нужно для чтения MP3, M4A, OGG):
```
   conda install -c conda-forge ffmpeg
```

6. Запусти сервис:
```
   streamlit run app.py
```

---
