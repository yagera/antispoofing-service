FROM python:3.10-slim

WORKDIR /app

COPY ui/requirements.txt .

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY ui/app.py .
COPY ui/config.yaml .

ENV STREAMLIT_SERVER_PORT=8931
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

EXPOSE 8931

CMD ["streamlit", "run", "app.py", "--server.port=8931", "--server.address=0.0.0.0"]
