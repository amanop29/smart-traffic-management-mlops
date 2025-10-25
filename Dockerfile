FROM python:3.10-slim

WORKDIR /app
COPY smart_traffic_management_dataset.csv .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]