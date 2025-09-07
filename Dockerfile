FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8086
CMD ["uvicorn", "apis.reports.service:app", "--host", "0.0.0.0", "--port", "8086", "--workers", "2"]

