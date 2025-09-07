FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy code into container
COPY . .
# Compile check: fail early if syntax/indent errors
RUN python -m py_compile ui_minimal.py \
 && python -m py_compile src/services/providers/fmp_provider.py

EXPOSE 8086
CMD ["uvicorn", "apis.reports.service:app", "--host", "0.0.0.0", "--port", "8086", "--workers", "2"]


