# ─────────────────────────────────────────
# Dockerfile — preprocessing step
# Prepares training and inference datasets
# Compatible con uso local y SageMaker BYOC*
# ─────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prep.py .
COPY prep_inference.py .
COPY utils/ utils/
COPY __main__.py .

# CMD en lugar de ENTRYPOINT:
CMD ["python", "__main__.py"]