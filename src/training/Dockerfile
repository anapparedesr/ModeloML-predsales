# ─────────────────────────────────────────
# Dockerfile — training + serving (SageMaker BYOC)
# ─────────────────────────────────────────

FROM python:3.12-slim

# SageMaker espera los scripts aquí
ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY train.py .
COPY predictor.py .
COPY utils/ utils/
COPY __main__.py .

# Copiar entry points y darles permisos de ejecución
COPY train train
COPY serve serve
RUN chmod +x train serve