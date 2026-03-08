#!/usr/bin/env python3
"""
predictor.py
------------
Servidor de inferencia Flask para SageMaker BYOC.

SageMaker invoca el serving ejecutando: docker run <image> serve
El script 'serve' arranca gunicorn, que a su vez carga esta app Flask.

Implementa el contrato de inferencia de SageMaker:
    GET  /ping         -> health check. 200 si el modelo cargó, 404 si no.
    POST /invocations  -> recibe features en CSV, devuelve predicciones en CSV.

Input esperado en el filesystem del container:
    /opt/ml/
    └── model/
        └── random_forest_lags.pkl

Formato del request POST /invocations:
    Content-Type: text/csv
    Body: filas CSV sin header, con columnas lag_1,lag_3,lag_6,lag_12
    Ejemplo: 5.0,4.0,3.0,2.0

Formato del response:
    Content-Type: text/csv — una predicción por línea
"""

from __future__ import print_function

import io
import os

import flask
import joblib
import numpy as np
import pandas as pd

# Paths de SageMaker
prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

FEATURE_LAGS = ["lag_1", "lag_3", "lag_6", "lag_12"]


# -----
# Singleton para el modelo
class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Carga el modelo desde disco si no está en memoria y lo retorna."""
        if cls.model is None:
            cls.model = joblib.load(
                os.path.join(model_path, "random_forest_lags.pkl")
            )
        return cls.model

    @classmethod
    def predict(cls, input_data):
        """
        Ejecuta la predicción sobre el DataFrame recibido.

        Args:
            input_data (pd.DataFrame): features de entrada (lag_1, lag_3, lag_6, lag_12)

        Returns:
            numpy array con una predicción por fila, clippeado entre 0 y 20.
        """
        rf_model = cls.get_model()
        predictions = rf_model.predict(input_data)
        return np.clip(predictions, 0, 20)


# -----
# App Flask
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check — 200 si el modelo cargó, 404 si no."""
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """
    Endpoint de inferencia.
    Recibe un batch de observaciones en CSV sin header,
    ejecuta la predicción y devuelve resultados en CSV.
    """
    data = None

    if flask.request.content_type == "text/csv":
        raw = flask.request.data.decode("utf-8")
        s = io.StringIO(raw)
        data = pd.read_csv(s, header=None, names=FEATURE_LAGS)
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )

    print(f"Invoked with {data.shape[0]} records")

    predictions = ScoringService.predict(data)

    out = io.StringIO()
    pd.DataFrame({"predictions": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)