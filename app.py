from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import datetime
import os

import matplotlib
matplotlib.use('Agg')  # Fix for macOS: avoids GUI crash
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and data
model = load_model("model/model.h5")
df = pd.read_csv("data/wiki_data.csv")
date_cols = df.columns.drop("Page", errors="ignore")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    page_selected = None
    traffic_level = None
    confidence = None
    error_message = None
    image_path = None

    pages = sorted(df["Page"].unique())

    if request.method == "POST":
        page_selected = request.form["page"]

        ts = df[df["Page"] == page_selected].drop("Page", axis=1).values.flatten()
        ts = pd.Series(ts).interpolate().bfill().ffill().values

        if len(ts) < 60 or np.sum(ts[-30:]) < 1:
            error_message = "Sorry, this page has too little recent traffic to make a confident prediction."
        else:
            input_window = ts[-30:].reshape(1, -1)
            y_pred = model.predict(input_window).flatten()[0]

            # If prediction is too small to be meaningful
            if y_pred < 1:
                error_message = "Prediction too low to display — this page has very little recent traffic."
            else:
                prediction = int(y_pred)

                # Confidence score (heuristic)
                mae = abs(ts[-1] - y_pred)
                confidence = round(1 - (mae / (np.mean(ts[-30:]) + 1)), 2)
                confidence = max(0, min(confidence, 1))

                # Traffic level
                mean_traffic = np.mean(ts[-30:])
                std_traffic = np.std(ts[-30:])
                if y_pred > mean_traffic + std_traffic:
                    traffic_level = "High"
                elif y_pred < mean_traffic - std_traffic:
                    traffic_level = "Low"
                else:
                    traffic_level = "Average"

                # Plot prediction vs last 30 days
                plt.figure(figsize=(8, 3))
                plt.plot(range(30), ts[-30:], label="Past 30 Days")
                plt.axhline(y=prediction, color="orange", linestyle="--", label="Predicted")
                plt.xlabel("Days")
                plt.ylabel("Page Views")
                plt.title("Traffic History & Forecast")
                plt.legend()
                image_path = "static/traffic_plot.png"
                plt.tight_layout()
                plt.savefig(image_path)
                plt.close()

    return render_template(
        "index.html",
        prediction=prediction,
        page_selected=page_selected,
        pages=pages,
        traffic_level=traffic_level,
        confidence=confidence,
        error_message=error_message,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)

