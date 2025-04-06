from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import datetime
import os
import logging
import matplotlib
matplotlib.use('Agg')  # Fix for macOS: avoids GUI crash
import matplotlib.pyplot as plt

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Health check endpoint
@app.route('/health')
def health():
    return 'OK', 200

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load data
try:
    df = pd.read_csv("data/wiki_data.csv")
    app.logger.info("Successfully loaded data/wiki_data.csv")
except Exception as e:
    app.logger.error(f"Error loading data: {e}")
    # Try the alternative path
    try:
        df = pd.read_csv("attached_assets/wiki_data.csv")
        app.logger.info("Successfully loaded attached_assets/wiki_data.csv")
    except Exception as e:
        app.logger.error(f"Error loading alternative data path: {e}")
        # Create a sample dataframe
        df = pd.DataFrame({
            "Page": ["Wikipedia", "Python", "Flask"],
            "2023-01-01": [1000, 500, 200],
            "2023-01-02": [1100, 550, 220],
            "2023-01-03": [1050, 525, 210]
        })
        app.logger.info("Created sample dataframe as fallback")

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
            # Simple prediction logic (average of last 7 days with a random factor)
            last_7_days_avg = np.mean(ts[-7:])
            y_pred = int(last_7_days_avg * (1 + np.random.uniform(-0.1, 0.1)))

            # Check for invalid prediction values
            if y_pred < 0:
                error_message = "Incompatible dataset: Unable to generate prediction for this page due to inconsistent traffic patterns."
            elif y_pred < 1:
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
                if prediction > mean_traffic + std_traffic:
                    traffic_level = "High"
                elif prediction < mean_traffic - std_traffic:
                    traffic_level = "Low"
                else:
                    traffic_level = "Average"

                # Plot prediction vs last 30 days with vertical line for prediction
                plt.figure(figsize=(8, 3))
                plt.plot(range(30), ts[-30:], label="Past 30 Days")
                plt.vlines(x=29, ymin=0, ymax=prediction, color="orange", linestyle="--")
                plt.plot(29, prediction, 'o', color="orange", label=f"Prediction: {prediction}")
                plt.xlabel("Days")
                plt.ylabel("Page Views")
                plt.title("Traffic History & Next Day Forecast")
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
