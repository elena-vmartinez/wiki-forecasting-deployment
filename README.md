# Wikipedia Page Views Forecasting
This project is a web application built using Flask that forecasts the next day's Wikipedia page views based on the last 30 days of data. The app uses a deep learning model to make predictions and visualizes past traffic data for the selected page.

View publicly at: https://wikiforecasting.replit.app/

## Features
- Predicts the next day's page views for a selected Wikipedia page.
- Shows the confidence score for the prediction.
- Categorizes the prediction as high, low, or average compared to past traffic levels.
- Displays a graph of the last 30 days' traffic along with the prediction.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/elena-vmartinez/Wiki-forecast-deployment.git
    ```

2. Navigate to the project folder:
    ```bash
    cd Wiki-forecast-deployment
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. To run the Flask app:
    ```bash
    python app.py
    ```

2. Open your web browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).
3. Select a Wikipedia page from the dropdown to see the forecast for the next day.
