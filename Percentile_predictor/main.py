# Flask Web Interface for JEE Main 2024 Percentile Prediction

from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Shift-wise JEE Main 2024 data with Marks and Percentile per shift
marks = [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
data = {
    'Marks': marks,
    'Jan_27_Shift_1_Percentile': [99.50, 99.00, 98.00, 97.00, 96.00, 95.00, 94.00, 93.00, 92.00, 91.00, 90.00],
    'Jan_27_Shift_2_Percentile': [99.40, 98.80, 97.90, 96.90, 96.00, 94.80, 93.90, 92.80, 91.90, 91.00, 89.80],
    'Jan_29_Shift_1_Percentile': [99.45, 99.00, 98.10, 97.20, 96.10, 95.00, 94.10, 93.00, 92.00, 91.00, 90.00],
    'Jan_29_Shift_2_Percentile': [99.50, 98.90, 98.00, 97.00, 95.90, 94.70, 93.80, 92.90, 91.80, 90.90, 89.70],
    'Jan_30_Shift_1_Percentile': [99.48, 98.95, 97.90, 96.80, 95.70, 94.60, 93.50, 92.30, 91.20, 90.00, 89.00],
    'Jan_30_Shift_2_Percentile': [98.90, 97.80, 96.70, 95.60, 94.50, 93.40, 92.30, 91.20, 90.10, 89.00, 88.00],
    'Jan_31_Shift_1_Percentile': [98.80, 97.70, 96.60, 95.50, 94.40, 93.30, 92.20, 91.10, 90.00, 88.90, 87.80],
    'Jan_31_Shift_2_Percentile': [99.00, 98.00, 96.90, 95.80, 94.70, 93.60, 92.50, 91.40, 90.30, 89.20, 88.10],
    'Feb_1_Shift_1_Percentile': [99.40, 98.80, 97.70, 96.60, 95.50, 94.40, 93.30, 92.20, 91.10, 90.00, 89.00],
    'Feb_1_Shift_2_Percentile': [99.30, 98.70, 97.60, 96.50, 95.40, 94.30, 93.20, 92.10, 91.00, 90.00, 88.90],
    'Apr_4_Shift_1_Percentile': [99.50, 99.00, 98.00, 97.00, 96.00, 95.00, 94.00, 93.00, 92.00, 91.00, 90.00],
    'Apr_4_Shift_2_Percentile': [99.40, 98.90, 98.00, 97.00, 96.00, 95.00, 94.00, 93.00, 92.00, 91.00, 90.00],
    'Apr_5_Shift_1_Percentile': [99.45, 99.00, 98.10, 97.20, 96.30, 95.40, 94.50, 93.60, 92.70, 91.80, 90.90],
    'Apr_5_Shift_2_Percentile': [99.38, 98.85, 97.90, 96.95, 96.00, 95.05, 94.10, 93.15, 92.20, 91.25, 90.30],
    'Apr_6_Shift_1_Percentile': [99.42, 98.90, 97.85, 96.80, 95.75, 94.70, 93.65, 92.60, 91.55, 90.50, 89.45],
    'Apr_6_Shift_2_Percentile': [99.35, 98.80, 97.75, 96.70, 95.65, 94.60, 93.55, 92.50, 91.45, 90.40, 89.35],
    'Apr_8_Shift_1_Percentile': [99.40, 98.85, 97.80, 96.75, 95.70, 94.65, 93.60, 92.55, 91.50, 90.45, 89.40],
    'Apr_8_Shift_2_Percentile': [99.30, 98.75, 97.70, 96.65, 95.60, 94.55, 93.50, 92.45, 91.40, 90.35, 89.30]
}

df = pd.DataFrame(data)

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>JEE Main Percentile Predictor</title>
</head>
<body style="font-family: Arial; text-align: center; padding: 30px">
    <h1>📊 JEE Main 2024 Percentile Predictor</h1>
    <form method="POST">
        <label>Select your shift:</label>
        <select name="shift">
            {% for col in shifts %}
            <option value="{{ col }}" {% if selected_shift == col %}selected{% endif %}>{{ col }}</option>
            {% endfor %}
        </select>
        <br><br>
        <label>Enter your marks:</label>
        <input type="number" name="marks" min="0" max="300" value="{{ marks }}">
        <br><br>
        <button type="submit">Predict Percentile</button>
    </form>

    {% if percentile is not none %}
        <h2>Predicted Percentile: {{ percentile|round(2) }} 🎯</h2>
        <img src="data:image/png;base64,{{ plot_url }}" alt="Plot">
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    shifts = [col for col in df.columns if col.endswith('_Percentile')]
    shift = shifts[0]
    marks = 150
    percentile = None
    plot_url = None

    if request.method == 'POST':
        shift = request.form['shift']
        marks = int(request.form['marks'])

        X = df['Marks'].values.reshape(-1, 1)
        y = df[shift].values
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        user_input_poly = poly.transform(np.array([[marks]]))
        percentile = model.predict(user_input_poly)[0]

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(X, y, color="blue", label="Original Data")
        X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        y_pred = model.predict(X_range_poly)
        ax.plot(X_range, y_pred, color="red", label="Polynomial Regression")
        ax.axvline(x=marks, linestyle="--", color="green", label=f"Your Marks ({marks})")
        ax.axhline(y=percentile, linestyle="--", color="orange", label=f"Predicted: {percentile:.2f}")
        ax.set_xlabel("Marks")
        ax.set_ylabel("Percentile")
        ax.set_title(f"Shift: {shift.replace('_Percentile','').replace('_',' ')}")
        ax.legend()
        ax.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode('utf8')
        plt.close()

    return render_template_string(TEMPLATE, shifts=shifts, selected_shift=shift, marks=marks, percentile=percentile, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
