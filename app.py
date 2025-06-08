from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("modulation_model.pkl")
le = joblib.load("label_encoder.pkl")

# Read dataset for visualization
df = pd.read_csv("modulation_dataset.csv")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(request.form[feature]) for feature in [
        'Bandwidth', 'Symbol_Rate', 'PAPR', 'Mod_Index', 
        'Spectral_Entropy', 'Freq_Variance', 'Phase'
    ]]
    input_df = pd.DataFrame([inputs], columns=[
        'Bandwidth', 'Symbol_Rate', 'PAPR', 'Mod_Index',
        'Spectral_Entropy', 'Freq_Variance', 'Phase'
    ])
    prediction_encoded = model.predict(input_df)[0]
    prediction = le.inverse_transform([prediction_encoded])[0]
    return render_template("index.html", prediction=prediction)

@app.route('/visualizations')
def visualizations():
    return render_template("visualization.html")

@app.route('/plot/<plot_type>')
def plot(plot_type):
    path = f"static/plots/{plot_type}.png"

    if not os.path.exists("static/plots"):
        os.makedirs("static/plots")

    # Clear the plot figure before generating a new one
    plt.clf()

    try:
        if plot_type == "pairplot":
            sns.pairplot(df.sample(300), hue='Modulation_Type', corner=True)
        elif plot_type == "histogram":
            df.drop("Modulation_Type", axis=1).hist(figsize=(12, 8), bins=30, edgecolor='black')
        elif plot_type == "countplot":
            sns.countplot(data=df, x='Modulation_Type', order=df['Modulation_Type'].value_counts().index)
        elif plot_type == "heatmap":
            sns.heatmap(df.drop("Modulation_Type", axis=1).corr(), annot=True, cmap="coolwarm", square=True)

        plt.tight_layout()
        plt.savefig(path)  # Save the plot as an image
        print(f"Plot saved at {path}")

    except Exception as e:
        print(f"Error while generating plot: {e}")
        return f"Error: {str(e)}"

    return render_template("visualization.html", plot_url=path)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)  # Run Flask without threading to avoid conflicts
