from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            X = df[model.feature_names_in_]
            df['Prediction'] = model.predict(X)
            df['Probability'] = model.predict_proba(X)[:, 1]
            table_html = df.to_html(classes='table table-striped', index=False)
    return render_template('index.html', table_html=table_html)


if __name__ == '__main__':
    app.run(debug=True)


