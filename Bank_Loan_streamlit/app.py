import streamlit as st
import pandas as pd
import pickle

# Load your trained pipeline
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Loan Prediction App")
st.write("Upload a CSV file to get predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure the input has the right columns
    try:
        X = df[model.feature_names_in_]
    except AttributeError:
        st.error("Your model is missing the 'feature_names_in_' attribute. Make sure to use scikit-learn >= 1.0.")
    except KeyError:
        st.error("Uploaded file is missing one or more required feature columns.")
    else:
        # Predict and append to the DataFrame
        df['Prediction'] = model.predict(X)
        df['Probability'] = model.predict_proba(X)[:, 1]

        st.success("Predictions generated!")
        st.dataframe(df)

        # Optional: Downloadable CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results as CSV", csv, "predictions.csv", "text/csv")
