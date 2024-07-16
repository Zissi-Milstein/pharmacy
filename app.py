import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tempfile import NamedTemporaryFile
import os
import base64
from sklearn.metrics import accuracy_score

# Global variables to store trained model
classifier = None
vectorizer = None

# Function to read CSV or Excel file with fallback encoding
def read_csv_with_fallback(file_path):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')

# Function to train model if not already trained
def train_model():
    global classifier, vectorizer

    known_drugs = read_csv_with_fallback('knowndrugs.csv')

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(known_drugs['Description'])
    y_train = known_drugs['Category']

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

# Function to classify drugs and calculate accuracy percentage
def classify_drugs(drugs):
    global classifier, vectorizer

    if classifier is None or vectorizer is None:
        train_model()

    X_new = vectorizer.transform(drugs['Description'])
    drugs['Predicted_Category'] = classifier.predict(X_new)
    
    # Calculate accuracy percentage if true labels are available
    if 'Category' in drugs.columns:
        correct_predictions = (drugs['Category'] == drugs['Predicted_Category']).sum()
        total_predictions = len(drugs)
        drugs['Accuracy (%)'] = round((correct_predictions / total_predictions) * 100, 2)

    return drugs

# Function to create a download link for a DataFrame as CSV
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="classified_drugs.csv">Download CSV File</a>'
    return href

# Streamlit UI
def main():
    st.title('Drug Classification App')
    st.write('Upload a CSV or Excel file to classify drugs.')

    uploaded_file = st.file_uploader('Choose a file', type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Read file and classify drugs
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error('Unsupported file format. Upload a CSV or Excel file.')
            return

        # Classify drugs and calculate accuracy percentage
        classified_df = classify_drugs(df)

        # Show classified data including accuracy percentage in Streamlit
        st.write('Classified Drugs:')
        st.write(classified_df)

        # Provide a download link for the classified data
        st.markdown(get_table_download_link(classified_df), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
