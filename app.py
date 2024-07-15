import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tempfile import NamedTemporaryFile
import os

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

# Function to classify drugs
def classify_drugs(drugs):
    global classifier, vectorizer

    if classifier is None or vectorizer is None:
        train_model()

    X_new = vectorizer.transform(drugs['Drug'])
    drugs['Category'] = classifier.predict(X_new)

    return drugs

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

        # Classify drugs
        classified_df = classify_drugs(df)

        # Download classified file
        with NamedTemporaryFile(delete=False) as tmp_file:
            classified_df.to_csv(tmp_file.name, index=False)
            st.download_button('Download Classified File', tmp_file.name, label='Click here')

        # Remove temporary file
        os.remove(tmp_file.name)

        # Show classified data in Streamlit
        st.write('Classified Drugs:')
        st.write(classified_df)

if __name__ == '__main__':
    main()
