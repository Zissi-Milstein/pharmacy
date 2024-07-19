import streamlit as st
import pandas as pd
from transformers import pipeline
import base64
import os 
import requests
from zipfile import ZipFile

# Global variable to store the model pipeline
classifier_pipeline = None

# Function to load the fine-tuned model pipeline
def load_model():
    global classifier_pipeline
    model_path = "./fine-tuned-model"
    classifier_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path)

def classify_drugs(drugs):
    global classifier_pipeline

    if classifier_pipeline is None:
        load_model()

    # Obtain predictions from the model
    descriptions = drugs['Description'].tolist()
    predictions = classifier_pipeline(descriptions)

    # Debug: Print the predictions to understand their structure
    st.write(predictions)

    # Define label mapping for the catagories
    label_mapping = {
        "LABEL_10" : "RX",
        "LABEL_13" : "SUPPLY-NON IV",
        "LABEL_3" : "IV DRUG",
        "LABEL_7" : "MISC",
        "LABEL_8" : "OTC"
    }

    # Function to map the label to the category
    def map_label(label):
        return label_mapping.get(label, 'unknown')  # Default to 'unknown' if label is not in mapping

    try:
        # Extract and map the labels
        drugs['Predicted_Category'] = [map_label(pred['label']) for pred in predictions]
    except KeyError as e:
        st.error(f"KeyError: {e} - Check if the label from the model is in the label_mapping.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

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
