import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import base64

# Global variable to store the model pipeline
classifier_pipeline = None

# Function to load the fine-tuned model pipeline
def load_model():
    global classifier_pipeline
    model_path = "./fine-tuned-model"
    classifier_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Function to classify drugs and calculate accuracy percentage
def classify_drugs(drugs):
    global classifier_pipeline

    if classifier_pipeline is None:
        load_model()

    # Use the fine-tuned model to classify the drug descriptions
    drugs['Predicted_Category'] = drugs['Description'].apply(lambda x: classifier_pipeline(x)[0]['label'])
    
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
