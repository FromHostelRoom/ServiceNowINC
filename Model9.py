import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load the trained models and vectorizer
model_category = joblib.load('model_category.pkl')
model_root_cause = joblib.load('model_root_cause.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.title("ServiceNow INC Impacted Module/Functionality and Root Cause Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Load data from the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    
    # Display the data
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Ensure the required columns exist
    required_columns = ['Short description', 'Category(u_category)', 'Configuration Item', 'Close notes']
    missing_columns = [column for column in required_columns if column not in df.columns]
    
    if missing_columns:
        st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
    else:
        # Drop rows with missing values in the required columns
        df.dropna(subset=required_columns, inplace=True)
        
        # Dropdown for selecting Configuration Item
        application = df['Configuration Item'].unique()
        selected_application = st.selectbox("Select Application:", application)
        
        # Filter data based on selected Configuration Item
        df_filtered = df[df['Configuration Item'] == selected_application].copy()
        
        # Combine text columns for feature extraction
        df_filtered.loc[:, 'Combined Text'] = df_filtered['Short description'] + ' ' + df_filtered['Close notes'] + ' ' + df_filtered['Category(u_category)']
        
        # Split data into features and labels
        X = df_filtered['Combined Text']
        y_category = df_filtered['Category(u_category)']
        y_root_cause = df_filtered['Resolution Code/Category'].astype(str) if 'Resolution Code/Category' in df_filtered.columns else None
        
        # Feature extraction using TF-IDF
        X_tfidf = vectorizer.transform(X)
        
        # Predict category
        df_filtered.loc[:, 'Predicted Impacted Module/Functionality'] = model_category.predict(X_tfidf)
        
        # Predict root cause if the column exists
        if y_root_cause is not None:
            df_filtered.loc[:, 'Predicted Root Cause'] = model_root_cause.predict(X_tfidf).astype(str)
        
        # Display predictions in tabular form
        st.write("Predictions:")
        st.dataframe(df_filtered[['Short description', 'Close notes', 'Description', 'Predicted Impacted Module/Functionality', 'Predicted Root Cause']])
        # Evaluate the models
        accuracy_category = accuracy_score(y_category, df_filtered['Predicted Impacted Module/Functionality'])
        report_category = classification_report(y_category, df_filtered['Predicted Impacted Module/Functionality'], zero_division=1, output_dict=True)
        
        st.write("Issue Model Accuracy:", accuracy_category)
        
        # Plot classification report for categorization
        st.write("Classification Report for Module/Func Impacted:")
        report_category_df = pd.DataFrame(report_category).transpose()
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        report_category_df.iloc[:-1, :-1]['precision'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Precision')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45, labelsize=10)
        
        report_category_df.iloc[:-1, :-1]['recall'].plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Recall')
        axes[1].set_ylabel('Score')
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        
        report_category_df.iloc[:-1, :-1]['f1-score'].plot(kind='bar', ax=axes[2], color='salmon')
        axes[2].set_title('F1-Score')
        axes[2].set_ylabel('Score')
        axes[2].tick_params(axis='x', rotation=45, labelsize=10)
        
        # Add space between x-axis labels
        plt.xticks(ticks=np.arange(len(report_category_df.index[:-1])), labels=report_category_df.index[:-1], rotation=45, ha='right', fontsize=10)
        plt.xlabel('Classes')
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)
        # Plot confusion matrix for categorization
        st.write("Confusion Matrix :")
        conf_matrix_category = confusion_matrix(y_category, df_filtered['Predicted Impacted Module/Functionality'])
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(conf_matrix_category, annot=True, fmt='d', cmap='Blues', xticklabels=model_category.classes_, yticklabels=model_category.classes_, ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Categorization')
        st.pyplot(fig)
        # Evaluate and plot root cause prediction if the column exists
        if y_root_cause is not None:
            accuracy_root_cause = accuracy_score(y_root_cause, df_filtered['Predicted Root Cause'])
            report_root_cause = classification_report(y_root_cause, df_filtered['Predicted Root Cause'], zero_division=1, output_dict=True)
            
            st.write("Root Cause Prediction Model Accuracy:", accuracy_root_cause)
            
            st.write("Classification Report for Root Cause Prediction:")
            report_root_cause_df = pd.DataFrame(report_root_cause).transpose()
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            
            report_root_cause_df.iloc[:-1, :-1]['precision'].plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title('Precision')
            axes[0].set_ylabel('Score')
            axes[0].tick_params(axis='x', rotation=45, labelsize=10)
            
            report_root_cause_df.iloc[:-1, :-1]['recall'].plot(kind='bar', ax=axes[1], color='lightgreen')
            axes[1].set_title('Recall')
            axes[1].set_ylabel('Score')
            axes[1].tick_params(axis='x', rotation=45, labelsize=10)
            
            report_root_cause_df.iloc[:-1, :-1]['f1-score'].plot(kind='bar', ax=axes[2], color='salmon')
            axes[2].set_title('F1-Score')
            axes[2].set_ylabel('Score')
            axes[2].tick_params(axis='x', rotation=45, labelsize=10)
            
            # Add space between x-axis labels
            plt.xticks(ticks=np.arange(len(report_root_cause_df.index[:-1])), labels=report_root_cause_df.index[:-1], rotation=45, ha='right', fontsize=10)
            plt.xlabel('Classes')
            plt.tight_layout(pad=3.0)
            st.pyplot(fig)
