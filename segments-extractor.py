import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# Function to load data based on user input
def load_data(file_option):
    if file_option == "Use default file":
        return pd.read_csv('Total_segments_Val.csv')
    elif file_option == "Upload my own file":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            return pd.read_csv(uploaded_file)
    return None


def transform_to_log(data, measurements):
    transformed_data = data.copy()
    for measure in measurements:
        transformed_data[measure] = np.log1p(transformed_data[measure])
    return transformed_data


def categorize_mast_measures(data):
    categorized_data = data.copy()
    for subject in categorized_data['Subjects'].unique():
        subject_data = categorized_data[categorized_data['Subjects'] == subject]
        baseline_value = subject_data[subject_data['Events'] == 'rest baseline']['RMSSD'].mean()
        mast_value = subject_data[subject_data['Events'] == 'MAST']['RMSSD'].mean()

        if mast_value > baseline_value:
            category = 'Higher'
        elif mast_value < baseline_value:
            category = 'Lower'
        else:
            category = 'Unchanged'

        categorized_data.loc[categorized_data['Subjects'] == subject, 'MAST_Category'] = category

    return categorized_data


# Visualization functions for segmented and non-segmented data
def visualize_data(data, subjects, events, measurements, plot_types, selected_plot, lower_bound=None, upper_bound=None):
    filtered_data = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    if lower_bound and upper_bound:
        filtered_data = filtered_data[(filtered_data[measurements] >= lower_bound) & (filtered_data[measurements] <= upper_bound)]

    if selected_plot == "Line Plot":
        plot_line(filtered_data, subjects, events, measurements)
    elif selected_plot == "Line Plot 3d":
        plot_line_3d(filtered_data, subjects, events, measurements)
    # Add more elif conditions for other plot types


# Main Streamlit app
def main():
    st.title("Visualization App for Total_segments_Val.csv")

    # Choose a CSV file source
    file_option = st.radio("Choose a CSV file source:", ["Use default file", "Upload my own file"])
    st.sidebar.markdown("## Advanced Analysis Options")

    # Load data
    data = load_data(file_option)
    if data is None:
        st.warning("Please upload a CSV file.")
        return

    st.write("Preview of the Data")
    st.write(data.head())

    # Advanced Analysis Options
    if st.sidebar.checkbox("Categorize MAST Measures"):
        categorized_data = categorize_mast_measures(data)
        # visualize_mast_categories(categorized_data)  # Implement this function for visualization

    # Visualization for segmented data
    all_segmented_subjects = data[~data['Subjects'].str.startswith('Full_')]['Subjects'].unique().tolist()
    selected_segmented_subjects = st.multiselect('Select Segmented Subjects', all_segmented_subjects, default=all_segmented_subjects)
    all_events = data['Events'].unique().tolist()
    selected_events = st.multiselect('Select Events', all_events, default=all_events)
    measurements = ['RMSSD', 'SDNN', 'MHR']
    selected_measurements = st.selectbox('Select Measurements', measurements, index=0)
    plot_types = ["Line Plot", "Line Plot 3d", "Box Plot", "Violin Plot", "Swarm Plot", "Facet Grid"]
    selected_plot = st.selectbox('Select Visualization Type', plot_types)

    if st.button("Generate Plot for Segmented Data"):
        visualize_data(data, selected_segmented_subjects, selected_events, selected_measurements, plot_types, selected_plot)

    # Visualization for non-segmented data
    st.markdown("## Visualization for non-segmented Data")
    all_full_subjects = data[data['Subjects'].str.startswith('Full_')]['Subjects'].unique().tolist()
    selected_full_subjects = st.multiselect('Select Full Subjects', all_full_subjects, default=all_full_subjects)

    if st.button("Generate Plot for Non-Segmented Data"):
        visualize_data(data, selected_full_subjects, selected_events, selected_measurements, plot_types, selected_plot)


if __name__ == "__main__":
    main()

