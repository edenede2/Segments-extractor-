# StreamlitApp.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Your functions and data loading here ...

# Streamlit interface
def main():
    st.title("Visualization App for Total_segments_Val.csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.write("Preview of the Data")
        st.write(data.head())

        # Select Subjects
        all_subjects = data['Subjects'].unique().tolist()
        selected_subjects = st.multiselect('Select Subjects', all_subjects, default=all_subjects)

        # Select Events
        all_events = data['Events'].unique().tolist()
        selected_events = st.multiselect('Select Events', all_events, default=all_events)

        # Select Segments
        all_segments = data['Segments'].unique().tolist()
        selected_segments = st.multiselect('Select Segments', all_segments, default=all_segments)

        # Select Measurements
        measurements = ['RMSSD', 'SDNN']
        selected_measurements = st.selectbox('Select Measurements', measurements, index=0)

        # Select Visualization type
        plot_types = ["Line Plot", "Box Plot", "Violin Plot", "Swarm Plot", "Facet Grid"]
        selected_plot = st.selectbox('Select Visualization Type', plot_types)

        # Additional options
        remove_outliers = st.checkbox("Remove Outliers", value=False)

        if st.button("Generate Plot"):
            if selected_plot == "Line Plot":
                plot_line(selected_subjects, selected_events, selected_measurements, remove_outliers)
            elif selected_plot == "Box Plot":
                plot_box(selected_subjects, selected_events, selected_measurements, x_var='Events', remove_outliers=remove_outliers)
            elif selected_plot == "Violin Plot":
                plot_violin(selected_subjects, selected_events, selected_measurements, x_var='Events', remove_outliers=remove_outliers)
            elif selected_plot == "Swarm Plot":
                plot_swarm(selected_subjects, selected_events, selected_measurements, x_var='Events', remove_outliers=remove_outliers)
            elif selected_plot == "Facet Grid":
                plot_facet(selected_subjects, selected_events, selected_measurements)

if __name__ == "__main__":
    main()
