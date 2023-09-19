import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_line(data, subjects, events, measure, lower_bound, upper_bound):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data_subset, x='Segments', y=measure, hue='Events', style='Subjects')
    plt.title(f"{measure} across Events for selected subjects")
    plt.show()
    plt.close()

def plot_box(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=data_subset, x=x_var, y=measure, hue='Events')
    plt.title(f"Box plot of {measure} across {x_var}")
    plt.show()
    plt.close()

def plot_violin(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    plt.figure(figsize=(14, 7))
    sns.violinplot(data=data_subset, x=x_var, y=measure, hue='Events')
    plt.title(f"Violin plot of {measure} across {x_var}")
    plt.show()
    plt.close()
    
def plot_swarm(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    plt.figure(figsize=(14, 7))
    sns.swarmplot(data=data_subset, x=x_var, y=measure, hue='Events')
    plt.title(f"Swarm plot of {measure} across {x_var}")
    plt.show()
    plt.close()

def plot_facet(data, subjects, events, measure):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    g = sns.FacetGrid(data_subset, col="Subjects", col_wrap=4, height=4)
    g = g.map(plt.plot, 'Segments', measure).add_legend()
    plt.show()
    plt.close()

def main():
    st.title("Visualization App for Total_segments_Val.csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of the Data")
        st.write(data.head())

        all_subjects = data['Subjects'].unique().tolist()
        selected_subjects = st.multiselect('Select Subjects', all_subjects, default=all_subjects)

        all_events = data['Events'].unique().tolist()
        selected_events = st.multiselect('Select Events', all_events, default=all_events)

        all_segments = data['Segments'].unique().tolist()
        selected_segments = st.multiselect('Select Segments', all_segments, default=all_segments)

        measurements = ['RMSSD', 'SDNN']
        selected_measurements = st.selectbox('Select Measurements', measurements, index=0)

        plot_types = ["Line Plot", "Box Plot", "Violin Plot", "Swarm Plot", "Facet Grid"]
        selected_plot = st.selectbox('Select Visualization Type', plot_types)

        remove_outliers = st.checkbox("Remove Outliers", value=False)
        lower_bound, upper_bound = None, None
        if remove_outliers:
            lower_bound = st.number_input('Enter Lower Bound for Outliers', value=data[selected_measurements].quantile(0.20))
            upper_bound = st.number_input('Enter Upper Bound for Outliers', value=data[selected_measurements].quantile(0.80))

        if st.button("Generate Plot"):
            if selected_plot == "Line Plot":
                plot_line(data, selected_subjects, selected_events, selected_measurements, lower_bound, upper_bound)
            elif selected_plot == "Box Plot":
                plot_box(data, selected_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
            elif selected_plot == "Violin Plot":
                plot_violin(data, selected_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
            elif selected_plot == "Swarm Plot":
                plot_swarm(data, selected_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
            elif selected_plot == "Facet Grid":
                plot_facet(data, selected_subjects, selected_events, selected_measurements)

if __name__ == "__main__":
    main()

