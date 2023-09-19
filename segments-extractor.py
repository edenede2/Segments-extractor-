import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def plot_line_3d(data, subjects, events, measure, lower_bound, upper_bound):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]
    
    fig = go.Figure()
    
    # Iterate through each event to plot separate lines
    for event in events:
        event_data = data_subset[data_subset['Events'] == event]
        fig.add_trace(go.Scatter3d(x=event_data['Segments'], 
                                   y=event_data[measure],
                                   z=event_data['Events'].astype(str),  # Convert event to string for plotting
                                   mode='lines',
                                   name=event))
    
    fig.update_layout(scene=dict(
            xaxis_title='Segments',
            yaxis_title=measure,
            zaxis_title='Events'
        ),
        title=f"{measure} across Events for selected subjects in 3D"
    )
    st.plotly_chart(fig)

def plot_line(data, subjects, events, measure, lower_bound, upper_bound):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]
    
    fig = px.line(data_subset, x='Segments', y=measure, color='Events', line_dash='Subjects', 
                  title=f"{measure} across Events for selected subjects", 
                  labels={'Segments': 'Segments', measure: measure}, 
                  hover_data=['Subjects', 'Events'])

    st.plotly_chart(fig)

def plot_box(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    fig = go.Figure()

    for event in events:  # corrected variable here
        event_data = data_subset[data_subset['Events'] == event]
        fig.add_trace(go.Box(y=event_data[measure], x=event_data[x_var], name=event, boxpoints='all', jitter=0.3, pointpos=-1.8))
        
    fig.update_layout(title=f"Box plot of {measure} across {x_var}", xaxis_title=x_var, yaxis_title=measure)
    st.plotly_chart(fig)
    
def plot_violin(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    fig = px.violin(data_subset, x=x_var, y=measure, color='Events', box=True, points="all", hover_data=['Subjects', 'Events'])
    st.plotly_chart(fig)

    
def plot_swarm(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    plt.figure(figsize=(14, 7))
    sns.swarmplot(data=data_subset, x=x_var, y=measure, hue='Events')
    plt.title(f"Swarm plot of {measure} across {x_var}")
    st.pyplot()
    plt.close()

def plot_facet(data, subjects, events, measure):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    fig = px.line(data_subset, x='Segments', y=measure, color='Events', facet_col='Subjects', hover_data=['Subjects', 'Events'])
    st.plotly_chart(fig)

def plot_full_data_bar(data, measure):
    fig = px.bar(data, x='Subjects', y=measure, color='Events', title=f"{measure} for Full Subjects")
    st.plotly_chart(fig)

def plot_full_data_box(data, measure):
    fig = px.box(data, x='Events', y=measure, title=f"Box Plot of {measure} for Full Subjects")
    st.plotly_chart(fig)

def plot_full_data_violin(data, measure):
    fig = px.violin(data, x='Events', y=measure, box=True, points="all", title=f"Violin Plot of {measure} for Full Subjects")
    st.plotly_chart(fig)

def plot_full_data_histogram(data, measure):
    fig = px.histogram(data, x=measure, color='Events', title=f"Histogram of {measure} for Full Subjects")
    st.plotly_chart(fig)

def plot_full_data_swarm(data, measure):
    plt.figure(figsize=(14, 7))
    sns.swarmplot(data=data, x='Events', y=measure)
    plt.title(f"Swarm plot of {measure} for Full Subjects")
    st.pyplot()
    plt.close()

def main():
    st.title("Visualization App for Total_segments_Val.csv")
    
    file_option = st.radio("Choose a CSV file source:", ["Use default file", "Upload my own file"])
    
    if file_option == "Use default file":
        data = pd.read_csv('Total_segments_Val.csv')
        full_data = data[data['Subjects'].str.startswith("Full_")]
        st.write("Preview of the Default Data")
    elif file_option == "Upload my own file":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of the Uploaded Data")
        else:
            st.warning("Please upload a CSV file.")
            return  # This ensures the rest of the code doesn't run until a file is uploaded

    st.write(data.head())

    all_subjects = data['Subjects'].unique().tolist()
    selected_subjects = st.multiselect('Select Subjects', all_subjects, default=all_subjects)

    all_events = data['Events'].unique().tolist()
    selected_events = st.multiselect('Select Events', all_events, default=all_events)
    
    # If specific events are selected, display segments relevant to those events
     # Filter segments based on selected subjects and events
    filtered_data = data[(data['Subjects'].isin(selected_subjects)) & (data['Events'].isin(selected_events))]
    relevant_segments = filtered_data['Segments'].unique().tolist()
    selected_segments = st.multiselect('Select Segments', relevant_segments, default=relevant_segments)
    filtered_data = filtered_data[filtered_data['Segments'].isin(selected_segments)]

    measurements = ['RMSSD', 'SDNN','MHR']
    selected_measurements = st.selectbox('Select Measurements', measurements, index=0)

    plot_types = ["Line Plot", "Line Plot 3d", "Box Plot", "Violin Plot", "Swarm Plot", "Facet Grid"]
    selected_plot = st.selectbox('Select Visualization Type', plot_types)

    remove_outliers = st.checkbox("Remove Outliers", value=False)
    lower_bound, upper_bound = None, None
    if remove_outliers:
        lower_bound = st.number_input('Enter Lower Bound for Outliers', value=data[selected_measurements].quantile(0.20))
        upper_bound = st.number_input('Enter Upper Bound for Outliers', value=data[selected_measurements].quantile(0.80))

    if st.button("Generate Plot"):
        if selected_plot == "Line Plot":
            plot_line(filtered_data, selected_subjects, selected_events, selected_measurements, lower_bound, upper_bound)
        elif selected_plot == "Line Plot 3d":
            plot_line_3d(filtered_data, selected_subjects, selected_events, selected_measurements, lower_bound, upper_bound)
        elif selected_plot == "Box Plot":
            plot_box(filtered_data, selected_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
        elif selected_plot == "Violin Plot":
            plot_violin(filtered_data, selected_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
        elif selected_plot == "Swarm Plot":
            plot_swarm(filtered_data, selected_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
        elif selected_plot == "Facet Grid":
            plot_facet(filtered_data, selected_subjects, selected_events, selected_measurements)

st.markdown("## Visualization for Full Data")
 
full_data = data[data['Subjects'].str.startswith('Full_')]

full_measurements = ['RMSSD', 'SDNN', 'MHR']
selected_full_measurement = st.selectbox('Select Measurement for Full Data', full_measurements)

full_plot_types = ["Box Plot", "Violin Plot", "Histogram", "Swarm Plot"]
selected_full_plot = st.selectbox('Select Visualization Type for Full Data', full_plot_types)

all_full_events = full_data['Events'].unique().tolist()
selected_full_events = st.multiselect('Select Events for Full Data', all_full_events, default=all_full_events)

filtered_full_data = full_data[full_data['Events'].isin(selected_full_events)]

if st.button("Generate Full Data Plot"):
    # Depending on the type of plot selected, call the appropriate function
    if selected_full_plot == "Box Plot":
        plot_full_data_box(full_data, selected_full_measurement)
    elif selected_full_plot == "Violin Plot":
        plot_full_data_violin(full_data, selected_full_measurement)
    elif selected_full_plot == "Histogram":
        plot_full_data_histogram(full_data, selected_full_measurement)
    elif selected_full_plot == "Swarm Plot":
        plot_full_data_swarm(full_data, selected_full_measurement)

if __name__ == "__main__":
    main()
