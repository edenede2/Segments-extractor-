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
                                   y=event_data['subjects'],
                                   z=event_data['Events'].astype(str),  # Convert event to string for plotting
                                   mode='lines',
                                   name=event))
    
    fig.update_layout(scene=dict(
            xaxis_title='Segments',
            yaxis_title='subjects',
            zaxis_title='Events'
        ),
        title=f"{measure} across Events for selected subjects in 3D"
    )
    st.plotly_chart(fig)



def plot_box(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    fig = go.Figure()

    for event in selected_events:
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


# ... [other parts of the code remain unchanged]

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
        
        # If specific events are selected, display segments relevant to those events
        if selected_events:
            filtered_data = data[data['Events'].isin(selected_events)]
            relevant_segments = filtered_data['Segments'].unique().tolist()
        else:
            relevant_segments = data['Segments'].unique().tolist()
        
        selected_segments = st.multiselect('Select Segments', relevant_segments, default=relevant_segments)

        # ... [rest of the code remains unchanged]

        measurements = ['RMSSD', 'SDNN']
        selected_measurements = st.selectbox('Select Measurements', measurements, index=0)

        plot_types = ["Line Plot 3d", "Box Plot", "Violin Plot", "Swarm Plot", "Facet Grid"]
        selected_plot = st.selectbox('Select Visualization Type', plot_types)

        remove_outliers = st.checkbox("Remove Outliers", value=False)
        lower_bound, upper_bound = None, None
        if remove_outliers:
            lower_bound = st.number_input('Enter Lower Bound for Outliers', value=data[selected_measurements].quantile(0.20))
            upper_bound = st.number_input('Enter Upper Bound for Outliers', value=data[selected_measurements].quantile(0.80))

        if st.button("Generate Plot"):
            if selected_plot == "Line Plot":
                plot_line_3d(data, selected_subjects, selected_events, selected_measurements, lower_bound, upper_bound)
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

