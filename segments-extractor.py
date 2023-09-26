import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scipy.stats as stats
import plotly.subplots as sp


def transform_to_log(data, measurements):

    transformed_data = data.copy()
    for measure in measurements:
        transformed_data[measure] = np.log1p(transformed_data[measure])  # Use log1p to handle values <= 0
    return transformed_data



def calculate_percentage_change(data, event1, event2):
    
    event1_data = data[data['Events'] == event1]
    event2_data = data[data['Events'] == event2]

    change_data = ((event2_data.set_index('Subjects')[['RMSSD', 'SDNN', 'MHR']] - event1_data.set_index('Subjects')[['RMSSD', 'SDNN', 'MHR']]) / event1_data.set_index('Subjects')[['RMSSD', 'SDNN', 'MHR']]) * 100
    change_data.reset_index(inplace=True)
    
    return change_data


def categorize_subjects(change_sc2_sc1, change_sc3_sc1, threshold):
    change_sc2_sc1['Category'] = change_sc2_sc1[['RMSSD', 'SDNN', 'MHR']].apply(lambda row: 'resilient' if max(abs(row)) <= threshold else 'sustainable', axis=1)
    change_sc3_sc1['Category'] = change_sc3_sc1[['RMSSD', 'SDNN', 'MHR']].apply(lambda row: 'resilient' if max(abs(row)) <= threshold else 'sustainable', axis=1)
    
    return change_sc2_sc1, change_sc3_sc1

def plot_line_3d(data, subjects, events, measure, lower_bound, upper_bound):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]
    
    fig = go.Figure()
    
    # Iterate through each subject to plot separate lines
    for subject in subjects:
        subject_data = data_subset[data_subset['Subjects'] == subject]
        fig.add_trace(go.Scatter3d(x=[subject]*len(subject_data),  # Setting constant x value for each subject
                                   y=subject_data['Events'],
                                   z=subject_data[measure],
                                   mode='lines',
                                   name=subject))
    
    fig.update_layout(scene=dict(
            xaxis_title='Subjects',
            yaxis_title='Events',
            zaxis_title=measure
        ),
        title=f"{measure} across Subjects for selected events in 3D"
    )
    st.plotly_chart(fig)


def plot_line(data, subjects, events, measure, lower_bound, upper_bound):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]
    
    fig = px.line(data_subset, x='Segments', y=measure, color='Subjects', line_dash='Events',
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

    fig = px.violin(data_subset, x=x_var, y=measure, color=data_subset['Subjects'].map(subject_categories), box=True, points="all", hover_data=['Subjects', 'Events'])
    st.plotly_chart(fig)

    
def plot_violin(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    fig = px.violin(data_subset, x=x_var, y=measure, box=True, points="all", hover_data=['Subjects', 'Events'])
    st.plotly_chart(fig)

def plot_swarm(data, subjects, events, measure, x_var='Subjects', lower_bound=None, upper_bound=None):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    
    if lower_bound and upper_bound:
        data_subset = data_subset[(data_subset[measure] >= lower_bound) & (data_subset[measure] <= upper_bound)]

    plt.figure(figsize=(14, 7))
    sns.swarmplot(data=data_subset, x=x_var, y=measure)
    plt.title(f"Swarm plot of {measure} across {x_var}")
    st.pyplot()
    plt.close()

def plot_facet(data, subjects, events, measure):
    data_subset = data[data['Subjects'].isin(subjects) & data['Events'].isin(events)]
    fig = px.line(data_subset, x='Segments', y=measure, color=data_subset['Subjects'].map(subject_categories), facet_col='Subjects', hover_data=['Subjects', 'Events'])
    st.plotly_chart(fig)
    
def plot_full_line_plot(data, measure):
    # Check whether there is any variation in the data
    if len(data[measure].unique()) > 1:
        fig = px.line(data, x='Events', y=measure, color='Subjects',
                      title=f"{measure} across Events for each Subject", 
                      hover_data=['Subjects', 'Events'])
        st.plotly_chart(fig)
    else:
        st.warning(f"No variation in {measure} across Events for the selected subjects.")



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
    fig = px.histogram(data, x=measure, title=f"Histogram of {measure} for Full Subjects")
    st.plotly_chart(fig)


def plot_full_data_swarm(data, measure):
    plt.figure(figsize=(14, 7))
    sns.swarmplot(data=data, x='Events', y=measure)
    plt.title(f"Swarm plot of {measure} for Full Subjects")
    st.pyplot()
    plt.close()

def plot_resilience_scatter(full_data, threshold, red_subjects, orange_subjects):
    # Calculate the mean RMSSD and SDNN values for each full subject across all events
    mean_values = full_data.groupby('Subjects')[['RMSSD', 'SDNN']].mean().reset_index()
    
    # Color the points based on whether the subject is in the red_subjects or orange_subjects list
    mean_values['color'] = mean_values['Subjects'].apply(lambda x: 'red' if x in red_subjects else ('orange' if x in orange_subjects else 'blue'))
    
    # Create the scatter plot
    fig = px.scatter(mean_values, x='SDNN', y='RMSSD', color='color', 
                     color_discrete_map={'red': 'red', 'orange': 'orange', 'blue': 'blue'},
                     labels={'SDNN': 'SDNN (Mean)', 'RMSSD': 'RMSSD (Mean)'},
                     hover_data=['Subjects'])  # Add Subjects to hover data
    st.plotly_chart(fig)




def resilience_sustainability_page():
    st.title("Resilience and Susceptible Analysis")
    
    # Load the data
    data = load_data()
    
    # Filter only full (non-segmented) subjects
    full_data = data[data['Subjects'].str.startswith('Full_')]

    # Allow user to select the events they want to compare
    event1 = st.selectbox("Select the first event (baseline):", all_events, index=all_events.index('rest baseline') if 'rest baseline' in all_events else 0)
    event2 = st.selectbox("Select the second event (comparison):", all_events, index=all_events.index('MAST') if 'MAST' in all_events else 1)
    
    # Calculate the mean HRV values for each subject
    mean_hrv_values = full_data.groupby('Subjects')[['SDNN', 'RMSSD', 'MHR']].mean().reset_index()
    
    # Define thresholds for categorization (you can modify these values)
    sdnn_threshold = mean_hrv_values['SDNN'].mean()
    rmssd_threshold = mean_hrv_values['RMSSD'].mean()
    mhr_threshold = mean_hrv_values['MHR'].mean()
    
    # Categorize subjects based on the mean HRV values
    mean_hrv_values['HRV_Category'] = np.where(
        (mean_hrv_values['SDNN'] >= sdnn_threshold) & 
        (mean_hrv_values['RMSSD'] >= rmssd_threshold) & 
        (mean_hrv_values['MHR'] >= mhr_threshold), 'High', 'Low'
    )
    
    # Merge the HRV category back to the full_data DataFrame
    full_data = full_data.merge(mean_hrv_values[['Subjects', 'HRV_Category']], on='Subjects', how='left')
    
    threshold = st.slider("Set Threshold for Highlighting Significant Change (%)", min_value=0, max_value=100, value=10, key='resilience_threshold')
    
    # Allow user to select the measurement they want to visualize
    measurement = st.selectbox("Select Measurement:", ['RMSSD', 'SDNN', 'MHR'])

    # Calculate percentage change for selected events
    change_data = calculate_percentage_change(full_data, event1, event2)
    

    # Identify the subjects that are marked red in the percentage change plots
    red_subjects_sc2 = change_sc2_sc1[change_sc2_sc1[['RMSSD', 'SDNN', 'MHR']].apply(lambda x: abs(x) > threshold, axis=1).any(axis=1)]['Subjects'].tolist()
    red_subjects_sc3 = change_sc3_sc1[change_sc3_sc1[['RMSSD', 'SDNN', 'MHR']].apply(lambda x: abs(x) > threshold, axis=1).any(axis=1)]['Subjects'].tolist()

    # Subjects that show change in both scenarios 1-2 and 1-3 are marked as red
    red_subjects = list(set(red_subjects_sc2) & set(red_subjects_sc3))

    # Subjects that show change only in scenario 1-2 are marked as orange
    orange_subjects = list(set(red_subjects_sc2) - set(red_subjects_sc3))

    def plot_with_threshold(change_data, scenario, measurement, threshold):
        # Define colors based on threshold
        colors = ['#d62728' if abs(val) > threshold else '#1f77b4' for val in change_data[measurement]]
        
        # Calculate the percentage of subjects under the threshold
        under_threshold_percentage = len(change_data[abs(change_data[measurement]) <= threshold]) / len(change_data) * 100
        
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Scatter Plot", "Bar Plot"))
        
        # Scatter plot
        scatter_trace = go.Scatter(x=change_data['Subjects'], y=change_data[measurement], mode='markers', marker=dict(color=colors))
        fig.add_trace(scatter_trace, row=1, col=1)
        
        # Bar plot
        bar_trace = go.Bar(x=change_data['Subjects'], y=change_data[measurement], marker=dict(color=colors))
        fig.add_trace(bar_trace, row=1, col=2)
        
        # Add annotation with percentage under the threshold
        fig.add_annotation(
            text=f"{under_threshold_percentage:.2f}% under threshold",
            xanchor='left',
            x=0,
            yanchor='top',
            y=1.05,
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            borderpad=4
        )
        
        # Check if yaxis range is not None, otherwise set default values
        if fig.layout.yaxis.range is not None:
            y1 = fig.layout.yaxis.range[1]
            y0 = fig.layout.yaxis.range[0]
        else:
            y1 = max(change_data[measurement])
            y0 = min(change_data[measurement])
        
        # Re-add reference line after adding annotation to ensure it's visible
        for col in range(1, 3):
            fig.add_shape(
                go.layout.Shape(
                    type='rect',
                    y0=threshold,
                    y1=y1,
                    xref='paper',
                    x0=0,
                    x1=1,
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    layer='below',
                    line_width=0
                ),
                row=1,
                col=col
            )
            fig.add_shape(
                go.layout.Shape(
                    type='rect',
                    y0=y0,
                    y1=-threshold,
                    xref='paper',
                    x0=0,
                    x1=1,
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    layer='below',
                    line_width=0
                ),
                row=1,
                col=col
            )
        fig.update_layout(title_text=f'Percentage Change in {measurement}: {scenario}')
        st.plotly_chart(fig)

     # Percentage Change in selected_measurements: event2 vs event1
    st.markdown(f"### Percentage Change in {measurement}: {event2} vs {event1}")
    plot_with_threshold(change_data, f"{event2} vs {event1}", measurement, threshold)

    st.markdown("### Resilience Scatter Plot")
    #     Call the modified scatter plot function
    plot_resilience_scatter(full_data, threshold, red_subjects, orange_subjects)

    
    # Move the affected/unaffected functionality to the bottom of the resilience page
    st.markdown("## Categorize Subjects Based on Z-Score")
    zscore_threshold = st.slider("Set Z-Score Threshold for Categorization", min_value=0.0, max_value=3.0, value=1.96, key='zscore_threshold')

    # Calculate the difference in measurements between MAST and rest baseline for full subjects
    mast_data = full_data[full_data['Events'] == 'MAST']
    baseline_data = full_data[full_data['Events'] == 'rest baseline']
    difference_data = mast_data.set_index('Subjects')[['RMSSD', 'SDNN']] - baseline_data.set_index('Subjects')[['RMSSD', 'SDNN']]
    difference_data.reset_index(inplace=True)

    # Calculate the z-score of the differences
    difference_data['RMSSD_zscore'] = stats.zscore(difference_data['RMSSD'])
    difference_data['SDNN_zscore'] = stats.zscore(difference_data['SDNN'])

    # Categorize subjects as "affected" or "unaffected" based on the z-score threshold
    difference_data['RMSSD_category'] = difference_data['RMSSD_zscore'].apply(lambda x: 'affected' if abs(x) > zscore_threshold else 'unaffected')
    difference_data['SDNN_category'] = difference_data['SDNN_zscore'].apply(lambda x: 'affected' if abs(x) > zscore_threshold else 'unaffected')

    # Display the categorized subjects
    st.write("Categorized Subjects based on RMSSD:", difference_data[['Subjects', 'RMSSD_category']])
    st.write("Categorized Subjects based on SDNN:", difference_data[['Subjects', 'SDNN_category']])

def load_data():
    """
    Load the data for the application.
    Return the loaded DataFrame.
    """
    data = pd.read_csv('Total_segments_Val.csv')
    return data    

def main_page():
    st.title("Visualization App for Total_segments_Val.csv")
    data = load_data()
    original_data = data.copy()
    
    file_option = st.radio("Choose a CSV file source:", ["Use default file", "Upload my own file"])
    
    if file_option == "Use default file":
        st.write("Preview of the Default Data")
    elif file_option == "Upload my own file":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            original_data = data.copy()
            st.write("Preview of the Uploaded Data")
        else:
            st.warning("Please upload a CSV file.")
            return  # This ensures the rest of the code doesn't run until a file is uploaded

        # Filter data for MAST and rest baseline events
    mast_data = data[data['Events'] == 'MAST']
    baseline_data = data[data['Events'] == 'rest baseline']

    st.write(data.head())
    
    st.markdown("## Transform Segment Data")
    log_transform = st.button("Transform Segment Data to Log Scale")
    if log_transform:
        measurements_to_transform = ['RMSSD', 'SDNN','MHR']
        data = transform_to_log(data, measurements_to_transform)
        st.write("Segment data has been transformed to log scale.")
        if st.button("Cancel Log Transformation"):
            if original_data is not None:
                data = original_data.copy()
                st.write("Log transformation has been reverted.")
            else:
                st.warning("Unable to revert the log transformation. Please reload or re-upload the original data.")
    

    all_full_subjects = data[data['Subjects'].str.startswith('Full_')]['Subjects'].unique().tolist()
    all_segmented_subjects = data[~data['Subjects'].str.startswith('Full_')]['Subjects'].unique().tolist()
    selected_segmented_subjects = st.multiselect('Select Segmented Subjects', all_segmented_subjects, default=all_segmented_subjects)

    
    all_events = data['Events'].unique().tolist()
    selected_events = st.multiselect('Select Events', all_events, default=all_events)
    
    # If specific events are selected, display segments relevant to those events
    # Filter segments based on selected subjects and events
    filtered_data = data[(data['Subjects'].isin(selected_segmented_subjects)) & (data['Events'].isin(selected_events))]
    relevant_segments = filtered_data['Segments'].unique().tolist()
    selected_segments = relevant_segments
    selected_segments = st.multiselect('Select Segments', relevant_segments, default=[s for s in selected_segments if s in relevant_segments])
    filtered_data = filtered_data[filtered_data['Segments'].isin(selected_segments)]
    

    measurements = ['RMSSD', 'SDNN','MHR']
    selected_measurements = st.selectbox('Select Measurements', measurements, index=0)

    plot_types = ["Line Plot", "(corrected) Line Plot 3d", "Box Plot", "Violin Plot", "Swarm Plot", "Facet Grid"]
    selected_plot = st.selectbox('Select Visualization Type', plot_types)

    remove_outliers = st.checkbox("Remove Outliers", value=False)
    lower_bound, upper_bound = None, None
    if remove_outliers:
        lower_bound = st.number_input('Enter Lower Bound for Outliers', value=data[selected_measurements].quantile(0.20))
        upper_bound = st.number_input('Enter Upper Bound for Outliers', value=data[selected_measurements].quantile(0.80))


    # Pass subject_categories to the plotting functions
    if st.button("Generate Plot"):
        if selected_plot == "Line Plot":
            plot_line(filtered_data, selected_segmented_subjects, selected_events, selected_measurements, lower_bound, upper_bound)
        elif selected_plot == "(corrected) Line Plot 3d":
            plot_line_3d(filtered_data, selected_segmented_subjects, selected_events, selected_measurements, lower_bound, upper_bound)
        elif selected_plot == "Box Plot":
            plot_box(filtered_data, selected_segmented_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
        elif selected_plot == "Violin Plot":
            plot_violin(filtered_data, selected_segmented_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
        elif selected_plot == "Swarm Plot":
            plot_swarm(filtered_data, selected_segmented_subjects, selected_events, selected_measurements, x_var='Events', lower_bound=lower_bound, upper_bound=upper_bound)
        elif selected_plot == "Facet Grid":
            plot_facet(filtered_data, selected_segmented_subjects, selected_events, selected_measurements)


    st.markdown("## Visualization for non-segmented Data")
    st.markdown("## Transform non-segmented Data")
    log_transform_full = st.button("Transform non-segmented Data to Log Scale")
    if log_transform_full:
        measurements_to_transform_full = ['RMSSD', 'SDNN', 'MHR']
        full_data = transform_to_log(full_data, measurements_to_transform_full)
        st.write("Full data has been transformed to log scale.")
        if st.button("Cancel Full Data Log Transformation"):
            if original_full_data is not None:
                full_data = original_full_data.copy()
                st.write("Full data log transformation has been reverted.")
            else:
                st.warning("Unable to revert the full data log transformation.")
    
    full_data = data[data['Subjects'].str.startswith('Full_')]
    original_full_data = full_data.copy()

    
    full_measurements = ['RMSSD', 'SDNN', 'MHR']
    selected_full_subjects = st.multiselect('Select Full Subjects', all_full_subjects, default=all_full_subjects)
    selected_full_measurement = st.selectbox('Select Measurement for Full Data', full_measurements)

    full_plot_types = ["Box Plot", "Violin Plot", "Histogram", "Swarm Plot", "Plot Line"]
    selected_full_plot = st.selectbox('Select Visualization Type for Full Data', full_plot_types)
    
    all_full_events = full_data['Events'].unique().tolist()
    selected_full_events = st.multiselect('Select Events for Full Data', all_full_events, default=all_full_events)
    filtered_full_data = full_data[full_data['Subjects'].isin(selected_full_subjects) & full_data['Events'].isin(selected_full_events)]

    

    # Pass full_subject_categories to the full data plotting functions
    if st.button("Generate Full Data Plot"):
        if selected_full_plot == "Box Plot":
            plot_full_data_box(filtered_full_data, selected_full_measurement)
        elif selected_full_plot == "Violin Plot":
            plot_full_data_violin(filtered_full_data, selected_full_measurement)
        elif selected_full_plot == "Histogram":
            plot_full_data_histogram(filtered_full_data, selected_full_measurement)
        elif selected_full_plot == "Swarm Plot":
            plot_full_data_swarm(filtered_full_data, selected_full_measurement)
        elif selected_full_plot == "Plot Line":
            plot_full_line_plot(filtered_full_data, selected_full_measurement)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main Page", "Resilience and Susceptible Analysis"])
    
    if page == "Main Page":
        main_page()
    elif page == "Resilience and Susceptible Analysis":
        resilience_sustainability_page()

if __name__ == "__main__":
    main()
