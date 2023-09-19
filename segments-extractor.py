import streamlit as st
import pandas as pd
import os
import glob

# Constants
REPO_PATH = "/edenede2/Segments-extractor-/tree/main"  # Replace with your repository name
TOTAL_SEGMENTS_FILE = os.path.join(REPO_PATH, "Total_segments_Val.csv")

def extract_data_from_hrv_file_corrected(hrv_file):
    # Read the HRV data
    hrv_data = pd.read_excel(hrv_file)
    
    # Extract subject ID from filename
    subject_id = "sub" + hrv_file.split("_")[1]
    
    # Create an empty DataFrame to store extracted data
    extracted_data = pd.DataFrame(columns=["Subjects", "Events", "Segments", "RMSSD", "SDNN", "MHR"])
    
    # Mapping of Start and End events to their respective event names
    event_mapping = {
        ("Keyboard:F1:Keyboard 1", "Keyboard:F2:Keyboard 2"): "rest baseline",
        ("Keyboard:F3:Keyboard 3", "Keyboard:F4:Keyboard 4"): "scenario 1",
        ("Keyboard:F5:Keyboard 5", "Keyboard:F6:Keyboard 6"): "MAST",
        ("Keyboard:F7:Keyboard 7", "Keyboard:F8:Keyboard 8"): "scenario 2",
        ("Keyboard:F9:Keyboard 9", "Keyboard:F10:Keyboard 10"): "scenario 3"
    }
    
    # Iterate over each segment
    for col in hrv_data.columns:
        # Extract event type
        start_event = hrv_data.at[0, col]
        end_event = hrv_data.at[2, col]
        event_type = event_mapping.get((start_event, end_event), "Unknown")
        
        # Extract other required values
        segment_number = col  # This is the corrected part
        mean_heart_rate = hrv_data.at[7, col]
        sdnn = hrv_data.at[11, col]
        rmssd = hrv_data.at[13, col]
        
        # Append to extracted_data DataFrame
        extracted_data = extracted_data.append({
            "Subjects": subject_id,
            "Events": event_type,
            "Segments": segment_number,
            "RMSSD": rmssd,
            "SDNN": sdnn,
            "MHR": mean_heart_rate
        }, ignore_index=True)
    
    # Append the extracted data to total_segments_file
    total_segments_data = pd.read_excel(total_segments_file)
    total_segments_data = total_segments_data.append(extracted_data, ignore_index=True)
    
    # Save the updated total_segments_data back to the file
    total_segments_data.to_excel(total_segments_file, index=False)
    
    return total_segments_data

def process_files():
    # Identify all files in the repo directory
    all_files = glob.glob(os.path.join(REPO_PATH, "*"))
    
    # Filter out the Total_segments_Val.csv file
    hrv_files = [file for file in all_files if not file.endswith("Total_segments_Val.csv")]
    
    # Read the Total_segments_Val.csv file
    total_segments_data = pd.read_csv(TOTAL_SEGMENTS_FILE)
    
    # Process each HRV file
    for hrv_file in hrv_files:
        extracted_data = extract_data_from_hrv_file(hrv_file)
        total_segments_data = total_segments_data.append(extracted_data, ignore_index=True)
    
    # Save the updated total_segments_data back to the file
    total_segments_data.to_csv(TOTAL_SEGMENTS_FILE, index=False)
    
    return total_segments_data

# Streamlit App
st.title("HRV Data Extractor from GitHub Repository")

if st.button("Process and Append Data from GitHub", key='github'):
    result_data = process_files()
    st.write(result_data)
    st.download_button("Download Updated Total Segments File", TOTAL_SEGMENTS_FILE, "Total_segments_Val_updated.csv")

# Note: Before running the app, make sure to clone your GitHub repository to the local environment.
# You can do this manually or add a git clone command in your deployment script.




# Streamlit app code
st.title("HRV Data Extractor")

# Upload HRV data file
hrv_file = st.file_uploader("Upload your HRV Analysis Excel file")

# Upload Total Segments file
total_segments_file = st.file_uploader("Upload your Total Segments csv file")

# Button to trigger data extraction and append
if st.button("Process and Append Data from Uploaded Files", key='upload'):
    if hrv_file and total_segments_file:
        # Convert the uploaded files to DataFrames
        hrv_data_df = pd.read_excel(hrv_file)
        total_segments_df = pd.read_csv(total_segments_file)

        # Save the uploaded DataFrames to temporary files to use them in the function
        hrv_file_path = "/tmp/hrv_temp_file.xlsx"
        total_segments_file_path = "/tmp/total_segments_temp_file.csv"
        
        hrv_data_df.to_excel(hrv_file_path, index=False)
        total_segments_df.to_excel(total_segments_file_path, index=False)

        # Call the function
        result_data = extract_data_from_hrv_file_corrected(hrv_file_path, total_segments_file_path)
        
        # Display the updated data
        st.write(result_data)

        # Provide an option to download the updated total_segments file
        st.download_button("Download Updated Total Segments File", total_segments_file_path, "total_segments_updated.xlsx")
    else:
        st.warning("Please upload both files before proceeding.")
