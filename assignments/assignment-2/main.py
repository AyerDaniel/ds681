# DS681 Daniel Ayer  Assignment_2 Spring 2026

## Dependencies ##

import torch
import datasets
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

## Environmental Variables ##
frames_uri = "frames"

## Define Functions ##

def train():

    # Load a pretrained segmentation model like YOLO26n-seg
    model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

    # Try for CUDA device.
    try:
        model.cuda()

    except Exception as e:
        print(f"model could not be moved to CUDA device.\nError:{e}")
        
    # Train the model on the Carparts Segmentation dataset
    results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)

def eval():

    # Load a fine tuned, locally trained and saved model.
    model = YOLO("/home/johnsmith/Desktop/njit/ds681/github/ds681/ds681/yolo26n-seg.pt")  # load a pretrained model (recommended for training)

    # Try for CUDA device.
    try:
        model.cuda()

    except Exception as e:
        print(f"model could not be moved to CUDA device.\nError:{e}")

    # After training, you can validate the model's performance on the validation set
    try:
        results = model.val()

    except Exception as e:
        print(e)


'''
1. Video processing and detection

You must:

    Sample frames from the input video.
    Run an object detector on each frame.
    Produce bounding boxes and class labels for detected exterior components.

Detections should be temporally indexed by frame number or timestamp.

'''

#########################

'''
    Frame file names are their frame location in the video.
'''

#########################


def vpd():

    # Load a fine tuned, locally trained and saved model.
    model = YOLO("/home/johnsmith/Desktop/njit/ds681/github/ds681/ds681/yolo26n-seg.pt")  # load a pretrained model (recommended for training)

    ### Sample frames from input video.  Frames have been preprocessed from video as described in the assignment.
 
    ### Run an object detector on each frame.
    predictions = model.predict(frames_uri, batch = 30, name = "source_frames_detection", save = True, save_frames = True, save_txt = True, save_conf = True, exist_ok = True)

    ### Produce bounding boxes and class labels for detected exterior components.

    # DatafFrame to store boxes.
    df_boxes = pd.DataFrame(columns = ['frame_index', 'class_label', 'bounding_box', 'confidence_score'])
    df_boxes.index.name = 'boxes'

    for frame in predictions:

        # Get frame identifier.  Here the filename.
        frame_index = frame.path.split("/")[-1]  ##  <--- Split path uri on "/" and grab the file name which is the last entry in that resultant list.

        for box in frame.boxes:
                        
            # Check for detected match.

                # Store box details in datafraame.
                
                time_index = frame_index.replace("frame_","").split('.')[0] ## <--- Strip out leading "frame_" from filename: 'frame_{time index}.jpg'
                class_label = model.names[int(box.cls)]
                xyxy = box.xyxy.tolist()[0]
                confidence_score = box.conf.tolist()[0]

                df_boxes.loc[len(df_boxes)] = [time_index, class_label, xyxy, confidence_score]

    # Save DataFrame.
    df_boxes.to_csv("df_boxes.csv")


'''

2. Image semantic search

For each query image:
    Run the same detector (or a compatible image encoder).
    Identify the detected component class(es).
    Match these against detected components in the video index.
    Retrieve contiguous time intervals where the component is present.

Simple matching (e.g., class label overlap) is sufficient, but you may incorporate confidence thresholds or similarity scores.

'''
def iss():

    from PIL import Image

    from datasets import load_dataset

    # Load query images dataset.
    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")

    # Check if DataFrame of boxes exists.
    if 'df_boxes' not in locals():

        # Load dataframe from save.
        df_boxes = pd.read_csv('df_boxes.csv', index_col=0)

    # Check if model has been loaded.  If not, load.
    if 'model' not in locals():

        # Load a fine tuned, locally trained and saved model.
        model = YOLO("/home/johnsmith/Desktop/njit/ds681/github/ds681/ds681/yolo26n-seg.pt")  # load a pretrained model (recommended for training)

    # Create dataframe to hold results to report.
    '''
    df_ds_found DataFrame structure:

        ds_index: Which query image are we using.  
        object_class: What object did we detect in the query.
        time_spans:  Which time spans in the source video contain a similar object.
    '''

    # Create dataframe to hold found matches.
    df_ds_found = pd.DataFrame() 

    # Name index to prevent thrown error.
    df_ds_found.index.name = 'found'

    # Make class predictions on query images.
    results = model.predict([x for x in ds['image']], name = "query_match_results", save = True, save_conf = True, exist_ok = True)

    # Counter to track which query image we are searching for.
    ds_index = 0

    # Find all frames of source video containing object.
    for image in results:

        for box in image.boxes:
                
            # Get class of object.
            box_class = model.names[int(box.cls)]

            # Locate all such instances in DataFrame.
            df_found = df_boxes[df_boxes['class_label'] == box_class].copy()
            
            # Add ds_index.
            df_found.insert(0, 'ds_index', ds_index)
            
            # Add results to dataframe.
            df_ds_found = pd.concat([df_ds_found, df_found], ignore_index=True)

        # Increment ds_index for next image.
        ds_index += 1


    # Save results.
    df_ds_found.to_csv('df_ds_found.csv')

def output():

    '''

        All detection results must be uploaded to Hugging Face as a Parquet file.
        Each row in the Parquet file must correspond to a single detection in the video and contain at least the following fields:
            video_id
            frame_index or timestamp
            class_label
            bounding_box (x_min, y_min, x_max, y_max)
            confidence_score
        You may add additional fields (e.g., detector_name, embedding_id), but these are optional.

        The Parquet file serves as the sole interface between detection and retrieval.
    

    '''

    # Load in DataFrame from previous steps.
    df_ds_found = pd.read_csv('df_ds_found.csv')

    # Rename first column.
    df_ds_found.rename(columns={"Unnamed: 0":'video_id'}, inplace=True)
    
    # Display results.
    print(df_ds_found.info())

    # Create Paraquet File.
    df_ds_found.to_parquet('ds_matches.paraquet')


def process_clips():
    '''
        This function takes in the parquet file uploaded to HuggingFace and parses it into videos.  
        Each video will consist of the segments of the original video which contain the object found in the reference query image.
    '''

    # Read Parquet File.
    df_ds_found = pd.read_parquet('ds_matches.paraquet')

    # Create video clips of each query image identified within the source video clip.
    
    # Get list of object types detected.
    objects_found = np.sort(df_ds_found['class_label'].unique())

    '''
    
        I want to see how many segments of the video contain the object.  delta_t controls how many seconds the object needs to be out of frame to be considered 
    a discontinuity.  I want to see when the increase in delta_t stops returning a meaningful increase in the segment counts.  I plan to plot this and hope for an elbow.
    
    '''

    # List to hold plot vectors. [delta_t, count of segments for each k thing]  Vector will be k + 1.
    plot_me = []

    # Check for time changes from 1 second to 10 seconds.
    for delta_t in range(10):
        
        # Vector to plot.
        del_t_plot = [delta_t]
        
        # Iterate through objects found.
        for thing in objects_found:

            # Get clips of object.
            df_returns = df_ds_found[df_ds_found['class_label'] == thing]

            # Sort by frame.
            df_returns.sort_values(by='frame_index', inplace = True)

            # Get segments where the difference in time stamps is 1.  Greater than 1 would indicate discontinuity.  1 is out atomic time unit.
            segments = (df_returns['frame_index'].diff() > delta_t).cumsum()

            time_spans = df_returns.groupby(segments)['frame_index'].agg(['min','max']).values.tolist()

            # Get count of segments.
            del_t_plot.append(len(time_spans))

        # Store results.
        plot_me.append(del_t_plot)

    # Plot results.
    # Extract x values
    x = [row[0] for row in plot_me]

    # Extract each y-series
    y_series = list(zip(*[row[1:] for row in plot_me]))

    # Plot each series
    for i, y in enumerate(y_series):
        plt.plot(x, y, marker='o', label=f'{objects_found[i]}')

    plt.title('Delta_t (s) vs Segment Counts')
    plt.xlabel("Seconds between Segments")
    plt.ylabel("Number of Segments")
    plt.legend()
    plt.grid(True)
    plt.show()
            
    # Based on generated plot chose delta_t = 2.
    delta_t = 2

    # VariabDictionaryle to hold list of video URLs.
    playlist = {}

    # List to store new video ids.
    video_ids = []

    # Create dataframe for parquet creation and submission.
    df_to_parquet = pd.DataFrame()

    # Iterate through objects found.
    for thing in objects_found:

        # Create counter for incrementing segments of object clips.
        seg_count = 0

        # Testing
        print(f"Thing: {thing}")

        # Get clips of object.
        df_returns = df_ds_found[df_ds_found['class_label'] == thing].copy()

        # Sort by frame.
        df_returns.sort_values(by='frame_index', inplace = True)

        # Get segments where the difference in time stamps is > delta_t.  Greater than delta_t would indicate discontinuity.
        segments = (df_returns['frame_index'].diff() > delta_t).cumsum()

        # Add segments to df.
        df_returns['segment'] = segments

        # Sort by segments.
        df_returns.sort_values('segment', inplace = True)

        # Change Video ID to 'class_label'_'segment'.
        df_returns['video_id'] = df_returns['class_label'] + "_" + df_returns['segment'].astype(str)

        # Store results to convert to parquet later.
        df_to_parquet = pd.concat([df_to_parquet, df_returns], ignore_index = True)

        # Get min, max of timespan.
        time_spans = df_returns.groupby(segments)['frame_index'].agg(['min','max']).values.tolist()

        # Set minimum length of clip in seconds.
        min_length = 3
        
        # Remove clips shorter than 1 second.
        time_spans = [x for x in time_spans if abs(x[1]) - abs(x[0]) > min_length]

        # Variable to hold list of this things clips urls.
        tmp_urls = []

        # Iterate over timespans.
        for timespan in time_spans:
            
            # Create url for video clip.

            url = f"https://www.youtube.com/embed/YcvECxtXoxQ?start={timespan[0]}&end={timespan[1]}"

            # Store url.
            tmp_urls.append(url)
        
        # Store things urls.
        playlist[thing] = tmp_urls
    
    with open("playlist.json", "w") as file:

        file.write(json.dumps(playlist))

    # Create Parquet file to submit.
    df_to_parquet.to_parquet('video_detections')

    # Write to csv for analysis.
    df_to_parquet.to_csv("df_to_parquet.csv")


def final_report():
    # This function processes and confirms everything for the final report submission.

    # Verify Parquet.

    # Read Parquet File.
    df_parquet = pd.read_parquet('video_detections')

    df_parquet['timespans'] = abs(df_parquet['frame_index'].diff(periods = -1))

    # Display details of DataFrame.
    print(df_parquet.info())
    print(df_parquet.head())

    # Get list of objects detected.
    objects_detected = df_parquet['class_label'].unique()

    # Create dictionary of counts and stats for each object detected.
    results = {}

    for label in df_parquet['class_label'].unique():

        # Get each object one by one.
        block = df_parquet[df_parquet['class_label'] == label]

        # Store analytics for the object.
        results[label] = {
            'videos' : len(block['segment'].unique())
                                                             
        }

    # Display results.
    for result in results:
        
        print(f"{result}\n{results[result]}\n")

    

## Run ##

exit_ = False

# Create interface for user.

steps = {"train" : train,
         "eval" : eval,
         "vpd" : vpd,
         "iss" : iss,
         "output" : output,
         "process_clips" : process_clips,
         "final_report" : final_report,
         "exit" : ""}

menu = [f"{index}: {key}" for index, key in enumerate(steps)]
        
while not exit_:

    # Display menu.
    print(menu)

    # Select user input.
    pos = int(input(f"Please select from the options above."))   

    # Check input.
    if not isinstance(pos, int):
        
        input("Please enter an integer matching the choice you would like to perform.")
        continue

    if not (0 <= pos <= len(steps)):

        input(f"Please enter a value between 0 and {len(steps)}")
        continue


    # Get command to run.
    do = list(steps.keys())[pos]

    # Testing
    print(do)

    # Check for exit.
    if do == "exit":
        
        # Set flag to exit.
        exit_ = True

        # Display message.
        print(f"\n\nThanks for running my project!\n\n")


        continue

    # Run step.
    steps[do]()