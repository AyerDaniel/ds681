This parquet file consists of data in the following format.

Each row represents a video segment from the parent video.  Each segment is a section within the parent video where a query image's object was detected.

It was converted form the DataFrame below into parquet with to_parquet().

<class 'pandas.DataFrame'>
RangeIndex: 44971 entries, 0 to 44970
Data columns (total 8 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   video_id          44971 non-null  str    
 1   ds_index          44971 non-null  int64  
 2   frame_index       44971 non-null  int64  
 3   class_label       44971 non-null  str    
 4   bounding_box      44971 non-null  str    
 5   confidence_score  44971 non-null  float64
 6   segment           44971 non-null  int64  
 7   timespans         44970 non-null  float64
dtypes: float64(2), int64(3), str(3)
memory usage: 6.3 MB

video_id:
	The video_id indicates the name of the video segment.  Naming convention is: 'object_detected'_'segment counter.' i.e. truck_1 is the first segment containing a truck.

df_index:
	This indicates which of the query images was used to for this identification.
	
	video_id: 'truck_1'
	ds_index: 0

	The above would indicate that this video is the first instance of a truck in the parent video, and was found in response to the query image '0' in the original query image set..

frame_index:
	This indicates which frame from the original video clip begins this segment video.

class_label:
	This is the object deteccted within the video clip.

bounding_box:
	These are the coordinates of the bounding box identifying this object.  In the raw pre-processed parquet (not this one) this was a per-frame metric.  It indicated where the bounding box was in the frame being considered.

confidence_score:
	This is teh confidence score of the bounding box.

segment:
	This is a tracker used to identify continuous segments.  It is based on a delta_t to identify discontinuity.  delta_t is how long the item must be absent from a clip before the clip is discontinuous in that item.

timespans:
	This indicates the duration of the clip iin seconds.