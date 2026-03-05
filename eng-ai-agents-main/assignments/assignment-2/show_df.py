import pandas as pd 

df = pd.read_parquet("video_detections")

df.to_csv('video_detections.csv')
print(df.info())

print( df.head())

print(df)

