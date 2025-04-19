import cv2
import os
import tqdm

frame_dir = "JuliaImgs"
output_file = "output.mp4"
fps = 12

# Get sorted list of frames
frames = sorted([f for f in os.listdir(frame_dir)])
height, width = cv2.imread(os.path.join(frame_dir, frames[0])).shape[:2]

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Write frames
for frame_name in tqdm.tqdm(frames):
    frame = cv2.imread(os.path.join(frame_dir, frame_name))
    video.write(frame)

video.release()