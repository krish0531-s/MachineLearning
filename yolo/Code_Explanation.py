# YOLO inference script with line-by-line explanations.
# Each original statement is followed by an explanatory comment.

import os  # OS functions like path checks and file operations
import sys  # System functions like exiting the program
import argparse  # Parse command-line arguments
import glob  # File pattern matching (used to list files in a folder)
import time  # Time measurement (used for FPS calculation)

import cv2  # OpenCV for image/video I/O and drawing
import numpy as np  # Numeric operations and arrays
from ultralytics import YOLO  # Load and run the Ultralytics YOLO model

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)  # Required: path to the .pt model
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)  # Required: source to process
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)  # Optional: confidence threshold; default 0.5
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)  # Optional: resize display/record resolution
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')  # Optional flag: save output video if present

args = parser.parse_args()  # Parse the command-line arguments into `args`


# Parse user inputs
model_path = args.model  # Path to model file provided by user
img_source = args.source  # Source provided by user (file/folder/video/camera)
min_thresh = args.thresh  # Confidence threshold string or number
user_res = args.resolution  # Optional resolution string
record = args.record  # Boolean flag whether to record or not

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)  # Exit if model file not found

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')  # Load YOLO model for detection
labels = model.names  # Get class name mapping from model

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']  # Supported image extensions
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']  # Supported video extensions

if os.path.isdir(img_source):
    source_type = 'folder'  # Source is a directory
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)  # Get file extension
    if ext in img_ext_list:
        source_type = 'image'  # Single image file
    elif ext in vid_ext_list:
        source_type = 'video'  # Video file
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)  # Unsupported file type
elif img_source.isdigit():
    source_type = 'usb'  # Numeric string indicates USB camera index
    usb_idx = int(img_source)  # Convert camera index to int

elif 'picamera' in img_source:
    source_type = 'picamera'  # Picamera string indicates Raspberry Pi camera
    picam_idx = int(img_source[8:])  # Extract camera index after 'picamera'
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)  # Not a valid input

# Parse user-specified display resolution
resize = False  # Flag indicating whether to resize frames
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])  # Parse WxH into integers

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)  # Recording unsupported for image/folder
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)  # Must provide resolution to record
    
    # Set up recording
    record_name = 'demo1.avi'  # Output filename for recording
    record_fps = 30  # Desired frames per second for output
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))  # Initialize video writer

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]  # Single image in list for unified processing
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')  # List all files in directory
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)  # Append supported image files to imgs_list
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)  # Open video file or camera stream

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)  # Set frame width property
        ret = cap.set(4, resH)  # Set frame height property

elif source_type == 'picamera':
    from picamera2 import Picamera2  # Import Picamera2 when needed
    cap = Picamera2()  # Initialize PiCamera object
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))  # Configure capture settings
    cap.start()  # Start camera

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]  # Predefined colors for boxes

# Initialize control and status variables
avg_frame_rate = 0  # Running average FPS
frame_rate_buffer = []  # Buffer to smooth FPS measurements
fps_avg_len = 200  # Number of frames to average FPS over
img_count = 0  # Index for iterating through images when source is folder or single image

# Begin inference loop
while True:

    t_start = time.perf_counter()  # Start time for FPS calc

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)  # No more images, exit
        img_filename = imgs_list[img_count]  # Get current image filename
        frame = cv2.imread(img_filename)  # Read image into a BGR numpy array
        img_count = img_count + 1  # Increment index
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()  # Read next frame
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break  # End of video reached
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break  # Camera read failed

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame = cap.capture_array()  # Capture frame as numpy array from Picamera2
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))  # Resize frame to user resolution

    # Run inference on frame
    results = model(frame, verbose=False)  # Run the YOLO model on the frame

    # Extract results
    detections = results[0].boxes  # Get bounding box container for the first (and only) image

    # Initialize variable for basic object counting example
    object_count = 0  # Counter for how many detections passed threshold

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu()  # Move tensor to CPU memory
        xyxy = xyxy_tensor.numpy().squeeze()  # Convert to numpy array and remove extra dims
        xmin, ymin, xmax, ymax = xyxy.astype(int)  # Convert coordinates to int for drawing

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())  # Class index as Python int
        classname = labels[classidx]  # Lookup class name from model labels

        # Get bounding box confidence
        conf = detections[i].conf.item()  # Confidence as float

        # Draw box if confidence threshold is high enough
        if conf > 0.5:  # Compare against hardcoded threshold (note: args.thresh parsed earlier as min_thresh but not used here)

            color = bbox_colors[classidx % 10]  # Choose a color based on class id modulo number of colors
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)  # Draw bounding box on frame

            label = f'{classname}: {int(conf*100)}%'  # Create label text showing class and confidence percent
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Compute label text size
            label_ymin = max(ymin, labelSize[1] + 10)  # Ensure label doesn’t go above top of image
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)  # Draw filled box for label background
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Put label text on top of background

            # Basic example: count the number of objects in the image
            object_count = object_count + 1  # Increment count for each drawn detection

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)  # Draw FPS on frame
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)  # Draw object count
    cv2.imshow('YOLO detection results',frame)  # Create window and display frame
    if record: recorder.write(frame)  # If recording, write current frame to video file

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()  # Wait indefinitely for a key press
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)  # Wait 5 milliseconds for a key press (real-time)
    
    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
        cv2.waitKey()  # Wait indefinitely until another key is pressed
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)  # Write frame to file
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()  # End time for frame
    frame_rate_calc = float(1/(t_stop - t_start))  # Instantaneous FPS

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)  # Remove oldest entry to maintain buffer length
        frame_rate_buffer.append(frame_rate_calc)  # Append newest FPS
    else:
        frame_rate_buffer.append(frame_rate_calc)  # Append newest FPS if buffer not full

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)  # Compute mean FPS from buffer


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')  # Print average FPS to terminal
if source_type == 'video' or source_type == 'usb':
    cap.release()  # Release OpenCV capture resource
elif source_type == 'picamera':
    cap.stop()  # Stop Picamera
if record: recorder.release()  # Release video writer if used
cv2.destroyAllWindows()  # Close all OpenCV windows
