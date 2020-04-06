# Importing all necessary libraries
import os

import cv2
import numpy as np

# Read the video from specified path
cam = cv2.VideoCapture(
  # '/Users/creager/Downloads/rollouts_with_attn_thresh_0.02.mp4'  # MMN
  '/Users/creager/Downloads/rollouts_with_attn_thresh_0.03.mp4'  # SSA
)

try:

  # creating a folder named data
  if not os.path.exists('data'):
    os.makedirs('data')

  # if not created then raise error
except OSError:
  print('Error: Creating directory of data')

# frame
currentframe = 0

# selected_frame_range = range(133, 141)  # MMN thresh 0.016
selected_frame_range = range(32, 37)  # MMN thresh 0.02
selected_frames = []

while (True):

  # reading from frame
  ret, frame = cam.read()

  if ret:
    # if video is still left continue creating images
    name = './data/frame' + str(currentframe) + '.jpg'
    print('Creating...' + name)

    # writing the extracted images
    cv2.imwrite(name, frame)

    # increasing counter so that it will
    # show how many frames are created
    currentframe += 1

    if currentframe in selected_frame_range:
      selected_frames.append(frame)

  else:
    break


selected_frames = np.vstack(selected_frames)
cv2.imwrite('./data/selected_frames.jpg', selected_frames)

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

