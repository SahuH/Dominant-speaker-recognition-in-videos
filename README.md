# dominant-speaker-recognition-in-videos
Frame-wise recognition of dominant speakers in videos
# Introduction
This is a frame-wise fae regonition project in videos. I have created my own dataset.

# Preparing Training Data
I have selected a set of 6 diverse-appearing youtuberes, namely, Atul Khatri, Flute Raman, Sadhguru, Sandeep Maheswari, Saurabh Pant. After selecting a number of videos on youtube for each speaker, I have extracted frames out of videos using Opencv. The frames from videos of different speakers were saved in separate directories so as to assign naive class labels to training data. 

# Removing noisy images
In order to remove noisy images from training data, openCV’s haar cascade classifier is used. I've used cascade face detector to identify human faces in each of the frame from the video and then crop the identified region. Two types of cropping was applied which is explained in detail in next section.

# Introducing background noise for generalization
