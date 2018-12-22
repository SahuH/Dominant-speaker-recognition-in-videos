# dominant-speaker-recognition-in-videos
Frame-wise recognition of dominant speakers in videos
# Introduction
This is a frame-wise face regonition project in videos. I have created my own dataset.

# Preparing Training Data
I have selected a set of 6 diverse-appearing youtuberes, namely, Atul Khatri, Flute Raman, Sadhguru, Sandeep Maheswari, Saurabh Pant. After selecting a number of videos on youtube for each speaker, I have extracted frames out of videos using Opencv. The frames from videos of different speakers were saved in separate directories so as to assign naive class labels to training data.
Dominant speaker's faces are croppped from images using openCV’s haar cascade classifier. I've used cascade face detector to identify human faces in each of the frame from the video and then crop the identified region. Two types of cropping was applied which is explained in detail in later section.

# Removing noisy images. 
In order to remove noisy images (e.g. images showing audience) from training data, a condition is imposed to assign the label of a particular speaker to a frame only when the number of detected faces in the frame was exactly equal to one. This strategy seemed to work fine for the five classes other than the Sadhguru. Due to the presence of a long beard on Sadhguru’s face the cascade face detector was not able to identify the facial features and therefore didn’t identify it as a human face. Hence, for the videos of Sadguru, Cascade eye detector is used to filter relevant frames as shown in the fig 2.1.3. We gave the label of Sadguru to a particular frame only when the number of detected eyes in a frame was exactly equal to two. The noisy images are introduced as a separate seventh class in training data.

## Other Heuristics

# Introducing background noise for generalization
In order to let model generalize well, a different type of cropping is also implemented in which some background behind the speaker is allowed in the images, I call this moderate-cropped. The model is trained on both the datasets, namely, super-cropped and moderate-cropped and finally the accuracies are compared between the two.

# Preparing Test Data

# Model Architecture
Pre-trained inception net is used in keras. After removing last layer, two fully connected layers are added, with the last one being a softmax layer. Output being a seven dimensional vector, corresponding to the seven classes as mentioned previously.

# Results
Model is trained on the two datasets: super-cropped and moderate-cropped. We did this experiment as we wanted to assess the effect of introducing noise in the training data on the model accuracy. We has suspected that the supercrop model will likely overfit and do worse on the new frames which had different background noise than the training data. The results of our experiment is shown in the fig 4.1.1. We observed that the model performed better for the less cropped dataset with the highest accuracy of 52.5%. The highest accuracy observed for the more cropped dataset model was 46% approximately.

