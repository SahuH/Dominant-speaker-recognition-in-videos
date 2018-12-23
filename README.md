# Dominant-speaker-recognition-in-videos
Frame-wise recognition of dominant speakers in videos
# Introduction
This is a frame-wise face regonition project in videos. I have created my own dataset.

# Preparing Training Data
I have selected a set of 6 diverse-appearing youtuberes, namely, Atul Khatri, Flute Raman, Sadhguru, Sandeep Maheswari, Saurabh Pant. After selecting a number of videos on youtube for each speaker, I have extracted frames out of videos using Opencv. The frames from videos of different speakers were saved in separate directories so as to assign naive class labels to training data.
Dominant speaker's faces are croppped from images using openCV’s haar cascade classifier. I've used cascade face detector to identify human faces in each of the frame from the video and then crop the identified region. Two types of cropping was applied which is explained in detail in later section.

## Removing noisy images. 
In order to remove noisy images (e.g. images showing audience) from training data, a condition is imposed to assign the label of a particular speaker to a frame only when the number of detected faces in the frame was exactly equal to one. This strategy seemed to work fine for the five classes except Sadhguru. Due to the presence of a long beard on Sadhguru’s face, the cascade face detector was not able to identify the facial features and therefore didn’t identify it as a human face. Hence, for the videos of Sadguru, Cascade eye detector is used to filter relevant frames as shown in the fig 2.1.3. We gave the label of Sadguru to a particular frame only when the number of detected eyes in a frame was exactly equal to two. The noisy images are introduced as a separate seventh class in training data.

In spite of using above mentioned methodology, some noisy images still appears in data. One can remove them manually.

## Other Heuristics to remove noise
There are other techniques which can be used to remove noisy frames. I mention some techniques here.

### Alternate approach #1
Measure the correlation between successive frames. You can achieve this by making use of [numpy correlation function](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.corrcoef.html). Then, plot correlation factor value with frames and remove the frames with low correlation factor value. The idea is that speaker and noise frames would have a very low correlation factor. 

### Alternate approach #2
For each of the speakers, keep a reference image during filtering. For each frame calculate its correlation with the reference image and the the previous frame. If the maximum of the two correlations is higher than a threshold than the frame get labelled as the particular speaker else noise. And, if a particular frame is identified as noise, don't use that to calculate the correlation with next frame. Basically, skip that frame and calculate the correlation of next frame with the reference image and the last identified speaker frame.

## Introducing background noise for generalization
In order to let model generalize well, a different type of cropping is also implemented in which some background behind the speaker is allowed in the images, I call this moderate-cropped. The model is trained on both the datasets, namely, super-cropped and moderate-cropped and finally the accuracies are compared between the two.

# Preparing Test Data

# Model Architecture
Pre-trained inception net is used in keras. After removing last layer, two fully connected layers are added, with the last one being a softmax layer. Output being a seven dimensional vector, corresponding to the seven classes as mentioned previously.

# Results
Model is trained on the two datasets: super-cropped and moderate-cropped. We did this experiment as we wanted to assess the effect of introducing noise in the training data on the model accuracy. We has suspected that the supercrop model will likely overfit and do worse on the new frames which had different background noise than the training data. The results of our experiment is shown in the fig 4.1.1. We observed that the model performed better for the less cropped dataset with the highest accuracy of 52.5%. The highest accuracy observed for the more cropped dataset model was 46% approximately.

## Gradient Ascent
Based on our learnt weights, we tried to construct the image which will maximise the probability of getting a particular class given the trained model.

## Visualizing Test Data
To visualise the test data, we made use of the tSNE algorithm. For the two sets of test data, we first reduced the
dimension of the data to 50 by doing a PCA(Principal Component Analysis). After this dimension reduction, we
further reduced the dimension of the data by using the tSNE algorithm. The results obtained for the two datasets, the one
given earlier on the one given out just the day before are shown in the Fig4.4.2. we observed clusters corresponding to each of the seven classes in our data. There were multiple clusters for the same type of speaker in the data. Also, clusters for the
class of noise were sparser than the other classes. The possible reason for observing different clusters for the same
speaker, is that since different frames of a particular. speaker has been taken from videos that have very different background noise. Therefore, though the images share some commonality, there are quite different from each
other. The class of noise is sparser because very diverse frames have been classified with the same label in this
class.
