# Dominant-speaker-recognition-in-videos
Frame-wise recognition of dominant speakers in videos
# Introduction
This is a frame-wise face recognition project in videos. I have created my own dataset through Youtube.

# Preparing Training Data
I have selected a set of 6 diverse-appearing youtubers, namely, [Atul Khatri](https://www.youtube.com/user/gutterguppie), [Flute Raman](https://www.youtube.com/user/fluteraman), [Sadhguru](https://www.youtube.com/user/sadhguru), [Shailendra Kumar](https://www.youtube.com/channel/UCc6tYtmPd-1aEtr5TCu3a8Q), [Sandeep Maheswari](https://www.youtube.com/user/SandeepSeminars) and [Saurabh Pant](https://www.youtube.com/user/PantOnFireComedy). After selecting a number of videos on youtube for each speaker, I have extracted frames out of videos using Opencv. The frames from videos of different speakers were saved in separate directories so as to assign naive class labels to training data. Dominant speaker's faces are croppped from images using OpenCV’s [haar cascade](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html) classifier. I've used cascade face detector to identify human faces in each of the frame from the video and then crop the identified region. Two types of cropping was applied which is explained in detail in later section.

## Removing noisy images. 
In order to remove noisy images (e.g. images showing audience) from training data, a condition is imposed to assign the label of a particular speaker to a frame only when the number of detected faces in the frame was exactly equal to one. This strategy seemed to work fine for the five classes except Sadhguru. Due to the presence of a long beard on Sadhguru’s face, the cascade face detector was not able to identify the facial features and therefore didn’t identify it as a human face. Hence, for the videos of Sadguru, Cascade eye detector is used to filter relevant frames as shown in the figure below. Label of Sadguru was given to a particular frame only when the number of detected eyes in a frame was exactly equal to two. The noisy images are introduced as a separate seventh class in training data.

![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/sadhguru_eye_detector.jpg)

In spite of using above mentioned methodology, some noisy images still appears in data. One can remove them manually.

## Other Heuristics to remove noise
There are other techniques which can be used to remove noisy frames. I mention some techniques here.

### Alternate approach #1
Measure the correlation between successive frames. You can achieve this by making use of [numpy correlation function](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.corrcoef.html). Then, plot correlation factor value with frames and remove the frames with low correlation factor value. The idea is that speaker and noise frames would have a very low correlation factor. 

### Alternate approach #2
For each of the speakers, keep a reference image during filtering. For each frame calculate its correlation with the reference image and the the previous frame. If the maximum of the two correlations is higher than a threshold than the frame get labelled as the particular speaker else noise. And, if a particular frame is identified as noise, don't use that to calculate the correlation with next frame. Basically, skip that frame and calculate the correlation of next frame with the reference image and the last identified speaker frame.

## Introducing background noise for generalization
In order to let model generalize well, a different type of cropping is also implemented in which some background behind the speaker is allowed in the images, I call this moderate-cropped. The model is trained on both the datasets, namely, super-cropped and moderate-cropped and finally the accuracies are compared between the two.

![Alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/super_moderate_cropped.jpg)

# Model Architecture
Pre-trained inception net is used in keras. After removing last layer, two fully connected layers are added, with the last one being a softmax layer. Output being a seven dimensional vector, corresponding to the seven classes as mentioned previously.

# Results
Model is trained on the two datasets: super-cropped and moderate-cropped, in order to assess the effect of introducing noise in the training data on the model accuracy. The supercrop model should overfit and do worse on the new frames which had different background noise than the training data. The results are consistent with intuitions with super-cropped model performing better achieving highest accuracy of **52.5%**. The highest accuracy observed for the super-cropped dataset model is 46% approximately.

![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/results.jpg)

## Visualizing Test Data
To visualise the test data, I made use of the [tSNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) algorithm. The Dimensions of test data is first reduced to 50 by doing a PCA(Principal Component Analysis). Then, tSNE is used to reduce it further to 2 dimensions. Clusters can be observed corresponding to each of the seven classes in our data. There were multiple clusters for the same type of speaker in the data. Also, clusters for the class of noise were sparser than the other classes. The possible reason for observing different clusters for the same speaker can be different frames of a particular speaker has been taken from videos that have very different background noise. Therefore, though the images share some commonality, they are quite different from each other. The class of noise is sparser because very diverse frames have been classified with the same label in this class.

![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/visualization_test_data.jpg)

## Visualizing the Model
I, here, make use of a technique called [Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf). It allows to obtain a localization map for any target class, highlighting regions (in the form of heatmap) in input image that positively correlates with the chosen class. For this, I have selected one image (which are not present in training data) for each class belonging to six speakers.

![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/grad_cam_AK.jpg)
![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/grad_cam_FR.jpg)
![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/grad_cam_SG.jpg)
![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/grad_cam_SK.jpg)
![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/grad_cam_SM.jpg)
![alt text](https://github.com/harsh-sahu/Dominant-speaker-recognition-in-videos/blob/master/images/grad_cam_SP.jpg)

As you can see, in all the images either face or its near region is highlighted showing that our model is looking at right place to classify images and therefore has **learnt well!**
