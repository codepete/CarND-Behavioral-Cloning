#**Behavioral Cloning**

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Nvidia_Architecture.PNG "Model Visualization"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md that summarizes my results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

![image1]

For my model architecture I based my model off of Nvidia's "End to End Learning for Self-Driving Cars" whitepaper to train my car to drive in the simulator. However, what the diagram doesn't show you is that that after every convolution layer I added relu activation function and added subsampling. After 5 layers of convolution I added a dropout of 0.5 to provide overfitting of data. In the 4 fully-connected layers I added relu activation. I also normalize the data between -0.5 and 0.5.

####2. Attempts to reduce overfitting in the model

The model contains a dropout of 0.5 at the very last convolution layer in order to reduce overfitting.

I made it a personal challenge to only use the data provided by the project; however, the data set isn't large enough by itself. To ensure that I provided a data set that had enough variance I utilized augmentation techniques; horizontal/vertical shifting, vertical flipping of image, and randomization of brightness. This ensured that my model was not overfitting and provided much more data for the model to train on.

After realizing that low loss and validation loss values didn't actually provide an indicator to how well the car would make it around the track, it was essential that I tested my model on the track after training.

####3. Model parameter tuning

The model utilized the adam optimizer where I manually set the learning rate to 0.0001 rather then the 0.001 default learning rate. This meant that I had to make sure I had a large enough data set to achieve optimal results, which was provided through data augmentation.

####4. Appropriate training data

I saw that many students were able to utilize only the provided data to fully train their models. I wanted to make this my goal. However, this isn't possible without augmenting the data. Many suggested techniques mentioned in https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yfybvx6fg. I utilized the horizontal/vertical shifting and randomization of brightness techniques mentioned in the article.

I first pulled all the data from the CSV file and loaded it into a list (**get_training_data**). I then created a second method called **generate_training_data** that would accept an arbitrary number of data points to generate. This method utilized a randomizer that would pick X amount of data points from the list provided by Udacity. Each row contains a steering angle and array of references to left, middle, and right images provided by the simulator training data. Since there was no way my computer could handle loading this much data into memory I had to utilize generators to create batches for training.

When a batch is requested we do the following:
- Shuffle the data
- Randomly pick from left, right, or center image
- Add steering correction of 0.25 to left images and -0.25 to right images (same angle for center)
- Gave 50% probability that the image would be flipped on the vertical axis
- Randomly changed the brightness of the images
- Randomly shifted the images left or right by a range 50 pixels and up and down by a range of 20 pixels. For each pixel shift we would correct the steering by 0.0004 per pixel.
- Resized images to 66x200x3 to match Nvidia model input size

This ensured that the model would not overfit. This was definitely a problem with the data set by iteslf - there was a lot of steering angles that were close to 0. My shifting our images left/right/up/down we introduced more variance in steering angles making our model more capable of handling hills and turns.

Also by utilizing both left and right images from the simulator we could better train our models on how to correct itself when it was going off or pointing off track.

###Model Architecture and Training Strategy

####1. Solution Design Approach & 2. Final Model Architecture

Architecture was discussed in **1. An appropriate model architecture has been employed**

The overall strategy for deriving a model architecture was to utilize a model that known to work. Based on student discussions I decided between CommaAI and Nvidia model. I ultimately decided to use the Nvidia model as it seemed more robust. The whitepaper provided a lot of information I could leverage in building a working solution for this project.

As I mentioned in the **4. Appropriate Training Data** section I augmented the Udacity data set to provide enough training data for my model. Using the techniques I mentioned I generated 30000 images - 20% was used for validation set. The techniques mentioned in that section is ultimately what allowed my car to make it around the tracks. However, that isn't how it started...

Initially, instead of using the technique I mentioned to generate random data, I grabbed all the images and shoved the left, right, and center image (each in its own row) into a list. I didn't introduce randomization of data selection. This made my data limited to Udacity dataset X3. Was that enough? It wasn't. I trained my model using this data set and found was that the car would drive off the track or wouldn't make it around corners. Why? Looking at the distribution of the data I found that there was simply still too much angles close to 0. Also the data simply didn't provide enough variance, which ultimately led me to the what I outlined in **4. Appropriate Training Data** section.

Even though the first approach I utilized actually gave me low loss numbers on the validation set it didn't mean anything significant in terms of success rate around the track. No matter how much dropout I added or additional epochs or tuning I did helped my car around the track. Although frustrating, it led me to the conclusion that data preprocessing and variance of data set was extremely important in training our model to get the car around the track.

After refining the generation of data through randomizing data selection and augmentation the car was finally able to make it around the track 1. As a surprise, I also found that my car was able to make it around track 2. Why? This is attributed to the vertical/horizontal shifting of my images AND the randomization of brightness. Randomization of brightness taught my model to rely less on colors/brightness/saturation.  

####3. Creation of the Training Set & Training Process

I only utilized Udacity's provided data and did not collect any additional data. I outlined how I augmented and created the data set in **4. Appropriate training data**. Many of the previous sections answer this section.


### Videos
I've recorded a video of my car going through track 1 and track 2 in my the main directory of my project.
