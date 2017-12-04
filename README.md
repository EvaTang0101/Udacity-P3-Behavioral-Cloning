The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

Helper Function to Pre-process Data
To preprocess the data, I firstly read data as RGB, and then went through the following steps:
- Brightness adjustment: The car was not performing well where there are shadows on the track, and therefore I adjusted the brightness of the image to capture it.
- Capturing left, center and right cameras: Since most of the data only captures correct driving, when the car is drifting away from the center, the car will not know where to steer. Therefore, I randomly choose between left, center and right cameras to capture side pictures. In addition, a correction angle is added for the steering angle of the left images and dedeucted deducted for the right images.
- Flip images: Since the training data are all recorded by driving counter-clockwise, which mostly involves left turns, the model could not generalize well. Thus, I randomly flipped 75% of the images, and assigned a opposite sign of the steering measure to simulate driving in the opposite direction so the model can better generalize.
- Zero Bias Adjustment: Since most of the training data has steering angle of zero, the model will tend to drive straight ahead, and get stuck easily. In order to correct the bias towards zero, I added a random threshold so that the steering measures with angles below the threshold will be excluded.
    
Collect and Load Data
I started with the Udacity data, but the data mainly captures normal driving. In order to better train the model, I used the simulator to capture more data:
- 2 laps of normal driving to expand the training data
- 1 lap of weaving driving: when the car drifts away from the center, it needs to know how to recover back to center. Therefore, I added some data of weaving driving.
- 1 lap of revsered lap: I drove the simulator clockwise to generalize. However, due to bad driving skills, this dataset did not help much and I excluded it from the final training set.
- 2 times weaving cross the bridge and the big left turn after bridge: the model performs badly at the bridge since the bridge edge is straight, and the old data tends to set steering angle to zero on the bridge. When the car enters the bridge at an angle, it gets stuck. Also, the car easily get stuck at the big left turn after the bridge. Therefore I collected extra data for those sections and the model performs well on those sections.
- 4 times the big right turn - since the counter-clockwise lap mainly consists of left turns, the steering measures are skewed to negative angles, and the model performs badly at the big right turn towards the end of the lap. Also, the right turn is really sharp and requires a very big angle, while most of the training data have smaller angles. In order to perform better on that section, I added 8 times of the right turn section to the training data. This corrects the skewness of the angle distributions.

Model Training
- Data Normalization and Cropping: I first normalized the image data to be between -0.5 and 0.5. Then I chopped out the upper and lower part of the images, which mainly consists of sky and the hood of the car, and do not help the training process.
- Regularization- I applied 20% dropout on the fuly connected layers to avoid overfitting.
- Optimizer- I used Adam optimizer to minimize the mean square error of the prediction loss.
- Train/Valid split- I splitted 20% of training data as validation data. No test data is needed because running the model in the simulator is equivalent to testing the model.
- Epoch- The validatio loss starts to increase after epoch 7, which is a sign of overfitting. Therefore, I stopped at epoch 7 for training the model.
- Generator- I did not use generator because the code runs pretty fast already, and the current structure is more convinent for adding/excluding extra training data.
- Model Structure- I constructed the model based on the NVIDIA Architecture.

Performance
- The car can run on track one smoothly without stepping on the lines.

Potential Improvements
1. the car seem to oscilate a bit when it is not necessary. This is caused by the addition weaving data added and the adjustment for skewness. The data pre-processing can be fine-tuned to produce more smooth driving.
2. the car doesn't perform well on the second track. It is because the model couldn't generalize well enough on windy mountain road with two lanes. More training data from the second track can be added to further train the model.
3. data augmentation can be added to better generalize the model.
4. different architectures can be experimented.

