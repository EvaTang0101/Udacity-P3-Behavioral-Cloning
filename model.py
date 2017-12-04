import csv
import cv2
import numpy as np
from scipy.misc import imread

### helper function to adjust brightness
def augmentbrightness(image):
    temp = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    #Compute a random brightness value and apply to the image
    brightness = 0.5 + np.random.uniform()
    temp[:,:,2] = temp[:,:,2] * brightness
    temp[:,:,2][temp[:,:,2]>225] = 225
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

### read data    
def readdata(dircsv, dirimage):
    lines = []
    with open(dircsv) as csvfile:
        reader = csv.reader(csvfile)
        next(reader,None)
        for line in reader:
            lines.append(line)
    
    images = []
    measurements = []
    for line in lines:
        img_choice = np.random.randint(3)
        
        #randomly discard image with angle below a threshold
        threshold = np.random.uniform(low=0,high=0.01)
        if abs(float(line[3])) > threshold:
            #Randomly choose between left, center and right images
            source_path = line[img_choice]
            filename = source_path.split('/')[-1]
            current_path = dirimage +filename
            #print(current_path)
            image = imread(current_path)
        
            if img_choice==0: #center camera
                correction = 0
            elif img_choice == 1: #left camera
                 correction = 0.2 
            elif img_choice==2: #right camera
                correction = -0.2

            #print(i,filename, correction)
            measurement = float(line[3]) + correction
        
            #adjust brightness
            images.append(augmentbrightness(image))
            measurements.append(measurement)
 
            #flip image: randomly flip 75% of the images
            if np.random.randint(4) < 3:
                images.append(cv2.flip(image,1))
                measurements.append(measurement*-1.0)
               
    return (images, measurements)
    
    
    X1, Y1 = readdata('data1/driving_log.csv', 'data1/IMG/')
X2, Y2 = readdata('data2/driving_log.csv', 'data2/IMG/') #2x normal driving
X3, Y3 = readdata('data3/driving_log.csv', 'data3/IMG/') #1x weaving driving
X4, Y4 = readdata('data4/driving_log.csv', 'data4/IMG/') #1x reversed lap
X5, Y5 = readdata('data5/driving_log.csv', 'data5/IMG/') #2x Bridge and the big left turn after bridge
X6, Y6 = readdata('data6/driving_log.csv', 'data6/IMG/') #1 Right turn
X7, Y7 = readdata('data7/driving_log.csv', 'data7/IMG/') #3 Right turn
X8, Y8 = readdata('data8/driving_log.csv', 'data8/IMG/') #3 Right turn
X9, Y9 = readdata('data9/driving_log.csv', 'data9/IMG/') #1 Right turn
X10, Y10 = readdata('data10/driving_log.csv', 'data10/IMG/') #1 end part


X_train = []
y_train = []
X_train = np.concatenate((X1,X2,X3,X5[300:],X6,X7,X8,X9,X10), axis=0)
X_train = np.array(X_train)
y_train = np.concatenate((Y1,Y2,Y3,Y5[300:],Y6,Y7,Y8,Y9,Y10), axis=0)
y_train = np.array(y_train)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) #normalize data
model.add(Cropping2D(cropping=((50,20), (0,0))))
#model.add(Lambda(lambda x: K.tf.image.resize_images(x, (64, 64)))) #resize to 64x64 to expedite training
model.add(Convolution2D(24,5,5,subsample = (2,2), activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2), activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2), activation = "relu"))
model.add(Convolution2D(64,3,3,subsample = (1,1), activation = "relu"))
model.add(Convolution2D(64,3,3,subsample = (1,1), activation = "relu"))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
###Relu
# model.add(Convolution2D(6,5,5,activation = "relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation = "relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
filepath = 'weighs-improvement-{epoch:02d}-{val_loss:.2f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor="val_loss",verbose=1,save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X_train, y_train, validation_split = 0.2,shuffle=True, nb_epoch =6,batch_size =128, callbacks=callbacks_list, verbose=1)

model.save('model.h5')



