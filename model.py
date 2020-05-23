from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam ,RMSprop ,SGD ,Nadam ,Adamax

from keras.models import Sequential
model = Sequential()

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(MaxPooling2D(pool_size=(2, 2)))


def architecture(option):
    if option == 1:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
    elif option == 2:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
    elif option == 3:
        #two convolutional and 2 max pooling layers
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    elif option == 4:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    
    else:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))


architecture(random.randint(1,4))

model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))

model.add(Flatten())

def fullyconnected(option):
    if option == 1:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 2:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 3:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 4:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        
    else:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    
          
        
fullyconnected(random.randint(1,5))

model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer=random.choice((RMSprop(lr=.0001),Adam(lr=.0001),SGD(lr=.001),Nadam(lr=.001),Adamax(lr=.001))),loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=800)
        
print(out.history['accuracy'][0])


mod =str(model.layers)
accuracy = str(out.history['accuracy'][0]


if out.history['accuracy'][0] >= .75:
    import smtplib
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls()
    s.login("user@gmail.com", "password") 
    message1 = accuracy
    message2 = mod
    s.sendmail("usernam@gmail.com", "receiver@gmail.com", message1)
    s.sendmail("user@gmail.com", "receiver@gmail.com", message2)
        
        


