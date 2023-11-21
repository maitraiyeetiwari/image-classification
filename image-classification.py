import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

# converting the data into train and test set as numpy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 

#if I try to view fashion_mnist directly, I cannot. So, trying to view the train_image which is a numpy array and this will work as shown below.

#looking at the data

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#storing class names in an array format in the right order (we found the order online) so that we can use them later for plotting 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.copper)
    #plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#plt.savefig('data.png', dpi =300)
plt.show()

#explore data
print('train data:',train_images.shape)
print('train data labels len:',len(train_labels))
print('train data labels:',train_labels)

print('test data:',test_images.shape)
print('test data labels len:',len(test_labels))
print('test data labels:',test_labels)


#preprocessing the data by dividing it by 255, which is the number of pixels in every image. 

train_images = train_images 
test_images = test_images
#train_images = train_images / 255.0
#test_images = test_images / 255.0

#building the model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


#compiling the model

#Optimizer —This is how the model is updated based on the data it sees and its loss function.
#Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
#Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#train the model

model.fit(train_images, train_labels, epochs=10) # 10 is the number of total labels

#accuracy

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#predicting models

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) #With the model trained, you can use it to make predictions about some images. Attach a softmax layer to convert the model's linear outputs—logits—to probabilities, which should be easier to interpret.
predictions = probability_model.predict(test_images)


print(predictions[0])

#print('highest confidence label',np.argmax(predictions[0]) #which label has the highest confidence


np.argmax(predictions[0]) #which label has the highest confidence






