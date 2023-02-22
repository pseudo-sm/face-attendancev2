
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import Model,model_from_json,load_model
from keras.layers import Layer,Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import keras.backend as K



# Define Encoder Model Architecture ( Inception-ResNet-v1 )
# Update model path according to your working environment

model_path = "Models/Inception_ResNet_v1.json"

json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
enc_model = model_from_json(loaded_model_json)


# Set encoder model to be trainable

enc_model.trainable = True


# Initialize a MTCNN face detector

mtcnn_detector = MTCNN()

def detect_face(filename, required_size=(160, 160)):

	img = Image.open(filename)
	
  # convert to RGB
	img = img.convert('RGB')
 
	# convert to array
	pixels = np.asarray(img)
 
	# detect faces in the image
	results = mtcnn_detector.detect_faces(pixels)
 
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']

	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	# extract the face
	face = pixels[y1:y2, x1:x2]
  
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
 
	return face_array
     
#      # Detect and store faces(with labels) in the training dataset
# # Keep different images of a single person in the same folder named as his/her name(LABEL)
# # Update dataset path according to your working environment

cropped_faces =[]
face_labels = []

dataset_path = "data/data/train"

i=0

for person in os.listdir(dataset_path):
  for filename in os.listdir(os.path.join(dataset_path,person)):

    # Detect faces
    try :
      face = detect_face(dataset_path+'/'+person+'/'+filename)
    except:
      print(filename  + " can't be loaded !")
      continue    
    cropped_faces.append (face)

    # Save labels
    label = person
    face_labels.append(label)

    i+=1
    if i%50 == 0 :
      print(str(i)+" images loaded !") 
    
print("\nTotal "+str(i)+" images loaded !")

face_labels = np.array (face_labels)
cropped_faces = np.array(cropped_faces)
print(face_labels.shape)
print(cropped_faces.shape)  

# # Compute No. of IDs loaded

n_IDs = len(np.unique(face_labels))
print(n_IDs)
     
# # Visualize dataset loaded

index = 70

plt.imsave('test.png',cropped_faces[index])
print(face_labels[index])
     

# # Function to find Euclidean distance between 2 Faces

def distance(a,b):

  a/= np.sqrt(np.maximum(np.sum(np.square(a)),1e-10))
  b/= np.sqrt(np.maximum(np.sum(np.square(b)),1e-10))

  dist = np.sqrt(np.sum(np.square(a-b)))

  return dist


# # Function to find Euclidean distances between 2 batch of Faces

def distance_batch(a,b):

  a/= np.sqrt(np.maximum(np.sum(np.square(a),axis=1,keepdims=True),1e-10))
  b/= np.sqrt(np.maximum(np.sum(np.square(b),axis=1,keepdims=True),1e-10))

  dist = np.sqrt(np.sum(np.square(a-b),axis=1))

  return dist
     

# # Function to normalize a Single face

def normalize_single(X):

  axis = (0,1,2)

  mean = np.mean(X,axis)
  std = np.std(X,axis)

  size= X.size
  adj_std = np.maximum(std,1/np.sqrt(size))

  X = (X-mean)/adj_std
  return X
     

# # Function to normalize a Batch of faces

def normalize_batch(X):

  axis = (1,2,3)

  mean = np.mean(X,axis,keepdims=True)
  std = np.std(X,axis,keepdims=True)

  size = X[0].size
  adj_std = np.maximum(std,1/np.sqrt(size))

  X = (X-mean)/adj_std
  return X
     

# # Function to normalize a Triplet batch

def normalize_triplet_batch(X):

  axis = (2,3,4)

  mean = np.mean(X,axis,keepdims=True)
  std = np.std(X,axis,keepdims=True)

  size = X[0][0].size
  adj_std = np.maximum(std,1/np.sqrt(size))

  X = (X-mean)/adj_std
  return [X[0],X[1],X[2]]
     

# # Generate all possible Anchor-Positive Combinations 

X_anchor = []
X_positive = []

X_anchor_labels = []

persons_list = np.unique(face_labels)

for person in persons_list:

  filter_person = (face_labels==person).reshape(cropped_faces.shape[0])
  X_person = cropped_faces[filter_person]

  for face1 in range(X_person.shape[0]):
    for face2 in range(face1+1,X_person.shape[0]):

      X_anchor.append (X_person[face1])
      X_positive.append (X_person[face2])

      X_anchor_labels.append (person)

X_anchor = np.array(X_anchor)
X_positive = np.array(X_positive)

X_anchor_labels = np.array(X_anchor_labels)

print(X_anchor.shape)
print(X_positive.shape) 

# # Function to generate a batch of Triplets ( Anchor-Positive-Negative )

def get_batch(n_rand_triplets,n_hard_triplets):

  # Generate n_rand_triplets no. of Random Triplets

  filter = np.random.choice(list(range(0,X_anchor.shape[0])),int(n_rand_triplets))

  X_anchor_random  = X_anchor[filter]
  X_positive_random = X_positive[filter]
  X_negative_random = []

  X_anchor_random_labels = X_anchor_labels[filter]

  for i in range(int(n_rand_triplets)):
    flag=False
    while(flag==False):
      index = np.random.randint(0,cropped_faces.shape[0])
      if face_labels[index]!=X_anchor_random_labels[i]:
        X_negative_random.append(cropped_faces[index])
        flag=True

  X_negative_random = np.array(X_negative_random)

  # If only hard triplets required
  if n_rand_triplets == 0:
    X_negative_random = X_anchor_random # Empty arrays 

  # Generate n_hard_triplets no. of Hard Triplets ( Positive distance > Negative distance )

  X_negative_samples = []

  for i in range(X_anchor.shape[0]):

    flag=False
    while(flag==False):
      index = np.random.randint(0,cropped_faces.shape[0])
      if face_labels[index]!=X_anchor_labels[i]:
        X_negative_samples.append(cropped_faces[index])
        flag=True
    
  X_negative_samples = np.array(X_negative_samples)  

  Negative_distances = distance_batch(enc_model.predict(normalize_batch(X_anchor)),
                                      enc_model.predict(normalize_batch(X_negative_samples)))
  
  Positive_distances = distance_batch(enc_model.predict(normalize_batch(X_anchor)),
                                      enc_model.predict(normalize_batch(X_positive)))

  distances =  Negative_distances - Positive_distances

  filter = np.argsort(distances)[:int(n_hard_triplets)]

  X_anchor_hard = X_anchor[filter]
  X_positive_hard = X_positive[filter]
  X_negative_hard = X_negative_samples[filter]

#   # Concatenate to form a Triplet batch of required size

  X_anchor_batch  = np.concatenate((X_anchor_random,X_anchor_hard))
  X_positive_batch = np.concatenate((X_positive_random,X_positive_hard))
  X_negative_batch = np.concatenate((X_negative_random,X_negative_hard))

  return [X_anchor_batch,X_positive_batch,X_negative_batch]
     

# # Define Triplet Loss Custom Keras Layer

class TripletLossLayer(Layer):

    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
      
        anchor, positive, negative = inputs

        anchor = anchor/K.sqrt(K.maximum(K.sum(K.square(anchor),axis=1,keepdims=True),1e-10))
        positive = positive/K.sqrt(K.maximum(K.sum(K.square(positive),axis=1,keepdims=True),1e-10))
        negative = negative/K.sqrt(K.maximum(K.sum(K.square(negative),axis=1,keepdims=True),1e-10))

        p_dist = K.sqrt(K.sum(K.square(anchor-positive), axis=1))
        n_dist = K.sqrt(K.sum(K.square(anchor-negative), axis=1))

        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0))
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
     

# # Function to create a model that trains our Face Encoder

def create_enc_trainer(input_shape,enc_model,margin = 0.5):

    # Define Input tensors
    anchor = Input(input_shape, name="anchor_input")
    positive = Input(input_shape, name="positive_input")
    negative = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    enc_anchor = enc_model(anchor)
    enc_positive = enc_model(positive)
    enc_negative = enc_model(negative)
    
    # TripletLoss Layer
    loss_layer=TripletLossLayer(margin=margin,name='triplet_loss')([enc_anchor,enc_positive,enc_negative])
    
    # Connect the inputs with the outputs
    enc_trainer = Model(inputs=[anchor,positive,negative],outputs=loss_layer,name = "Trainer Model")
    
    # return the model
    return enc_trainer
     

# Create and compile Trainer model

enc_trainer_model = create_enc_trainer((160,160,3),enc_model,margin = 0.5)
enc_trainer_model.compile(optimizer=Adam(lr=0.0005))

enc_trainer_model.summary()
# plot_model(enc_trainer_model,to_file='enc_trainer_model.png')


# Train the Encoder Model using Trainer model

epochs = 100
random_batch_size = 25
hard_batch_size = 75

losses = []

for e in range(1,epochs+1):

  mini_batch = get_batch(random_batch_size,hard_batch_size)
  loss = enc_trainer_model.train_on_batch(normalize_triplet_batch(mini_batch),None)  
  losses.append(loss)

  if(e%5 == 0):
    print("Triplet Loss after "+str(e)+' epochs : '+str(loss))

# Plot Triplet Loss over training period

e = list(range(1,epochs+1))

plt.plot(e,losses)
plt.xlabel('Epochs')
plt.ylabel('Triplet Loss')
plt.show()

test_batch = get_batch(128,0)



# Visualize Triplets with positive and negative distances

index = 0

anchor = test_batch[0][index]
positive  = test_batch[1][index]
negative = test_batch[2][index]

plt.figure(figsize=(12,12))

plt.subplot(1,3,1)
plt.imshow(anchor)
plt.title('ANCHOR')
plt.subplot(1,3,2)
plt.imshow(positive)
plt.title('POSITIVE')
plt.subplot(1,3,3)
plt.imshow(negative)
plt.title('NEGATIVE')

plt.show()

anchor_enc  = enc_model.predict(normalize_single(anchor).reshape(1,160,160,3))
positive_enc  = enc_model.predict(normalize_single(positive).reshape(1,160,160,3))
negative_enc  = enc_model.predict(normalize_single(negative).reshape(1,160,160,3))

print("Distance between Anchor and Positive : "+str(distance(anchor_enc,positive_enc)))
print("Distance between Anchor and Negative : "+str(distance(anchor_enc,negative_enc)))


save_location = "results/models/enc_model_weights.h5"
enc_model.save_weights(save_location)