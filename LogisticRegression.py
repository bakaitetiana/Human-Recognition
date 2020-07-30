# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from PIL import Image
import cv2, os, re, csv
from numpy import array
from sklearn import metrics, cross_validation
import glob
import time
from IPython.display import display
from IPython.display import Image as _Imgdis
from time import time
from time import sleep
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import *
from scipy.special import expit
#from sklearn.linear_model import LogisticRegressionCV
from matplotlib import cm
import math
from skimage.morphology import skeletonize
from skimage.feature import ORB, match_descriptors, plot_matches
#from sklearn.linear_model import LogisticRegression

filenames = glob.glob("C:\\Users\\Tanya\\Downloads\\socofing\\Real\\*.bmp")
file_path = "C:\\Users\\Tanya\\Downloads\\socofing\\Real"
#print(filenames)
#print(len(filenames))
images = [Image.open(fn).convert('L') for fn in filenames]  #conversion to gray scale
images = [i.resize((96, 103)) for i in images]
data = np.dstack([np.array(im) for im in images])

print(filenames[0].split('\\')[-1].split('_')[1]) # for i in range(len(filenames))all pic lab = filenames[i].split('\\')[-1].split('_')[2]
# label
cont =  0
for i in range(len(filenames)):
    bw = images[i].point(lambda x: 0 if x<128 else 255, '1')  #binarization
    lb = filenames[i].split('\\')[-1].split('_')[2]
    #display(bw)
   # print(lb)
    #display(lp_G.astype(int))
    cont +=1
print(cont)

# perform skeletonization
def thin_features(data):
    skeleton=[]
    for i in range(data.shape[2]):
       #skeleton.append(skeletonize((data[:,:,i] > 127).astype(np.int_)))
        #bw = images[i].point(lambda x: 0 if x<128 else 255, '1')
        skeleton = skeletonize(((data[:,:,i] > 127).astype(np.int_)))
        
        descriptor_extractor = ORB(n_keypoints=30)
        descriptor_extractor.detect_and_extract(skeleton)
        responses = descriptor_extractor.responses
        return ["{0:0.2f}".format(i) for i in responses.tolist()]


'''with open("C:\\Users\\Tanya\\Downloads\\socofing\\image_features.csv", 'w', newline='') as myfile:
    d_writer = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
    for file in os.listdir(file_path):
        label = re.search("__(\w)_", file).group(1)
        name = re.search("^(\d+)__", file).group(1)
        dt = thin_features(data)     
        d_writer.writerow([name]+[label] + dt)
        print(file)
#skeleton = skeletonize(data)
        '''
X = []
Y = []

with open("C:\\Users\\Tanya\\Downloads\\socofing\\image_features.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        X.append([float(x) for x in row[2:]])
        #Y.append(row[0])
        Y.append(row[1])
X = np.array(X)
Y = (array(Y) == 'M').astype('float')
#Y = array(Y).astype('float')
Y = np.expand_dims(Y, -1)

def train_test_split(X, Y, split=0.2):
    indices = np.random.permutation(X.shape[0])
    split = int(split * X.shape[0])

    train_indices = indices[split:]
    test_indices = indices[:split]

    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = Y[train_indices], Y[test_indices]

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = train_test_split(X, Y)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

class LogisticRegression:
    
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept =  fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
        
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __cross_entropy(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        #weights initialization
        self.theta = np.zeros(X.shape[1])
        self.theta = self.theta.reshape(self.theta.shape[0],-1)
       # print(self.theta.shape)
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            z = z.reshape(z.shape[0],-1)
           # print(z.shape)
            h = self.__sigmoid(z)
            #print(h.shape)
            #print(y.shape)
            #print(X.T.shape)
           
            gradient = np.dot(X.T, (h - y)) / y.size
            #print(gradient.shape)
            self.theta -= self.lr * gradient
            
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print (f'loss: {self.__cross_entropy(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
 
    '''def accuracy(self, X, y):
        preds = self.predict(X, 170)
        return np.mean(preds == y)
    '''


model = LogisticRegression()
model.fit(x_train, y_train)
l = model.predict_prob(x_test)
print(l)
print("\n")
#n = model.accuracy(x_test, y_test)
#print(n)

#print('Accuracy on test set: {:.2f}%'.format(model.predict(x_test, 54) * 100))
#print('Loss on test set: {:.2f}'.format(model.__cross_entropy(x_test, y_test)))


'''
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
#model.predict_proba(x_test)
print(score)
'''

'''

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(skeleton[-1], cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('skeleton', fontsize=20)
    
ax[1].imshow(data[:,:,-1] , cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('original', fontsize=20)

ax[2].imshow((data[:,:,i] > 127).astype(np.int_), cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('binarized', fontsize=20)

fig.tight_layout()
plt.show()
'''

'''
# edge detection
img = cv2.imread("C:\\Users\\Tanya\\Downloads\\socofing\\Real\\100__M_Left_index_finger.bmp", cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (11, 11), 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
canny = cv2.Canny(img, 100, 150)
cv2.imshow("Image", img)
cv2.imshow("Sobelx", sobelx)
cv2.imshow("Sobely", sobely)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
img = cv2.imread("C:\\Users\\Tanya\\Downloads\\socofing\\Real\\100__M_Left_index_finger.bmp", 0) 
#orb = cv2.ORB_create(nfeatures=20)        # Initiate SIFT detector
def extract_features(img, vector_size=5):
    orb = cv2.ORB_create(nfeatures = 4)
    # find the keypoints and descriptors with SIFT
   # img = skeletonize(img)
   
    kps1 = orb.detect(img)
    kps1 = sorted(kps1, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
    kps1, des1 = orb.compute(img, kps1)
            # Flatten all of them in one big vector - our feature vector
    des1 = des1.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 64
    needed_size = (vector_size * 10)
    if des1.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
        des1 = np.concatenate([des1, np.zeros(needed_size - des1.size)])
        #img2 = cv2.drawKeypoints(img,kp1,None)
       # cv2.imshow("Image", img2)
      #  cv2.waitKey(0)
       # cv2.destroyAllWindows()
        #plt.imshow(img2),plt.show()
    return des1, kps1

l, n = extract_features(img, vector_size=5)
print(l)
'''




    


