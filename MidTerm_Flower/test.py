#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from train import fd_hu_moments
from train import fd_haralick
from train import fd_histogram

# fixed-sizes for image
fixed_size = tuple((500, 500))
# no.of.trees for Random Forests
num_trees = 100
# train_test_split size
test_size = 0.10
# seed for reproducing same results
seed = 9

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100, random_state=9)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# path to test data
test_path = "dataset/test/"
train_labels = "output/labels.h5"


#### GET TEN LABEL HOA
def getTenHoa(i):
    switcher={
            0:'bluebell',
            1:'buttercup',
            2:'coltsfoot',
            3:'cowslip',
            4:'crocus',
            5:'daffodil',
            6:'daisy',
            7: 'dandelion',
            8: 'fritillary',
            9: 'iris',
            10: 'lilyvalley',
            11: 'pansy',
            12: 'snowdrop',
            13: 'sunflower',
            14: 'tigerlily',
            15: 'tulip',
            16: 'windflower'
        }
    return switcher.get(i,"I don't know your flowers - Hung Thinh")




# loop through the test images
for file in glob.glob(test_path + "*.jpg"):
    # read the image
    image = cv2.imread(file)    

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    

    # show predicted label on image
    print("Hung Thinh - Ten hoa la: " + getTenHoa(prediction))
    cv2.putText(image, getTenHoa(prediction), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
