import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import plot_learning_curves as plc




# Importing Keras libraries
from tensorflow import keras 
from tensorflow.keras import backend as k 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D,GlobalAveragePooling1D, Conv1D
from tensorflow.keras.preprocessing.image import img_to_array



# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score 


import itertools

from PIL import ImageFile              
ImageFile.LOAD_TRUNCATED_IMAGES = True       

 


import random
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



#######################################################
  
# training a Classifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



import xgboost as xgb
clf= xgb.XGBClassifier()

print(clf)

import warnings
warnings.filterwarnings('ignore')




train = [os.path.join("train",img) for img in os.listdir("train")]

test= [os.path.join("test",img) for img in os.listdir("test")]


print(len(train),len(test))
print(type(train))
print(type(test))




train2=[os.path.basename(path) for path in train]


test2=[os.path.basename(path) for path in test]


print(len(train2),len(test2))
print(type(train2))
print(type(test2))

print((train2==train))
print((test2==test))
##############################################################


columns_train=['retinal_hemorrhage', 'hard_exudate','microaneurysm']


print("columns_train=",columns_train)


df_train=pd.read_csv('lesions_train.csv')
 
df_test=pd.read_csv('lesions_test.csv',) 
print(",,,,",df_test.shape)

columns_test=['microaneurysm','exudate','hemorrhage']



#############################################################

# load the VGG16 network
print("[INFO] loading network...")


 
# chop the top dense layers, include_top=False
model = VGG16(weights="imagenet", include_top=False)
model.summary()

print("@@@@@@@@@@@@@@@@@")

# chop the top dense layers, include_top=False
model2 = ResNet50(weights="imagenet", include_top=False)
model2.summary()

print("&&&&&&&&&&&&&&&&&")





def create_features(dataset, pre_model):
 
    x_scratch = []
 
    # loop over the images
    for imagePath in dataset:
 
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
 

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
 
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512)) #np.concatenate((features.shape[0], lesions), axis=1)
    return x, features, features_flatten



def create_features2(dataset, pre_model):
 
    x_scratch = []
 
    # loop over the images
    for imagePath in dataset:
 
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
 
        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
 
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 2048)) #np.concatenate((features.shape[0], lesions), axis=1)
    return x, features, features_flatten

##############################################################


train_x, train_features, train_features_flatten = create_features(train, model) #print(train.shape)  #train_x is the image array
test_x, test_features, test_features_flatten = create_features(test, model)

 
print("\*****",train_x.shape, train_features.shape, train_features_flatten.shape)
print("\*****",test_x.shape, test_features.shape, test_features_flatten.shape)

print("#####################")
print(type(train_x))
print(type(test_x))

print("#####################")
print(type(train_features))
print(type(test_features))


print(type(train_features_flatten))
print(type(test_features_flatten))

print("#####################")

train_x50, train_features50, train_features_flatten50 = create_features2(train, model2) #
test_x50, test_features50, test_features_flatten50 = create_features2(test, model2)

 
print("\*****",train_x50.shape, train_features50.shape, train_features_flatten50.shape)
print("\*****",test_x50.shape, test_features50.shape, test_features_flatten50.shape)


print("#####################")
print(type(train_x50))
print(type(test_x50))

print("#####################")
print(type(train_features50))
print(type(test_features50))

print("#####################")


print(type(train_features_flatten50))
print(type(test_features_flatten50))





###########################################################################################

y_train = df_train.iloc[: , -1] 
y_test = df_test.iloc[: , -1]


print(",,,,",y_train.shape)
print(",,,,",y_test.shape)



print("WWWW",df_test.shape)
print(df_test)


#drop 1st column of samples_name 
df_train= df_train.iloc[: , 1:]
df_test= df_test.iloc[: , 1:]

print("0000",df_test.shape)
print(df_test)


lesions_features_train = df_train[columns_train].to_numpy()
df_test = df_test.iloc[: , :-1]
lesions_features_test = df_test[columns_test].to_numpy()

print("DDDDD",lesions_features_test.shape)
print(lesions_features_test)

print("^^^^",train_features_flatten.shape)
print("&&&&&",lesions_features_train.shape)

###################################################
#extract features:




#features_train=train_features_flatten
#features_test=test_features_flatten

#features_train=train_features_flatten50
#features_test=test_features_flatten50

#features_train=lesions_features_train
#features_test=lesions_features_test


features_train=np.concatenate((train_features_flatten,lesions_features_train),axis=1) #Vgg-16+lesion

#features_train=np.concatenate((train_features_flatten50,lesions_features_train),axis=1) #Resnet50 +lesion

#features_train=np.concatenate((train_features_flatten, train_features_flatten50),axis=1) #Vgg-16+Resnet50

#features_train=np.concatenate((train_features_flatten, train_features_flatten50,lesions_features_train),axis=1) #all features


print("----",features_train.shape)


features_test=np.concatenate((test_features_flatten,lesions_features_test),axis=1)   #Vgg-16+lesion

#features_test=np.concatenate((test_features_flatten50,lesions_features_test),axis=1) #Resnet50 +lesion

#features_test=np.concatenate((test_features_flatten, test_features_flatten50),axis=1) #Vgg-16+Resnet50

#features_test=np.concatenate((test_features_flatten, test_features_flatten50,lesions_features_test),axis=1) #all features


print("----",features_test.shape)








features_train= scaler.fit_transform(features_train)


features_test= scaler.fit_transform(features_test)





#################################################################


print("validation_part with ML")

X_train=features_train
X_test=features_test





print("@@@",X_train.shape)
print("@@@",y_train.shape)


model = clf.fit(X_train, y_train)



###end of lesion part


y_pred  = model.predict(X_test)     

print("*****",y_pred.shape)

print("@@@",y_pred)
print("@@@",y_test)
##########################################

# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix
# creating a confusion matrix 
#cm = confusion_matrix(y_test, y_pred )
print("y_pred.shape=",y_pred.shape)
print(y_pred[0: 5])
print("y_test.shape=",y_test.shape)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred,multi_class="ovr",average=None)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)


l=confusion_matrix(y_test, y_pred)
print(l)
print('TN',l.item((0, 0)))
print('FP',l.item((0, 1)))
print('FN',l.item((1, 0)))
print('TP',l.item((1, 1)))
##################################
print("end of validation_part with ML for lesions part")
print(clf)
print(columns_test)


y_prob = model.predict_proba(X_test)
y_prob = y_prob[:,1]

#plot learning curve

import matplotlib.pyplot as plt


print(type(X_train))
print(type(X_test))


plc.plot_learning_curves(clf, X_train, y_train, X_test, y_test)

# Create plot
#plt.title("Learning Curve")
plt.xlabel("Training Set (Size)"), plt.ylabel("Accuracy")
plt.xticks(fontsize=18, weight = 'bold') 
plt.yticks(fontsize=18, weight = 'bold')

plt.tight_layout()
#plt.show()
plt.savefig('Xgboost_LC_validation_VGG-16_lesions.png',dpi=300)
plt.savefig('Xgboost_LC_validation_VGG-16_lesions.svg',dpi=300)
plt.close()

#plot ROC curve
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt1

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt1.plot(fpr,tpr)
#plt1.title("ROC Curve")
#axis labels
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt.xticks(fontsize=12, weight = 'bold') 
plt.yticks(fontsize=12, weight = 'bold')
plt1.legend(loc="best")
#plt.show() 
plt1.savefig('Xgboost_ROC_validation_VGG-16_lesions.png',dpi=300)
plt1.savefig('Xgboost_ROC_validation_VGG-16_lesions.svg',dpi=300)
plt1.close()

#############################################


