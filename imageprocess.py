#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 03:16:15 2022

@author: morteza shokraneh
adapted from https://github.com/ardamavi/HandwritingRecognition 
"""


from sklearn import tree
from os import listdir
from PIL import Image
import joblib 
import numpy 
import sys 


# return image information from given address
def get_image(img_address):
    img = Image.open(img_address()).convert('L')
    img = numpy.array(img).tolist()
    img_list = [value for line in img for value in line]
    return img_list  
        

# give train and test datasets and their labels 
def get_datasets(path):
    X_train, y_train, X_test, y_test = [], [], [], []
    img_dirs = [] 
    for char_dir in listdir(path):
        img_dirs = listdir(path+'/'+char_dir) 
        split_ratio = int(0.8*len(img_dirs)) 
        train_imgs, test_imgs = img_dirs[:split_ratio], img_dirs[split_ratio:]
        for img_dir in train_imgs:
            X_train.append(get_image(path+'/'+char_dir+'/'+img_dir))
            y_train.append(ord(char_dir))
        for img_dir in test_imgs:
            X_test.append(get_image(path+'/'+char_dir+'/'+img_dir))
            y_test.append(ord(char_dir))
            
    return X_train, y_train, X_test, y_test
        

img = get_image(sys.argv[1]) #get the test image from commandline 

clf = tree.DecisionTreeClassifier() 
X_train, y_train, X_test, y_test = get_datasets('data/train/trainingSample') 
clf = clf.fit(X_train,y_train)
joblib.dump(clf, 'data/classifier.pkl')


print('Train Score:', clf.score(X_train, y_train)) 
print('Test Score:', clf.score(X_test, y_test))  
print(chr(clf.predict([img])[0]))  #inverse of ord  










    
    
    
