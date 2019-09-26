import numpy as np
from mnist import MNIST
import cv2
import time

# Note: This program requires the cv2 and mnist library installed. If you don't have these
# libraries installed on your computer, then please install them py using following comminads:
# pip install mnist
# Please run cmd prompt with admin privileges in order to install these libraries.
# You need to download the MNIST dataset from Yann Lecun's website first.

startTime = time.time()
mndata = MNIST('D:\Study\Pattern Recognition\Project1')
mndata.gz = True
trainImages, trainLabels = mndata.load_training()
testImages,testLabels = mndata.load_testing()

trainImages = np.array(trainImages)
trainLabels = np.array(trainLabels)
testImages = np.array(testImages)
testLabels = np.array(testLabels)

print(np.shape(np.array(trainImages[1])))
classes = np.unique(trainLabels)
seperatedTrainData = []
for i in classes:
    idx2 = np.where(trainLabels == i)
    seperatedTrainData.append(trainImages[np.array(idx2)])
print(len(seperatedTrainData))
c = 0
allMeans = []
for j in seperatedTrainData:
    print("calculating for digit: "+str(c))

    digit = np.array(j[0])
    #print(np.shape(digit))
    means = np.sum(digit, axis=0) / np.shape(digit)[0]
    allMeans.append(means)
    out = np.reshape(means,(28,28))
    cv2.imwrite("mean_"+str(c)+".jpg",out)
    diffs = np.abs(digit - means)
    variances = np.sum(np.square(diffs), axis=0) / np.shape(digit)[0]
    variances = np.reshape(variances, (28, 28))
    variances = np.sqrt(variances)
    #print(np.shape(covariances))
    #print(np.max(covariances))
    cv2.imwrite("dev_" + str(c) +".jpg", variances)
    c += 1
allCovar = []
for k in range(10):
    print("Computing covar for "+str(k))
    d = np.array(seperatedTrainData[k][0])
    mean0 = allMeans[k]
    subtractedMean = d - mean0
    covar = np.dot(np.transpose(subtractedMean),subtractedMean)/(np.shape(d)[0]-1)
    np.fill_diagonal(covar,np.diagonal(covar)+1)
    covar = np.linalg.inv(covar)
    allCovar.append(covar)

print(np.shape(testImages))
corrects = 0
incorrects = 0
count = 0
for idx1 in testImages:
    img = idx1
    probs = []
    for idx2 in range(10):

        p1 = np.dot((img - allMeans[idx2]), allCovar[idx2])
        p2 = -(np.dot(p1, np.transpose(img - allMeans[idx2])) / 2)
        probs.append(p2)
    decision = probs.index(max(probs))
    if decision == testLabels[count]:
        corrects += 1
    else:
        incorrects += 1
    count += 1
    print("Decision is: "+str(decision))
acc = corrects/(corrects+incorrects)
print("Corrects: "+str(corrects)+"   Incorrects: "+str(incorrects))
print("Accuracy is: "+str(acc))
endTime = time.time()
print("Finished in: "+str((endTime - startTime)/60) + " minutes.")