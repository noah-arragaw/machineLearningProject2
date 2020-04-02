import pandas as pd
import numpy as np
import math
import csv
import pickle

# function to calculate euclidean distance
def euclideanDistance(row1, row2):
    sum = 0
    for i in range(2):
        sum += pow((row1[i+1] - row2[i+1]), 2)
    
    eucDist = np.sqrt(sum)
    return eucDist

def cosineSimilarity(user1, user2):
    dotProduct = np.dot(user1, user2)

    row1 = np.sum(np.power(user1, 2))
    row1 = np.sqrt(row1)

    row2 = np.sum(np.power(user2, 2))
    row2 = np.sqrt(row2)

    distance = dotProduct / (row1 * row2)
    return distance

# function to calculate k closest users to user passed in
def findKClosest(currentUserID, trainingData, k):
    neighbors = dict()
    for i in range(len(trainingData)):
        if i == currentUserID:
            continue
        else:
            currDistance = cosineSimilarity(trainingData[currentUserID], trainingData[i])
            if len(neighbors) < k:
                neighbors.update({i : currDistance})
            else:
                for key in list(neighbors.keys()):
                    if currDistance > neighbors[key]:
                        del neighbors[key]
                        neighbors.update({i : currDistance})
                        break
    # sort neighbors by similarity for referencing later with different number of neighbors <= 5
    neighbors = {a: b for a, b in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)}
    return neighbors

# function to predict a given user's review of a given movie
def predictReview(trainingData, neighborList, userID, movieToPredict):
    neighborsWhoViewed = dict()
    for i in neighborList[userID]:
        if trainingData[i][movieToPredict] != 0:
            neighborsWhoViewed.update({i : trainingData[i][movieToPredict]})
    
    # if none of the neighbors have viewed the given movie, give default review of 3
    if bool(neighborsWhoViewed) == False:
        return 3
    else:
        review = 0
        runningSimilarity = 0
        for i in neighborsWhoViewed:
            review += neighborsWhoViewed[i] * neighborList[userID][i]
            runningSimilarity += neighborList[userID][i]
        review = review / runningSimilarity
    return review

# function to calculate error using test data
def calculateError(trainingData, neighborList, testingData):
    errors = []
    for i in range(0, 50):
        for j in range(1682):
            if testingData[i][j] == 0:
                continue
            else:
                errors.append(np.power((testingData[i][j] - predictReview(trainingData, neighborList, i, j)), 2))
    error = sum(errors) / len(errors)
    return error

def leaveOneOut(originalTrainingData, trainingData, neighborList, currentLine, k):
    movieToPredict = int(originalTrainingData[currentLine - 1][1])
    currentUser = int(originalTrainingData[currentLine - 1][0])
    neighborListKLength = {}
    for j in range(k):
        key = int(list(neighborList[currentUser])[j])
        value = float(list(neighborList[currentUser].values())[j])
        neighborListKLength.update({key : value})
    neighborsWhoViewed = dict()
    for key in neighborListKLength:
        if trainingData[key][movieToPredict] != 0:
            neighborsWhoViewed.update({key : trainingData[key][movieToPredict]})
    if bool(neighborsWhoViewed) == False:
        prediction = 3
    else:
        prediction = 0
        runningSimilarity = 0
        for i in neighborsWhoViewed:
            prediction += float(neighborList[currentUser][i] * trainingData[i][movieToPredict])
            runningSimilarity += neighborList[currentUser][i]
        prediction = prediction / runningSimilarity
    error = float(np.power((float(originalTrainingData[currentLine - 1][2]) - float(prediction)), 2))
    return error

# def main():

# pull in KNN data previously pickled
fileName = 'trainingData'
infile = open(fileName,'rb')
trainingData = pickle.load(infile)
infile.close()
fileName = 'neighborList'
infile = open(fileName,'rb')
neighborList = pickle.load(infile)
infile.close()
fileName = 'testingData'
infile = open(fileName,'rb')
testingData = pickle.load(infile)
infile.close()
fileName = 'originalTrainingData'
infile = open(fileName,'rb')
originalTrainingData = pickle.load(infile)
infile.close()

print('ERROR: ' + str(calculateError(trainingData, neighborList, testingData)))
runningError = 0
for i in range(5):
    for j in range(len(originalTrainingData) - 250):
        error = leaveOneOut(originalTrainingData, trainingData, neighborList, j + 1, i + 1)
        runningError += float(error)
    avgError = runningError / (len(originalTrainingData) - 1)
    print('average error for k = ' + str(i + 1) + ': ' + str(avgError))
    runningError = 0

# all code that was used to get pickled data

# # create originalTrainingData list of lists to hold training file as it is given
# originalTrainingData = []
# with open("u1-base.base",'r') as f:    
#     lines = f.readlines()
#     for line in lines:
#         entries = line.strip().split("\t")
#         originalTrainingData.append(entries)

# # initialize trainingData to be zeroed out list of lists
# trainingData = []
# for i in range(500):
#     trainingData.append([])
#     for k in range(1682):
#         trainingData[i].append(0)

# file = open("u1-test.test", encoding="ISO-8859-1")
# trainingRows = file.readlines()

# # initialize testingData to be zeroed out list of lists
# testingData = []
# for i in range(500):
#     testingData.append([])
#     for k in range(1682):
#         testingData[i].append(0)

# file = open("u1-test.test", encoding="ISO-8859-1")
# testingRows = file.readlines()

# # loop through testing data writing to new list
# # every row is an individual user; every column is a unique movie
# for row in testingRows:
#     sepRow = row.split('\t')
#     userID = sepRow[0]
#     movie = sepRow[1]
#     rating = sepRow[2]
#     testingData[int(userID) - 1][int(movie)] = int(rating)

# # loop through every user writing a list of their KNN to neighborList
# neighborList = []
# for user in range(500):
#     neighborList.append(findKClosest(user, trainingData, 5))
#     print('neighbor list appended.')

# # pickle data into file called 'neighborList'
# filename = 'neighborList'
# outfile = open(filename,'wb')
# pickle.dump(neighborList, outfile)
# outfile.close()

# # pickle data into file called 'trainingData'
# filename = 'trainingData'
# outfile = open(filename,'wb')
# pickle.dump(trainingData, outfile)
# outfile.close()

# # pickle data into file called 'testingData'
# filename = 'testingData'
# outfile = open(filename,'wb')
# pickle.dump(testingData, outfile)
# outfile.close()

# # pickle data into file called 'originalTrainingData'
# filename = 'originalTrainingData'
# outfile = open(filename,'wb')
# pickle.dump(originalTrainingData, outfile)
# outfile.close()
# print(originalTrainingData)