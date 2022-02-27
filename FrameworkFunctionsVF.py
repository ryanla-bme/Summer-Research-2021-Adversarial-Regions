"""This file contains the functions used to implement the Active Learning using
Poisoning Attacks with Density Regularisation framework aswell as some other
useful functions.
"""

import numpy as np 
import DatasetGeneratorsVF as DG
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def SplitTrain(full_trainX, full_trainY, random_state):
    """Splits a full training set into a 50-50 split for train and validation 
    set for poisoning attack. 
    """
    splitter = StratifiedKFold(n_splits=2, shuffle = True,\
        random_state = random_state)
    trainIndex, valIndex = splitter.split(full_trainX, full_trainY)
    trainX = full_trainX[trainIndex[0],:]
    trainY = full_trainY[trainIndex[0]]
    valX = full_trainX[valIndex[0],:]
    valY = full_trainY[valIndex[0]]

    return trainX, trainY, valX, valY

def GridFindMax(gridX,gridY,funcGrid):
    """"Finds the maximum value of funcGrid and returns its corresponding X,Y
    coordinates.

    Inputs:
        gridX <- (2D numpy array)
        gridY <- (2D numpy array)
        funcGrid <- (2D numpy array)
    
    Outputs:
        maxCoords <- [X,Y] (1D numpy array)
        maxVal <- (float)
    """

    ind = np.unravel_index(np.argmax(funcGrid), funcGrid.shape)
    maxCoords = np.array([gridX[ind],gridY[ind]])
    maxVal = funcGrid[ind]

    return maxCoords, maxVal    


def EuclideanDist(A,B):
    """Returns the euclidean distance between 
    A and B (two N dimensional vectors)
    """

    return np.sqrt(np.sum(np.square(A-B)))


def EzNHSearch(A, Dataset, sigma):
    """Returns the datapoints within euclidean distance of sigma (radius) from
    point A.
    """
    return np.array([i for i in Dataset if EuclideanDist(A,i) <= sigma])

def LogQuadDensity(NH_dps,nu):
    """Returns -density- value of a point (X) according to a 
    negative log quadratic function.

    Inputs:
        NH_dps <- datapoints in neighbourhood of point X (2D numpy array)
        nu <- number of points at the maximum of the quadratic function (int)

    Outputs:
        -density- value of log quadratic function
    """
    n = np.shape(NH_dps)[0]
    return -1*np.log(((n-nu)**2)+1)

def PoisDReg(L,N, lambda_prm):
    """Returns density regularised poisoning attack

    Inputs: 
        L <- loss function evaluated for dataset with poisoning attack
        D <- density of points around poisoning poiunt
        lamda <- tradeoff parameter

    Output:
        Density regularsied poisoning attack function evaluation 
    """
    return L + lambda_prm*N

def ALPADR(datasetX, datasetY, clf,\
    f1_b, f2_b, resol,\
    lambda_prm, sigma, nu, density_func = LogQuadDensity,\
    normalise = True, regularise = True, random_state = 1):
    """Main function for Active Learning using Poisoning Attacks and Density
    Regularisation (ALPADR).
    """
    # Split dataset into train and validation set for poisoning attack #
    trainX, trainY, valX, valY = SplitTrain(datasetX, datasetY, random_state)

    # Create grid of poisoning points  #
    f1_pois = np.linspace(f1_b[0], f1_b[1], resol)
    f2_pois = np.linspace(f2_b[0], f2_b[1], resol)

    F1_pois, F2_pois = np.meshgrid(f1_pois, f2_pois)
    f1f2_pois = np.vstack([F1_pois.ravel(), F2_pois.ravel()]).T

    n = np.shape(f1f2_pois)[0]
    Loss1 = np.zeros(n) #Loss if class 1 was injected
    Loss0 = np.zeros(n) #Loss if class 0 was injected
    LossMax  = np.zeros(n) #Loss maximum between both classes
    LossMaxLabel = np.zeros(n) #Loss maximum between both classes label
    Densities = np.zeros(n)
    DensityRegLoss = np.zeros(n)

    for i in range(n):

        # Add poisoning point to temporary training set
        tempTrainX = np.vstack([trainX,f1f2_pois[i]])

        #Try injecting with label 1
        tempTrainY = np.hstack([trainY,1])
        clf.fit(tempTrainX,tempTrainY)
        Loss1[i] = 1 - clf.score(valX,valY)

        #Try injecting with label 0
        tempTrainY = np.hstack([trainY,0])
        clf.fit(tempTrainX,tempTrainY)
        Loss0[i] = 1 - clf.score(valX,valY)

        if Loss1[i]> Loss0[i]:
            LossMax[i] = Loss1[i]
            LossMaxLabel[i] = 1
        else:
            LossMax[i] = Loss0[i]
            LossMaxLabel[i] = 0

        if regularise == True:
            # Density calculation
            Densities[i] = density_func(EzNHSearch(f1f2_pois[i], trainX, sigma),nu)
     
    #### Regularised ####
    if regularise == True:
        if normalise == True:
            Loss1 = Loss1/Loss1.max()
            Loss0 = Loss0/Loss0.max()    
            LossMax = LossMax/LossMax.max()
            Densities = Densities/np.abs(Densities.min())
        for i in range(n):
            DensityRegLoss[i] = PoisDReg(LossMax[i], Densities[i], lambda_prm)
        LossMax, LossMaxLabel, Densities, DensityRegLoss\
            = LossMax.reshape(F1_pois.shape), LossMaxLabel.reshape(F1_pois.shape),\
            Densities.reshape(F1_pois.shape), DensityRegLoss.reshape(F1_pois.shape)
        return LossMax, LossMaxLabel, Densities, DensityRegLoss, F1_pois, F2_pois

    #### Unregularised ####
    else:
        if normalise == True:
            Loss1 = Loss1/Loss1.max()
            Loss0 = Loss0/Loss0.max()    
            LossMax = LossMax/LossMax.max()
       
        Loss1, Loss0, LossMax, LossMaxLabel \
            = Loss1.reshape(F1_pois.shape), Loss0.reshape(F1_pois.shape),\
            LossMax.reshape(F1_pois.shape), LossMaxLabel.reshape(F1_pois.shape)
        return Loss1, Loss0, LossMax, LossMaxLabel, F1_pois, F2_pois

def UncertaintySampling(clf, f1_b, f2_b, resol):
    """Main function used for an iteration of Uncertainty sampling baseline 
    active learning strategy.

    Note: Classifier should already be trained on the biased tranining set
    """

    f1 = np.linspace(f1_b[0], f1_b[1], resol)
    f2 = np.linspace(f2_b[0], f2_b[1], resol)
    F1grid,F2grid = np.meshgrid(f1, f2)
    f1f2 = np.vstack([F1grid.ravel(), F2grid.ravel()]).T

    n = np.shape(f1f2)[0]

    probsMax = np.zeros(n)
    probs = clf.predict_proba(f1f2)

    for i in range(n):
        if probs[i,0] > probs[i,1]:
            probsMax[i] = probs[i,0]
        else:
            probsMax[i] = probs[i,1]

    Uncertainties = -1*probsMax + 1
    Uncertainties = Uncertainties.reshape(F1grid.shape)

    return F1grid, F2grid, Uncertainties

def IdealRandomSampling(means,covs,numberIter, RANDOM_STATE):
    """Main function used for Ideal Random baseline active learning strategy.
    Returns all points at once specified by numberIter.
    """
    new_1, new_0 = DG.GenerateMultiVarGaussians(means, covs, np.array([numberIter,numberIter]), RANDOM_STATE + 10)
    new_points = np.vstack((new_1, new_0))
    new_points = shuffle(new_points,random_state = RANDOM_STATE)

    return new_points[:numberIter,:]


def ALPADRUnRegMorePoints(datasetX, datasetY, clf, f1_b, f2_b, resol, random_state, num_points):
    """Returns grid of loss for ALPADR UNREGULARISED with more than one point 
    added onto a coordinate in dataspace. 
    Note: THIS FUNCTION WAS ONLY USED TO SET UP THE INTUITION FOR THE 
          GAN FRAMEWORK. NOT USED FOR ANY ALPADR EXPERIMENTS.
    """
    # Split dataset into train and validation set for poisoning attack #
    trainX, trainY, valX, valY = SplitTrain(datasetX, datasetY, random_state)

    # Create grid of poisoning points  #
    f1_pois = np.linspace(f1_b[0], f1_b[1], resol)
    f2_pois = np.linspace(f2_b[0], f2_b[1], resol)

    F1_pois, F2_pois = np.meshgrid(f1_pois, f2_pois)
    f1f2_pois = np.vstack([F1_pois.ravel(), F2_pois.ravel()]).T

    n = np.shape(f1f2_pois)[0]
    Loss1 = np.zeros(n) #Loss if class 1 was injected
    Loss0 = np.zeros(n) #Loss if class 0 was injected
    LossMax  = np.zeros(n) #Loss maximum between both classes
    LossMaxLabel = np.zeros(n) #Loss maximum between both classes label

    for i in range(n):

        # Add poisoning point to temporary training set
        tempTrainX = np.vstack([trainX,f1f2_pois[i]])
        for j in range(int(num_points-1)):
            tempTrainX = np.vstack([tempTrainX,f1f2_pois[i]])

        #Try injecting with label 1
        tempTrainY = np.hstack([trainY,1])
        for j in range(int(num_points-1)):
            tempTrainY = np.hstack([tempTrainY,1])
        clf.fit(tempTrainX,tempTrainY)
        Loss1[i] = 1 - clf.score(valX,valY)

        #Try injecting with label 0
        tempTrainY = np.hstack([trainY,0])
        for j in range(int(num_points-1)):
            tempTrainY = np.hstack([tempTrainY,0])
        clf.fit(tempTrainX,tempTrainY)
        Loss0[i] = 1 - clf.score(valX,valY)

        if Loss1[i]> Loss0[i]:
            LossMax[i] = Loss1[i]
            LossMaxLabel[i] = 1
        else:
            LossMax[i] = Loss0[i]
            LossMaxLabel[i] = 0

    Loss1, Loss0, LossMax, LossMaxLabel = Loss1.reshape(F1_pois.shape), Loss0.reshape(F1_pois.shape), LossMax.reshape(F1_pois.shape), LossMaxLabel.reshape(F1_pois.shape),
    return Loss1, Loss0, LossMax, LossMaxLabel, F1_pois, F2_pois