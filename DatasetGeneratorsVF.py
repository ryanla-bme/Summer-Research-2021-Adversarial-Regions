"""This file contains the functions used to generate the toy datasets and 
induce selection biases.
"""

import numpy as np

def GenerateMultiVarGaussians(means,covs,num_samples, RANDOM_STATE):
    """
    Generates the multivariate gaussian datasets used for experiments

    Inputs: 
        means <- mean vectors of each gaussian dataset (2D numpy array)
        covs <- covariance matrices of each gaussian dataset (3D numpy array)
        num_samples <- number of samples to generate for each gaussian dataset
            (1D numpy array)
        RANDOM_STATE <- for reproducibility (int) 

    Outputs:
        datasets <- samples for each dataset (3D numpy array)
    """
    
    sampler = np.random.default_rng(seed = RANDOM_STATE)
    return np.array([sampler.multivariate_normal(means[i],covs[i],num_samples[i]) for i in range(np.shape(means)[0])])


def KathBiasPlane2D(ds, alpha, rot_centre, prob, RANDOM_STATE, RetInBiased = False):
    """
    Generates the same bias as in -Your Best Guess When You Know Nothing:
    Identification and Mitigation of Selection Bias- Katharina Dost et al (2020).
    Uses a linear plane and the area 'underneath' the plane as the area to bias. 
    Datapoints in the biased area is kept a probability of prob.
    
    Inputs: 
        ds <- dataset to generate bias in (2D numpy array)
        alpha <- plane rotation (radians)
        rot_centre <- centre of rotation (1D numpy array)
        prob <- probability a datapoint is kept in the biased area (float)
        RANDOM_STATE <- for reproducibility (int) 
        RetInBiased <- if True returns datapoints in ds that are in biased area 

    Ouputs:
        dsBiased <- dataset with the generated bias (2D numpy array)
        plane <- dictionary containing keys that parameterise the plane (dict)
            coeff <- coefficient of linear plane (float)
            intercept <- 'y' (second feature) intercept when 'x' (first feature)
                          equal zero. 
        dsInBiased(OPTIONAL) <- datapoints in biased area (2D numpy array)
    """
    alphaInv = -1*alpha

    #### Rotataing Matrices ####
    R = np.array([[np.cos(alpha), -np.sin(alpha)],\
        [np.sin(alpha),np.cos(alpha)]])

    RInv = np.array([[np.cos(alphaInv), -np.sin(alphaInv)],\
        [np.sin(alphaInv),np.cos(alphaInv)]])

    #### Finding Characteristics of Plane ####    
    # find coefficient of plane
    v = np.dot(np.array([1,0]),R) 
    coeff = v[1]/v[0]

    # find 'y' intercept 
    intercept = -rot_centre[0] * coeff + rot_centre[1]
    plane = {"coeff": coeff, "intercept": intercept}

    #### Finding datapoints that are in the bias area ####
    # Uses inverse rotation. Derivation: p' = pR+T -> (p'-T)R^-1 = p

    rng = np.random.default_rng(seed = RANDOM_STATE)

    rotPoints = np.array([np.dot((i-rot_centre),RInv) for i in ds])
    inBias = rotPoints[:,1] < 0

    dsBiased = np.array([ds[i] for i in range(np.shape(ds)[0]) if (inBias[i] == False) or (rng.uniform() <= prob)])

    if RetInBiased == True:
        dsInBiased = np.array([ds[i] for i in range(np.shape(ds)[0]) if inBias[i] == True])
        return dsBiased, plane, dsInBiased
    
    return dsBiased, plane


