'''
@author: Ankit Swarnkar
Input : mXn
Kernelify : mXk
'''
import numpy as np
import random
from scipy.linalg import norm
from sklearn.cluster import KMeans
import classalgorithms as algs
import dataloader as dtl

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

class RadialBasis:
    def __init__(self,number_center=40):
        self.model=-1
        self.beta=1.4
        self.number_center=number_center
        self.center=[]
        pass
    def kmeancenter(self,xtest):
        km = KMeans(n_clusters = self.number_center)
        km.fit(xtest)
        self.center = km.cluster_centers_
        #print(self.center.shape)
    def radialfunc(self,center,xrow):
        val=self.beta*norm(center-xrow)**2
        return np.exp(-1*val)   
    def centerpoints(self,xdata):
        inx=random.sample(range(0, xdata.shape[0]), self.number_center)
        self.center = np.array([xdata[i,:] for i in inx])
    
    def calculatekmeanBeta(self,xdata):
        km = KMeans(n_clusters = 1)
        km.fit(self.center)
        mu=km.cluster_centers_
        sum=0
        for x1 in xdata:
            sum=sum+norm(x1-mu)
        leng=xdata.shape[0]
        sigma=sum/leng
        self.beta=1/(2*sigma**2)
        #print("Beta Taken : ",self.beta)
        
    def learn(self,xtest,ytest):
        #**Learn Center
        #self.centerpoints(xtest)
        self.kmeancenter(xtest)
        #print("pass1")
        #get Beta
        self.calculatekmeanBeta(xtest)
        #print("pass2")
        #Calculate distance/activation represenation
        D=np.zeros((xtest.shape[0],self.number_center),float)
        ci=0
        xi=0
        for c in self.center:
            xi=0
            for x in xtest:
                D[xi,ci]=self.radialfunc(c,x)
                xi=xi+1
            ci=ci+1
        #print(D)
        #print("pass3")
        self.model=algs.LogitReg({'regularizer': 'l2'})
        #print("pass4")
        #print(D.shape)
        self.model.learn(D,ytest)
        #print("pass5")

    def predict(self,xtest):
        D=np.zeros((xtest.shape[0],self.number_center),float)
        ci=0
        xi=0
        for c in self.center:
            xi=0
            for x in xtest:
                D[xi,ci]=self.radialfunc(c,x)
                xi=xi+1
            ci=ci+1
        pred=self.model.predict(D)
        return pred

        
    
if __name__ == '__main__':
    trainsize = 500
    testsize = 1000
    trainset, testset = dtl.load_susy(trainsize,testsize)
    learner_radial=RadialBasis(300)
    
    learner_radial.learn(trainset[0], trainset[1])
    # Test model
    predictions = learner_radial.predict(testset[0])
    error_radial = geterror(testset[1], predictions)
    print("Radial error:",error_radial)
    
    #Logistic Execution
    model = algs.LogitReg({'regularizer': 'l2'})
    model.learn(trainset[0], trainset[1])
    predictions = model.predict(testset[0])
    error = geterror(testset[1], predictions)
    print("Standard Logit error",error)
    
    
    
