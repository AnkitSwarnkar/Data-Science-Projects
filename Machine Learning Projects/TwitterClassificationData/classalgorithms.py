####
#Ankit Swarnkar
#References:
#Naive bayes reference:
#Naive Bayes Tutorial,http://machinelearningmastery.com/category/machine-learning-algorithms/,2016
#Logistic Regression:
#Logistic Regression, https://bryantravissmith.com/, 2016
#Couresa Machine Learning Specialization
#Neural Network:
#http://machinelearningmastery.com/
#[3]https://www.quora.com/Why-do-initial-weights-and-bias-in-a-neural-network-affect-so-heavily-the-speed-of-convergence
####
from __future__ import division  # floating point division
import numpy as np
import math
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.meanWeight={}
        self.reset(parameters)
            
    def reset(self, parameters):
        self.resetparams(parameters)
        # TODO: set up required variables for learning
        
    # TODO: implement learn and predict functions
    def learn(self,xin,yin):
        no_of_attribute=xin.shape[1]
        #Learning the classes
        classLearned={}
        #Learn the prob_c
        self.probc={}
        i=0
        while i<len(yin):
            ele=yin[i]
            if not ele in classLearned:
                classLearned[ele]=list()
                classLearned[ele].append(i)
            else:
                classLearned[ele].append(i)
            i+=1
        #print classLearned
        for cls in classLearned:
            self.probc[cls]=len(classLearned[cls])/yin.shape[0]
        #Calculate Mean
        meanSd={}
        for cls in classLearned:
           # print cls
            self.meanWeight[cls] = list()
            for col in np.transpose(xin[classLearned[cls]]):
                mean_val=np.mean(col)
                sd_val=np.std(col)
                self.meanWeight[cls].append((mean_val,sd_val))

    def predict(self, Xtest):
        Yreturn=np.zeros(Xtest.shape[0])
        probability={}
        for rowc in range(Xtest.shape[0]):
            for classVal in self.meanWeight:
                probability[classVal]=1
                for col in range(len(self.meanWeight[classVal])):
                    meanVal,sdVal=self.meanWeight[classVal][col]
                    #print Xtest[rowc][col]
                    probability[classVal] *= self.Gaussian(meanVal,sdVal,Xtest[rowc][col])*self.probc[classVal]
            Yreturn[rowc]= max(probability, key=probability.get)

            #print Yreturn[rowc]
        return Yreturn

    def Gaussian(self,meanVal, sdVal, x):
        if sdVal == 0:
            return 1
        else:
            exponent = math.exp(-(math.pow(x - meanVal, 2) / ( 2 * math.pow(sdVal, 2))))
            return ( 1 / (math.sqrt(2 * math.pi) * sdVal)) * exponent

class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.01, 'regularizer': 'None'}
        self.l1=False
        self.l2=False
        self.l3=False
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        #print(self.params)
        if self.params['regularizer'] == 'l1':
            self.regularizer = (utils.l1, utils.dl1)
            self.lamba=self.params['regwgt']
            self.l1=True
            self.l2=False
        elif self.params['regularizer'] == 'l2':
            self.regularizer = (utils.l2, utils.dl2)
            self.lamba = self.params['regwgt']
            #print(self.lamba)
            self.l2 = True
            self.l1 = False
        elif self.params['regularizer'] == 'l3':
            self.lamba1 = self.params['regwgt']
            self.lamba2= self.params['regwgt']
            self.l3 = True
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions
    def prob_sigmoid(self,x):
        retprob=1 / (1 + np.exp(-1 * np.dot(x,self.weights)))
        return retprob

    def indicator(self,y):
        #Assuming label as 1
        if y==1:
            return 1
        else:
            return 0
    def llhood(self,x,y):
        prob=self.prob_sigmoid(x)
        #ll= self.weight*np.log(prob+1e-24) + (1-self.labels)*np.log(1-prob+1e-24)
        ll=y*np.log(prob)+( 1 - y )*np.log(1-prob)
        ret_ll=-1*ll.sum()
        if(self.l1 is True):
            ret_ll+=self.lamba*np.abs(self.weights).sum()
        if(self.l2 is True):
            ret_ll+=self.lamba*np.power(self.weights,2).sum()/2
        if (self.l3 is True):
            alpha1 = self.lamba2 / (self.lamba1 + self.lamba2)
            ret_ll += (alpha1 * np.abs(self.weights).sum())+((1-alpha1)* np.power(self.weights, 2).sum() / 2)
        return ret_ll


    def learn(self, Xtrain, ytrain):
        ytrain = ytrain.reshape(ytrain.size,1)#for the brodcasting array mutiplication
        self.weights=np.zeros((Xtrain.shape[1], 1))
        #print (self.weights.shape)
        difference_ll=1
        tolerance=1*np.exp(-6)
        alpha=0.001
        cur_ll=self.llhood(Xtrain,ytrain)
        iteration=1
        while(np.abs(difference_ll)>tolerance):
            intern=Xtrain * (ytrain-self.prob_sigmoid(Xtrain))
            grad=intern.sum(axis=0).reshape(self.weights.shape)
            if (self.l1 is True):
                grad += self.lamba * np.sign(self.weights)
            if (self.l2 is True):
                grad += self.lamba * self.weights
            if (self.l3 is True):
                grad += (1/np.sqrt(1 + self.lamba2)) * np.sign(self.weights)
            self.weights=self.weights+alpha*grad
            #print (self.weights.shape)
            new_ll=self.llhood(Xtrain,ytrain)
            difference_ll=new_ll-cur_ll
            #print difference_ll

            iteration+=1
            cur_ll=new_ll
        #print iteration
    def predict(self,Xtest):
        #print(self.weights)
        ytest=np.zeros(Xtest.shape[0])
        prob = self.prob_sigmoid(Xtest)
        i=0
        for i in range(len(prob)):

            if prob[i]>=0.5:
                ytest[i]=1
            else:
                ytest[i]=0
        #print ytest
        return ytest


class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.001,
                        'epochs': 200}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
        
    # TODO: implement learn and predict functions
    def predict(self, Xtest):
        W1, W2 = self.model['W1'], self.model['W2']
        yhat = np.zeros((Xtest.shape[0]))
        z1 = Xtest.dot(W2)
        a1 = self.sigmoid(z1)
        z2 = a1.dot(W1)
        a2 = self.sigmoid(z2)
        i = 0
        for i in range(len(a2)):

            if a2[i] >= 0.5:
                yhat[i] = 1
            else:
                yhat[i] = 0
        return yhat

    def sigmoid(self,xvec):

        return 1.0 / (1.0 + np.exp(np.negative(xvec)))

    def learn(self, Xtrain, ytrain):
        #np.random.seed(0)
        outpd=1
        hidden_node=self.params['nh']
        alpha=self.params['stepsize']
        ytrain=ytrain.reshape(ytrain.size,1)#Reshaping for broadcast
        yhat = np.zeros((Xtrain.shape[0],1))
        W2 = np.random.randint(1,size=(Xtrain.shape[1], hidden_node))
        from random import randrange
        W1 = np.array([randrange(-140,140) for i in range(0,hidden_node)])#[3]
        cur_ll = 1
        for iteration in range(self.params['epochs']):
            state = np.random.get_state()
            np.random.shuffle(Xtrain)
            np.random.set_state(state)
            np.random.shuffle(ytrain)
            for sample in range(0, Xtrain.shape[0]):
                #Forward Propogation
                #w2->d X hidden
                z2=np.dot(Xtrain[sample], W2)
                h2 = self.sigmoid(z2)
                #w1->hidden X 1
                z1=np.dot(h2, W1)
                h1 = self.sigmoid(z1)
                #Backward Propogation
                delta1 = h1 - ytrain[sample]
                dW1 = delta1.T * h2
                delta2=np.array([(W1 * delta1)[i] * h2[i] * (1 - h2[i]) for i in range(len(W1))])
                dW2 = np.array([Xtrain[sample] * i for i in delta2]).T
                #Apply Gradient
                W2 = W2 - alpha * dW2
                W1 = W1 - alpha * dW1

            self.model = { 'W1': W1, 'W2': W2,}

            #self.crossEntropyloss(Xtrain, ytrain)
            #new_ll = self.crossEntropyloss(Xtrain, ytrain)
            #difference_ll = new_ll - cur_ll
            #print "Loss after iteration %i: %f" % (i, difference_ll)

    def crossEntropyloss(self,x,y):
        W1,W2 = self.model['W1'], self.model['W2'],
        yhat = np.zeros((x.shape[0], 1))
        z1 = x.dot(W2)
        h1=self.sigmoid(z1)

        z2=h1.dot(W1)
        h2=self.sigmoid(z2)
        i = 0
        for i in range(len(h2)):

            if h2[i] >= 0.5:
                yhat[i] = 1
            else:
                yhat[i] = 0
        #print(yhat)
        ll = y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
        return -1*ll.sum()

    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)


class LogitRegAlternative(Classifier):


    def __init__( self, parameters={} ):
        self.reset(parameters)

    def prob(self,x):
        retprob = 0.5 * (1+ (np.dot(x, self.weights)/np.sqrt(1 + np.power(np.dot(x, self.weights),2))))
        return retprob

    def llhood(self,x,y):
        prob=self.prob(x)
        #ll= self.weight*np.log(prob+1e-24) + (1-self.labels)*np.log(1-prob+1e-24)
        ll=y*np.log(prob)+( 1 - y )*np.log(1-prob)
        ret_ll=-1*ll.sum()
        return ret_ll

    def learn(self, Xtrain, ytrain):
        ytrain = ytrain.reshape(ytrain.size,1)#for the brodcasting array mutiplication
        self.weights=np.zeros((Xtrain.shape[1], 1))
        #print (self.weights.shape)
        difference_ll=1
        tolerance=1*np.exp(-6)
        alpha=0.001
        cur_ll=self.llhood(Xtrain,ytrain)
        iteration=1
        while(np.abs(difference_ll)>tolerance):
            intern=Xtrain * (ytrain-self.prob(Xtrain))
            #Extra Part
            intern=intern*(0.5*(1/np.sqrt(1+np.power(np.dot(Xtrain,self.weights),2))))
            grad=intern.sum(axis=0).reshape(self.weights.shape)
            self.weights=self.weights+alpha*grad
            #print (self.weights.shape)
            new_ll=self.llhood(Xtrain,ytrain)
            difference_ll=new_ll-cur_ll
            #
            # print difference_ll

            iteration+=1
            cur_ll=new_ll
        #print iteration

    def predict(self,Xtest):
        ytest=np.zeros(Xtest.shape[0])
        prob = self.prob(Xtest)
        i=0
        for i in range(len(prob)):

            if prob[i]>=0.5:
                ytest[i]=1
            else:
                ytest[i]=0
        #print ytest
        return ytest

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

           
    
