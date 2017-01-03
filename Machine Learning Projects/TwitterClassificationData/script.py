"""
@author: ankitswarnkar
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as pt
import numpy as np
import copy
import classalgorithms as algs
import csvloader as cv
import RadialBasis as rd
model_evalution={}
model_result=[]
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))
#For internal CV
def cv_error(model,xdata,ydata,k=10):
    subset_size=int(xdata.shape[0]/k)
    error_block=list()

    for i in range(0,k):
        #print(i)
        xtest = xdata[i*subset_size:][:subset_size]
        ytest = ydata[i*subset_size:][:subset_size]

        xtrain1 = xdata[:i*subset_size]
        ytrain1 = ydata[:i*subset_size]
        xtrain2= xdata[(i+1)*subset_size:]
        ytrain2= ydata[(i+1)*subset_size:]
        if xtrain1.shape[0]==0:
            xtrain=xtrain2
            ytrain=ytrain2
        elif xtrain2.shape[0]==0:
            xtrain=xtrain1
            ytrain=ytrain1
        else:
            xtrain=np.concatenate((xtrain1,xtrain2),axis=0)
            ytrain= np.concatenate((ytrain1,ytrain2),axis=0)
        #print(xtrain)
        model.learn(xtrain,ytrain)
        ypredict=model.predict(xtest)
        err=geterror(ytest,ypredict)
        #print(err)
        error_block.append(err)
    average=sum(error_block)/error_block.__len__()
    return average

#Statistically evalution function    
def model_er_evalution(model,xdata,ydata,k=10,name="no_name"):
    subset_size=int(xdata.shape[0]/k)
    error_block=list()
    model_evalution[name]=[]

    for i in range(0,k):
        #print(i)
        xtest = xdata[i*subset_size:][:subset_size]
        ytest = ydata[i*subset_size:][:subset_size]

        xtrain1 = xdata[:i*subset_size]
        ytrain1 = ydata[:i*subset_size]
        xtrain2= xdata[(i+1)*subset_size:]
        ytrain2= ydata[(i+1)*subset_size:]
        if xtrain1.shape[0]==0:
            xtrain=xtrain2
            ytrain=ytrain2
        elif xtrain2.shape[0]==0:
            xtrain=xtrain1
            ytrain=ytrain1
        else:
            xtrain=np.concatenate((xtrain1,xtrain2),axis=0)
            ytrain= np.concatenate((ytrain1,ytrain2),axis=0)
        #print(xtrain)
        model.learn(xtrain,ytrain)
        ypredict=model.predict(xtest)
        err=geterror(ytest,ypredict)
        #print(err)
        error_block.append(err)
    model_evalution[name].append(np.mean(error_block))
    model_evalution[name].append(np.std(error_block))
    model_result.append(error_block)

dataread=cv.load_data("../DataSet/gender-classifier-DFE-791531.csv",'latin1')

#Normalizing text Data
dataread=cv.normalize_data(dataread)
#Feature Selection Trial
x= dataread[['tweet_count','fav_number','sidebar_color','retweet_count','link_col_dec']]
#check= dataread['']
#finalx=pd.concat([x,xvec],axis=1)
encoder=LabelEncoder()
y=encoder.fit_transform(dataread['gender'])

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
xtrain=xtrain.as_matrix()
xtest=xtest.as_matrix()
ytrain1=copy.deepcopy(ytrain)
ytrain=ytrain.reshape((ytrain.shape[0],1))
#print((xtrain.shape[0]))
#Module Selection Trial

#Model1 : Naive Bayes Model

model_Naive=algs.NaiveBayes()
model_er_evalution(model_Naive,xtrain,ytrain1,k=10,name="Gaussian")
model_Naive.learn(xtrain,ytrain1)
predictions =model_Naive.predict(xtest)
naive_error=geterror(ytest,predictions)
ploty={}
#Model2 : Logistic Regression
#Selecting Regularizer lambda value

for parms in [0.01,0.005,0.04,0.03,0.05,0.06]:
    params = {'regwgt': parms}
    log_class=algs.LogitReg({'regularizer': 'l2'})
    log_class.reset(params)
    train_error_logit=cv_error(log_class, xtrain , ytrain ,10)
    ploty[parms]=train_error_logit

#Using the above best selected one
best_par=min(ploty,key=ploty.get)
params = {'regwgt': best_par}
model_log=algs.LogitReg({'regularizer': 'l2'})
model_log.reset(params)
model_er_evalution(model_log,xtrain,ytrain,k=10,name="Logistic")

#Final Prediction
model_log.learn(xtrain,ytrain)
predictions=model_log.predict(xtest)
log_error=geterror(predictions,ytest)

#Model3 : Radial Basis
plotnrd={}
for node_count in [2,3,4]:
    modelrdf=rd.RadialBasis(node_count)
    plotnrd[node_count]=cv_error(modelrdf,xtrain,ytrain,10)
best_par_rf=min(plotnrd,key=plotnrd.get)
modelrdf=rd.RadialBasis(best_par_rf)
model_er_evalution(modelrdf,xtrain,ytrain,k=10,name="Radial")
modelrdf.learn(xtest,ytest)
ypr=modelrdf.predict(xtest)
#accrry=cv_error(model, xtrain , ytrain ,10)
rdf_error=geterror(ytest,ypr)


print("****************REPORT************************")
'''Box Plot'''
boxfig = pt.figure()
boxfig.suptitle('Classification Comparison')
ax = boxfig.add_subplot(111)
pt.boxplot(model_result)
ax.set_xticklabels(["Naive Bayes","Logistic L2","Radial basis"])
pt.show()

'''Numerical Report'''

print("Naive Bayes Model error", naive_error)
print("Logistic Regression L2 error",log_error)
print("Best parameter for Logistic Regression Regularizer:",best_par)
print("Radial Network error",rdf_error)
print("Best parameter for Radial Basis Function:",best_par_rf)
