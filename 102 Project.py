import pandas as pd
from pandas import read_excel, DataFrame
from pandas.plotting import scatter_matrix
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import pylab as py
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics


#Test Data uses 20% of 9568 = 1913 (First 1913 data used)
#Using Sheet 2 and initilizing name below

Test = 'Test_Data.xlsx'
Training = 'Training_Data.xlsx'

#Coose the data set below
file_name = Test
df = pd.read_excel(file_name)

#file_name_csv = 'Folds5x2.csv'
#df_ccv = pd.read_excel( file_name_csv )

print(df.mode())

"""
#----------------------#

#Mean of Data
data_mean = df.mean()
print(data_mean)


#Median of Data
data_median = df.median()
print(data_median)

#Mode of Data
data_mode = df.mode()
print(data_mode)

#Minimum of Data
data_min = df.min()
print(data_min)

#Maximum of Data
data_max = df.max()
print(data_max)


#Varaince of Data
data_var = df.var()
print(data_var)

#Standard Diviation of Data
data_std = df.std()
print(data_std)



#Basically Everything above in one code
data_des = df.describe()
print(data_des)

#----------------------#


#----------------------#

#Scatter Plot of AT vs. PE
def scatterPlot(name):
    plt.scatter(df[name], df['PE'])
    plt.xlabel(name)
    plt.ylabel('PE')
    plt.title('Scatter Plot')
    plt.show()
scatterPlot('AT')

#----------------------#

#----------------------#

#Creates a Scatter Plot comparing all the collums
scatter_plot = scatter_matrix(df.loc[:,'AT':'PE'],figsize=[10,7],diagonal= 'hist', alpha= 0.2)
for ax in scatter_plot.flatten():
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

#----------------------#

#----------------------#

#Correlation Coefficient for input and output
def cor_coe(x ):
    i = df[x]
    o = df['PE']
    corr,_ = pearsonr(i, o)
    print('Pearsons correlation for ' + x + ' : %.4f'  % corr)

#Place collumn name here
cor_coe('AP')
cor_coe('AT')
cor_coe('RH')
cor_coe('V')
#----------------------#


#----------------------#
def scaleTest(name):
    min = df[name].min()
    max = df[name].max()

    index = 0
    numRows = len(df.axes[0])
    inputNormalized = []

    while index < numRows:
        x = df[name].values[index]
        xp = ( (x - min )/ (max - min) )
        inputNormalized.append(xp)
        index = index + 1

    inputNormalized = np.array(inputNormalized).reshape(-1,1)
    return inputNormalized

#----------------------#


#----------------------#
#Used to find the MSE for 1 input 
def MSE1(num):
    names = ["AT", "V", "AP", "RH"]
    for index, n in enumerate(names):
        if index == num:
            name = n
    
    i = scaleTest(name)
    o = scaleTest ('PE')
    
    
    reg = linear_model.LinearRegression()
    reg.fit(i,o)

    index = 0
    numRows = len(df.axes[0])
    Total = 0

    while index < numRows:
        value = i[index]
        predictedValue = (reg.coef_ * value) + reg.intercept_
        actualValue = o[index]
        Total =  (actualValue - predictedValue)**2 + Total
        index = index + 1 
    
    MSE = Total / numRows
    return MSE


#----------------------#


#----------------------#
#Shows linear regression for 1 variables 
def lineRegression (num):
    #In my project these were the input names
    names = ["AT", "V", "AP", "RH"]
    for index, n in enumerate(names):
        if index == num:
            name = n

    i = scaleTest(name)
    o = scaleTest ('PE')
    
    #i = df.iloc[:, num ].values.reshape(-1, 1)
    #o = df.iloc[:, 4].values.reshape(-1, 1)
    
    reg = linear_model.LinearRegression()
    reg.fit(i,o)
    MS = MSE1(num)
    print("For {}".format(name))
    print("{0} * {1} + {2} = PE".format(reg.coef_, name, reg.intercept_ ))
    print("MSE is {}".format(MS))
    print("R^2 is equal to {0} \n".format(reg.score(i,o)))



#Place collum number here 
lineRegression(0)
lineRegression(1)
lineRegression(2)


#----------------------#

#----------------------#
#Used to find the MSE for two inputs
def MSE2(num1, num2):
    names = ["AT", "V", "AP", "RH"]
    for index, n in enumerate(names):
        if index == num1:
            name1 = n
        if index == num2:
            name2 = n
    
    i1 = scaleTest(name1)
    i2 = scaleTest(name2)
    o = scaleTest ('PE')
    i = np.column_stack( (i1,i2) )
    
    reg = linear_model.LinearRegression()
    reg.fit(i,o)

    index = 0
    numRows = len(df.axes[0])
    Total = 0

    while index < numRows:
        value1 = i1[index]
        value2 = i2[index]
        predictedValue = (reg.coef_[0][0] * value1) + (reg.coef_[0][1] * value2) + reg.intercept_
        actualValue = o[index]
        Total =  (actualValue - predictedValue)**2 + Total
        index = index + 1 
    
    MSE = Total / numRows
    return MSE
#----------------------#

#----------------------#

#Linear Regression with two variables
def lineRegression2(num1,num2):
    names = ["AT", "V", "AP", "RH"]
    for index, n in enumerate(names):
        if index == num1:
            name1 = n
        if index == num2:
            name2 = n

    i1 = scaleTest(name1)
    i2 = scaleTest(name2)
    o = scaleTest ('PE')


    i = np.column_stack( (i1,i2) )
    reg = linear_model.LinearRegression(normalize = True, fit_intercept=True)
    reg.fit(i,o)
    MS = MSE2(num1,num2)
    print("For {} and {}".format(name1, name2))
    print("{} * {} + {} * {} + {} = PE".format(reg.coef_[0][0], name1, 
    reg.coef_[0][1], name2,reg.intercept_ ))
    print("MSE is {}".format(MS))
    print("R^2 is equal to {0}\n".format(reg.score(i,o)))
      
#Place Column number here
lineRegression2(0,1)
lineRegression2(0,2)
lineRegression2(1,2)

#----------------------#

#----------------------#
#Used to find the MSE for three inputs
def MSE3(num1, num2, num3):
    names = ["AT", "V", "AP", "RH"]
    for index, n in enumerate(names):
        if index == num1:
            name1 = n
        if index == num2:
            name2 = n
        if index == num3:
            name3 = n
    
    i1 = scaleTest(name1)
    i2 = scaleTest(name2)
    i3 = scaleTest(name3)
    o = scaleTest ('PE')
    i = np.column_stack( (i1,i2,i3) )
    
    reg = linear_model.LinearRegression()
    reg.fit(i,o)

    index = 0
    numRows = len(df.axes[0])
    Total = 0

    while index < numRows:
        value1 = i1[index]
        value2 = i2[index]
        value3 = i3[index]
        predictedValue = (reg.coef_[0][0] * value1) + (reg.coef_[0][1] * value2) + (reg.coef_[0][2] * value3) + reg.intercept_
        actualValue = o[index]
        Total =  (actualValue - predictedValue)**2 + Total
        index = index + 1 
    
    MSE = Total / numRows
    return MSE
#----------------------#

#----------------------#
# Shows linear regression for 3 input 
def lineRegression3(num1,num2,num3):
    names = ["AT", "V", "AP", "RH"]
    for index, n in enumerate(names):
        if index == num1:
            name1 = n
        if index == num2:
            name2 = n
        if index == num3:
            name3 = n

    i1 = scaleTest(name1)
    i2 = scaleTest(name2)
    i3 = scaleTest(name3)

    i = np.column_stack( (i1,i2,i3) )
    o = scaleTest ('PE')   
    
    reg = linear_model.LinearRegression(normalize = True, fit_intercept=True)
    reg.fit(i,o)
    MS = MSE3(num1,num2,num3)
    print("For {}, {}, and {}".format(name1, name2, name3))
    print("{} * {} + {} * {} + {} * {} + {} = PE".format(reg.coef_[0][0], name1, 
    reg.coef_[0][1], name2,reg.coef_[0][2], name3, reg.intercept_ ))
    print("MSE is {}".format(MS))
    print("R^2 is equal to {0}\n".format(reg.score(i,o)))

#Put Column numbers here 
lineRegression3(0,1,2)
#----------------------#
"""