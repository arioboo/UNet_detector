#import tools.analyzer as analyzer

###---LINEAR_REGRESSIONS---###
#-<sp_linregress>-#
from scipy import stats

def sp_linregress(x,y):   
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print(' slope= %.2f \n intercept= %.2f'%(slope,intercept))
    return (slope,intercept)

'''
import numpy as np
x = np.random.random(10)
y = np.random.random(10)
sp_linregress(x,y)
'''

#-<sklearn_linregress>-#
from sklearn.linear_model import LinearRegression

def sklearn_linregress(x,y):
    reg = LinearRegression().fit(x,y)
    #reg.coef_
    #reg.intercept_ 
    return reg

'''
import numpy as np
x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3             # y = 1 * x_0 + 2 * x_1 + 3
sklearn_linregress(x,y)
'''


#-<np_linregress>-#

import numpy as np



###---POLYFIT---###
#-<np_polyfit>-#   

def np_polyfit(x,y,order=1):
    fit = np.polyfit(x,y,1)             #y=mx+b
    fit_fn = np.poly1d(fit) 
    # fit_fn is now a function which takes in x and returns an estimate for y
    
    slope=fit[0] ; intercept=fit[1]
    print(' slope= %.2f \n intercept= %.2f'%(slope,intercept))
    return fit,fit_fn

'''
x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
z,z_theo = np.polyfit(x, y, 1)
'''

###---CURVEFIT---###
#-<sp_curvefit>-#   From a defined "func", fit the data x , returning (a,b,c parameters)
def func(x, a, b, c):
     return a * np.exp(-b * x) + c    #a,b,c parameters

    
from scipy import optimize
def sp_curvefit(xdata,ydata):
    popt, pcov = optimize.curve_fit(func,xdata,ydata)
    #
    return popt,pcov
    
'''
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
popt,pcov = sp_curvefit(xdata,ydata)
'''  


###-------------------------------------END---------------------------------------###
