# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:53:24 2020
@author: Tahar FERHATI
Paper: SVI Model Free Wings, 12/04/2020
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from scipy.optimize import minimize  
from scipy import integrate

start=datetime.now()

#============================================================================================
# Input data Model - SVI Parameters : change the SVI parameters in order to get an arbitrage  
# a > 0, 0 < b < 1, -1 < rho < 1, m real Number , sigma > 0 
#============================================================================================

# Arbitrage example ===========================================================
# Synthetic Input Data ( total variance) : change the SVI parameters to get new input data model 
#a_input =  0.08
#b_input =  0.68
#rho_input = -0.99
#m_input =  0.05
#sigma_input = 0.05
 
a_input = 0.02
b_input =   0.6
rho_input = -0.9
m_input =0.2
sigma_input = 0.01

# ============  SVI Functions ==================================================
# Attention à l'ordre des paramètres

def SVI(k, a, b , rho, m, sig):
    total_variance = a + b*(rho*(k - m) + np.sqrt( (k - m)*(k - m) + sig*sig))
    return total_variance

def SVI_two_arguments(theta, k):
    a, b , rho, m, sig = theta
    return SVI(k, a, b , rho, m, sig)

#================= g(k) function for the convexity test ========================     
def test_convexity(k, a, b, rho, m, sig):
    square_rt = np.sqrt( (k - m)*(k - m) + sig*sig)
    w = SVI(k, a, b, rho, m, sig)
    first_der_w = b*rho + b*(k - m) / square_rt
    second_der_w = b*sig*sig / (square_rt**3)
    g = second_der_w / 2 - first_der_w**2 * (1/w + 1./4) / 4
    g = g + ( 1 - k*first_der_w / (2*w) )**2
    return g

# Right wing constraints ======================================================    
def constraint1(theta, log_mon):
    a, b, rho, m, sig = theta
    return ((4-a+b*m*(rho+1))*(a-b*m*(rho+1)))-(b*b*(rho+1)*(rho+1))

def constraint2(theta, log_mon):
    a, b, rho, m, sig = theta
    return  4 -(b*b*(rho+1)*(rho+1))

# Left wing constraints ======================================================    
def constraint3(theta, log_mon):
    a, b, rho, m, sig = theta
    return (((4-a+b*m*(rho-1))*(a-b*m*(rho-1)))-(b*b*(rho-1)*(rho-1)))

def constraint4(theta, log_mon):
    a, b, rho, m, sig = theta
    return 4-(b*b*(rho-1)*(rho-1))

#================= Objective function to optimize ==============================  
    
def fct_least_squares(theta, log_mon, tot_implied_variance):    
    return np.linalg.norm((SVI_two_arguments(theta, log_mon) - tot_implied_variance), 2)

#=============== k = Log Forward Moneyness =====================================
mkt_log_mon = np.linspace(-1,1, num=40)

#====== syntechitc total implied variance (input date) ========================
mkt_tot_variance = SVI(mkt_log_mon, a_input, b_input , rho_input, m_input, sigma_input)

#===========  SVI’s Parameters Boundaries ======================================
a_low = 0.000001 
a_high = np.max(mkt_tot_variance)
b_low = 0.001
b_high = 1
rho_low = -0.999999
rho_high = 0.999999
m_low = 2*np.min(mkt_log_mon )
m_high = 2*np.max(mkt_log_mon)
sigma_low = 0.001
sigma_high = 2

#===========  SVI’s Parameters Initial Guess =====================================
a_init = np.min(mkt_tot_variance)/2
b_init = 0.1
rho_init = -0.5
m_init = 0.1 
sig_init =  0.1 

############################################################################
## The bounds we impose on the SVI parameters : SVI( a, b , rho, m, sig)
############################################################################
        
SVI_param_bounds = ((a_low,a_high),(b_low, b_high),(rho_low,rho_high),(m_low,m_high),(sigma_low,sigma_high))  
theta_init = np.array([a_init, b_init, rho_init, m_init, sig_init])
        
#Constraint Function : g(k) > 0 
cons1 = {'type': 'ineq', 'fun': lambda x:constraint1(x , mkt_log_mon )}
cons2 = {'type': 'ineq', 'fun': lambda x:constraint2(x , mkt_log_mon ) }
cons3= {'type': 'ineq', 'fun': lambda x:constraint3(x , mkt_log_mon ) }
cons4 = {'type': 'ineq', 'fun': lambda x:constraint4(x , mkt_log_mon )}
        
#############################################################################
# Optimisation Function : min Loss function = ( SVI_model - Variance_Market )
# We can use these bunded opt function : trust-constr , SLSQP,  COBYLA
#############################################################################
       
result = minimize (lambda x:fct_least_squares(x,mkt_log_mon,mkt_tot_variance), theta_init,method='SLSQP',
                  bounds=SVI_param_bounds, constraints=[cons1,cons2,cons3,cons4], options={'ftol': 1e-9, 'disp': True})
        
# Optimal SVI vector : a*, b*, rho*, m*, sigma* 
a_star, b_star, rho_star, m_star, sig_star = result.x
    
#########################################
## Plot of SVI fit from least squares
#########################################
total_variances_fit = SVI(mkt_log_mon, a_star, b_star, rho_star, m_star, sig_star)
    
plt.figure(figsize=(9,6))
    
plt.plot(mkt_log_mon, mkt_tot_variance,
             color='blue', linewidth=0., marker="o",linestyle="--", markersize=3, label="SVI with Arbitrage ")
    
plt.plot(mkt_log_mon, total_variances_fit, color='red', linewidth=2, label= "SVI Arbitrage Free" ) 
    
plt.axvline(0., linestyle="--", linewidth=0.5, color="k")
    
plt.xlabel(r"Log moneyness $k$", fontsize=18)
plt.ylabel("Implied Total Variance", fontsize=18)
plt.legend(fontsize=15)
plt.title(r"SVI Model Calibration", fontsize=15)
#plt.title(r"$T$=%1.2f,"%(maturity), fontsize=15)
#plt.legend(loc=9, fontsize=12, bbox_to_anchor=(1.6, 1.0), ncol=1)
     
#==========================================================================
## We check a posteriori the positivity and Slope conditions
# min SVI =  a + b*sigma*np.sqrt(1 - rho*rho) positif !
#(1) Right Slope b(rho+1) < 2
#(2) Right Slope b(rho-1) > -2
#==========================================================================
    
def test_positivity(a, b, rho, sigma):
    assert( rho < 1)
    assert(rho > -1)
    assert(b > 0)
    ## On vérifie la condition de positivité du minimum
    minimum = a + b*sigma*np.sqrt(1 - rho*rho)
    assert(minimum > 0)
    print("\n Positivity test OK and SVI min is :",minimum)
    return 0
    
def test_slope(b, rho):
    right_slope = b * (rho+1)
    left_slope = b * (rho -1)
    print("\n Right slope is b*(rho+1) < 2 and it's value is: %1.7f" %right_slope)
    print("\n Left slope is b*(rho-1) > -2 and it's value is: %1.7f" %left_slope)
    assert(right_slope < 2)
    assert(left_slope > -2)
    pass
    
# Positivity test min (SVI) > 0 
test_positivity(a_star, b_star, rho_star,sig_star)
# Test Slope b(1+rhou) = right_slope < 2
test_slope(b_star, rho_star)
    
##################################################
## SVI Butterfly Arbitrage 
## Test positivity of the function g(k) 
##################################################
    
def visual_test_convexity(a, b, rho, k_0, sig):
    
    test_fct_g = test_convexity(mkt_log_mon, a, b, rho, k_0, sig)
    test_fct_arbitrage = test_convexity(mkt_log_mon, a_input, b_input, rho_input, m_input, sigma_input)
    
    plt.figure(figsize=(9,6))
    plt.plot(mkt_log_mon, test_fct_arbitrage, color="blue", 
             linewidth=1, marker="o",linestyle="--", 
             markersize=2.5,label="g(k) with Arbitrage")
    plt.plot(mkt_log_mon, test_fct_g, color="red", 
             linewidth=1., marker="*", markersize=2.5, 
             label="g(k) Arbitrage Free")
        
    plt.axhline(0., linestyle="--", color="k")
    
    plt.xlabel("Log moneyness $k$", fontsize=14)
    plt.ylabel("Function g(k)", fontsize=14)
    plt.title(r"Test of butterfly arbitrage g(k) function for SVI parameterisation", fontsize=15)
    
    plt.legend(fontsize=14)
        
    assert( (test_fct_g > 0).all )
    pass
    
visual_test_convexity(a_star, b_star, rho_star, m_star, sig_star)
    
##################################################
## Test Positivity of the density function 
##################################################
    
def dminus(x, a,b,rho,m,sigma):
    vsqrt = np.sqrt(SVI(x, a,b,rho,m,sigma))
    return -x/vsqrt - 0.5*vsqrt  
    
def densitySVI(x, a,b,rho,m,sigma):
        
    dm = dminus(x, a,b,rho,m,sigma)
    return test_convexity(x, a,b,rho,m,sigma)*np.exp(-0.5*dm*dm)/np.sqrt(2.*np.pi*SVI(x, a,b,rho,m,sigma))
    
def visual_density (a,b,rho,m,sigma):
       
    test_fct_f = densitySVI(mkt_log_mon, a,b,rho,m,sigma)
    test_fct_arbitrage = densitySVI(mkt_log_mon, a_input, b_input, rho_input, m_input, sigma_input)
    
    plt.figure(figsize=(9,6))
    plt.plot(mkt_log_mon, test_fct_arbitrage, color="blue", linewidth=1,linestyle="--", marker="o", markersize=2.5,label="Density with Arbitrage")
    plt.plot(mkt_log_mon, test_fct_f, color="red", linewidth=1., marker="*", markersize=2.5, label="Density Arbitrage Free")
        
    plt.axhline(0., linestyle="--", color="k")
    
    plt.xlabel("Log moneyness $k$", fontsize=14)
    plt.ylabel("Density function", fontsize=14)
    plt.title(r"Density function f(k) for SVI parameterisation", fontsize=15)
    
    plt.legend(fontsize=14)
   # print(test_fct_f)
    assert( (test_fct_f > 0).all )
    pass

visual_density (a_star, b_star, rho_star, m_star, sig_star) 
    
#================================================
# Check if the density integrate to one 
#================================================
    
print( "Check that the density integrates to unity: ", 
           integrate.quad(lambda x: densitySVI(x, a_star, b_star, rho_star, m_star, sig_star ), -np.inf, np.inf, epsrel=1.49e-15, limit=200)[0])
   
#=====================================================
# Display the SVI Parameters : Model & Market
#=====================================================
    
print( "=============== SVI Parameters Calibration (arbitrage Free) ================ ")
print( " a_star = ", a_star)
print( " b_star = ",b_star)
print( " rho_star = ", rho_star)
print( " m_star = ", m_star)
print( " sigma_star = ", sig_star,"\n" )
# ==============================================================================
print( "=============== SVI Parameters Market Data ================= ")
print( " a_star = ", a_input)
print( " b_star = ",b_input)
print( " rho_star = ", rho_input)
print( " m_star = ", m_input)
print( " sigma_star = ", sigma_input,"\n" )
print( "====================================================== ")
    
print(datetime.now()-start)
plt.show() 
 
