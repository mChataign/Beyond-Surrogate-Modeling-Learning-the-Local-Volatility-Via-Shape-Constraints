__author__ = 'dixon'
from scipy.stats import norm
import scipy.stats as st
import numpy as np
import pandas as pd
from math import *

def bsformula( callput, S0, K, r, T, sigma, q=0.):
    """
    :param callput: Indicates if the option is a Call or Put option
    :param S0: Stock price
    :param K: Strike price
    :param r: Risk-free rate
    :param T: Time to expiration
    :param sigma: Volatility
    :param q: Dividend rate
    :return: Value of the option, its Delta, its Vega
    """
    d1=(log(float(S0)/K)+((r-q)+sigma*sigma/2.)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    if callput==1:
        optionValue=S0*exp(-q*T)*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
        delta=norm.cdf(d1)
    elif callput==-1:
        optionValue=K*exp(-r*T)*norm.cdf(-d2)-S0*exp(-q*T)*norm.cdf(-d1)
        delta=-norm.cdf(-d1)

    vega=S0*sqrt(T)*norm.pdf(d1)
    return optionValue,delta,vega

#Density derivative
def dpdf(x):
    v = 1
    return -x*np.exp(-x**2/(2.0*v**2))/(v**3*np.sqrt(2.0*np.pi))      

def generalizedGreeks(cp, s, k, t, v, rfInt, divInt):
        """ Price an option using the Black-Scholes model.
        cp: +1/-1 for call/put
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rfInt: deterministic risk-free rate integrated between 0 and t
        divInt: deterministic dividend integrated between 0 and t
        """

        d1 = (np.log(s/k)+(rfInt-divInt+0.5*v*v*t))/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)
        
        Nd1 = st.norm.cdf(cp*d1)
        Nd2 = st.norm.cdf(cp*d2)

        discountFactor = np.exp(-rfInt)
        forwardFactor = np.exp(-divInt)
        avgDiv = divInt/t
        avgRf = rfInt/t

        optprice = (cp*s*forwardFactor*Nd1) - (cp*k*discountFactor*Nd2)

        delta = cp*Nd1
        vega  = s*np.sqrt(t)*st.norm.pdf(d1)
        delta_k = -s*forwardFactor*Nd1/(v*np.sqrt(t)*k) - cp*discountFactor*Nd2 + k*discountFactor*Nd2/(v*np.sqrt(t)*k)
        
        gamma_k = s*forwardFactor/((v*np.sqrt(t)*k)**2)*(Nd1*v*np.sqrt(t) + cp*dpdf(cp*d1)) - k*discountFactor/((v*np.sqrt(t)*k)**2)*(Nd2*v*np.sqrt(t) + cp*dpdf(cp*d2)) +  2.0*discountFactor*Nd2/(v*np.sqrt(t)*k)  

        dd1_dt = (avgRf-avgDiv+0.5*v*v)/(v*np.sqrt(t)) - 0.5*(np.log(s/k)+(rfInt-divInt+0.5*v*v*t))/(v*v*t**(3/2))
        dd2_dt = dd1_dt - 0.5*v/np.sqrt(t)
        delta_T = avgRf*cp*k*discountFactor*Nd2 - avgDiv*cp*s*forwardFactor*Nd1 + s*forwardFactor*Nd1*dd1_dt- k*discountFactor*Nd2*dd2_dt
        
        return optprice, delta, vega, delta_k, gamma_k, delta_T
        


#Change of variable for constant discount and dividend short rate 
def changeOfVariable_BS(s, t, q, r):
  
  factorPrice = np.exp( - q*t )

  divSpread = (q-r)*t

  factorStrike = np.exp( divSpread )
  adjustedStrike = np.multiply(s, factorStrike)
  return adjustedStrike, factorPrice


def bs_price(cp, s, k, rf, t, v, div):
        """ Price an option using the Black-Scholes model.
        cp: +1/-1 for call/put
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        """
    
        d1 = (np.log(s/k)+(rf-div+0.5*v*v)*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)

        optprice = (cp*s*np.exp(-div*t)*st.norm.cdf(cp*d1)) - (cp*k*np.exp(-rf*t)*st.norm.cdf(cp*d2))
        
        return optprice

def bissectionMethod(S0, r, q, implied_vol0, maturity, Strike, refPrice, epsilon, optionType):
    calibratedSigma = implied_vol0
    #Call black-scholes price function for initial value
    priceBS = bs_price(optionType , S0, Strike, r, maturity, calibratedSigma, q)
    sigmaUp = 2.0
    sigmaInf = epsilon
    lossSerie = []
    
    priceMax = bs_price(optionType ,S0, Strike, r, maturity, sigmaUp, q)
    if priceMax < refPrice:
        return priceMax, sigmaUp, pd.Series(lossSerie)
    
    priceMin = bs_price(optionType ,S0, Strike, r, maturity, sigmaInf, q)
    if priceMin > refPrice:
        return priceMin, sigmaInf, pd.Series(lossSerie) 

    #Stop the optimization when the error is less than epsilon
    while(abs(priceBS - refPrice) > epsilon):
        #Update the upper bound or the lower bound 
        #by comparing calibrated price and the target price 
        if priceBS < refPrice : 
            sigmaInf = calibratedSigma
        else :
            sigmaUp = calibratedSigma
        #Update calibratedSigma
        calibratedSigma = (sigmaUp + sigmaInf) / 2
        #Update calibrated price
        priceBS = bs_price(optionType ,S0, Strike, r, maturity, calibratedSigma, q)
        #Record the calibration error for this step
        lossSerie.append(abs(priceBS - refPrice)) 
        
    return priceBS, calibratedSigma, pd.Series(lossSerie)

def vectorizedImpliedVolatilityCalibration(S0,
                                           bootstrap,
                                           maturity,
                                           strike,
                                           optionType,
                                           marketPrice):
    epsilon = 1e-6
    discountRate = bootstrap.discountIntegral(maturity) / maturity
    dividendRate = bootstrap.dividendIntegral(maturity) / maturity
    data = np.vstack([np.array(maturity),  np.array(strike), np.array(optionType),
                      np.array(marketPrice), np.array(discountRate), np.array(dividendRate)]).T
    dataSet = pd.DataFrame(data, columns = ["Maturity", "Strike", "OptionType", "Price", "r", "q"])

    calibFunction = lambda x : bissectionMethod(S0, x["r"], x["q"], 0.2,
                                                x["Maturity"],
                                                x["Strike"],
                                                x["Price"],
                                                epsilon,
                                                x["OptionType"])[1]
    impVol = dataSet.apply(calibFunction, axis=1)

    return np.ravel(impVol)
