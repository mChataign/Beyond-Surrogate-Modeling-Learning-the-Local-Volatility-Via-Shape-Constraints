"""
Created on Sat Mar 14 22:53:24 2020
@author: Tahar FERHATI
Paper: SVI Model Free Wings, 12/04/2020
"""
import numpy as np
import pandas as pd 
import scipy
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from scipy.optimize import minimize  
from scipy import integrate
import BS
import bootstrapping
import sys, os

impliedVolColumn = BS.impliedVolColumn

#####################################################################################    Final calibration
# ============  SVI Functions ==================================================
# Attention à l'ordre des paramètres

def SVI(k, a, b , rho, m, sig):
    total_variance = a + b*(rho*(k - m) + np.sqrt( (k - m)*(k - m) + sig*sig))
    return total_variance

def SVI_two_arguments(theta, k):
    a, b , rho, m, sig = theta
    return SVI(k, a, b , rho, m, sig)

def fct_least_squares(theta, log_mon, tot_implied_variance):    
    return np.linalg.norm((SVI_two_arguments(theta, log_mon) - tot_implied_variance), 2)

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

##################################################
## Test Positivity of the density function 
##################################################
    
def dminus(x, a,b,rho,m,sigma):
    vsqrt = np.sqrt(SVI(x, a,b,rho,m,sigma))
    return -x/vsqrt - 0.5*vsqrt  
    
def densitySVI(x, a,b,rho,m,sigma):
    dm = dminus(x, a,b,rho,m,sigma)
    return test_convexity(x, a,b,rho,m,sigma)*np.exp(-0.5*dm*dm)/np.sqrt(2.*np.pi*SVI(x, a,b,rho,m,sigma))

def generateRandomStartValues(lb=None, ub=None):
    # function to generate random initial values for the parameters
    lb[~ np.isfinite(lb)] = -1000.0
    ub[~ np.isfinite(ub)] = 1000.0
    param0 = lb + np.random.rand(size(lb)) * (ub - lb)
    return param0

def isAdmissible(x, constraintList):
    for constraint in constraintList :
        if constraint["fun"](x) < 0.0 : 
            return False
    return True

def generateAdmissibleRandomStartValues(lb=None, ub=None, constraintList = []):
    nbIter = 0
    x = generateRandomStartValues(lb=lb, ub=ub)
    while not isAdmissible(x, constraintList):
        x = generateRandomStartValues(lb=lb, ub=ub)
        if nbIter >= 10000 :
            raise Exception("Too many attempts")
        nbIter += 1
    return x

def fit_svi(mkt_tot_variance=None,
            maturity=None,
            log_moneyness=None,
            initialGuess=None,
            S0=None,
            lambdaList = None,
            param_slice_before = None):
    #############################################################################
    # Optimisation Function : min Loss function = ( SVI_model - Variance_Market )
    # We can use these bunded opt function : trust-constr , SLSQP,  COBYLA
    #############################################################################
    #===========  SVI’s Parameters Boundaries ======================================
    a_low = 1e-6 
    a_high = np.max(mkt_tot_variance)
    b_low = 0.001
    b_high = 1
    rho_low = -0.999999
    rho_high = 0.999999
    m_low = 2*np.min(log_moneyness)
    m_high = 2*np.max(log_moneyness)
    sigma_low = 0.001
    sigma_high = 2

    #===========  SVI’s Parameters Initial Guess =====================================
    a_init = np.min(mkt_tot_variance)/2
    b_init = 0.1
    rho_init = -0.5
    m_init = 0.1 
    sig_init =  0.1 

    SVI_param_bounds = ((a_low,a_high),(b_low, b_high),(rho_low,rho_high),(m_low,m_high),(sigma_low,sigma_high))  
    theta_init = initialGuess
    if initialGuess is None :
        theta_init = np.array([a_init, b_init, rho_init, m_init, sig_init])
    if param_slice_before is not None :
        theta_init = fit_svi(mkt_tot_variance=mkt_tot_variance,
                             maturity=maturity,
                             log_moneyness=log_moneyness,
                             initialGuess=initialGuess,
                             S0=S0,
                             lambdaList = lambdaList,
                             param_slice_before = None)
    
    #Constraint Function : g(k) > 0 
    cons1 = {'type': 'ineq', 'fun': lambda x : lambdaList[0] * constraint1(x , log_moneyness )}
    cons2 = {'type': 'ineq', 'fun': lambda x : lambdaList[1] * constraint2(x , log_moneyness )}
    cons3 = {'type': 'ineq', 'fun': lambda x : lambdaList[2] * constraint3(x , log_moneyness )}
    cons4 = {'type': 'ineq', 'fun': lambda x : lambdaList[3] * constraint4(x , log_moneyness )}
    
    gridPenalization = np.linspace(np.log(0.3),
                                   np.log(3.0),
                                   num=200)
    
    constraintList = [cons1,cons2,cons3,cons4]
    if param_slice_before is not None :
        def calendarConstraint(theta, mkt_log_mon, param_slice_before):
            sliceBefore = SVI_two_arguments(param_slice_before, mkt_log_mon)
            sliceCurrent = SVI_two_arguments(theta, mkt_log_mon)
            epsilon = 1e-3
            #return - np.sqrt(np.mean(np.square(np.clip(sliceBefore - sliceCurrent + epsilon, 0.0, None))))
            return - np.mean(np.abs(np.clip(sliceBefore - sliceCurrent + epsilon, 0.0, None)))
        cons5 = {'type': 'ineq', 'fun': lambda x : lambdaList[4] * calendarConstraint(x, gridPenalization, param_slice_before)}
        constraintList.append(cons5)
    #constraintList = []
    
    nbTry = 1#20
    parameters = np.zeros((size(theta_init), nbTry))
    funValue = np.zeros((nbTry, 1))
    for i in range(nbTry):
        #param0 = generateRandomStartValues(lb=np.array(list(map(lambda x : x[0], SVI_param_bounds))), 
        #                                   ub=np.array(list(map(lambda x : x[1], SVI_param_bounds))))
        #param0 = generateAdmissibleRandomStartValues(lb=np.array(list(map(lambda x : x[0], SVI_param_bounds))), 
        #                                             ub=np.array(list(map(lambda x : x[1], SVI_param_bounds))), 
        #                                             constraintList=constraintList)
        result = minimize(lambda x : fct_least_squares(x, log_moneyness, mkt_tot_variance), 
                          theta_init,
                          method='SLSQP',
                          bounds=SVI_param_bounds, 
                          constraints=constraintList, 
                          options={'ftol': 1e-9, 'disp': True})
        parameters[:, i] = result.x
        funValue[i,0] = result.fun
    
    
    idMin = idxmin(funValue) 
                
    # Optimal SVI vector : a*, b*, rho*, m*, sigma* 
    a_star, b_star, rho_star, m_star, sig_star = parameters[:, idMin[0]]
        
    total_variances_fit = SVI(log_moneyness, a_star, b_star, rho_star, m_star, sig_star)
        
    return parameters[:, idMin[0]]








#####################################################################################    Black-scholes
def blsprice(close,
             strike,
             bootstrap,
             tau_interp,
             implied_volatility,
             optionType) :
    cp = optionType
    return BS.bs_price(cp, close, strike,
                       bootstrap.discountIntegral(tau_interp)/tau_interp,
                       tau_interp, implied_volatility,
                       bootstrap.dividendIntegral(tau_interp)/tau_interp)

def blsimpv(close,
            K_t,
            bootstrap,
            tau_interp,
            optionPrice,
            optionType):
    maturities = tau_interp * np.ones_like(K_t)
    return BS.vectorizedImpliedVolatilityCalibration(close, bootstrap, maturities,
                                                     K_t, optionType, optionPrice)

#####################################################################################    Utilities
def isempty(l):
    return ((l is None) or (numel(l)==0))

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

def error(message):
    raise Exception(message)
    return 

def _assert(predicate, message):
    assert predicate, message
    return

def ismember(elt, l):
    return np.isin(elt, l)

def sqrt(x):
    return np.sqrt(x)

def sign(x):
    return np.sign(x)

def numel(l):
    if type(l)==np.float :
        return 1
    return len(l) if (type(l)==type([]) or type(l)==type(())) else l.size

def ones(shape):
    return np.ones(shape)

def size(array, dim = None):
    return numel(array) if (dim is None) else array.shape[dim] 
    
def isequal(*args):
    return (len(set(args)) <= 1)

def exp(x):
    return np.exp(x)

def unique(x):
    return np.unique(x)

def zeros(x):
    return np.zeros(x)

def idxmin(x):
    return np.unravel_index(np.argmin(x), x.shape)

def unsortedUniquePairs(a):
    _, idx = np.unique(np.ravel([p[0] for p in a]), return_index=True)
    return [a[i] for i in np.sort(idx)]

def interp1(x, v, xq, method, extrapolationMethod):
    if method=="linear":
        funInter = scipy.interpolate.interp1d(np.ravel(x),
                                              np.ravel(v),
                                              kind=method,
                                              fill_value="extrapolate")
    else :
        sortedPairs = unsortedUniquePairs([(x,v) for x,v in sorted(zip(np.ravel(x),np.ravel(v)))])
        funInter = scipy.interpolate.PchipInterpolator(np.ravel([p[0] for p in sortedPairs]),
                                                       np.ravel([p[1] for p in sortedPairs]),
                                                       extrapolate=(extrapolationMethod == "extrapolate"))
    
    return funInter(xq)





#####################################################################################    Converting SVI parametrization

def svi_convertparameters(param_old=None, _from=None, to=None, tau=None):
    #svi_convertparameters converts the parameter set of one type of SVI
    #formulation to another. The parameterizations are assumed to be:
    # * raw =(a,b,m,rho, sigma)
    # * natural = (delta, mu, rho, omega, zeta)
    # * jumpwing = (v, psi, p, c, vt)
    #
    # Input:
    # * param_old = (5x1) = original parameters
    # * from = string = formulation of original parameters (raw, natural,
    # jumpwing)
    # * to = string = formulation of new parameters (raw, natural, jumpwings)
    #
    # Output:
    # param_new = (5x1) = new parameters
    # test that input is correct
    _assert(numel(param_old) == 5, ('There have to be five original parameters'))
    if not ((_from == 'raw') or (_from == 'natural') or (_from == 'jumpwing')):
        error('from has to be one of: raw, natural, jumpwing')
    
    if not ((to == 'raw') or (to == 'natural') or (to == 'jumpwing')):
        error('from has to be one of: raw, natural, jumpwing')
    
    if ((to == 'jumpwing') or (_from == 'jumpwing')) and (tau is None):
        error('tau is required for tailwings formulation')
    
    __switch_0__ = _from
    if __switch_0__ == 'raw':
        a = param_old[0]
        b = param_old[1]
        m = param_old[3]
        rho = param_old[2]
        sigma = param_old[4]
        __switch_1__ = to
        if __switch_1__ == 'raw':
            param_new = param_old
        elif __switch_1__ == 'natural':
            omega = 2 * b * sigma / sqrt(1 - rho ** 2)
            delta = a - omega / 2 * (1 - rho ** 2)
            mu = m + rho * sigma / sqrt(1 - rho ** 2)
            zeta = sqrt(1 - rho ** 2) / sigma
            param_new = [delta, mu, rho, omega, zeta]
        elif __switch_1__ == 'jumpwing':
            w = a + b * (-rho * m + sqrt(m ** 2 + sigma ** 2))
            v = w / tau
            psi = 1 / np.sqrt(w) * b / 2 * (-m / sqrt(m ** 2 + sigma ** 2) + rho)
            p = 1 / np.sqrt(w) * b * (1 - rho)
            c = 1 / np.sqrt(w) * b * (1 + rho)
            vt = 1 / tau * (a + b * sigma * sqrt(1 - rho ** 2))
            param_new = [v, psi, p, c, vt]
        
    elif __switch_0__ == 'natural':

        __switch_1__ = to
        if __switch_1__ == 'raw':
            delta = param_old[0]
            mu = param_old[1]
            rho = param_old[2]
            omega = param_old[3]
            zeta = param_old[4]

            a = delta + omega / 2 * (1 - rho ** 2)
            b = omega * zeta / 2
            m = mu - rho / zeta
            sigma = np.sqrt(1 - rho ** 2) / zeta
            param_new = [a, b, rho, m, sigma]
        elif __switch_1__ == 'natural':
            param_new = param_old
        elif __switch_1__ == 'jumpwing':
            param_temp = svi_convertparameters(param_old, 'natural', 'raw', tau)
            param_new = svi_convertparameters(param_temp, 'raw', 'jumpwing', tau)
        
    elif __switch_0__ == 'jumpwing':

        __switch_1__ = to
        if __switch_1__ == 'raw':
            v = param_old[0]
            psi = param_old[1]
            p = param_old[2]
            c = param_old[3]
            vt = param_old[4]
            w = v * tau

            b = np.sqrt(w) / 2 * (c + p)
            rho = 1 - p * np.sqrt(w) / b
            beta = rho - 2 * psi * np.sqrt(w) / b
            alpha = np.sign(beta) * np.sqrt(1 / beta ** 2 - 1)
            m = ((v - vt) * tau / 
                 (b * (-rho + np.sign(alpha) * np.sqrt(1 + alpha ** 2) - alpha * np.sqrt(1 - rho ** 2))))
            if m == 0:
                sigma = (vt * tau - w) / b / (sqrt(1 - rho ** 2) - 1)
            else:
                sigma = alpha * m
            
            a = vt * tau - b * sigma * np.sqrt(1 - rho ** 2)

            if sigma < 0:
                sigma = 0
            
            param_new = [a, b, rho, m, sigma]
        elif __switch_1__ == 'natural':
            param_temp = svi_convertparameters(param_old, 'jumpwing', 'raw', tau)
            param_new = svi_convertparameters(param_temp, 'raw', 'natural', tau)
        elif __switch_1__ == 'jumpwing':
            param_new = param_old
        
    return param_new

#######################################################################################


def fit_svi_surface(implied_volatility=None,
                    maturity=None,
                    log_moneyness=None,
                    phifun=None,
                    S0=None,
                    lambdaList=None):
    #fit_svi_surface calibrates the SVI surface to market data. First, the entire Surface SVI is fitted
    #to all log-moneyness-theta observations. Second, each slice is fitted again using the SSVI fit as
    #initial guess.
    #
    # Input:
    # * impliedvolatility, maturity, moneyness, phi
    # Output:
    # * parameters = (5xT) = parameters of SSVI = [a, b, rho, m, sigma]
    # * maturities = (Tx1) = corresponding time to maturity
    # * S0 = () = underlying value
    # step one: estimate total implied variance
    total_implied_variance = np.multiply(np.square(implied_volatility), maturity)
    # step two: use linear interpolation for ATM total implied variance
    maturities = np.sort(np.unique(maturity))
    T = size(maturities)
    theta = np.zeros(T)#np.zeros((T, 1))
    for t in np.arange(T):
        pos = (maturity == maturities[t]) #position corresponding to a slice i.e. a smile
        tiv_t = total_implied_variance[pos]
        if np.isin(0, log_moneyness[pos]):
            theta[t] = tiv_t[log_moneyness[pos] == 0] #ATM total implied variance 
        else:#Interpolate ATM total implied variance from the smile
            theta[t] = max(interp1(log_moneyness[pos], tiv_t, 0, 'linear', 'extrapolate'), theta[t-1] if t > 0 else 0)
    

    # step three: fit SVI surface by estimating parameters = [rho, lambda] subject to parameter bounds:
    # -1 < rho < 1, 0 < lambda
    # and constraints: in heston_like: (1 + |rho|) <= 4 lambda, in power-law: eta(1+|rho|) <= 2
    
    
    
    v = np.divide(theta, maturities)
    
    #print()
    # step five: iterate through each maturity and fit c and vt for best fit
    parameters = np.zeros((5, T))
    for t in np.arange(T):#[::-1]:
        pos = (maturity == maturities[t]) #position for the slice
        log_moneyness_t = log_moneyness[pos] 
        total_implied_variance_t = total_implied_variance[pos] #smile
        print("time step", t, " : ", maturities[t])
        if t == 0: 
            parameters[:, t] = fit_svi(mkt_tot_variance=total_implied_variance_t,
                                       maturity=maturities[t],
                                       log_moneyness=log_moneyness_t,
                                       initialGuess=None,
                                       S0=S0,
                                       lambdaList = lambdaList,
                                       param_slice_before = None)
        else:
            parameters[:, t] = fit_svi(mkt_tot_variance=total_implied_variance_t,
                                       maturity=maturities[t],
                                       log_moneyness=log_moneyness_t,
                                       initialGuess=parameters[:, t-1],
                                       S0=S0,
                                       lambdaList = lambdaList,
                                       param_slice_before = None)#parameters[:, t-1])
        theta[t] = SVI_two_arguments(parameters[:, t], 0.0)
    
    return parameters, theta, maturities



#####################################################################################    Interpolating different SVI smiles
def svi_interpolation(log_moneyness=None,
                      tau_interp=None,
                      forward_interp=None,
                      interest_interp=None,
                      parameters=None,
                      theta=None,
                      maturities=None,
                      forward_theta=None,
                      interest_rate_theta=None,
                      S0=None,
                      bootstrap=None ,
                      optionType = None):
    #svi_interpolation estimates inter/extrapolated SVI
    #
    #Input:
    # * log_moneyness = (Kx1) = log-moneyness at which to evalute volatility slices
    # * tau_interp = scalar = maturity at which to generate volatility slices
    # * forward_interp = scalar = forward prices corresponding to maturities
    # * interest_interp = scalar = interest rates corresponding to maturities
    # * parameters (5xL) = estimated parameters of SVI in jumpwing parameterization
    # * theta (Lx1) = ATM total variance time at which the parameters were estimated
    # * maturities (Lx1) = time to maturity corresponding to theta
    # * forward_theta = (Lx1) = forward prices corresponding to theta
    # * interest_rate_theta = (Lx1) = interest rates corresponding to theta (can be scalar)
    # * S0 = () = underlying value
    # * bootstrap = class = provided srvices for discoounting and dividend
    #
    #Output:
    # * total_implied_variance (Kx1) = total_implied_variances for each log_moneyness and tau_interp
    # * implied_volatility (Kx1) = implied volatilities corresponding to total_implied_variance
    # * call_price (Kx1) = call option prices correspongin to total_implied_variance
    #ensure column vectors
    log_moneyness = log_moneyness.flatten()
    theta = theta.flatten()
    maturities = maturities.flatten()
    forward_theta = forward_theta.flatten()
    interest_rate_theta = interest_rate_theta.flatten()
    #ensure scalar input
    _assert(type(tau_interp)==np.float, ('tau_interp has to be scalar'))
    _assert(type(forward_interp)==np.float, ('forward_interp has to be scalar'))
    _assert(type(interest_interp)==np.float, ('interest_interp has to be scalar'))
    #expand scalar input
    if numel(interest_rate_theta) == 1:
        interest_rate_theta = interest_rate_theta * ones(size(theta))
    
    # ensure correct size of input
    _assert(size(parameters, 1) == size(theta, 0), ('parameter set for each theta required'))
    _assert(isequal(size(theta), size(forward_theta), size(interest_rate_theta)), 
            ('theta, forward_theta, and interestrate_theta have to have the same size'))
    # estimate theta for interpolated maturity
    theta_interp = interp1(maturities, theta, tau_interp, 'linear', 'extrapolate')
    close = S0#np.multiply(forward_interp, exp(-interest_interp * tau_interp))
    
    paramJumpwing = np.zeros_like(parameters)
    for i in range(parameters.shape[1]) :
        paramJumpwing[:,i] = svi_convertparameters(parameters[:,i], 'raw', 'jumpwing', maturities[i])
    if ismember(tau_interp, maturities):
        indexMaturity = np.argwhere(maturities == tau_interp)[0][0]
        total_implied_variance = SVI_two_arguments(parameters[:, indexMaturity], log_moneyness) 
        implied_volatility = sqrt(total_implied_variance / tau_interp)
        strike = forward_interp * exp(log_moneyness)
        optionPrice = np.array(blsprice(close, strike,
                                        bootstrap,
                                        tau_interp,
                                        implied_volatility,
                                        optionType))
    else:
        if min(maturities) < tau_interp and tau_interp < max(maturities):
            # interpolation
            idx = idxmin(abs(tau_interp - maturities))[0]
            # if closest maturity is smaller than tau_interp, make idx one unit larger --> idx is index of
            # smallest maturity larger than tau_interp
            if maturities[idx] < tau_interp:
                idx = idx + 1
            epsilon = 1e-6
            thetaBefore = SVI_two_arguments(parameters[:, idx - 1], 0.0) 
            thetaAfter = SVI_two_arguments(parameters[:, idx], 0.0) 
            
            if abs(thetaAfter - thetaBefore) > epsilon :
                alpha_t = ((sqrt(thetaAfter) - sqrt(theta_interp)) / (sqrt(thetaAfter) - sqrt(thetaBefore)))
            else :
                alpha_t = ((maturities[idx] - tau_interp) / (maturities[idx]- maturities[idx-1]))

            param_interp = alpha_t * paramJumpwing[:, idx - 1] + (1 - alpha_t) * paramJumpwing[:, idx]
            param_interp = svi_convertparameters(param_interp, 'jumpwing', 'raw', tau_interp)
            
            total_implied_variance = SVI_two_arguments(param_interp, log_moneyness) 
            implied_volatility = sqrt(total_implied_variance / tau_interp)
            strike = forward_interp * exp(log_moneyness)
            optionPrice = np.array(blsprice(close, strike,
                                            bootstrap,
                                            tau_interp,
                                            implied_volatility,
                                            optionType))
        elif tau_interp < min(maturities):
            # extrapolation for small maturities
            forward_0 = interp1(maturities, forward_theta, 0.0, 'linear', 'extrapolate')
            strike_1 = forward_0 * exp(log_moneyness)
            isCall = np.where(optionType==1, True, False)
            optionPrice_1 = np.where(isCall,
                                     np.maximum(close - strike_1, 0.0),
                                     np.maximum(strike_1 - close, 0.0))

            idx = 0
            total_implied_variance_2 = SVI_two_arguments(parameters[:, idx], log_moneyness) 
            implied_volatility_2 = sqrt(total_implied_variance_2 / maturities[idx])
            strike_2 = forward_theta[idx] * exp(log_moneyness)
            optionPrice_2 = np.array(blsprice(close, strike_2,
                                              bootstrap,
                                              maturities[idx],
                                              implied_volatility_2,
                                              optionType))
            
            thetaAfter = SVI_two_arguments(parameters[:, idx], 0.0) 
            alpha_t = (sqrt(thetaAfter) - sqrt(theta_interp)) / sqrt(thetaAfter)
            K_t = forward_interp * exp(log_moneyness)

            optionPrice = np.multiply(K_t,
                                      ( np.divide(alpha_t * optionPrice_1, strike_1) + np.divide((1 - alpha_t) * optionPrice_2, strike_2) ))
            
            implied_volatility = blsimpv(close, K_t, bootstrap, tau_interp, optionPrice, optionType)
            total_implied_variance = np.power(implied_volatility, 2) * tau_interp

            if any((total_implied_variance - total_implied_variance_2) >= 0) : #Arbitrage are caused by
                param_slope = (paramJumpwing[:, idx + 1] - paramJumpwing[:, idx]) / (theta[idx + 1] - theta[idx])
                param_interp = paramJumpwing[:, idx] + (theta_interp - theta[idx]) * param_slope
                param_interp = svi_convertparameters(param_interp, 'jumpwing', 'raw', tau_interp)

                total_implied_variance = SVI_two_arguments(param_interp, log_moneyness) 
                implied_volatility = sqrt(total_implied_variance / tau_interp)
                strike = forward_interp * exp(log_moneyness)
                optionPrice = np.array(blsprice(close, strike,
                                                bootstrap,
                                                tau_interp,
                                                implied_volatility,
                                                optionType))

        else:
            # extrapolation for large maturities
            total_implied_variance = SVI_two_arguments(parameters[:, -1], log_moneyness)  
            total_implied_variance = total_implied_variance + theta_interp - theta[-1]
            implied_volatility = sqrt(total_implied_variance / tau_interp)
            strike = forward_interp * exp(log_moneyness)
            optionPrice = np.array(blsprice(close, strike,
                                            bootstrap,
                                            tau_interp,
                                            implied_volatility,
                                            optionType))
        
    return optionPrice, implied_volatility, total_implied_variance
 



#####################################################################################    Main functions
def interpolateGrid(df,
                    parameters,
                    theta,
                    maturities,
                    interestrate_theta,
                    forward_theta,
                    S0,
                    bootstrap):
    impliedVolInterpolated = pd.Series()
    #print(df.head())
    for smile in df.rename({"Maturity" : "MaturityColumn"}, axis=1).groupby("MaturityColumn"):
        maturity = smile[0]
        tau_interp = maturity
        k = smile[1]["logMoneyness"].values
        #print(maturities)
        #print(forward_theta)
        #print(tau_interp)
        forward_interp = interp1(maturities, 
                                 forward_theta,
                                 tau_interp,
                                 'linear',
                                 "extrapolate")
        interest_interp = bootstrap.discountShortRate(tau_interp)
        optionType = smile[1]["OptionType"].values
        call_price, implied_volatility, total_implied_variance = svi_interpolation(k,
                                                                                   tau_interp, 
                                                                                   float(forward_interp), 
                                                                                   float(interest_interp), 
                                                                                   parameters, 
                                                                                   theta, 
                                                                                   maturities, 
                                                                                   forward_theta, 
                                                                                   interestrate_theta,
                                                                                   S0,
                                                                                   bootstrap,
                                                                                   optionType)
        impliedVolInterpolated = impliedVolInterpolated.append(pd.Series(implied_volatility, index=smile[1].index))

    index = pd.MultiIndex.from_tuples(impliedVolInterpolated.index.tolist(), 
                                      names=["Strike", "Maturity"])
    
    return pd.Series(impliedVolInterpolated.values, index=index).sort_index()



def impliedVariance(impVol, mat=None):
    Ts = impVol.index.get_level_values("Maturity") if mat is None else mat
    return np.square(impVol) * Ts
    
    
#####################################################################################    Local volatility
def finiteDifferenceSVI(xSet, sviEvalModel):
    strikeStep = 0.0001
    maturityStep = 0.0001
    
    moneynesses = np.exp(xSet.logMoneyness) 
    x = moneynesses = xSet.logMoneyness
    maturities = xSet.Maturity
    
    xSetShifted = xSet.copy(deep=True)
    xSetShifted["logMoneyness"] = xSetShifted["logMoneyness"]  + strikeStep  
    gridStrikeUp = impliedVariance(sviEvalModel(xSetShifted))
    
    xSetShifted["logMoneyness"] = xSetShifted["logMoneyness"] - 2 * strikeStep  
    gridStrikeLow = impliedVariance(sviEvalModel(xSetShifted)) 
    
    gridStrikeMid = impliedVariance(sviEvalModel(xSet)) 
    
    hk = pd.Series((gridStrikeUp + gridStrikeLow - 2 * gridStrikeMid) / (strikeStep**2), 
                   index = xSet.index)
    dK = pd.Series((gridStrikeUp - gridStrikeLow ) / (2 * strikeStep),
                   index = xSet.index)
    
    xSetShifted["logMoneyness"] = xSetShifted["logMoneyness"]  + strikeStep  
    xSetShifted["Maturity"] = xSetShifted["Maturity"]  + maturityStep  
    gridMaturityUp = impliedVariance(sviEvalModel(xSetShifted), mat = xSetShifted["Maturity"].values)
    
    xSetShifted["Maturity"] = xSetShifted["Maturity"] - 2 * maturityStep  
    gridMaturityLow = impliedVariance(sviEvalModel(xSetShifted), mat = xSetShifted["Maturity"].values)
    dT = pd.Series((gridMaturityUp - gridMaturityLow) / (2 * maturityStep), 
                   index = xSet.index)
    
    numerator = (1 - np.divide(x, gridStrikeMid) * dK + 
                 0.25 * ( -0.25 - np.divide(1, gridStrikeMid) + 
                         np.square(np.divide(x, gridStrikeMid.values)) ) * np.square(dK) + 
                 0.5 * hk )
    locVolGatheral = np.sqrt(dT / numerator)
    
    return dT, hk, dK, locVolGatheral, numerator

def removeMaturityInvalidData(df):
    initialist = df["Maturity"].unique()
    maturitiesToKeep = np.ravel( list( filter(lambda x : (df[df["Maturity"]==x]["logMoneyness"].unique().shape[0] > 1), initialist) ) )
    return df[df["Maturity"].isin(maturitiesToKeep)]

########################################################################################## Main Class
class SSVIModelFerhati:
    def __init__(self, S0, bootstrap):
        #Hyperparameters
        self.phi = "power_law"
        self.bootstrap = bootstrap
        self.S0 = S0
        self.tau_interp = 30 / 365.25
        self.interpMethod = 'linear'
        self.extrapolationMethod = "extrapolate"

        #Fitting results
        self.parameters = None
        self.theta = None
        self.maturities = None
        self.interestrate_theta = None
        self.forward_theta = None
        self.lambdaList = [1.0, 1.0, 1.0, 1.0, 1.0]

    def fit(self, df):
        filteredDf = removeMaturityInvalidData(df)
        self.parameters, self.theta, self.maturities = fit_svi_surface(filteredDf[impliedVolColumn].values,
                                                                       filteredDf["Maturity"].values,
                                                                       filteredDf["logMoneyness"].values,
                                                                       self.phi,
                                                                       S0 = self.S0, 
                                                                       lambdaList = self.lambdaList)
        #dataSet = dataSet.copy()
        forward = np.exp(-filteredDf["logMoneyness"]) * filteredDf["Strike"]
        # round for float comparaison
        self.forward_theta = forward.groupby("Maturity").mean().values
        self.interestrate_theta = self.bootstrap.discountIntegral(self.maturities) / self.maturities
        return
    
    def assessArbitrageViolations(self, df):
        nbViolationBut = 0
        #logMoneynessGrid = df["logMoneyness"].unique()
        logMoneynessGrid = np.linspace(np.log(0.3),
                                       np.log(3.0),
                                       num=200)
        for m in range(self.parameters.shape[1]):
            a, b, rho, m, sig = self.parameters[:,m]
            g = test_convexity(logMoneynessGrid, a, b, rho, m, sig)
            nbViolationBut += np.sum(g < 0.0)
        
        
        slicePrevious = np.zeros_like(logMoneynessGrid)
        nbViolationCal = 0
        for m in range(self.parameters.shape[1]):
            a, b, rho, m, sig = self.parameters[:,m]
            sliceSVI = SVI(logMoneynessGrid, a, b, rho, m, sig)
            nbViolationCal += np.sum((slicePrevious - sliceSVI) > 0.0)
            slicePrevious = sliceSVI
        return  nbViolationBut, nbViolationCal
        
    def automaticHyperparametersTuning(self, df):
        #Block print
        formerStdOut = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        def multiplyList(liste, factor):
            return list(map(lambda y : factor * y, liste))
        
        #Iterate on a grid of values for butterfly arbitrage constraint
        formerLambdaList = self.lambdaList
        lambdaButterfly = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
        numberOfarbitrageButterfly = []
        rmseBut = []
        firstArbitrageFreeLambda = None
        for l in lambdaButterfly : 
            self.lambdaList = multiplyList(formerLambdaList, l)
            self.lambdaList[4] = 0.0
            self.fit(df)
            numberOfarbitrageButterfly.append(self.assessArbitrageViolations(df)[0]) 
            
            pred = self.eval(df)
            rmseBut.append( mean_squared_error( pred, df[impliedVolColumn]))
            if (firstArbitrageFreeLambda is None) and (numberOfarbitrageButterfly[-1]==0) :  
                firstArbitrageFreeLambda = l
        
        #Iterate on a grid of values for calendar arbitrage constraint
        lambdaCalendar = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
        numberOfArbitrageCalendar = []
        rmseCal = []
        for l in lambdaCalendar : 
            self.lambdaList = multiplyList(formerLambdaList, l)
            self.lambdaList[0] = firstArbitrageFreeLambda
            self.lambdaList[1] = firstArbitrageFreeLambda
            self.lambdaList[2] = firstArbitrageFreeLambda
            self.lambdaList[3] = firstArbitrageFreeLambda
            self.fit(df)
            numberOfArbitrageCalendar.append(self.assessArbitrageViolations(df)[1]) 
            
            pred = self.eval(df)
            rmseCal.append( mean_squared_error( pred, df[impliedVolColumn]))
        
        
        self.lambdaList = formerLambdaList
        #Activate print
        #sys.stdout = formerStdOut
        sys.stdout = formerStdOut
        res = {"ButterflyArbitrage" : pd.Series(numberOfarbitrageButterfly, index = lambdaButterfly),
               "CalendarArbitrage" : pd.Series(numberOfArbitrageCalendar, index = lambdaCalendar),
               "ButterflyRMSE" : pd.Series(rmseBut, index = lambdaButterfly),
               "CalendarRMSE" : pd.Series(rmseCal, index = lambdaCalendar)}
        
        plt.plot(res["ButterflyArbitrage"]) 
        plt.title("Number of arbitrages")
        plt.xscale('symlog')
        plt.show()
        
        plt.plot(res["ButterflyRMSE"]) 
        plt.title("RMSES")
        plt.xscale('symlog')
        plt.show()
        
        plt.plot(res["CalendarArbitrage"]) 
        plt.title("Number of arbitrages")
        plt.xscale('symlog')
        plt.show()
        
        plt.plot(res["CalendarRMSE"]) 
        plt.title("RMSES")
        plt.xscale('symlog')
        plt.show()
        
        #Dichotomy on parameter values for which we can assume monotonicity : the higher the penalization, the worst the accuracy and less arbitrage occured 
        return res 
        

    def eval(self, df):
        serie = interpolateGrid(df[df["Maturity"] > 0],
                                self.parameters,
                                self.theta,
                                self.maturities,
                                self.interestrate_theta,
                                self.forward_theta,
                                self.S0,
                                self.bootstrap)
        return serie
