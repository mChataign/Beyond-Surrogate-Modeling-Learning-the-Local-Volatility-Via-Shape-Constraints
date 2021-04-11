import numpy as np
import pandas as pd
import scipy
import BS
from scipy import *
import bootstrapping
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import time

impliedVolColumn = BS.impliedVolColumn

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


def monotonicInterpolation(theta, maturities):
    previousMaxIdx = 0
    previousMax = 0
    monotonicCurve = []
    
    #Select observations which preserve monotonicity
    originalCurve = pd.Series(theta, index = maturities).sort_index()
    prevMax = 0.0
    increasingCurve = []
    for k in originalCurve.dropna().index :
        if originalCurve[k] >= prevMax :
             prevMax = originalCurve[k]
        increasingCurve.append(prevMax)
    
    for m in maturities :
        monotonicCurve.append(interp1(originalCurve.dropna().index.values, np.array(increasingCurve), m, 'linear', 'extrapolate'))
    return pd.Series(np.ravel(monotonicCurve), index = maturities)
#####################################################################################    First SSVI calibration
def fit_ssvi(phifun=None, log_moneyness=None, theta_expanded=None, total_implied_variance=None, tau_expanded = None):
    lb = np.array([-1, 0])
    ub = np.array([1, np.inf])
    lambdaConstraint = 1#10000
    c = lambda x: lambdaConstraint * mycon(x, phifun)
    cJac = lambda x: lambdaConstraint * myconJac(x, phifun)
    cHess = lambda x,v: lambdaConstraint * myconHess(x, v, phifun)
    #options = optimset('fmincon')
    #options = optimset(options, 'algorithm', 'interior-point')
    #options = optimset(options, 'Display', 'off')
    targetfun = lambda x: fitfunctionSSVI(x, phifun, log_moneyness, theta_expanded, total_implied_variance, tau_expanded)
    # perform optimization N times with random start values
    N = 100
    parameters = np.zeros((size(lb), N))
    funValue = np.zeros((N, 1))
    for n in np.arange(N):
        param0 = generateRandomStartValues(lb, ub)
        constraints = {"type" : "ineq",
                       "fun" : lambda x : np.array([ - c(x) ]),
                       "jac" : lambda x : np.array([ - cJac(x) ])}
        res = scipy.optimize.minimize(targetfun, param0,
                                      method= "SLSQP",#"L-BFGS-B", #"trust-constr",#"trust-constr",
                                      bounds=list(zip(lb, ub)),#[lb, ub],
                                      constraints = constraints) 
        parameters[:, n] = res.x
        funValue[n,0] = res.fun


    idMin = idxmin(funValue) 
    parameters = parameters[:, idMin[0]]

    print("Contrainte : ", c(parameters))

    if phifun == 'power_law':
        parameters = np.concatenate((parameters, [0.5]))
    return parameters

def fitfunctionSSVI(x=None, phifun=None, log_moneyness=None, theta=None, total_implied_variance=None, tau_expanded = None):
    # extract parameters from x
    rho = x[0]
    __switch_0__ = phifun
    if __switch_0__ == 'heston_like':
        phi_param = x[1]
    elif __switch_0__ == 'power_law':
        phi_param = [x[1], 0.5]
    else:
        raise Exception('Incorrect function for phi') 
    model_total_implied_variance, _ = svi_surface(log_moneyness, theta, rho, 
                                                  phifun, phi_param, 
                                                  tau_expanded)
    value = norm(total_implied_variance - model_total_implied_variance)
    return value

def mycon(x=None, phifun=None):
    ceq = 0
    __switch_0__ = phifun
    rho = x[0]
    if __switch_0__ == 'heston_like':
        _lambda = x[1]
        c = 1 + abs(rho) - 4 * _lambda    #by construction, rho is the first parameter, lamda the second
    elif __switch_0__ == 'power_law':
        eta = x[1]
        c = eta * (1 + abs(rho)) - 2    #by construction, rho is the first parameter, lamda the second
    else:
        raise Exception('Incorrect function for phi') 
    return c#(c, ceq)

def myconJac(x=None, phifun=None):
    ceq = np.array([0,0])
    __switch_0__ = phifun
    rho = x[0]
    if __switch_0__ == 'heston_like':
        c = np.array([np.sign(rho), -4])    #by construction, rho is the first parameter, lamda the second
    elif __switch_0__ == 'power_law':
        eta = x[1]
        c = np.array([eta * np.sign(rho), (1 + abs(rho))])  #by construction, rho is the first parameter, lamda the second
    else:
        raise Exception('Incorrect function for phi')
    return c#(c, ceq)

def myconHess(x=None, v=None, phifun=None):
    ceq = np.array([[0,0],[0,0]])
    __switch_0__ = phifun
    if __switch_0__ == 'heston_like':
        c = np.array([[0,0],[0,0]])   #by construction, rho is the first parameter, lamda the second
    elif __switch_0__ == 'power_law':
        c = np.array([[0, np.sign(x[0])], [np.sign(x[0]), 0]]) #by construction, rho is the first parameter, lamda the second
    else:
        raise Exception('Incorrect function for phi')
    return c#(c, ceq)


def generateRandomStartValues(lb=None, ub=None):
    # function to generate random intial values for the parameters
    lb[~ np.isfinite(lb)] = -1000
    ub[~ np.isfinite(ub)] = 1000
    param0 = lb + np.random.rand(size(lb)) * (ub - lb)
    return param0

#####################################################################################    Smile calibration
def fit_svi(x0=None, k=None, 
            total_implied_variance=None, 
            slice_before=None, slice_after=None, 
            tau=None,
            gridPenalization = None,
            param_slice_before = None,
            param_slice_after = None,
            tau_before = None,
            tau_after = None):
    # fit_svi fits a volatility slice to observed implied volatilities. 
    # get variable bounds
    large = 1e5
    small = 1e-6
    lb = [small, -large, small, small, small]
    ub = [large, large, large, large, large]
    lambdaConstraint = 10000
    targetfun = lambda x: fitfunctionSVI(x, k,
                                         total_implied_variance,
                                         slice_before, slice_after,
                                         tau,
                                         gridPenalization,
                                         param_slice_before,
                                         param_slice_after,
                                         tau_before,
                                         tau_after)
    # only optimize first three variables, final two are set by no-arbitrage condition

    x0 = x0[:3] # order of parameters [v, psi, p, c, vt]
    lb = lb[:3]
    ub = ub[:3]
    # linear inequality: -p <= 2*psi
    #A = np.array([0 , -2, -1])
    A = np.array([0, 2, 1])
    #b = 0
    #print("slice_before",slice_before)
    parameters = None
    fval = np.inf
    constraints = {"type" : "ineq",
                   "fun" : lambda x : np.array([ A @ x ]),
                   "jac" : lambda x : np.array([ [0, 2, 1] ])}
    
    x3 = x0[2] + 2 * x0[1]
    x4 = x0[0] * 4 * x0[2] * x3 / (x0[2] + x3) ** 2
    newX = np.concatenate((x0, np.array([x3,x4]))) 
    calendarConstraint = crossedNess(gridPenalization, 
                                     newX, 
                                     tau,
                                     param_slice_before, 
                                     tau_before,
                                     param_slice_after, 
                                     tau_after)
    print()
    print("x0")
    print("Calendar constraint : ", calendarConstraint, "  ", "violated" if calendarConstraint > 0 else "satisfied")
    print("Butterfly constraint : ", A @ x0, "  ", "violated" if A @ x0 <= 0 else "satisfied")
    model_total_implied_variance,_ = svi_jumpwing(k, newX, tau)
    print("RMSE : ", norm(total_implied_variance - model_total_implied_variance))
    print()
    
    nbRestart = 50
    for i in np.arange(nbRestart):
        res = scipy.optimize.minimize(targetfun, x0, 
                                      method="SLSQP",#"L-BFGS-B", #"trust-constr",#"trust-constr",
                                      bounds=list(zip(lb, ub)),#[lb, ub],
                                      constraints = constraints) 
        #res = fmincon(targetfun, x0, A, b, [], [], lb, ub, [], options)
        if (i==0) or (fval > res.fun) :
            fval =  res.fun
            parameters = res.x
    
    
    x3 = res.x[2] + 2 * res.x[1]
    x4 = res.x[0] * 4 * res.x[2] * x3 / (res.x[2] + x3) ** 2
    newX = np.concatenate((res.x, np.array([x3,x4]))) 
    calendarConstraint = crossedNess(gridPenalization, 
                                     newX, 
                                     tau,
                                     param_slice_before, 
                                     tau_before,
                                     param_slice_after, 
                                     tau_after)
    print("Calendar constraint : ", calendarConstraint, "  ", "violated" if calendarConstraint > 0 else "satisfied")
    print("Butterfly constraint : ", A @ res.x, "  ", "violated" if A @ res.x <= 0 else "satisfied")
    model_total_implied_variance,_ = svi_jumpwing(k, newX, tau)
    print("RMSE : ", norm(total_implied_variance - model_total_implied_variance))
    print("--------------------------------------------------------------------")
    print()
    return parameters

def crossedNess(gridPenalization, 
                newX, 
                tau,
                param_slice_before, 
                tau_before,
                param_slice_after, 
                tau_after):
    
    model_total_implied_variance,_ = svi_jumpwing(gridPenalization, newX, tau)
    crossedness = 0.0 
    threshold = 1e-4
    if not isempty(param_slice_before):
        before_total_implied_variance,_ = svi_jumpwing(gridPenalization, param_slice_before, tau_before)
        if any(model_total_implied_variance < before_total_implied_variance) :
            crossedness += np.amax(np.maximum( before_total_implied_variance - model_total_implied_variance + threshold, 0))
            
    if not isempty(param_slice_after):
        after_total_implied_variance,_ = svi_jumpwing(gridPenalization, param_slice_after, tau_after)
        if any(model_total_implied_variance > after_total_implied_variance) :
            crossedness += np.amax(np.maximum( model_total_implied_variance - after_total_implied_variance + threshold, 0))
    
    return crossedness

def fitfunctionSVI(x=None, k=None, 
                   total_implied_variance=None, 
                   slice_before=None, slice_after=None, 
                   tau=None,
                   gridPenalization = None,
                   param_slice_before = None,
                   param_slice_after = None,
                   tau_before = None,
                   tau_after = None):
    # fitfunction is the objective function of the minimization of fit_svi. The objective is the 2-norm
    # of the error in total implied variance. If specified, the function tests whether the current total
    # variance slice lies between the prior and later slice. If the arbitrage bound is violated, the
    # function is penalized with a very large value.
    #
    # Input:
    # * x = (5x1) = parameters of SVI model
    # * k = (Nx1) = log moneyness at which total implied variance is calculated
    # * total_implied_variance = (Nx1) = market observed total implied variance
    # * slice_before = (Nx1) = total implied variance with smaller maturity than current slice
    # * slice_after = (Nx1) = total implied variance with larter maturity than current slice
    #
    # Output:
    # * value
    # check that input is consistent
    _assert(numel(k) == numel(total_implied_variance), 
            'moneyness and total implied variance need to have the same size')
    if not isempty(slice_before):
        _assert((numel(k) == numel(slice_before)), 
                'moneyness and slice_before need to have the same size')
    
    if not isempty(slice_after):
        _assert((numel(k) == numel(slice_after)), 
                'moneyness and slice_after need to have the same size')
    
    # expand x
    x3 = x[2] + 2 * x[1]
    x4 = x[0] * 4 * x[2] * x3 / (x[2] + x3) ** 2
    newX = np.concatenate((x, np.array([x3,x4]))) 
    # calculate model total implied variance given current parameter estimate
    model_total_implied_variance,_ = svi_jumpwing(k, newX, tau)
    # calculate value of objective function
    value = norm(total_implied_variance - model_total_implied_variance)
    # if the current model total implied variance crosses the earlier or later slice, set value to 1e6
    #if not isempty(slice_before) and any(model_total_implied_variance < slice_before):
    #    value = 1e6
    
    #if not isempty(slice_after) and any(model_total_implied_variance > slice_after):
    #    value = 1e6
    
    #if np.isnan(value):
    #    value = 1e6
    
    #if not isempty(gridPenalization) and ((not isempty(param_slice_before)) or
    #                                      (not isempty(param_slice_after))):
    if not isempty(gridPenalization) :
        lambdaCalendar = 1e8
        value += lambdaCalendar * crossedNess(gridPenalization, 
                                              newX, 
                                              tau,
                                              param_slice_before, 
                                              tau_before,
                                              param_slice_after, 
                                              tau_after)
    
    return value


#####################################################################################    Final calibration
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
    
def fit_svi_surface(implied_volatility=None,
                    maturity=None,
                    log_moneyness=None,
                    phifun=None,
                    S0=None):
    #fit_svi_surface calibrates the SVI surface to market data. First, the entire Surface SVI is fitted
    #to all log-moneyness-theta observations. Second, each slice is fitted again using the SSVI fit as
    #initial guess.
    #
    # Input:
    # * impliedvolatility, maturity, moneyness, phi
    # Output:
    # * parameters = (2x1) = parameters of SSVI = [rho, lambda]
    # * theta      = (Tx1) = ATM total implied variance
    # * maturities = (Tx1) = corresponding time to maturity
    # * S0 = () = underlying value
    # step one: estimate total implied variance
    total_implied_variance = np.multiply(np.square(implied_volatility), maturity)
    # step two: use linear interpolation for ATM total implied variance
    maturities = np.sort(np.unique(maturity))
    T = size(maturities)
    theta = np.zeros(T)#np.zeros((T, 1))
    theta_expanded = np.zeros(size(maturity))
    for t in np.arange(T):
        pos = (maturity == maturities[t]) #position corresponding to a slice i.e. a smile
        tiv_t = total_implied_variance[pos]
        if np.isin(0, log_moneyness[pos]):
            theta[t] = tiv_t[log_moneyness[pos] == 0] #ATM total implied variance 
        else:#Interpolate ATM total implied variance from the smile when 0 is contained in the logmoneyness domain
            if 0.0 <= np.amax(log_moneyness[pos]) :
                theta[t] = interp1(log_moneyness[pos], tiv_t, 0, 'linear', 'extrapolate')
            else :
                theta[t] = np.nan
            #plt.plot(log_moneyness[pos], tiv_t)
            #plt.title(str(maturities[t]))
            #plt.show()
    thetaSeries = monotonicInterpolation(theta, maturities)
    theta = thetaSeries.values
    maturities = thetaSeries.index
    for t in np.arange(T):
        pos = (maturity == maturities[t]) #position corresponding to a slice i.e. a smile
        theta_expanded[pos] = theta[t]
    
    
    # step three: fit SVI surface by estimating parameters = [rho, lambda] subject to parameter bounds:
    # -1 < rho < 1, 0 < lambda
    # and constraints: in heston_like: (1 + |rho|) <= 4 lambda, in power-law: eta(1+|rho|) <= 2
    parameters = fit_ssvi(phifun, 
                          log_moneyness, 
                          theta_expanded, 
                          total_implied_variance,
                          maturity)
    parametersSSVI = parameters
    print("Parameters of SSVI model : ", parametersSSVI)
    
    rho = parameters[0]
    phi_param = parameters[1:]
    # step four: transform SSVI parameters to SVI-JW parameters
    __switch_0__ = phifun
    #print(phi_param)
    if __switch_0__ == 'heston_like':
        phi = heston_like(theta, phi_param)
    elif __switch_0__ == 'power_law':
        phi = power_law(theta, phi_param)
    else:
        raise Exception('Incorrect function for phi') 
    
    v = np.divide(theta, maturities)
    #print("v.shape", v.shape)
    #print("theta.shape", theta.shape)
    #print("maturities.shape", maturities.shape)
    psi = np.multiply(0.5 * rho * np.sqrt(theta),  phi)
    p = np.multiply(0.5 * np.sqrt(theta), phi * (1 - rho))
    c = p + 2 * psi
    vt = np.divide(np.multiply(v, (4 * np.multiply(p, c))), np.square((p + c)))
    gridPenalization = np.linspace(np.log(0.3),
                                   np.log(3.0),
                                   num=200)
    
    #print()
    # step five: iterate through each maturity and fit c and vt for best fit
    parameters = np.zeros((5, T))
    for t in np.arange(T)[::-1]:
        pos = (maturity == maturities[t]) #position for the slice
        log_moneyness_t = log_moneyness[pos] 
        total_implied_variance_t = total_implied_variance[pos] #smile
        #
        if t == (T-1): 
            param_before = [v[t - 1], psi[t - 1], p[t - 1], c[t - 1], vt[t - 1]]
            slice_before,_ = svi_jumpwing(log_moneyness_t, param_before, maturities[t - 1])
            param_after = None
            slice_after = []
        elif t == 0:
            slice_before = []
            param_before = None
            param_after = [v[t + 1], psi[t + 1], p[t + 1], c[t + 1], vt[t + 1]]
            slice_after,_ = svi_jumpwing(log_moneyness_t, param_after, maturities[t + 1])
        else:
            param_before = [v[t - 1], psi[t - 1], p[t - 1], c[t - 1], vt[t - 1]]
            slice_before,_ = svi_jumpwing(log_moneyness_t, param_before, maturities[t - 1])
            param_after = [v[t + 1], psi[t + 1], p[t + 1], c[t + 1], vt[t + 1]]
            slice_after,_ = svi_jumpwing(log_moneyness_t, param_after, maturities[t + 1])
        
        print("time step", t, " : ", maturities[t])
        param0 = [v[t], psi[t], p[t], c[t], vt[t]]
        parameters[:3, t] = fit_svi(param0,
                                    log_moneyness_t,
                                    total_implied_variance_t, 
                                    slice_before, slice_after, 
                                    maturities[t], 
                                    gridPenalization,
                                    param_before,
                                    param_after,
                                    None if t == 0 else maturities[t - 1],
                                    None if t == (T-1) else maturities[t + 1])
        parameters[3, t] = parameters[2, t] + 2 * parameters[1, t]
        parameters[4, t] = np.divide(parameters[0, t] * 4 * parameters[2, t] * parameters[3, t] ,
                                     np.square(parameters[2, t] + parameters[3, t])) 
        #theta[t] = svi_jumpwing(np.array([0.0]), parameters[:, t], maturities[t])[0][0]

    return parameters, theta, maturities, parametersSSVI

#####################################################################################    ATM Total variance parametrization
def fit_svi_surface_nonlin(x=None):
    ceq = np.zeros((2, 1))
    c = np.zeros((5, 1))
    # c = p+2*psi
    ceq[0] = x[3] - x[2] - 2 * x[1]
    # vt = v*(4*p*c)/(p+c)^2
    ceq[1] = x[4] - (x[0] * 4 * x[2] * x[3]) / ((x[2] + x[3]) ** 2)
    # -p <= 2 psi
    c[0] = -x[2] - 2 * x[1]
    # 2 psi <= c
    c[1] = 2 * x[1] - x[3]
    # vt <= v
    c[2] = x[4] - x[0]
    # -p <= 2psi
    c[3] = -x[2] - 2 * x[1]
    # 2psi <= c
    c[4] = 2 * x[1] - c[3]
    return c, ceq 

def heston_like(theta=None, param=None):
    # Heston-like parameterization
    _lambda = param#[0]
    value = np.multiply(1. / (_lambda * theta), 
                        1 - np.divide( (1 - exp(-_lambda * theta)), (_lambda * theta) ) )
    return value


def power_law(theta=None, param=None):
    # Power-law parameterization
    eta = param[0]
    gamma = param[1]
    value = np.divide(eta, 
                      np.multiply(np.power(theta, (gamma)), 
                                  np.power((1 + theta), (1 - gamma))))
    #value = eta * np.power(theta, - gamma)
    return value 


def svi(parameterization=None, k=None, param=None, tau=None):
    #SVI - Stochastic Volatility Inspired parameterization of the implied
    #volatility smile.
    # Input:
    # * parameterization = string = one of: raw, natural, or jumpwing
    # * k = (Nx1) = log-moneyness at which to evaluate the total implied
    # variance
    # * param = (5x1) = parameters:
    #   * a = scalar = level of variance
    #   * b = scalar = slope of wings
    #   * m = scalar = translates smile to right
    #   * rho = scalar = counter-clockwise rotation of smile
    #   * sigma = scalar = reduces ATM curvature of the smile
    # Output:
    # * totalvariance = (Nx1) = estimated total variance at k
    # * impliedvolatility = (Nx1) = estimated implied volatility at (k,tau)
    __switch_0__ = parameterization
    if __switch_0__ == 'raw':
        [totalvariance, impliedvolatility] = svi_raw(k, param, tau)
    elif __switch_0__ == 'natural':
        [totalvariance, impliedvolatility] = svi_natural(k, param, tau)
    elif __switch_0__ == 'jumpwing':
        [totalvariance, impliedvolatility] = svi_jumpwing(k, param, tau)
    else:
        raise Exception('Unknown parameterization specified') 
        
    return totalvariance, impliedvolatility

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
        m = param_old[2]
        rho = param_old[3]
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
            param_new = [a, b, m, rho, sigma]
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
            #print("m",m)
            #print("tau",tau)
            #print("v",v)
            #print("vt",vt)
            #print("param_old", param_old)
            if m == 0:
                sigma = (vt * tau - w) / b / (sqrt(1 - rho ** 2) - 1)
            else:
                sigma = alpha * m
            
            a = vt * tau - b * sigma * np.sqrt(1 - rho ** 2)

            if sigma < 0:
                sigma = 0
            
            param_new = [a, b, m, rho, sigma]
        elif __switch_1__ == 'natural':
            param_temp = svi_convertparameters(param_old, 'jumpwing', 'raw', tau)
            param_new = svi_convertparameters(param_temp, 'raw', 'natural', tau)
        elif __switch_1__ == 'jumpwing':
            param_new = param_old
        
    return param_new

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
    if ismember(tau_interp, maturities):
        indexMaturity = np.argwhere(maturities == tau_interp)[0][0]
        total_implied_variance,_ = svi_jumpwing(log_moneyness, parameters[:, indexMaturity], tau_interp)
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
            if abs(theta[idx] - theta[idx - 1]) > epsilon :
                alpha_t = ((sqrt(theta[idx]) - sqrt(theta_interp)) / (sqrt(theta[idx]) - sqrt(theta[idx - 1])))
            else :
                alpha_t = ((maturities[idx] - tau_interp) / (maturities[idx]- maturities[idx-1]))

            param_interp = alpha_t * parameters[:, idx - 1] + (1 - alpha_t) * parameters[:, idx]
            total_implied_variance,_ = svi_jumpwing(log_moneyness, param_interp, tau_interp)
            implied_volatility = sqrt(total_implied_variance / tau_interp)
            strike = forward_interp * exp(log_moneyness)
            optionPrice = np.array(blsprice(close, strike,
                                            bootstrap,
                                            tau_interp,
                                            implied_volatility,
                                            optionType))
            optionPrice[optionPrice < 0] = 0
        elif tau_interp < min(maturities):
            # extrapolation for small maturities
            forward_0 = interp1(maturities, forward_theta, 0.0, 'linear', 'extrapolate')
            strike_1 = forward_0 * exp(log_moneyness)
            isCall = optionType==1 #np.where(optionType==1, True, False)
            optionPrice_1 = np.where(isCall,
                                     np.maximum(close - strike_1, 0),
                                     np.maximum(strike_1 - close, 0))

            idx = 0
            total_implied_variance_2,_ = svi_jumpwing(log_moneyness, parameters[:, idx], maturities[idx])
            implied_volatility_2 = sqrt(total_implied_variance_2 / maturities[idx])
            strike_2 = forward_theta[idx] * exp(log_moneyness)
            optionPrice_2 = np.array(blsprice(close, strike_2,
                                              bootstrap,
                                              maturities[idx],
                                              implied_volatility_2,
                                              optionType))
            optionPrice_2[optionPrice_2 < 0] = 0

            alpha_t = (sqrt(theta[idx]) - sqrt(theta_interp)) / sqrt(theta[idx])
            K_t = forward_interp * exp(log_moneyness)

            optionPrice = np.multiply(K_t,
                                      (np.divide(alpha_t * optionPrice_1, strike_1) +
                                       np.divide((1 - alpha_t) * optionPrice_2, strike_2)))

            #print("optionPrice_1 : ", optionPrice_1)
            #print("optionPrice_2 : ", optionPrice_2)
            #print("optionPrice : ", optionPrice)

            #print("implied_volatility_2 : ", implied_volatility_2)
            #print("implied_volatility : ", implied_volatility)
            implied_volatility = blsimpv(close, K_t, bootstrap, tau_interp, optionPrice, optionType)
            total_implied_variance = np.power(implied_volatility, 2) * tau_interp

            if any((total_implied_variance - total_implied_variance_2) >= 0) : #Arbitrage are caused by
                param_slope = (parameters[:, idx + 1] - parameters[:, idx]) / (theta[idx + 1] - theta[idx])
                param_interp = parameters[:, idx] + (theta_interp - theta[idx]) * param_slope

                total_implied_variance,_ = svi_jumpwing(log_moneyness, param_interp, tau_interp)
                implied_volatility = sqrt(total_implied_variance / tau_interp)
                strike = forward_interp * exp(log_moneyness)
                optionPrice = np.array(blsprice(close, strike,
                                                bootstrap,
                                                tau_interp,
                                                implied_volatility,
                                                optionType))
                optionPrice[optionPrice < 0] = 0

        else:
            # extrapolation for large maturities
            total_implied_variance,_ = svi_jumpwing(log_moneyness, parameters[:, -1], maturities[-1])
            total_implied_variance = total_implied_variance + theta_interp - theta[-1]
            #print(total_implied_variance)
            #print(tau_interp)
            implied_volatility = sqrt(total_implied_variance / tau_interp)
            strike = forward_interp * exp(log_moneyness)
            optionPrice = np.array(blsprice(close, strike,
                                            bootstrap,
                                            tau_interp,
                                            implied_volatility,
                                            optionType))
        
    return optionPrice, implied_volatility, total_implied_variance
 
#####################################################################################    SVI paramerizations
def svi_jumpwing(k=None, param=None, tau=None):
    #SVI - Stochastic Volatility Inspired parameterization of the implied
    #volatility smile. This function implements the jump-wings formulation.
    # Input:
    # * k = (Nx1) = log-moneyness at which to evaluate the total implied
    # variance
    # * param = (5x1) = parameters:
    #   * v = scalar = ATM variance
    #   * psi = scalar = ATM skew
    #   * p = scalar = slope of left/put wing
    #   * c = scalar = slope of right/call wing
    #   * vt = scalar = minimum implied variance
    # Output:
    # * totalvariance = (Nx1) = estimated total variance at k
    # * impliedvolatility = (Nx1) = estimated implied volatility at (k,tau)
    # make sure that input is in column vector format
    k = k.flatten()
    # check that input is consistent
    _assert(numel(param) == 5, 'There have to be five parameters: v, psi, p, c, vt')
    if tau is None:
        error('tau needs to be specified for jump-wings formulation')
    
    v = param[0]
    p = param[2]
    c = param[3]
    vt = param[4]

    # make sure that parameter restrictions are satisfied
    # assert(v >= 0, 'v has to be non-negative');
    # assert(p >= 0, 'p has to be non-negative');
    # assert(c >= 0, 'c has to be non-negative');
    # assert(vt >= 0, 'vt has to be non-negative');
    # convert parameters to raw formulation
    param_raw = svi_convertparameters(param, 'jumpwing', 'raw', tau)
    # calculate total variance
    totalvariance, impliedvolatility = svi_raw(k, param_raw, tau)
    return totalvariance, impliedvolatility


def svi_natural(k=None, param=None, tau=None):
    #SVI - Stochastic Volatility Inspired parameterization of the implied
    #volatility smile. This function implements the natural formulation.
    # Input:
    # * k = (Nx1) = log-moneyness at which to evaluate the total implied
    # variance
    # * param = (5x1) = parameters:
    #   * delta = scalar = level of variance
    #   * mu = scalar = slope of wings
    #   * rho = scalar = translates smile to right,
    #   * omega = scalar = counter-clockwise rotation of smile
    #   * zeta = scalar = reduces ATM curvature of the smile
    # Output:
    # * totalvariance = (Nx1) = estimated total variance at k
    # * impliedvolatility = (Nx1) = estimated implied volatility at (k,tau)
    # make sure that input is in column vector format
    k = k.flatten()
    # check that input is consistent
    _assert(numel(param) == 5, 'There have to be five parameters: a, b, m, rho, sigma')
    if tau is None:
        error('tau needs to be specified in order to return implied volatilities')
    
    delta = param[0]
    mu = param[1]
    rho = param[2]
    omega = param[3]
    zeta = param[4]
    # make sure that parameter restrictions are satisfied
    _assert(omega >= 0, 'omega has to be non-negative')
    _assert(abs(rho) < 1, '|rho| has to be smaller than 1')
    _assert(zeta > 0, 'zeta has to be positive')
    _assert(delta + omega * (1 - rho ** 2) >= 0, 'delta + omega (1-rho^2) has to be non-negative')
    # calculate total variance
    totalvariance = delta + omega / 2 * (1 + zeta * rho * (k - mu) + 
                                         sqrt(np.square(zeta * (k - mu) + rho) + (1 - rho ** 2)))
    # if requested, calculate implied volatility
    impliedvolatility = sqrt(totalvariance / tau)
    return totalvariance, impliedvolatility


def svi_parameter_bounds(parameterization=None):
    # svi_parameter_bounds returns the parameter bounds of an SVI parameterization. The parameters are
    # assumed to be in the following order:
    # raw = [a b m rho sigma]
    # natural = [delta mu rho omega zeta]
    # jumpwing = [v psi p c vt]
    large = 1e5
    __switch_0__ = parameterization
    if 0:
        pass
    elif __switch_0__ == ('raw'):
        lb = [-large, 0 - large - 1, 0]
        ub = [large, large, large, 1, large]
    elif __switch_0__ == ('natural'):
        lb = [-large - large - 1, 0, 0]
        ub = [large, large, 1, large, large]
    elif __switch_0__ == ('jumpwing'):
        lb = [0 - large, 0, 0, 0]
        ub = [large, large, large, large, large]
    else:
        error('Unknown parameterization')
    return lb, ub

def svi_raw(k=None, param=None, tau=None):
    #SVI - Stochastic Volatility Inspired parameterization of the implied
    #volatility smile. This function implements the raw formulation.
    # Input:
    # * k = (Nx1) = log-moneyness at which to evaluate the total implied
    # variance
    # * param = (5x1) = parameters:
    #   * a = scalar = level of variance
    #   * b = scalar = slope of wings
    #   * m = scalar = translates smile to right
    #   * rho = scalar = counter-clockwise rotation of smile
    #   * sigma = scalar = reduces ATM curvature of the smile
    # Output:
    # * totalvariance = (Nx1) = estimated total variance at k
    # * impliedvolatility = (Nx1) = estimated implied volatility at (k,tau)
    # make sure that input is in column vector format
    k = k.flatten()
    # check that input is consistent
    _assert(numel(param) == 5, ('There have to be five parameters: a, b, m, rho, sigma'))
    if tau is None:
        error('tau needs to be specified in order to return implied volatilities')
    
    a = param[0]
    b = param[1]
    m = param[2]
    rho = param[3]
    sigma = param[4]

    # make sure that parameter restrictions are satisfied
    # assert(b >= 0, 'b has to be non-negative');
    # assert(abs(rho) <= 1, '|rho| has to be smaller than 1');
    # assert(sigma >= 0, 'sigma has to be positive');
    # assert(a + b*sigma*sqrt(1-rho^2)>=0, 'a + b sigma (1-rho^2)^0.5 has to be non-negative');
    # calculate total variance
    totalvariance = a + b * (rho * (k - m) + sqrt(np.square(k - m) + sigma ** 2))
    # if requested, calculate implied volatility
    impliedvolatility = sqrt(totalvariance / tau)
    return totalvariance, impliedvolatility

def svi_surface(k=None, theta=None, rho=None, phifun=None, phi_param=None, tau=None):
    #svi_surface calcualtes the surface SVI free of statis arbitrage.
    #
    # Input:
    # * k = (Kx1) = moneyness at which to evaluate the surface
    # * theta = (Tx1) = ATM variance time at which to evaluate the surface.
    # If theta and k have the same dimensions, then the surface is evaluated 
    # for each given (k,theta) pair. If the dimensions are different, the 
    # function evaluates the surface on the grid given by k and theta
    # * param = (Xx1) = parameters of the surface
    #
    # Output:
    # * totalvariance = (KxT) or (Kx1) = estimated total variance
    # * impliedvolatility = (KxT) or (Kx1) = estimated implied volatility
    # ensure correct input
    k = k.flatten()
    theta = theta.flatten()
    _assert(numel(rho) == 1, 'rho has to be a scalar')
    if tau is None:
        error(('Implied volatility calculation requires tau'))
    
    #evaluate phi
    __switch_0__ = phifun
    if __switch_0__ == 'heston_like':
        phi = heston_like(theta, phi_param)
    elif __switch_0__ == 'power_law':
        phi = power_law(theta, phi_param)
    else:
        error('Incorrect function for phi')
    

    if isequal(size(k), size(theta)):
        totalvariance = np.multiply(theta / 2,
                                    (1 + rho * np.multiply(phi, k) + 
                                     sqrt( np.square(np.multiply(phi, k) + rho) + (1 - rho ** 2))) )
    else:
        T = size(theta)
        K = size(k)
        totalvariance = zeros((K, T))
        for t in np.arange(T):
            totalvariance[:, t] = np.multiply(theta[t] / 2,
                                              (1 + rho * phi[t] * k + 
                                               sqrt(np.square(np.multiply(phi[t], k) + rho) + (1 - rho ** 2))))
        
    
    if isequal(size(k), size(theta), size(tau)):
        impliedvolatility = sqrt(np.divide(np.round(totalvariance, decimals = 4), tau))
    else:
        # expand tau
        tau_expanded = zeros(totalvariance.shape)
        maturities = unique(tau)
        T = size(maturities)
        for t in np.arange(T):
            tau_expanded[:, t] = maturities[t]
            
        impliedvolatility = sqrt(np.divide(totalvariance, tau_expanded))
        
    return totalvariance, impliedvolatility


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

def train_svi_surface(df, S0):
    parameters, theta, maturities, pSSVI = fit_svi_surface(df[impliedVolColumn].values, 
                                                           df["Maturity"].values, 
                                                           df["logMoneyness"].values, 
                                                           'power_law',
                                                           S0 = S0)
    return parameters, theta, maturities, pSSVI

def interpolateWithSSVI(df, parameters, theta, maturities, pSSVI):
    
    logMoneyness = df[df["Maturity"] > 0]["logMoneyness"].values
    tau = df[df["Maturity"] > 0]["Maturity"].values
    impliedATMTotalVar = interp1(maturities, theta, tau, 'linear', "extrapolate")
    impliedTotalVar, impliedVol =  svi_surface(k=logMoneyness, 
                                               theta=impliedATMTotalVar, 
                                               rho=pSSVI[0], 
                                               phifun="power_law", 
                                               phi_param=pSSVI[1:], 
                                               tau=tau)
    return pd.Series(impliedVol,index = df[df["Maturity"] > 0].index)

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
    xSetShifted["Maturity"] = xSetShifted["Maturity"]  + maturityStep
    
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


class SSVIModel:
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
        self.pSSVI = None
        self.interestrate_theta = None
        self.forward_theta = None

    def fit(self, df):
        start = time.time()
        filteredDf = removeMaturityInvalidData(df)
        self.parameters, self.theta, self.maturities, self.pSSVI = fit_svi_surface(filteredDf[impliedVolColumn].values,
                                                                                   filteredDf["Maturity"].values,
                                                                                   filteredDf["logMoneyness"].values,
                                                                                   self.phi,
                                                                                   S0 = self.S0)
        thetaSerie = monotonicInterpolation(self.theta, self.maturities)
        self.theta = thetaSerie.values
        self.maturities = thetaSerie.index.values
        
        #dataSet = dataSet.copy()
        forward = np.exp(-filteredDf["logMoneyness"]) * filteredDf["Strike"]
        # round for float comparaison
        self.forward_theta = forward.groupby("Maturity").mean().values
        #self.forward_theta = (forward[filteredDf["Maturity"].isin(self.maturities)]
        #                      .round(decimals=6)
        #                      .droplevel("Strike")
        #                      .drop_duplicates()
        #                      .loc[self.maturities].values)
        self.interestrate_theta = self.bootstrap.discountIntegral(self.maturities) / self.maturities

        #forward_interp = interp1(self.maturities,
        #                         self.forward_theta,
        #                         self.tau_interp,
        #                         self.interpMethod,
        #                         self.extrapolationMethod)
        #interest_interp = bootstrap.discountShortRate(tau_interp)
        end = time.time()
        print("Training Time : ", end - start)
        return

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
