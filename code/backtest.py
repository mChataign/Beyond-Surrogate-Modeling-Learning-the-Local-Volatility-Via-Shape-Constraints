import numpy as np
import pandas as pd
import dataSetConstruction
import bootstrapping

impliedVolColumn = bootstrapping.impliedVolColumn

################################################################################################## Monte Carlo
def interpolatedMCLocalVolatility(localVol,
                                  strikes,
                                  maturities):
    coordinates =  np.array( dataSetConstruction.customInterpolator(localVol,
                                                                    strikes,
                                                                    maturities) ).flatten()
    return pd.Series(coordinates, index = pd.MultiIndex.from_arrays([strikes, maturities],
                                                                    names=('Strike', 'Maturity')))
    

def MonteCarloPricer(S0,
                     Strike, 
                     Maturity, 
                     bootstrap,
                     randomIncrement, 
                     nbTimeStep, 
                     volLocaleFunction):
  singleMaturity = np.unique(Maturity)[0]
  time_grid = np.linspace(0, singleMaturity, int(nbTimeStep + 1))
  timeStep = singleMaturity / nbTimeStep
  gaussianNoise = np.sqrt(timeStep) * randomIncrement
  nbPaths = gaussianNoise.shape[1]

  logReturn = np.zeros((nbTimeStep + 1, nbPaths))
  logReturn[0,:] = 0

  for i in range(nbTimeStep) :
      t = time_grid[i]

      St = S0 * np.exp(logReturn[i,:])
      volLocale = volLocaleFunction(St, np.ones(nbPaths) * t)

      mu = bootstrap.discountShortRate(t) - bootstrap.dividendShortRate(t)
      drift = np.ones(nbPaths) * (mu - np.square(volLocale) / 2.0) 
      logReturn[i + 1, :] = logReturn[i,:] + drift * timeStep + gaussianNoise[i,:] * volLocale
  SFinal = S0 * np.exp(logReturn[-1, :])
  MCresult = lambda x : (np.mean(np.maximum(x - SFinal, 0)), np.std(np.maximum(x - SFinal, 0)))
  return Strike.apply(MCresult)

def MonteCarloPricerVectorized(S, 
                               dataSet,
                               bootstrap,
                               nbPaths, 
                               nbTimeStep, 
                               volLocaleFunction):
  gaussianNoise = np.random.normal(scale = 1.0, size=(nbTimeStep, nbPaths))
  func = lambda x : MonteCarloPricer(S, x["Strike"], x["Maturity"], 
                                     bootstrap,
                                     gaussianNoise, nbTimeStep, 
                                     volLocaleFunction)
  res = dataSet.groupby(level="Maturity", as_index = False).apply(func).droplevel(0).sort_index()
  priceMC = res.map(lambda x : x[0]) * np.exp(-bootstrap.discountIntegral(res.index.get_level_values("Maturity")))
  stdMC = res.map(lambda x : x[1])
  return pd.DataFrame(np.vstack([priceMC.values, stdMC.values]).T,
                      columns=["Price", "stdPrice"],
                      index=priceMC.index)

def loadMCPrices(fileName, parseHeader=None):
    renameDict = {1: "Maturity", 
                  0 : "Strike"}
    mcDf = pd.read_csv(fileName, 
                       decimal=".", 
                       header=parseHeader).apply(pd.to_numeric).rename(columns=renameDict)
    mcDf["Maturity"] = mcDf["Maturity"].round(decimals=3)
    return mcDf.set_index(["Strike","Maturity"])


################################################################################################## PDE
def triDiagonalSolver(a, b, c, d):

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

def CFLCondition(time_step, space_step, b, sigma):
  tmp = (space_step ** 2) / (sigma**2 + b * space_step ** 2) 
  return (time_step <= tmp)

def buildTriDiagonalMatrix(a, b, c, size):
  diagInf = np.diag(a * np.ones(size-1),k=-1)
  diag = np.diag(b * np.ones(size),k=0)
  diagSup = np.diag(c * np.ones(size-1),k=1)
  return diagInf + diag + diagSup


def priceEuropeanCallCranckNicholson(time_step, S,
                                     bootstrap,
                                     volLocaleFunction,
                                     t, T, K,
                                     sigmaSup):
    nbTimePoint = int(T/time_step) + 1 #Number of points in the time grid
    nbSpacePoint = 101 #Number of points in space grid
    timeGrid = np.linspace(0,T,num=nbTimePoint)
    
    b = bootstrap.discountShortRate(timeGrid) - bootstrap.dividendShortRate(timeGrid)
    bSup = np.amax(b)
    
    #Discretize logspace
    space_step = 2 * (sigmaSup * time_step ** 0.5) / (( 1 - bSup * time_step ) ** 0.5)
    space_grid = np.exp((np.arange(nbSpacePoint) - 50) * space_step) * S
    
    theta = 0.5
    grid = np.zeros((nbSpacePoint, nbTimePoint, K.size))
    
    
    #Compute the terminal condition : payoff of a call
    SMax = np.amax(space_grid)
    SMin = np.amin(space_grid)
    
    MdiagInf = np.zeros((nbSpacePoint - 1,nbTimePoint))
    MdiagSup = np.zeros((nbSpacePoint - 1,nbTimePoint))
    Mdiag = np.zeros((nbSpacePoint,nbTimePoint))
    N = np.zeros((nbSpacePoint, nbSpacePoint, nbTimePoint))
    w = np.zeros((nbSpacePoint, nbTimePoint, K.size))
    
    for i in range(nbTimePoint):
        tIter = timeGrid[::-1][i]
        bIter = b[::-1][i]
        
        sigma = volLocaleFunction(space_grid, tIter * np.ones_like(space_grid))
        log_drift = b[::-1][i] - np.square(sigma)/2
        alpha = sigma[1:] ** 2 / (2 * space_step ** 2) - log_drift[1:] / (2 * space_step)
        MdiagInf[:,i] = (-theta) * time_step * alpha
        NdiagInf = (1-theta) * time_step * alpha
        
        beta = - (np.square(sigma) / (space_step ** 2) + bIter)
        Mdiag[:,i] = 1 - (theta * time_step * beta)
        Ndiag = 1 + (1-theta) * time_step * beta

        gamma = np.ravel(sigma[:-1] ** 2 / (2 * space_step ** 2) + log_drift[:-1] / (2 * space_step))   
        MdiagSup[:,i] = (-theta) * time_step * gamma
        NdiagSup = (1-theta) * time_step * gamma
        
        N[:,:,i] = np.diag(NdiagInf,k=-1) + np.diag(Ndiag,k=0) + np.diag(NdiagSup,k=1)  
        
        #dirichlet condition
        w[0, i, :] = (theta * gamma[0] * (K * np.exp(- b[i] * (T - tIter) ) - SMin) 
                      + (1-theta) * gamma[0] * (K * np.exp(- b[i-1] * (T - timeGrid[::-1][i-1]) )  - SMin))
        w[-1, i, :] = np.zeros_like(K)
    
    
    for i in range(nbTimePoint): #From the end to the beginning 
        if i==0 :
            for j in range(K.size):
                payoff = np.maximum(K[j] - space_grid, 0)
                grid[:, 0, j] = payoff
        else :
            tIter = timeGrid[::-1][i]
            
            for j in range(K.size):
                y = np.matmul(N[:,:,i], grid[:, i-1, j]) + time_step * w[:, i, j]
                grid[:, i, j] = triDiagonalSolver(MdiagInf[:,i], Mdiag[:,i], MdiagSup[:,i], y)
    
    #Return price and delta
    price = grid[50, -1, :]
    delta = (grid[51, -1, :] - grid[49, -1, :])/(2*space_step*S)
    return (price,delta)
    
def pdeRepricing(K, T, S0, volLocaleFunction, bootstrap, sigmaSup):
    S = S0
    t = 0
    time_step = 1/360
    return pd.Series(priceEuropeanCallCranckNicholson(time_step, S, bootstrap,
                                                      volLocaleFunction, t, T, K, sigmaSup)[0],
                     index = K.index)

def PDEPricerVectorized(dataSet, S0,
                        volLocaleFunction,
                        bootstrap):
    
    #Estimation of local volatility supremum
    #strikeLow = min(dataSet["Strike"].min(),
    #                dataSetTest["Strike"].min())
    strikeLow = 0
    strikeUp = dataSet["Strike"].max()
    strikeGrid = np.linspace(strikeLow, strikeUp, 100)
    
    matLow = 0
    matUp = dataSet["Maturity"].max()
    matGrid = np.linspace(matLow, matUp, 100)
    
    volLocaleGrid = np.meshgrid(strikeGrid, matGrid)
    
    sigmaSup = np.amax(volLocaleFunction(volLocaleGrid[0].flatten(), 
                                         volLocaleGrid[1].flatten()))
                                         
    func = lambda x : pdeRepricing(x["Strike"], x["Maturity"].unique()[0], S0,
                                   volLocaleFunction, bootstrap, sigmaSup)
    return dataSet.groupby(level="Maturity", as_index = False).apply(func).droplevel(0).sort_index()

def interpolatedMCLocalVolatility(localVol, strikes, maturities):
    coordinates =  np.array( dataSetConstruction.customInterpolator(localVol, strikes, maturities) ).flatten()
    return pd.Series(coordinates, index = pd.MultiIndex.from_arrays([strikes, maturities],
                                                                    names=('Strike', 'Maturity')))


def rmse(a,b, relative=False):
    aFiltered = a[a.index.get_level_values("Maturity") > 0]
    bFiltered = b[b.index.get_level_values("Maturity") > 0]
    gap = np.abs(aFiltered-bFiltered)/aFiltered if relative else (aFiltered-bFiltered)
    return np.sqrt(np.mean(np.square(gap)))


######################################################################## Sanity check with implied volatilities
def MonteCarloPricerImplicit(S,
                             Strike,
                             Maturity,
                             bootstrap,
                             nbPaths,
                             nbTimeStep,
                             impliedVol):
  time_grid = np.linspace(0, Maturity, int(nbTimeStep + 1))
  timeStep = Maturity / nbTimeStep
  gaussianNoise = np.random.normal(scale = np.sqrt(timeStep), size=(nbTimeStep, nbPaths))

  logReturn = np.zeros((nbTimeStep + 1, nbPaths))
  logReturn[0,:] = 0

  for i in range(nbTimeStep) :
      t = time_grid[i]

      St = S0 * np.exp(logReturn[i,:])
      volLocale = impliedVol

      mu = bootstrap.discountShortRate(t) - bootstrap.dividendShortRate(t)
      drift = np.ones(nbPaths) * (mu - np.square(volLocale) / 2.0)
      logReturn[i + 1, :] = logReturn[i,:] + drift * timeStep + gaussianNoise[i,:] * volLocale
  SFinal = S0 * np.exp(logReturn[-1, :])
  return np.mean(np.maximum(Strike - SFinal, 0))

def MonteCarloPricerVectorizedImplicit(S,
                                       dataSet,
                                       bootstrap,
                                       nbPaths,
                                       nbTimeStep):
  func = lambda x : MonteCarloPricerImplicit(S, x["Strike"], x["Maturity"], bootstrap, nbPaths, nbTimeStep, x[impliedVolColumn])
  return dataSet.apply(func, axis=1) * np.exp(-bootstrap.discountIntegral(dataSet.index.get_level_values("Maturity")))
