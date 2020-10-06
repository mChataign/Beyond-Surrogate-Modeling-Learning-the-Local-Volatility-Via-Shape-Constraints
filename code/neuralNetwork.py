#Import tensorflow for 1.x version
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras import Model
import tensorflow.compat.v1.keras.backend as K
import tensorflow.compat.v1.keras.activations as Act
from functools import partial
import time
import dataSetConstruction
import bootstrapping
import numpy as np
import pandas as pd
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

activateScaling = False
transformCustom = dataSetConstruction.transformCustomMinMax if activateScaling else dataSetConstruction.transformCustomId
inverseTransform = dataSetConstruction.inverseTransformMinMax if activateScaling else dataSetConstruction.inverseTransformId
inverseTransformColumn = dataSetConstruction.inverseTransformColumnMinMax if activateScaling else dataSetConstruction.inverseTransformColumnId
inverseTransformColumnGreeks = dataSetConstruction.inverseTransformColumnGreeksMinMax if activateScaling else dataSetConstruction.inverseTransformColumnGreeksId

layerFactory = {}

def resetTensorflow():
    tf.reset_default_graph()
    layerFactory.clear()
    return
# Format result from training step
def evalAndFormatResult(price, loss, dataSet, scaler):
    scaledPredPrice = pd.Series(price.flatten(), index=dataSet.index).rename("Price")
    predPrice = inverseTransformColumn(dataSet, scaler, scaledPredPrice)
    lossEpochSerie = pd.Series(loss)
    return predPrice, lossEpochSerie


# Format result from training step when local volatility is computed
def evalAndFormatDupireResult(price, volDupire, theta, gamma,
                              dupireVar, loss, dataSet, scaler):
    predPrice, lossEpoch = evalAndFormatResult(price, loss, dataSet, scaler)

    predDupire = pd.Series(volDupire.flatten(), index=dataSet.index).rename("Dupire")

    scaledTheta = pd.Series(theta.flatten(), index=dataSet.index).rename("Theta")
    predTheta = inverseTransformColumnGreeks(dataSet, scaler, scaledTheta,
                                             "Price", "Maturity")

    scaledGammaK = pd.Series(gamma.flatten(), index=dataSet.index).rename("GammaK")
    predGammaK = inverseTransformColumnGreeks(dataSet, scaler, scaledGammaK,
                                              "Price", "ChangedStrike", order=2)

    return predPrice, predDupire, predTheta, predGammaK, lossEpoch

#Penalization for pseudo local volatility
def intervalRegularization(localVariance, vegaRef, hyperParameters):
  lowerVolBound = hyperParameters["DupireVolLowerBound"]
  upperVolBound = hyperParameters["DupireVolUpperBound"]
  no_nans = tf.clip_by_value(localVariance, 0, hyperParameters["DupireVarCap"])
  reg = tf.nn.relu(tf.square(lowerVolBound) - no_nans) + tf.nn.relu(no_nans - tf.square(upperVolBound))
  lambdas = hyperParameters["lambdaLocVol"] / tf.reduce_mean(vegaRef)
  return lambdas * tf.reduce_mean(tf.boolean_mask(reg, tf.is_finite(reg)))

#Add above regularization to the list of penalization
def addDupireRegularisation(priceTensor, tensorList, penalizationList,
                            formattingResultFunction, vegaRef, hyperParameters):
    updatedPenalizationList = penalizationList + [intervalRegularization(tensorList[-1], vegaRef, hyperParameters)]
    return priceTensor, tensorList, updatedPenalizationList, formattingResultFunction

#Mini-batch sampling methods for large datasets
def selectMiniBatchWithoutReplacement(dataSet, batch_size):
    nbObs = dataSet.shape[0]
    idx = np.arange(nbObs)
    np.random.shuffle(idx)
    nbBatches = int(np.ceil(nbObs/batch_size))
    xBatchList = []
    lastBatchIndex = 0
    for i in range(nbBatches):
        firstBatchIndex = i*batch_size
        lastBatchIndex = (i+1)*batch_size
        xBatchList.append(dataSet.iloc[idx[firstBatchIndex:lastBatchIndex],:])
    xBatchList.append(dataSet.iloc[idx[lastBatchIndex:],:])
    return xBatchList

def selectMiniBatchWithReplacement(dataSet, batch_size):
    nbObs = dataSet.shape[0]
    nbBatches = int(np.ceil(nbObs/batch_size)) + 1
    xBatchList = []
    lastBatchIndex = 0
    for i in range(nbBatches):
        idx = np.random.randint(nbObs, size = batch_size)
        xBatchList.append(dataSet.iloc[idx,:])
    return xBatchList


# Train neural network with a decreasing rule for learning rate
# NNFactory :  function creating the architecture
# dataSet : training data
# activateRegularization : boolean, if true add bound penalization to dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name under which tensorflow model is saved
def create_train_model(NNFactory,
                       dataSet,
                       activateRegularization,
                       hyperparameters,
                       scaler,
                       modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    fixedLearningRate = (None if hyperparameters["FixedLearningRate"] else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Strike = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    StrikePenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    factorPrice = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    vegaRef = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRef')
    vegaRefPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRefPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])

    # Get scaling for strike
    colStrikeIndex = dataSet.columns.get_loc("ChangedStrike")
    maxColFunction = scaler.data_max_[colStrikeIndex]
    minColFunction = scaler.data_min_[colStrikeIndex]
    scF = (maxColFunction - minColFunction)
    scaleTensor = tf.constant(scF, dtype=tf.float32)
    strikeMinTensor = tf.constant(minColFunction, dtype=tf.float32)

    #Grid on which is applied Penalization
    t = np.linspace(0, #scaler.data_min_[dataSet.columns.get_loc("Maturity")],
                    4 * scaler.data_max_[dataSet.columns.get_loc("Maturity")],
                    num=100)
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    #t = np.linspace(0, 4, num=100)

    k = np.linspace((- 0.5 * minColFunction) / scF,
                    (2.0 * maxColFunction - minColFunction) / scF,
                    num=50)
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    if activateRegularization:  # Add pseudo local volatility regularisation
        price_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       Strike,
                       Maturity,
                       scaleTensor,
                       strikeMinTensor,
                       vegaRef,
                       hyperparameters),
            vegaRef,
            hyperparameters)
        price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       StrikePenalization,
                       MaturityPenalization,
                       scaleTensor,
                       strikeMinTensor,
                       vegaRefPenalization,
                       hyperparameters),
            vegaRefPenalization,
            hyperparameters)
    else:
        price_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                        Strike,
                                                                                        Maturity,
                                                                                        scaleTensor,
                                                                                        strikeMinTensor,
                                                                                        vegaRef,
                                                                                        hyperparameters)
        price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                            StrikePenalization,
                                                                                            MaturityPenalization,
                                                                                            scaleTensor,
                                                                                            strikeMinTensor,
                                                                                            vegaRefPenalization,
                                                                                            hyperparameters)

    price_pred_tensor_sc = tf.multiply(factorPrice, price_pred_tensor)
    TensorList[0] = price_pred_tensor_sc

    # Define a loss function
    pointwiseError = tf.reduce_mean(tf.abs(price_pred_tensor_sc - y) / vegaRef)
    errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
    loss = tf.log(tf.reduce_mean(errors))

    # Define a train operation to minimize the loss
    lr = learningRate

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learningRateTensor,
    #                                       momentum=0.9,
    #                                       use_nesterov=True)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRateTensor,
    #                                      momentum=0.9,
    #                                      decay=0.9)
    train = optimizer.minimize(loss)
    # optimizer = tf.keras.optimizers.Nadam(learning_rate=learningRateTensor)
    # train = optimizer.minimize(loss, var_list = tf.trainable_variables())

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = dataSet.shape[0]
    scaledInput = dataSetConstruction.transformCustomMinMax(dataSet, scaler)

    maturity = dataSet["Maturity"].values.reshape(n, 1)
    loss_serie = []

    def createFeedDict(batch):
        batchSize = batch.shape[0]
        feedDict = {Strike: scaledInput["ChangedStrike"].values.reshape(batchSize, 1),#scaledInput["ChangedStrike"].loc[batch.index].values.reshape(batchSize, 1),
                    Maturity: batch["Maturity"].values.reshape(batchSize, 1),
                    y: batch["Price"].values.reshape(batchSize, 1),
                    StrikePenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    factorPrice: batch["DividendFactor"].values.reshape(batchSize, 1),
                    learningRateTensor: learningRate,
                    vegaRef: np.ones_like(batch["VegaRef"].values.reshape(batchSize, 1)),
                    vegaRefPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    # Learning rate is divided by 10 if no imporvement is observed for training loss after "patience" epochs
    def updateLearningRate(iterNumber, lr, lrEpoch):
        if not activateLearningDecrease:
            print("Constant learning rate, stop training")
            return False, lr, lrEpoch
        if learningRate > finalLearningRate:
            lr *= 0.1
            lrEpoch = iterNumber
            saver.restore(sess, modelName)
            print("Iteration : ", lrEpoch, "new learning rate : ", lr)
        else:
            print("Last Iteration : ", lrEpoch, "final learning rate : ", lr)
            return False, lr, lrEpoch
        return True, lr, lrEpoch

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid) : ", currentBestPenalizations1)
        return

    for i in range(nbEpoch):
        miniBatchList = [dataSet]
        penalizationResult = sess.run(penalizationList, feed_dict=epochFeedDict)
        lossResult = sess.run(pointwiseError, feed_dict=epochFeedDict)

        # miniBatchList = selectMiniBatchWithoutReplacement(dataSet, batch_size)
        for k in range(len(miniBatchList)):
            batchFeedDict = createFeedDict(miniBatchList[k])
            sess.run(train, feed_dict=batchFeedDict)

        loss_serie.append(sess.run(loss, feed_dict=epochFeedDict))

        if (len(loss_serie) < 2) or (loss_serie[-1] <= min(loss_serie)):
            # Save model as model is improved
            saver.save(sess, modelName)
        if (np.isnan(loss_serie[-1]) or  # Unstable model
                ((i - learningRateEpoch >= patience) and (min(loss_serie[-patience:]) > min(
                    loss_serie)))):  # No improvement for training loss during the latest 100 iterations
            continueTraining, learningRate, learningRateEpoch = updateLearningRate(i, learningRate, learningRateEpoch)
            if continueTraining:
                evalBestModel()
            else:
                break
    saver.restore(sess, modelName)

    evalBestModel()

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()
    end = time.time()
    print("Training Time : ", end - start)
    lossEpochSerie = pd.Series(loss_serie)
    lossEpochSerie.to_csv("loss" + modelName + ".csv")

    return formattingFunction(*evalList, loss_serie, dataSet, scaler)


# Evaluate neural network without training, it restores parameters obtained from a pretrained model
# NNFactory :  function creating the neural architecture
# dataSet : dataset on which neural network is evaluated
# activateRegularization : boolean, if true add bound penalization for dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name of tensorflow model to restore
def create_eval_model(NNFactory,
                      dataSet,
                      activateRegularization,
                      hyperparameters,
                      scaler,
                      modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]

    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Strike = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    StrikePenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    factorPrice = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    vegaRef = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRef')
    vegaRefPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRefPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])

    # Get scaling for strike
    colStrikeIndex = dataSet.columns.get_loc("ChangedStrike")
    maxColFunction = scaler.data_max_[colStrikeIndex]
    minColFunction = scaler.data_min_[colStrikeIndex]
    scF = (maxColFunction - minColFunction)
    scaleTensor = tf.constant(scF, dtype=tf.float32)
    strikeMinTensor = tf.constant(minColFunction, dtype=tf.float32)

    #Grid on which is applied Penalization
    t = np.linspace(0,#scaler.data_min_[dataSet.columns.get_loc("Maturity")],
                    4 * scaler.data_max_[dataSet.columns.get_loc("Maturity")],
                    num=100)
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    #t = np.linspace(0, 4, num=100)

    k = np.linspace((- 0.5 * minColFunction) / scF,
                    (2.0 * maxColFunction - minColFunction) / scF,
                    num=50)
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    if activateRegularization:
        price_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       Strike,
                       Maturity,
                       scaleTensor,
                       strikeMinTensor,
                       vegaRef,
                       hyperparameters,
                       IsTraining=False),
            vegaRef,
            hyperparameters)
        price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       StrikePenalization,
                       MaturityPenalization,
                       scaleTensor,
                       strikeMinTensor,
                       vegaRefPenalization,
                       hyperparameters),
            vegaRefPenalization,
            hyperparameters)
    else:
        price_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                        Strike,
                                                                                        Maturity,
                                                                                        scaleTensor,
                                                                                        strikeMinTensor,
                                                                                        vegaRef,
                                                                                        hyperparameters,
                                                                                        IsTraining=False)
        price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                            StrikePenalization,
                                                                                            MaturityPenalization,
                                                                                            scaleTensor,
                                                                                            strikeMinTensor,
                                                                                            vegaRefPenalization,
                                                                                            hyperparameters)

    price_pred_tensor_sc = tf.multiply(factorPrice, price_pred_tensor)
    TensorList[0] = price_pred_tensor_sc

    # Define a loss function
    pointwiseError = tf.reduce_mean(tf.abs(price_pred_tensor_sc - y) / vegaRef)
    errors = tf.add_n([pointwiseError] + penalizationList)
    loss = tf.log(tf.reduce_mean(errors))

    # Define a train operation to minimize the loss
    lr = learningRate

    # optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    # train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = dataSet.shape[0]
    scaledInput = dataSetConstruction.transformCustomMinMax(dataSet, scaler)

    maturity = dataSet["Maturity"].values.reshape(n, 1)
    loss_serie = []

    def createFeedDict(batch):
        batchSize = batch.shape[0]
        feedDict = {Strike: scaledInput["ChangedStrike"].values.reshape(batchSize, 1),#scaledInput["ChangedStrike"].loc[batch.index].values.reshape(batchSize, 1),
                    Maturity: batch["Maturity"].values.reshape(batchSize, 1),
                    y: batch["Price"].values.reshape(batchSize, 1),
                    StrikePenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    factorPrice: batch["DividendFactor"].values.reshape(batchSize, 1),
                    learningRateTensor: learningRate,
                    vegaRef: np.ones_like(batch["VegaRef"].values.reshape(batchSize, 1)),
                    vegaRefPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid) : ", currentBestPenalizations1)
        return

    saver.restore(sess, modelName)

    evalBestModel()

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return formattingFunction(*evalList, [0], dataSet, scaler)



############################################################################# Evaluate local volatility

def evalVolLocale(NNFactory,
                  strikes,
                  maturities,
                  dataSet,
                  hyperParameters,
                  scaler,
                  modelName="bestModel"):
    hidden_nodes = hyperParameters["nbUnits"]

    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Strike = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    StrikePenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    factorPrice = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    vegaRef = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRef')
    vegaRefPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRefPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])

    # Get scaling for strike
    colStrikeIndex = dataSet.columns.get_loc("ChangedStrike")
    maxColFunction = scaler.data_max_[colStrikeIndex]
    minColFunction = scaler.data_min_[colStrikeIndex]
    scF = (maxColFunction - minColFunction)
    scaleTensor = tf.constant(scF, dtype=tf.float32)
    strikeMinTensor = tf.constant(minColFunction, dtype=tf.float32)

    #Grid on which is applied Penalization
    t = np.linspace(0, #scaler.data_min_[dataSet.columns.get_loc("Maturity")],
                    4 * scaler.data_max_[dataSet.columns.get_loc("Maturity")],
                    num=100)
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    #t = np.linspace(0, 4, num=100)

    k = np.linspace((- 0.5 * minColFunction) / scF,
                    (2.0 * maxColFunction - minColFunction) / scF,
                    num=50)
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    price_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                    Strike,
                                                                                    Maturity,
                                                                                    scaleTensor,
                                                                                    strikeMinTensor,
                                                                                    vegaRef,
                                                                                    hyperParameters,
                                                                                    IsTraining=False)  # one hidden layer
    price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                        StrikePenalization,
                                                                                        MaturityPenalization,
                                                                                        scaleTensor,
                                                                                        strikeMinTensor,
                                                                                        vegaRefPenalization,
                                                                                        hyperparameters)

    price_pred_tensor_sc = tf.multiply(factorPrice, price_pred_tensor)
    TensorList[0] = price_pred_tensor_sc

    # Define a loss function
    pointwiseError = tf.reduce_mean(tf.abs(price_pred_tensor_sc - y) / vegaRef)
    errors = tf.add_n([pointwiseError] + penalizationList1)
    loss = tf.log(tf.reduce_mean(errors))

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = strikes.shape[0]
    changedVar = changeOfVariable(strikes, maturities)
    scaledStrike = (changedVar[0] - minColFunction) / scF
    dividendFactor = changedVar[1]

    def createFeedDict(s, t, d):
        batchSize = s.shape[0]
        feedDict = {Strike: np.reshape(s, (batchSize, 1)),
                    Maturity: np.reshape(t, (batchSize, 1)),
                    factorPrice: np.reshape(d, (batchSize, 1)),
                    StrikePenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization,1),
                    vegaRefPenalization : np.ones_like(np.expand_dims(kPenalization, 1)),
                    vegaRef: np.ones((batchSize, 1))}
        return feedDict

    epochFeedDict = createFeedDict(scaledStrike, maturities, dividendFactor)

    saver.restore(sess, modelName)

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return pd.Series(evalList[1].flatten(),
                     index=pd.MultiIndex.from_arrays([strikes, maturities], names=('Strike', 'Maturity')))

# Soft constraints for strike convexity and strike/maturity monotonicity
def arbitragePenaltiesPrice(priceTensor, strikeTensor, maturityTensor, scaleTensor, vegaRef, hyperparameters):
    dK = tf.gradients(priceTensor, strikeTensor, name="dK")
    hK = tf.gradients(dK[0], strikeTensor, name="hK") / tf.square(scaleTensor)
    theta = tf.gradients(priceTensor, maturityTensor, name="dT")

    lambdas = hyperparameters["lambdaSoft"] / tf.reduce_mean(vegaRef)
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    grad_penalty = lambdas * tf.reduce_mean(tf.nn.relu(-theta[0] + lowerBoundTheta))
    hessian_penalty = lambdas * hyperparameters["lowerBoundGamma"] * tf.reduce_mean(
        tf.nn.relu(-hK[0] + lowerBoundGamma))

    return [grad_penalty, hessian_penalty]

#Dupire formula from exact derivative computation
def dupireFormula(HessianStrike,
                  GradMaturity,
                  Strike,
                  scaleTensor,
                  strikeMinTensor,
                  IsTraining=True):
  twoConstant = tf.constant(2.0)
  dupireVar = tf.math.divide(tf.math.divide(tf.math.scalar_mul(twoConstant,GradMaturity),
                                            HessianStrike),
                             tf.square(Strike + strikeMinTensor / scaleTensor))
  #Initial weights of neural network can be random which lead to negative dupireVar
  dupireVolTensor = tf.sqrt(dupireVar)
  return dupireVolTensor, dupireVar

#Dupire formula with derivative obtained from native tensorflow algorithmic differentiation
def rawDupireFormula(priceTensor,
                     adjustedStrikeTensor,
                     maturityTensor,
                     scaleTensor,
                     strikeMinTensor,
                     IsTraining=True):
  batchSize = tf.shape(adjustedStrikeTensor)[0]
  dK = tf.reshape(tf.gradients(priceTensor, adjustedStrikeTensor, name="dK")[0], shape=[batchSize,-1])
  hK = tf.reshape(tf.gradients(dK, adjustedStrikeTensor, name="hK")[0], shape=[batchSize,-1])
  dupireDenominator = tf.square(adjustedStrikeTensor + strikeMinTensor / scaleTensor) * hK

  dT = tf.reshape(tf.gradients(priceTensor,maturityTensor,name="dT")[0], shape=[batchSize,-1])

  #Initial weights of neural network can be random which lead to negative dupireVar
  dupireVar = 2 * dT / dupireDenominator
  dupireVol = tf.sqrt(dupireVar)
  return  dupireVol, dT, hK / tf.square(scaleTensor), dupireVar

def saveDataModel(predictedPrices, volLocal, impliedVol, name):
  predictedPrices.to_csv("Price" + name + ".csv")
  volLocal.to_csv("localVol" + name + ".csv")
  impliedVol.to_csv("impliedVol" + name + ".csv")
  return

############################################################################# Tools function for Neural network architecture
############################################################################# Hard constraints

# Initilize weights as positive
def positiveKernelInitializer(shape,
                              dtype=None,
                              partition_info=None):
    return tf.abs(tf.keras.initializers.normal()(shape, dtype=dtype, partition_info=partition_info))


# Soft convex layer
def convexLayer(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.softplus(layer)


# Soft monotonic layer
def monotonicLayer(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor,
                                units=n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.sigmoid(layer)


# Soft convex layer followed by output layer for regression
def convexOutputLayer(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=2 * n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                activation='softplus')

        layer = tf.layers.dense(layer,
                                units=1,
                                kernel_initializer=positiveKernelInitializer,
                                activation='softplus')

        return layer


# Neural network factory for Hybrid approach : splitted network with soft contraints
def NNArchitectureConstrained(n_units,
                              strikeTensor,
                              maturityTensor,
                              scaleTensor,
                              strikeMinTensor,
                              vegaRef,
                              hyperparameters,
                              IsTraining=True):
    # First splitted layer
    hidden1S = convexLayer(n_units=n_units,
                           tensor=strikeTensor,
                           isTraining=IsTraining,
                           name="Hidden1S")

    hidden1M = monotonicLayer(n_units=n_units,
                              tensor=maturityTensor,
                              isTraining=IsTraining,
                              name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second and output layer
    out = convexOutputLayer(n_units=n_units,
                            tensor=hidden1,
                            isTraining=IsTraining,
                            name="Output")
    # Soft constraints
    penaltyList = arbitragePenaltiesPrice(out, strikeTensor,
                                          maturityTensor,
                                          scaleTensor,
                                          vegaRef,
                                          hyperparameters)

    return out, [out], penaltyList, evalAndFormatResult


############################################################################# Unconstrained architecture


# Unconstrained dense layer
def unconstrainedLayer(n_units, tensor, isTraining, name, activation=K.softplus):
    with tf.name_scope(name):
        if name not in layerFactory :
            layerFactory[name] = tf.keras.layers.Dense(n_units,
                                                       activation=activation,
                                                       kernel_initializer=tf.keras.initializers.he_normal())
        #layer = tf.layers.dense(tensor,
        #                        units=n_units,
        #                        activation=activation,
        #                        kernel_initializer=tf.keras.initializers.he_normal())
        return layerFactory[name](tensor)


# Factory for unconstrained network
def NNArchitectureUnconstrained(n_units,
                                strikeTensor,
                                maturityTensor,
                                scaleTensor,
                                strikeMinTensor,
                                vegaRef,
                                hyperparameters,
                                IsTraining=True):
    inputLayer = tf.concat([strikeTensor, maturityTensor], axis=-1)

    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")

    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)

    return out, [out], [], evalAndFormatResult

############################################################################# Manual differentiation

def exact_derivatives(Strike, Maturity):
    w1K = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
    w1T = tf.get_default_graph().get_tensor_by_name('dense_1/kernel:0')
    w2 = tf.get_default_graph().get_tensor_by_name('dense_2/kernel:0')
    w3 = tf.get_default_graph().get_tensor_by_name('dense_3/kernel:0')

    b1K = tf.get_default_graph().get_tensor_by_name('dense/bias:0')
    b1T = tf.get_default_graph().get_tensor_by_name('dense_1/bias:0')
    b2 = tf.get_default_graph().get_tensor_by_name('dense_2/bias:0')
    b3 = tf.get_default_graph().get_tensor_by_name('dense_3/bias:0')

    Z1K = tf.nn.softplus(tf.matmul(Strike, w1K) + b1K)
    Z1T = tf.nn.sigmoid(tf.matmul(Maturity, w1T) + b1T)

    Z = tf.concat([Z1K, Z1T], axis=-1)
    I2 = tf.matmul(Z, w2) + b2
    Z2 = tf.nn.softplus(I2)
    I3 = tf.matmul(Z2, w3) + b3
    F = tf.nn.softplus(I3)

    D1K = tf.nn.sigmoid(tf.matmul(Strike, w1K) + b1K)
    I2K = tf.multiply(D1K, w1K)
    Z2K = tf.concat([I2K, tf.scalar_mul(tf.constant(0.0), I2K)], axis=-1)

    dI2dK = tf.matmul(Z2K, w2)
    Z2w3 = tf.multiply(tf.nn.sigmoid(I2), dI2dK)
    dI3dK = tf.matmul(Z2w3, w3)
    dF_dK = tf.multiply(tf.nn.sigmoid(I3), dI3dK)

    D1T = sigmoidGradient(tf.matmul(Maturity, w1T) + b1T)
    I2T = tf.multiply(D1T, w1T)
    Z2T = tf.concat([tf.scalar_mul(tf.constant(0.0), I2T), I2T], axis=-1)

    dI2dT = tf.matmul(Z2T, w2)
    Z2w3 = tf.multiply(tf.nn.sigmoid(I2), dI2dT)
    dI3dT = tf.matmul(Z2w3, w3)
    dF_dT = tf.multiply(tf.nn.sigmoid(I3), dI3dT)

    d2F_dK2 = tf.multiply(sigmoidGradient(I3), tf.square(dI3dK))
    DD1K = sigmoidGradient(tf.matmul(Strike, w1K) + b1K)
    w1K2 = tf.multiply(w1K, w1K)
    ID2K = tf.multiply(DD1K, w1K2)
    ZD2K = tf.concat([ID2K, tf.scalar_mul(tf.constant(0.0), ID2K)], axis=-1)

    d2I2_dK2 = tf.matmul(ZD2K, w2)

    ZD2 = tf.multiply(sigmoidGradient(I2), tf.square(dI2dK))
    ZD2 += tf.multiply(tf.nn.sigmoid(I2), d2I2_dK2)
    d2I3dK2 = tf.matmul(ZD2, w3)

    d2F_dK2 += tf.multiply(tf.nn.sigmoid(I3), d2I3dK2)

    return dF_dT, dF_dK, d2F_dK2


############################################################################# Soft constraint
# Tools functions for neural architecture


# Neural network architecture
def convexLayer1(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.softplus(layer), layer


def monotonicLayer1(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor,
                                units=n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.sigmoid(layer), layer


def convexOutputLayer1(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=2 * n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                activation='softplus')

        layer = tf.layers.dense(layer,
                                units=1,
                                kernel_initializer=positiveKernelInitializer,
                                activation='softplus')

        return layer, layer


def convexLayerHybrid1(n_units,
                       tensor,
                       isTraining,
                       name,
                       activationFunction2=Act.softplus,
                       activationFunction1=Act.exponential,
                       isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_initializer=positiveKernelInitializer)
        l1, l2 = tf.split(layer, 2, 1)
        output = tf.concat([activationFunction1(l1), activationFunction2(l2)], axis=-1)
        return output, layer


def sigmoidGradient(inputTensor):
    return tf.nn.sigmoid(inputTensor) * (1 - tf.nn.sigmoid(inputTensor))


def sigmoidHessian(inputTensor):
    return (tf.square(1 - tf.nn.sigmoid(inputTensor)) -
            tf.nn.sigmoid(inputTensor) * (1 - tf.nn.sigmoid(inputTensor)))


def NNArchitectureConstrainedDupire(n_units,
                                    strikeTensor,
                                    maturityTensor,
                                    scaleTensor,
                                    strikeMinTensor,
                                    vegaRef,
                                    hyperparameters,
                                    IsTraining=True):
    # First splitted layer
    hidden1S, layer1S = convexLayer1(n_units=n_units,
                                     tensor=strikeTensor,
                                     isTraining=IsTraining,
                                     name="Hidden1S")

    hidden1M, layer1M = monotonicLayer1(n_units=n_units,
                                        tensor=maturityTensor,
                                        isTraining=IsTraining,
                                        name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second layer and output layer
    out, layer = convexOutputLayer1(n_units=n_units,
                                    tensor=hidden1,
                                    isTraining=IsTraining,
                                    name="Output")

    dT, dS, HS = exact_derivatives(strikeTensor, maturityTensor)

    # Local volatility
    dupireVol, dupireVar = dupireFormula(HS, dT,
                                         strikeTensor,
                                         scaleTensor,
                                         strikeMinTensor,
                                         IsTraining=IsTraining)

    # Soft constraints on price
    lambdas = hyperparameters["lambdaSoft"]
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    grad_penalty = lambdas * tf.reduce_mean(tf.nn.relu(-dT + lowerBoundTheta) / vegaRef)
    HSScaled = HS / tf.square(scaleTensor)
    hessian_penalty = lambdas * hyperparameters["lambdaGamma"] * tf.reduce_mean(
        tf.nn.relu(- HSScaled + lowerBoundGamma) / vegaRef)

    return out, [out, dupireVol, dT, HSScaled, dupireVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult


############################################################################# Soft constraint


def NNArchitectureConstrainedRawDupire(n_units,
                                       strikeTensor,
                                       maturityTensor,
                                       scaleTensor,
                                       strikeMinTensor,
                                       vegaRef,
                                       hyperparameters,
                                       IsTraining=True):
    # First splitted layer
    hidden1S = convexLayer(n_units=n_units,
                           tensor=strikeTensor,
                           isTraining=IsTraining,
                           name="Hidden1S")

    hidden1M = monotonicLayer(n_units=n_units,
                              tensor=maturityTensor,
                              isTraining=IsTraining,
                              name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second hidden layer and output layer
    out = convexOutputLayer(n_units=n_units,
                            tensor=hidden1,
                            isTraining=IsTraining,
                            name="Output")

    # Compute local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, strikeTensor,
                                                       maturityTensor,
                                                       scaleTensor,
                                                       strikeMinTensor,
                                                       IsTraining=IsTraining)

    # Soft constraints for no-arbitrage
    lambdas = hyperparameters["lambdaSoft"]
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    grad_penalty = lambdas * tf.reduce_mean(tf.nn.relu(-theta + lowerBoundTheta) / vegaRef)
    hessian_penalty = lambdas * hyperparameters["lambdaGamma"] * tf.reduce_mean(
        tf.nn.relu(-hK + lowerBoundGamma) / vegaRef)

    return out, [out, dupireVol, theta, hK, dupireVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult


############################################################################# Soft constraint

# Soft constraints for strike convexity and strike/maturity monotonicity
def arbitragePenalties(dT, gatheralDenominator, vegaRef, hyperparameters):
    lambdas = 1.0 / tf.reduce_mean(vegaRef)
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    calendar_penalty = lambdas * hyperparameters["lambdaSoft"] * tf.reduce_mean(tf.nn.relu(-dT + lowerBoundTheta))
    butterfly_penalty = lambdas * hyperparameters["lambdaGamma"] * tf.reduce_mean( tf.nn.relu(-gatheralDenominator + lowerBoundGamma) )

def NNArchitectureVanillaSoftDupire(n_units, strikeTensor,
                                    maturityTensor,
                                    scaleTensor,
                                    strikeMinTensor,
                                    vegaRef,
                                    hyperparameters,
                                    IsTraining=True):
    inputLayer = tf.concat([strikeTensor, maturityTensor], axis=-1)
    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, strikeTensor,
                                                       maturityTensor,
                                                       scaleTensor,
                                                       strikeMinTensor,
                                                       IsTraining=IsTraining)
    # Soft constraints for no arbitrage
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    lambdasDT = hyperparameters["lambdaSoft"]
    lambdasGK = hyperparameters["lambdaGamma"]
    grad_penalty = lambdasDT * tf.reduce_mean(tf.nn.relu(-theta + lowerBoundTheta) / vegaRef)
    hessian_penalty = lambdasGK * tf.reduce_mean( tf.nn.relu(-hK + lowerBoundGamma) / vegaRef )

    return out, [out, dupireVol, theta, hK, dupireVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult


############################################################################# Unconstrained arhcitecture

def NNArchitectureUnconstrainedDupire(n_units, strikeTensor,
                                      maturityTensor,
                                      scaleTensor,
                                      strikeMinTensor,
                                      vegaRef,
                                      hyperparameters,
                                      IsTraining=True):
    inputLayer = tf.concat([strikeTensor, maturityTensor], axis=-1)

    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Ouput layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, strikeTensor,
                                                       maturityTensor,
                                                       scaleTensor,
                                                       strikeMinTensor,
                                                       IsTraining=IsTraining)

    return out, [out, dupireVol, theta, hK, dupireVar], [], evalAndFormatDupireResult


# Tools functions for hard constrained neural architecture

def convexLayerHard(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.softplus(layer), layer


def monotonicLayerHard(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor,
                                units=n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.sigmoid(layer), layer


def convexOutputLayerHard(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=2 * n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                activation='softplus')

        layer = tf.layers.dense(layer,
                                units=1,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=positiveKernelInitializer,
                                activation='softplus')

        return layer, layer


def convexLayerHybridHard(n_units,
                          tensor,
                          isTraining,
                          name,
                          activationFunction2=Act.softplus,
                          activationFunction1=Act.exponential,
                          isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=positiveKernelInitializer)
        l1, l2 = tf.split(layer, 2, 1)
        output = tf.concat([activationFunction1(l1), activationFunction2(l2)], axis=-1)
        return output, layer


def sigmoidGradientHard(inputTensor):
    return tf.nn.sigmoid(inputTensor) * (1 - tf.nn.sigmoid(inputTensor))


def sigmoidHessianHard(inputTensor):
    return (tf.square(1 - tf.nn.sigmoid(inputTensor)) -
            tf.nn.sigmoid(inputTensor) * (1 - tf.nn.sigmoid(inputTensor)))

############################################################################# Hard constraint

def NNArchitectureHardConstrainedDupire(n_units, strikeTensor,
                                        maturityTensor,
                                        scaleTensor,
                                        strikeMinTensor,
                                        vegaRef,
                                        hyperparameters,
                                        IsTraining=True):
    # First layer
    hidden1S, layer1S = convexLayerHard(n_units=n_units,
                                        tensor=strikeTensor,
                                        isTraining=IsTraining,
                                        name="Hidden1S")

    hidden1M, layer1M = monotonicLayerHard(n_units=n_units,
                                           tensor=maturityTensor,
                                           isTraining=IsTraining,
                                           name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second layer and output layer
    out, layer = convexOutputLayerHard(n_units=n_units,
                                       tensor=hidden1,
                                       isTraining=IsTraining,
                                       name="Output")
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, strikeTensor,
                                                       maturityTensor,
                                                       scaleTensor,
                                                       strikeMinTensor,
                                                       IsTraining=IsTraining)

    return out, [out, dupireVol, theta, hK, dupireVar], [], evalAndFormatDupireResult



################################################################### Working on implied volatilities


# Train neural network with a decreasing rule for learning rate
# NNFactory :  function creating the architecture
# dataSet : training data
# activateRegularization : boolean, if true add bound penalization to dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name under which tensorflow model is saved
def create_train_model_gatheral(NNFactory,
                                dataSet,
                                activateRegularization,
                                hyperparameters,
                                scaler,
                                modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    fixedLearningRate = (None if hyperparameters["FixedLearningRate"] else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Moneyness = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    MoneynessPenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    vegaRef = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRef')
    vegaRefPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRefPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colMoneynessIndex = dataSet.columns.get_loc("logMoneyness")
    maxColFunction = scaler.data_max_[colMoneynessIndex]
    minColFunction = scaler.data_min_[colMoneynessIndex]
    scF = (maxColFunction - minColFunction)
    scaleTensor = tf.constant(scF, dtype=tf.float32)
    moneynessMinTensor = tf.constant(minColFunction, dtype=tf.float32)

    #Grid on which is applied Penalization
    t = np.linspace(scaler.data_min_[dataSet.columns.get_loc("Maturity")],
                    4 * scaler.data_max_[dataSet.columns.get_loc("Maturity")],
                    num=100)
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    #t = np.linspace(0, 4, num=100)

    k = np.linspace((np.log(0.5) - minColFunction) / scF,
                    (np.log(2.0) - minColFunction) / scF,
                    num=50)
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    if activateRegularization:  # Add pseudo local volatility regularisation
        vol_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       Moneyness,
                       Maturity,
                       scaleTensor,
                       moneynessMinTensor,
                       vegaRef,
                       hyperparameters),
            vegaRef,
            hyperparameters)
        vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       MoneynessPenalization,
                       MaturityPenalization,
                       scaleTensor,
                       moneynessMinTensor,
                       vegaRefPenalization,
                       hyperparameters),
            vegaRefPenalization,
            hyperparameters)
    else:
        vol_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                      Moneyness,
                                                                                      Maturity,
                                                                                      scaleTensor,
                                                                                      moneynessMinTensor,
                                                                                      vegaRef,
                                                                                      hyperparameters)
        vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                          MoneynessPenalization,
                                                                                          MaturityPenalization,
                                                                                          scaleTensor,
                                                                                          moneynessMinTensor,
                                                                                          vegaRefPenalization,
                                                                                          hyperparameters)

    vol_pred_tensor_sc = vol_pred_tensor
    TensorList[0] = tf.square(vol_pred_tensor_sc) * Maturity

    # Define a loss function
    pointwiseError = tf.reduce_mean(tf.abs(vol_pred_tensor_sc - y) / vegaRef)
    errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
    loss = tf.log(tf.reduce_mean(errors))

    # Define a train operation to minimize the loss
    lr = learningRate

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = dataSet.shape[0]
    scaledInput = dataSetConstruction.transformCustomMinMax(dataSet, scaler)

    maturity = dataSet["Maturity"].values.reshape(n, 1)
    loss_serie = []

    def createFeedDict(batch):
        batchSize = batch.shape[0]
        feedDict = {Moneyness: scaledInput["logMoneyness"].values.reshape(batchSize, 1),#scaledInput["logMoneyness"].loc[batch.index].drop_duplicates().values.reshape(batchSize, 1),
                    Maturity: batch["Maturity"].values.reshape(batchSize, 1),
                    y: batch["ImpliedVol"].values.reshape(batchSize, 1),
                    MoneynessPenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    learningRateTensor: learningRate,
                    vegaRef: np.ones_like(batch["logMoneyness"].values.reshape(batchSize, 1)),
                    vegaRefPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    # Learning rate is divided by 10 if no imporvement is observed for training loss after "patience" epochs
    def updateLearningRate(iterNumber, lr, lrEpoch):
        if not activateLearningDecrease:
            print("Constant learning rate, stop training")
            return False, lr, lrEpoch
        if learningRate > finalLearningRate:
            lr *= 0.1
            lrEpoch = iterNumber
            saver.restore(sess, modelName)
            print("Iteration : ", lrEpoch, "new learning rate : ", lr)
        else:
            print("Last Iteration : ", lrEpoch, "final learning rate : ", lr)
            return False, lr, lrEpoch
        return True, lr, lrEpoch

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid) : ", currentBestPenalizations1)
        return

    for i in range(nbEpoch):
        miniBatchList = [dataSet]
        penalizationResult = sess.run(penalizationList, feed_dict=epochFeedDict)
        lossResult = sess.run(pointwiseError, feed_dict=epochFeedDict)

        # miniBatchList = selectMiniBatchWithoutReplacement(dataSet, batch_size)
        for k in range(len(miniBatchList)):
            batchFeedDict = createFeedDict(miniBatchList[k])
            sess.run(train, feed_dict=batchFeedDict)

        loss_serie.append(sess.run(loss, feed_dict=epochFeedDict))

        if (len(loss_serie) < 2) or (loss_serie[-1] <= min(loss_serie)):
            # Save model as model is improved
            saver.save(sess, modelName)
        if (np.isnan(loss_serie[-1]) or  # Unstable model
                ((i - learningRateEpoch >= patience) and (min(loss_serie[-patience:]) > min(
                    loss_serie)))):  # No improvement for training loss during the latest 100 iterations
            continueTraining, learningRate, learningRateEpoch = updateLearningRate(i, learningRate, learningRateEpoch)
            if continueTraining:
                evalBestModel()
            else:
                break
    saver.restore(sess, modelName)

    evalBestModel()

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()
    end = time.time()
    print("Training Time : ", end - start)

    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    lossEpochSerie = pd.Series(loss_serie)
    lossEpochSerie.to_csv("loss" + modelName + ".csv")

    return formattingFunction(*evalList, loss_serie, dataSet, scaler)


# Evaluate neural network without training, it restores parameters obtained from a pretrained model
# NNFactory :  function creating the neural architecture
# dataSet : dataset on which neural network is evaluated
# activateRegularization : boolean, if true add bound penalization for dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name of tensorflow model to restore
def create_eval_model_gatheral(NNFactory,
                               dataSet,
                               activateRegularization,
                               hyperparameters,
                               scaler,
                               modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]

    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Moneyness = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    MoneynessPenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    vegaRef = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRef')
    vegaRefPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRefPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])

    # Get scaling for strike
    colMoneynessIndex = dataSet.columns.get_loc("logMoneyness")
    maxColFunction = scaler.data_max_[colMoneynessIndex]
    minColFunction = scaler.data_min_[colMoneynessIndex]
    scF = (maxColFunction - minColFunction)
    scaleTensor = tf.constant(scF, dtype=tf.float32)
    moneynessMinTensor = tf.constant(minColFunction, dtype=tf.float32)

    #Grid on which is applied Penalization
    t = np.linspace(scaler.data_min_[dataSet.columns.get_loc("Maturity")],
                    4 * scaler.data_max_[dataSet.columns.get_loc("Maturity")],
                    num=100)
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    #t = np.linspace(0, 4, num=100)

    k = np.linspace((np.log(0.5) - minColFunction) / scF,
                    (np.log(2.0) - minColFunction) / scF,
                    num=50)
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    if activateRegularization:  # Add pseudo local volatility regularisation
        vol_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       Moneyness,
                       Maturity,
                       scaleTensor,
                       moneynessMinTensor,
                       vegaRef,
                       hyperparameters),
            vegaRef,
            hyperparameters)
        vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       MoneynessPenalization,
                       MaturityPenalization,
                       scaleTensor,
                       moneynessMinTensor,
                       vegaRefPenalization,
                       hyperparameters),
            vegaRefPenalization,
            hyperparameters)
    else:
        vol_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                      Moneyness,
                                                                                      Maturity,
                                                                                      scaleTensor,
                                                                                      moneynessMinTensor,
                                                                                      vegaRef,
                                                                                      hyperparameters)
        vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                          MoneynessPenalization,
                                                                                          MaturityPenalization,
                                                                                          scaleTensor,
                                                                                          moneynessMinTensor,
                                                                                          vegaRefPenalization,
                                                                                          hyperparameters)

    vol_pred_tensor_sc = vol_pred_tensor
    TensorList[0] = tf.square(vol_pred_tensor_sc) * Maturity

    # Define a loss function
    pointwiseError = tf.reduce_mean(tf.abs(vol_pred_tensor_sc - y) / vegaRef)
    errors = tf.add_n([pointwiseError] + penalizationList1)
    loss = tf.log(tf.reduce_mean(errors))

    # Define a train operation to minimize the loss
    lr = learningRate

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = dataSet.shape[0]
    scaledInput = dataSetConstruction.transformCustomMinMax(dataSet, scaler)

    maturity = dataSet["Maturity"].values.reshape(n, 1)
    loss_serie = []

    def createFeedDict(batch):
        batchSize = batch.shape[0]
        feedDict = {Moneyness: scaledInput["logMoneyness"].values.reshape(batchSize, 1),#scaledInput["logMoneyness"].loc[batch.index].drop_duplicates().values.reshape(batchSize, 1),
                    Maturity: batch["Maturity"].values.reshape(batchSize, 1),
                    y: batch["ImpliedVol"].values.reshape(batchSize, 1),
                    MoneynessPenalization : np.expand_dims(kPenalization,1),
                    MaturityPenalization : np.expand_dims(tPenalization,1),
                    learningRateTensor: learningRate,
                    vegaRef: np.ones_like(batch["logMoneyness"].values.reshape(batchSize, 1)),
                    vegaRefPenalization : np.ones_like(np.expand_dims(kPenalization,1))}
        return feedDict

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid): ", currentBestPenalizations1)
        return

    saver.restore(sess, modelName)

    evalBestModel()

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return formattingFunction(*evalList, [0], dataSet, scaler)


def evalVolLocaleGatheral(NNFactory,
                          strikes,
                          maturities,
                          dataSet,
                          hyperparameters,
                          scaler,
                          bootstrap,
                          S0,
                          modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]

    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Moneyness = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    MoneynessPenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    vegaRef = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRef')
    vegaRefPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='vegaRefPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colMoneynessIndex = dataSet.columns.get_loc("logMoneyness")
    maxColFunction = scaler.data_max_[colMoneynessIndex]
    minColFunction = scaler.data_min_[colMoneynessIndex]
    scF = (maxColFunction - minColFunction)
    scaleTensor = tf.constant(scF, dtype=tf.float32)
    moneynessMinTensor = tf.constant(minColFunction, dtype=tf.float32)

    #Grid on which is applied Penalization
    t = np.linspace(scaler.data_min_[dataSet.columns.get_loc("Maturity")],
                    4 * scaler.data_max_[dataSet.columns.get_loc("Maturity")],
                    num=100)
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    #t = np.linspace(0, 4, num=100)

    k = np.linspace((np.log(0.5) - minColFunction) / scF,
                    (np.log(2.0) - minColFunction) / scF,
                    num=50)
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    vol_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                  Moneyness,
                                                                                  Maturity,
                                                                                  scaleTensor,
                                                                                  moneynessMinTensor,
                                                                                  vegaRef,
                                                                                  hyperparameters)
    vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                      MoneynessPenalization,
                                                                                      MaturityPenalization,
                                                                                      scaleTensor,
                                                                                      moneynessMinTensor,
                                                                                      vegaRefPenalization,
                                                                                      hyperparameters)

    vol_pred_tensor_sc = vol_pred_tensor
    TensorList[0] = tf.square(vol_pred_tensor_sc) * Maturity

    # Define a loss function
    pointwiseError = tf.reduce_mean(tf.abs(vol_pred_tensor_sc - y) / vegaRef)
    errors = tf.add_n([pointwiseError] + penalizationList1)
    loss = tf.log(tf.reduce_mean(errors))

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = strikes.shape[0]
    changedVar = bootstrap.changeOfVariable(strikes, maturities)

    moneyness = np.log(changedVar[0] / S0)
    scaledMoneyness = (moneyness - minColFunction) / scF

    def createFeedDict(m, t):
        batchSize = m.shape[0]
        feedDict = {Moneyness: np.reshape(m, (batchSize, 1)),
                    Maturity: np.reshape(t, (batchSize, 1)),
                    MoneynessPenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization,1),
                    vegaRef: np.ones((batchSize, 1)),
                    vegaRefPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    epochFeedDict = createFeedDict(scaledMoneyness, maturities)

    saver.restore(sess, modelName)

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return pd.Series(evalList[1].flatten(),
                     index=pd.MultiIndex.from_arrays([strikes, maturities], names=('Strike', 'Maturity')))


# Dupire formula from exact derivative computation
def dupireFormulaGatheral(HessianMoneyness,
                          GradMoneyness,
                          GradMaturity,
                          totalVariance,
                          ScaledMoneyness,
                          scaleTensor,
                          MoneynessMinTensor,
                          IsTraining=True):
    twoConstant = tf.constant(2.0)
    oneConstant = tf.constant(1.0)
    quarterConstant = tf.constant(0.25)
    halfConstant = tf.constant(0.5)

    moneyness = ScaledMoneyness * scaleTensor + MoneynessMinTensor

    dT = GradMaturity

    dMoneyness = GradMoneyness / scaleTensor
    dMoneynessFactor = (moneyness / totalVariance)
    dMoneynessSquaredFactor = quarterConstant * (
                -quarterConstant - oneConstant / totalVariance + tf.square(dMoneynessFactor))

    gMoneyness = HessianMoneyness / tf.square(scaleTensor)
    gMoneynessFactor = halfConstant
    denominator = oneConstant - dMoneynessFactor * (dMoneyness) + dMoneynessSquaredFactor * tf.square(
        dMoneyness) + gMoneynessFactor * gMoneyness

    gatheralVar = dT / denominator
    # Initial weights of neural network can be random which lead to negative dupireVar
    gatheralVolTensor = tf.sqrt(gatheralVar)
    return gatheralVolTensor, gatheralVar, gatheralDenominator

#Dupire formula with derivative obtained from native tensorflow algorithmic differentiation
def rawDupireFormulaGatheral(totalVarianceTensor,
                             scaledMoneynessTensor,
                             maturityTensor,
                             scaleTensor,
                             moneynessMinTensor,
                             IsTraining=True):
  batchSize = tf.shape(scaledMoneynessTensor)[0]
  twoConstant = tf.constant(2.0)
  oneConstant = tf.constant(1.0)
  quarterConstant = tf.constant(0.25)
  halfConstant = tf.constant(0.5)

  moneyness = scaledMoneynessTensor * scaleTensor + moneynessMinTensor

  dMoneyness = tf.reshape(tf.gradients(totalVarianceTensor, scaledMoneynessTensor, name="dK")[0], shape=[batchSize,-1]) / scaleTensor
  dMoneynessFactor = (moneyness/totalVarianceTensor)
  dMoneynessSquaredFactor = quarterConstant * (-quarterConstant - oneConstant/totalVarianceTensor + tf.square(dMoneynessFactor))

  gMoneyness = tf.reshape(tf.gradients(dMoneyness, scaledMoneynessTensor, name="hK")[0], shape=[batchSize,-1]) / scaleTensor
  gMoneynessFactor = halfConstant


  gatheralDenominator = oneConstant - dMoneynessFactor * (dMoneyness) + dMoneynessSquaredFactor * tf.square(dMoneyness) + gMoneynessFactor *  gMoneyness

  dT = tf.reshape(tf.gradients(totalVarianceTensor,maturityTensor,name="dT")[0], shape=[batchSize,-1])

  #Initial weights of neural network can be random which lead to negative dupireVar
  gatheralVar = dT / gatheralDenominator
  gatheralVol = tf.sqrt(gatheralVar)
  return  gatheralVol, dT, gMoneyness, gatheralVar, gatheralDenominator


# Soft constraints for strike convexity and strike/maturity monotonicity
def arbitragePenalties(dT, gatheralDenominator, vegaRef, hyperparameters):
    lambdas = 1.0 / tf.reduce_mean(vegaRef)
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    calendar_penalty = lambdas * hyperparameters["lambdaSoft"] * tf.reduce_mean(tf.nn.relu(-dT + lowerBoundTheta))
    butterfly_penalty = lambdas * hyperparameters["lambdaGamma"] * tf.reduce_mean( tf.nn.relu(-gatheralDenominator + lowerBoundGamma) )

    return [calendar_penalty, butterfly_penalty]


def NNArchitectureVanillaSoftGatheralAckerer(n_units,
                                             scaledMoneynessTensor,
                                             maturityTensor,
                                             scaleTensor,
                                             moneynessMinTensor,
                                             vegaRef,
                                             hyperparameters,
                                             IsTraining=True):
    inputLayer = tf.concat([scaledMoneynessTensor, maturityTensor], axis=-1)
    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Third layer
    hidden3 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden2,
                                 isTraining=IsTraining,
                                 name="Hidden3")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden3,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    gatheralVol, theta, hK, gatheralVar, gatheralDenominator = rawDupireFormulaGatheral(tf.square(out) * maturityTensor,
                                                                                        scaledMoneynessTensor,
                                                                                        maturityTensor,
                                                                                        scaleTensor,
                                                                                        moneynessMinTensor,
                                                                                        IsTraining=IsTraining)
    # Soft constraints for no arbitrage
    penalties = arbitragePenalties(theta, gatheralDenominator, vegaRef, hyperparameters)
    grad_penalty = penalties[0]
    hessian_penalty = penalties[1]

    return out, [out, gatheralVol, theta, gatheralDenominator, gatheralVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult


def NNArchitectureVanillaSoftGatheral(n_units,
                                      scaledMoneynessTensor,
                                      maturityTensor,
                                      scaleTensor,
                                      moneynessMinTensor,
                                      vegaRef,
                                      hyperparameters,
                                      IsTraining=True):
    inputLayer = tf.concat([scaledMoneynessTensor, maturityTensor], axis=-1)
    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    gatheralVol, theta, hK, gatheralVar, gatheralDenominator = rawDupireFormulaGatheral(tf.square(out) * maturityTensor,
                                                                                        scaledMoneynessTensor,
                                                                                        maturityTensor,
                                                                                        scaleTensor,
                                                                                        moneynessMinTensor,
                                                                                        IsTraining=IsTraining)
    # Soft constraints for no arbitrage
    penalties = arbitragePenalties(theta, gatheralDenominator, vegaRef, hyperparameters)
    grad_penalty = penalties[0]
    hessian_penalty = penalties[1]

    return out, [out, gatheralVol, theta, gatheralDenominator, gatheralVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult


################################################################### Dugas neural network
def convexDugasLayer(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        nbInputFeatures = tensor.get_shape().as_list()[1]
        bias = tf.Variable(initial_value=tf.zeros_initializer()([n_units], dtype=tf.float32),
                           trainable=True,
                           shape=[n_units],
                           dtype=tf.float32,
                           name=name + "Bias")
        weights = tf.exp(tf.Variable(
            initial_value=tf.keras.initializers.glorot_normal()([nbInputFeatures, n_units], dtype=tf.float32),
            trainable=True,
            shape=[nbInputFeatures, n_units],
            dtype=tf.float32,
            name=name + "Weights"))
        layer = tf.matmul(tensor, weights) + bias
        return K.softplus(layer)


def monotonicDugasLayer(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        nbInputFeatures = tensor.get_shape().as_list()[1]
        bias = tf.Variable(initial_value=tf.zeros_initializer()([n_units], dtype=tf.float32),
                           trainable=True,
                           shape=[n_units],
                           dtype=tf.float32,
                           name=name + "Bias")
        weights = tf.exp(tf.Variable(
            initial_value=tf.keras.initializers.glorot_normal()([nbInputFeatures, n_units], dtype=tf.float32),
            trainable=True,
            shape=[nbInputFeatures, n_units],
            dtype=tf.float32,
            name=name + "Weights"))
        layer = tf.matmul(tensor, weights) + bias
        return K.sigmoid(layer)


def convexDugasOutputLayer(tensor, isTraining, name):
    with tf.name_scope(name):
        nbInputFeatures = tensor.get_shape().as_list()[1]
        bias = tf.exp(tf.Variable(initial_value=tf.zeros_initializer()([], dtype=tf.float32),
                                  shape=[],
                                  trainable=True,
                                  dtype=tf.float32,
                                  name=name + "Bias"))
        weights = tf.exp(
            tf.Variable(initial_value=tf.keras.initializers.glorot_normal()([nbInputFeatures, 1], dtype=tf.float32),
                        shape=[nbInputFeatures, 1],
                        trainable=True,
                        dtype=tf.float32,
                        name=name + "Weights"))
        layer = tf.matmul(tensor, weights) + bias
        return layer


def NNArchitectureHardConstrainedDugas(n_units, strikeTensor,
                                       maturityTensor,
                                       scaleTensor,
                                       strikeMinTensor,
                                       vegaRef,
                                       hyperparameters,
                                       scaler,
                                       IsTraining=True):
    # First layer
    hidden1S = convexDugasLayer(n_units=n_units,
                                tensor=strikeTensor,
                                isTraining=IsTraining,
                                name="Hidden1S")

    hidden1M = monotonicDugasLayer(n_units=n_units,
                                   tensor=maturityTensor,
                                   isTraining=IsTraining,
                                   name="Hidden1M")

    hidden1 = hidden1S * hidden1M

    # Second layer and output layer
    out = convexDugasOutputLayer(tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Output")
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, strikeTensor,
                                                       maturityTensor,
                                                       scaleTensor,
                                                       strikeMinTensor,
                                                       IsTraining=IsTraining)

    return out, [out, dupireVol, theta, hK, dupireVar], [], evalAndFormatDupireResult


def selectHyperparameters(hyperparameters, parameterOfInterest, modelFactory,
                          modelName, activateDupireReg, scaledDataSet,
                          scaler,
                          trainedOnPrice = True,
                          logGrid=True):
    oldValue = hyperparameters[parameterOfInterest]
    gridValue = oldValue * (
        np.exp(np.log(10) * np.array([-2, -1, 0, 1, 2])) if logGrid else np.array([0.2, 0.5, 1, 2, 5]))

    oldNbEpochs = hyperparameters["maxEpoch"]
    hyperparameters["maxEpoch"] = int(oldNbEpochs / 10)
    trainLoss = {}
    arbitrageViolation = {}
    for v in gridValue:
        hyperparameters[parameterOfInterest] = int(v)
        if trainedOnPrice :
            pred, volLoc, theta, gammaK, loss = create_train_model(modelFactory,
                                                                   scaledDataSet,
                                                                   activateDupireReg,
                                                                   hyperparameters,
                                                                   scaler,
                                                                   modelName=modelName)
        else :
            pred, volLoc, theta, gammaK, loss = create_train_model_gatheral(modelFactory,
                                                                            scaledDataSet,
                                                                            activateDupireReg,
                                                                            hyperparameters,
                                                                            scaler,
                                                                            modelName=modelName)

        nbArbitrageViolation = np.sum((theta <= 0)) + np.sum((gammaK <= 0))
        trainLoss[v] = min(loss)
        arbitrageViolation[v] = nbArbitrageViolation
        print()
        print()

    hyperparameters["maxEpoch"] = oldNbEpochs
    hyperparameters[parameterOfInterest] = oldValue
    # Plot curves

    fig, ax1 = plt.subplots()
    if logGrid:
        plt.xscale('symlog')

    color = 'tab:red'
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(pd.Series(trainLoss), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Arbitrage violation', color=color)  # we already handled the x-label with ax1
    ax2.plot(pd.Series(arbitrageViolation), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    return


def selectHyperparametersRandom(hyperparameters,
                                parametersOfInterest,
                                modelFactory,
                                modelName,
                                activateDupireReg,
                                nbAttempts,
                                scaledDataSet,
                                scaler,
                                trainedOnPrice = True,
                                logGrid=True):
    oldValue = {}
    for k in parametersOfInterest:
        oldValue[k] = hyperparameters[k]

    gridValue = np.exp(np.log(10) * np.array([-2, -1, 0, 1, 2])) if logGrid else np.array([0.2, 0.5, 1, 2, 5])

    oldNbEpochs = hyperparameters["maxEpoch"]
    hyperparameters["maxEpoch"] = int(oldNbEpochs / 10)
    trainLoss = {}
    arbitrageViolation = {}
    nbTry = nbAttempts
    for v in range(nbTry):
        combination = np.random.randint(5, size=len(parametersOfInterest))
        for p in range(len(parametersOfInterest)):
            hyperparameters[parametersOfInterest[p]] = oldValue[parametersOfInterest[p]] * gridValue[
                int(combination[p])]
            print(parametersOfInterest[p], " : ", hyperparameters[parametersOfInterest[p]])
        if trainedOnPrice :
            pred, volLoc, theta, gammaK, loss = create_train_model(modelFactory,
                                                                   scaledDataSet,
                                                                   activateDupireReg,
                                                                   hyperparameters,
                                                                   scaler,
                                                                   modelName=modelName)
        else :
            pred, volLoc, theta, gammaK, loss = create_train_model_gatheral(modelFactory,
                                                                            scaledDataSet,
                                                                            activateDupireReg,
                                                                            hyperparameters,
                                                                            scaler,
                                                                            modelName=modelName)

        nbArbitrageViolation = np.sum((theta <= 0)) + np.sum((gammaK <= 0))
        print("loss : ", min(loss))
        print("nbArbitrageViolation : ", nbArbitrageViolation)
        print()
        print()
        print()

    hyperparameters["maxEpoch"] = oldNbEpochs
    for k in parametersOfInterest:
        hyperparameters[k] = oldValue[k]

    return
