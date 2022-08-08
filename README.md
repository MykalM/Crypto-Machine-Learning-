# Crypto-Machine-Learning
## Background
Building from our **Portfolio Risk Analyzer** we decided to incorporate a Machine-Learning-Cryptocurrency trading strategy to maximize profits for our clients. By utilizing the power of machine learning we are able to backtest our trading strategies, further improving our clients profits. With the addition of Amazon's Lexbot clients are able to easily convert currencies, and purchase select cryptocurrencies.

## Table of Contents
1. [Front-end Interface](#1-Front-end-Interface)

2. [Trading Strategy](#2-Trading-Strategy)

3. [Simulation Trajectories](#3-Simulation-Trajectories)

4. [Back-End Functionality](#4-Back-End-Functionality)

5. [Compose the data story](#5-compose-the-data-story)

---
## 1. Front-end Interface

## 2. Trading Strategy
Viewing the price data and the buy/sell signals

![ezgif com-gif-maker](https://user-images.githubusercontent.com/98198920/183314000-3e53f27c-c0db-48aa-9904-b27a726f346e.gif)

Volume & MACD

![image](https://user-images.githubusercontent.com/98198920/183315051-5661f139-fc16-4d9c-8a02-b9a463e33df1.png)

Finding the optimal parameters
![image](https://user-images.githubusercontent.com/98198920/183315281-68dfda73-19b9-4f91-9e9d-b2669d6dd5c3.png)

Backtest results
![image](https://user-images.githubusercontent.com/98198920/183315195-df2516bd-b64c-44db-8158-6ac47cacc98d.png)

## 3. Baseline Simulations
The following portions of code constitue the inital proof-of-concept for our machine learning functionality. We included Perceptron, Logistic Regression, and Decision Tree Classifier Models in our early phases. 

```python
perc_regression_model = Perceptron()
perc_model = perc_regression_model.fit(X_train_scaled, y_train)
perc_predictions= perc_regression_model.predict(X_test_scaled)
perc_predictions[0:10]
perc_report = classification_report(y_test, perc_predictions,zero_division=1)
print(perc_report)
```
This code initializes the Perceptron Model. The last two lines are for performing backtesting of the model for accuracy.  

```python
logistic_regression_model = LogisticRegression()
model = logistic_regression_model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
predictions[0:10]
lrm_report = classification_report(y_test, predictions, zero_division=1)
print(lrm_report)
```
This code initializes the Logisitc Regression Model. The last two lines are for performing backtesting of the model for accuracy. 

```python
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train_scaled, y_train)
tree_btc_predictions = tree_model.predict(X_test_scaled)
tree_btc_predictions[0:10]
tree_btc_report = classification_report(y_test, tree_btc_predictions, zero_division=1)
print(tree_btc_report)
```
This code initializes the Decision Tree Classifier Model. First we import the model in order to construct it. The last two lines are then used for backtesting purposes. 

**Initial Results**
The following images show some of the initial results for our modeling. These results would later be improved upon in further revisions (See section #2).

![BTC-PERC](https://i.postimg.cc/DymKsmng/btc-perc-6mos.png)


![ETH-LRM](https://i.postimg.cc/N0knZwQF/eth-lrm-6mos.png)

![BNB-TREE](https://i.postimg.cc/bwXFJvJd/bnb-tree-8mos.png)

## 4. Back-End Functionality

## 5. Compose the data story
