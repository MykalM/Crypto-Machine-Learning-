# **Crypto-Machine-Learning**

**Smart exposure to digital currencies**, This application uses an interactive client user interface to collect investment information from the user; it then recomends a cryptocurrency real-time conversaion rate using API calls. Building from our **Crypto Portfolio Risk Analyzer** We decided to incorporate different Machine Learning methodologies to our cryptocurrency trading strategies to maximize profits for our clients. By utilizing the power of Machine Learning we are able to backtest multiple trading strategies, further improving our clients portfolio performance and profits. With the addition of Amazon's Lexbot clients are able to easily convert currencies, and purchase select cryptocurrencies in real time. 

This upgraded **Crypto Porfolio Risk Analyzer** is an innovative addition to the fintech industry. We want to offer optimized cryptocurrency porfolios to everyone, even those who lack the financial knowledge and understanding of cryptocurrencies. We plan to build additional features into this like the ability to link to cryptocurrency wallets, interactive and real time education on Machine Learning strategies for clients to choose from. 

The class presentation is located here File: [Machine Learning Crypto Converter and Analyzer.pdf](https://github.com/MykalM/Crypto-Machine-Learning-/files/9278601/Machine.Learning.Crypto.Converter.and.Analyzer.pdf)


## **Table of Contents**
1. [Front-end Interface](#1-Front-end-Interface)

2. [Trading Strategy](#2-Trading-Strategy)

3. [Simulation Trajectories](#3-Simulation-Trajectories)

4. [Compose The Data Story](#4-Compose-The-Data-Story)

5. [Contributors](#5-Contributors) 

---
## 1. **Front-end Interface**
Our service is only available for people over 21 years old. Clients can get the current price of the most popular cryptocurrencies including Bitcoin(BTC), Binance Coin(BNB), Ethereum(ETH), Cardano(ADA), Ripple(XRP) and Solana(SOL) in top traded national currencies - US dollars(USD), Japanese Yen(JPY), Korean Won(KRW), Euro(ERU), British Pounds(GBP) and Canadian dollars(CAD). 

Below are videos showing how easy our clients can convert dollars to cryptocurrencies they wish for. 
 
 - **Convert USD to BTC** 

https://user-images.githubusercontent.com/103230949/183328929-217127a7-2c3c-43c9-a1a0-86992047304c.mp4

 - **Convert EUR to SOL**

https://user-images.githubusercontent.com/103230949/183328933-7d390948-d39c-492b-8b21-292ce17c37f4.mp4

We also conducted different testing to ensure the accuracy of our chatbox. 

 - If clients select any national currencies outside of our service package, they will get an error message saying: 

**_Sorry, I'm just able to convert from USD, JPY, KRW, EUR, GBP or CAD for now._**

![convertBRLerror_TEST](https://user-images.githubusercontent.com/103230949/183332822-e0d9a819-622c-46ac-82cc-8fe7389f44c0.png)

 - If clients select any cryptocurrencies outside of our service package, they will get an error message saying: 

**_Sorry, I'm just able to convert from BTC, BNB, ETH, ADA, XRP or SOL for now._**

![convertUNIerror_TEST](https://user-images.githubusercontent.com/103230949/183332826-23aa2a82-bbd4-4659-8be8-f21a812cbfd8.png)

 - If clients entered dollar amount less or equal to zero, they will get an error message saying: 

**_The amount to convert should be greater than zero, please provide a correct amount in dollars to convert._**

![convertZEROerror_TEST](https://user-images.githubusercontent.com/103230949/183332827-2a8999db-593a-413c-9511-2d45dd31aa7b.png)

## 2. **Trading Strategy**
The graph below shows Bitcoin Price data buy/sell signals using the SuperTrend trading strategy. 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/98198920/183314000-3e53f27c-c0db-48aa-9904-b27a726f346e.gif)


Volume & MACD
![image](https://user-images.githubusercontent.com/98198920/183315051-5661f139-fc16-4d9c-8a02-b9a463e33df1.png)


Finding the optimal parameters
![image](https://user-images.githubusercontent.com/98198920/183315281-68dfda73-19b9-4f91-9e9d-b2669d6dd5c3.png)


Backtest results
![image](https://user-images.githubusercontent.com/98198920/183315195-df2516bd-b64c-44db-8158-6ac47cacc98d.png)

## 3. **Baseline Simulations**
The following portions of code constitue the inital proof-of-concept for our machine learning functionality. We included **Perceptron**, **Logistic Regression**, and **Decision Tree** Classifier Models in our early phases. 

```python
perc_regression_model = Perceptron()
perc_model = perc_regression_model.fit(X_train_scaled, y_train)
perc_predictions= perc_regression_model.predict(X_test_scaled)
perc_predictions[0:10]
perc_report = classification_report(y_test, perc_predictions,zero_division=1)
print(perc_report)
```

This code initializes the **Perceptron** Model. The last two lines are for performing backtesting of the model for accuracy.  

```python
logistic_regression_model = LogisticRegression()
model = logistic_regression_model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
predictions[0:10]
lrm_report = classification_report(y_test, predictions, zero_division=1)
print(lrm_report)
```
This code initializes the **Logisitc Regression** Model. The last two lines are for performing backtesting of the model for accuracy. 

```python
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train_scaled, y_train)
tree_btc_predictions = tree_model.predict(X_test_scaled)
tree_btc_predictions[0:10]
tree_btc_report = classification_report(y_test, tree_btc_predictions, zero_division=1)
print(tree_btc_report)
```
This code initializes the **Decision Tree** Classifier Model. First we import the model in order to construct it. The last two lines are then used for backtesting purposes. 

**Initial Results**
The following images show some of the initial results for our modeling. These results would later be improved upon in further revisions (See section #2).

![BTC-PERC](https://i.postimg.cc/DymKsmng/btc-perc-6mos.png)

![ETH-LRM](https://i.postimg.cc/N0knZwQF/eth-lrm-6mos.png)

![BNB-TREE](https://i.postimg.cc/bwXFJvJd/bnb-tree-8mos.png)

## 4. Compose the data story
Our portfolios are designed to minimize long-term risk while benefiting from the accumulated growth of the most popular and leading cryptocurrencies. Tired of hearing about stocks that already made investors a lot of money? Our Crypto Porfolio Advisors will help you pick top value stocks and cryptocurrencies with strong long-term growth potential. Utilizing real-time data users will have access to their own personal crypto advisors at the palm of their hands, or on their computer. You are in good hands with us!

## 5. Contributors
Jeremy Pierce - Machine Learning Lead,
Jay Wiley - Machine Learning Developer,
Mykal Morton - Cluster Analyst,
Vicky Lee - Client User Interface Designer

