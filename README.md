# TensorFlow-StockPrediction

I was watching a stock trading conference video, the head of one of Canada’s top trading banks was talking about High Frequency Trading. It surprised me when he said that almost 60% of all trades in the stock market are decided by computers. It showed me the disadvantage any individual trader has when competing with big banks and trading firms in the stock market. 

Although it seems hopeless to compete with the professionals, maybe a machine learning model could help close this gap between these giants and the individual traders. 

As with most, if not all, machine learning models, it will not predict with perfect accuracy but if it is used in conjunction with other strategies and indicators, it could help these traders decide where to place their money to maximize their return on investments. 

After some research I decided to try to use a Long-Short Term Memory network, which is a type of Recurrent Neural Network. This network will predict the stock prices of Microsoft (MSFT) and Apple (AAPL). The model will be trained to work with large cap companies since long-term investors will be more interested in those companies rather than medium or small cap stocks. 

With stock price prediction and a well-rounded risk management method, I believe that machine learning will be more popular with individual traders and not only banks, hedge funds and other trading firms.

## Built With

* [Python v3.6.5](https://www.python.org/) - The programming language used
* [TensorFlow v1.12](https://www.tensorflow.org/) - Machine Learning library
* [pandas v0.23](https://pandas.pydata.org/) - Used to read csv files and manipulate data
* [numpy v1.14.3](http://www.numpy.org/) - Used to manipulate data arrays

## Authors

* **Lucas Magalhaes** - *Author*

## Paper

For more information on version 1 of this project, you can read the paper that I wrote.
[Paper](TensorFlow_StockPrediction_Paper.pdf)

## License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Future Work

* Add more tickers to the algorithm
* Add different timelines of the stock pricing(now it only does 1 price/day)
* Backtest algorithm to make sure it makes money
* Add trading algorithm that will take risk management into account
* Add portfolio management
