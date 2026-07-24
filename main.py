import numpy as np
import yfinance as yf

weights = np.array([.5, .3, .2])

prices = yf.download(["AAPL", "MSFT", "NVDA"], period="5y")

closing_prices = prices["Close"]

returns = closing_prices.pct_change()
returns = returns.dropna()

average_returns = returns.mean()
annualized_returns = (1 + returns).prod() ** (252 / len(returns)) - 1

covariance_matrix = returns.cov() * 252

#Caluculate expected portfolio return
portfolio_return = np.dot(weights, annualized_returns)
return_percentage = portfolio_return * 100

#Calculate portfolio variance using the covariance matrix
portfolio_variance = weights.T @ covariance_matrix @ weights

#Convert variance into volatility
portfolio_volatility = np.sqrt(portfolio_variance)
volatility_percentage = 100 * portfolio_volatility

#Calculate sharpe ratio
risk_free_rate = 0.03
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility


print("--------------------")
print("Portfolio Analysis")
print("--------------------")
print()
print(f"Portfolio Return: {return_percentage:.2f}%")
print(f"Portfolio Variance: {portfolio_variance:.4f}")
print(f"Portfolio Volatility: {volatility_percentage:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

