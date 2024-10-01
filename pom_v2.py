import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

df = pd.read_csv('Data/predicted_prices_csv.csv')

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    np.random.seed(42)
    for i in range(num_portfolios):
        weights = np.random.random(6)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


returns = df.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 10000
risk_free_rate = 0.0178

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients



def find_efficient_values(mean_returns, cov_matrix, risk_free_rate):
    results, random_weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_coef = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    std, returns = portfolio_annualised_performance(max_sharpe_coef['x'], mean_returns, cov_matrix)
    max_sharpe_coef_allocation = pd.DataFrame(max_sharpe_coef.x, index=df.columns, columns=['allocation'])
    max_sharpe_coef_allocation.allocation = [round(i*100,2) for i in max_sharpe_coef_allocation.allocation]
    max_sharpe_coef_allocation = max_sharpe_coef_allocation.T

    min_volatility = min_variance(mean_returns, cov_matrix)
    std_min, returns_min = portfolio_annualised_performance(min_volatility['x'], mean_returns, cov_matrix)
    min_volatility_allocation = pd.DataFrame(min_volatility.x, index = df.columns, columns=['allocation'])
    min_volatility_allocation.allocation = [round(i*100,2) for i in min_volatility_allocation.allocation]
    min_volatility_allocation = min_volatility_allocation.T

    return returns, std, max_sharpe_coef_allocation, returns_min, std_min, min_volatility_allocation, results



def show_efficient_values(mean_returns, cov_matrix, risk_free_rate):
    returns, std, max_sharpe_coef_allocation, returns_min, std_min, min_volatility_allocation, results = find_efficient_values(mean_returns, cov_matrix, risk_free_rate)
    print ("-" * 60)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(returns,2))
    print ("Annualised Volatility:", round(std,2))
    print ("\n")
    print (max_sharpe_coef_allocation)
    print ("-" * 60)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(returns_min,2))
    print ("Annualised Volatility:", round(std_min,2))
    print ("\n")
    print (min_volatility_allocation)


def show_efficient_graphic(mean_returns, cov_matrix, risk_free_rate):
    returns, std, max_sharpe_coef_allocation, returns_min, std_min, min_volatility_allocation, results = find_efficient_values(mean_returns, cov_matrix, risk_free_rate)
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='GnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(std,returns,marker='*',color='tomato',s=500, label='Maximum Sharpe ratio')
    plt.scatter(std_min,returns_min,marker='*',color='seagreen',s=500, label='Minimum Volatility')

    target = np.linspace(returns_min, 0.6, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='Efficient frontier')
    plt.title('Efficient Frontier', fontweight='bold')
    plt.xlabel('Annualised volatility', fontweight='bold')
    plt.ylabel('Annualised returns', fontweight='bold')
    plt.legend(labelspacing=0.9)
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()

show_efficient_graphic(mean_returns, cov_matrix, risk_free_rate)