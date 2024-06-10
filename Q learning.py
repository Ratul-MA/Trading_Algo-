#!/usr/bin/env python
# coding: utf-8


##Attempt_2.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('spy_data.csv')

# Strategy-based trading parameters
portfolio_value = 10000
total_holdings = 0
total_profit = 0
cumulative_portfolio_returns = []

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000
num_states = 2  # Price, Moving Average
num_actions = 2  # Buy, Sell
Q = np.zeros((num_states, num_actions))

# Define selling condition for both algorithms
sell_profit_threshold = 1.03  # 3% higher than SMA

# Initialize lists to store performance metrics
strategy_returns = []
q_learning_returns = []

# Strategy-based trading
for index, row in df.iterrows():
    if row['Moving_Average'] is not None:
        if total_holdings == 0 and row['Close_Price'] < row['Moving_Average']:
            # Buy condition
            buy_price = row['Close_Price']
            shares_to_buy = int(portfolio_value / buy_price)
            total_holdings += shares_to_buy
            portfolio_value -= buy_price * shares_to_buy
        elif total_holdings > 0 and row['Close_Price'] >= sell_profit_threshold * buy_price:
            # Sell condition
            selling_price = row['Close_Price']
            profit_loss = (selling_price - buy_price) * total_holdings
            total_profit += profit_loss
            portfolio_value += selling_price * total_holdings
            total_holdings = 0
    
    # Calculate cumulative portfolio returns
    total_portfolio_value = portfolio_value + total_holdings * row['Close_Price']
    cumulative_return = ((total_portfolio_value - 10000) / 10000) * 100
    cumulative_portfolio_returns.append(cumulative_return)
    strategy_returns.append(total_portfolio_value)

# Define discretization parameters
price_bins = np.linspace(df['Close_Price'].min(), df['Close_Price'].max(), num_states)
sma_bins = np.linspace(df['Moving_Average'].min(), df['Moving_Average'].max(), num_states)

# Q-learning
for episode in range(num_episodes):
    price = df['Close_Price'].iloc[0]
    sma = df['Moving_Average'].iloc[0]
    
    # Discretize state
    price_bin_index = np.digitize(price, price_bins) - 1
    sma_bin_index = np.digitize(sma, sma_bins) - 1
    state_index = price_bin_index
    
    done = False
    episode_return = 0
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(num_actions)  # Explore
        else:
            action = np.argmax(Q[state_index, :])  # Exploit
        
        if action == 0 and price < sma:
            # Buy action
            buy_price = price
            shares_to_buy = int(portfolio_value / buy_price)
            total_holdings += shares_to_buy
            portfolio_value -= buy_price * shares_to_buy
            next_state = state_index  # Next state is the same as current state
            reward = price * 0.03  # Reward for buying below SMA
        elif action == 1 and total_holdings > 0 and price >= sell_profit_threshold * buy_price:
            # Sell action
            selling_price = price
            profit_loss = (selling_price - buy_price) * total_holdings
            total_profit += profit_loss
            portfolio_value += selling_price * total_holdings
            total_holdings = 0
            next_state = state_index  # Next state is the same as current state
            reward = profit_loss  # Reward for selling at profit
        else:
            # Hold action
            next_state = state_index  # Next state is the same as current state
            reward = 0  # No reward
        
        # Update Q-value
        best_next_action = np.argmax(Q[next_state, :])
        td_target = reward + gamma * Q[next_state, best_next_action]
        td_delta = td_target - Q[state_index, action]
        Q[state_index, action] += alpha * td_delta
        
        # Transition to next state
        state_index = next_state
        episode_return += reward
    
    q_learning_returns.append(portfolio_value)


# Plot performance comparison
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], strategy_returns, label='Strategy-based Trading', color='blue')
plt.plot(df['Date'], q_learning_returns, label='Q-learning', color='green')
plt.title('Performance Comparison: Strategy-based Trading vs Q-learning')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




