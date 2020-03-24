# http://github.com



# Working through MEAP Machine Learning w/ TensorFlow Book
# added a few things to their sample code 

# first pass stock estimates using random policy reinforcement learning 
# Unimpressed - tried various look back time and other parameters
# still does about the same as random guesses but with a wider STD
# More important - it's not clear why the network makes the trades it does



import tensorflow as tf
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import layout, row


#####################################################################################
# constants
# stock, money, shares
budget = 1000.              # dollars
n_stocks = 0                # starting shares
stock = 'MSFT'
start_date = '2000-01-01'
end_date = '2017-04-01'



# network constants
history = 20                # about 20 trading days/month       
n_hidden1 = history

####################################################################################
# functions

# fetch prices from finance.yahoo.com or save and re-use?
def get_prices (share_symbol, start_date, end_date, ):
    df = pd.read_csv(f"./data/{stock}_US.csv", parse_dates=["Date"])
    dates = df["Date"][21:]
    df = df.set_index('Date')
    stock_prices = df["Open"]

    print(len(stock_prices ))

    return stock_prices, dates


def plot_prices(prices):

    plt.title('Opening stock prices')
    plt.xlabel('Day')
    plt.ylabel('Price $')
    plot.plot(prices)
    plt.savefig('prices.png')



class DecisionPolicy:

    def select_action(self, current_state):
        pass

    def update_Q(self, state, action, reward, next_state):
        pass 



class DecisionPolicy:

    def select_action(self, current_state, step):
        pass

    def update_Q(self, state, action, reward, next_state):
        pass 




class RandomDecisionPolicy(DecisionPolicy):

    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action 



class QLearningDecisionPolicy(DecisionPolicy):

    def __init__(self, actions, n_input):
        
        self.epsilon = 0.8      # probability of chosing best action over random action
        self.gamma = 0.01      # time discount for past data
        self.actions = actions
        learning_rate = 0.01
        n_output = len(actions)

        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [n_output])
        
        W1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
        b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden1]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        W2 = tf.Variable(tf.random_normal([n_hidden1, n_output]))
        b2 = tf.Variable(tf.constant(0.1, shape=[n_output]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def select_action(self, current_state, step):

        threshold = min(self.epsilon, step / 1000.)

        if random.random() < threshold:
            action_q_vals = self.sess.run(self.q, {self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]

        return action


    def update_Q(self, state, action, reward, next_state):

        action_q_vals = self.sess.run(self.q, {self.x: next_state})
        next_action_q_vals = self.sess.run(self.q, {self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        
        self.sess.run(self.train_op, {self.x: state, self.y: action_q_vals})



def run_simulation(policy, initial_budget, initial_number_stocks, prices, history, debug=False):

    budget = initial_budget
    n_stocks = initial_number_stocks
    share_value = 0
    plan = []
    transitions = list()

    for i in range(len(prices) - history - 1):

        current_state = np.asmatrix(np.hstack((prices[i:i+history], budget, n_stocks)))
        current_portfolio = budget + n_stocks + share_value
        action = policy.select_action(current_state, i)
        share_value = float(prices[i + history + 1])

        if action == 'Buy' and budget >= share_value:
            budget -= share_value 
            n_stocks += 1

        elif action == 'Sell' and n_stocks > 0:
            budget += share_value
            n_stocks -= 1 

        else:
            action == 'Hold'

        new_portfolio = budget + n_stocks * share_value
        reward = new_portfolio - current_portfolio
        next_state = np.asmatrix(np.hstack((prices[i+1:i+history+1], budget, n_stocks)))
        transitions.append((current_state, action, reward, next_state))

        policy.update_Q(current_state, action, reward, next_state)
        plan.append((action, share_value))
        
    portfolio = budget + n_stocks * share_value
    return portfolio, plan



def run_simulations(policy, budget, n_stocks, prices, history):
    n_tries = 10
    final_portfolios = list()
    final_policies = list()
    final_plans = list()

    for i in range(n_tries):
        final_portfolio, final_plan = run_simulation(policy, budget, n_stocks, prices, history)
        final_portfolios.append(final_portfolio)
        final_plans.append(final_plan)

    avg, std = np.mean(final_portfolios), np.std(final_portfolios)
    return avg, std, final_plans






##################################################################################################
# run_simulation
################################################################################################
prices, dates = get_prices(stock, start_date, end_date)

actions = ['Buy', 'Sell', 'Hold']
policy = RandomDecisionPolicy(actions)

budget = 1000.
n_stocks = 0
n_hidden1 = history

avg, std, plans = run_simulations(policy, budget, n_stocks, prices, history)


last_plan = plans[len(plans) -1]
prices = prices[history:-1]

fig = plt.figure(figsize=(24,16))
ax = fig.add_subplot(1,1,1)
d = 0
for p in last_plan:
    action, price = p 
    d += 1
   
    if action == 'Buy': 
        ax.scatter(d, price, c='green', alpha=0.7, s=12)
    if action == 'Sell': 
        ax.scatter(d, price, c='red', alpha=0.7, s=12)


# compute how well we did
years = len(last_plan) / 251
earnings = (avg - budget) / budget * 100.
net = avg - budget 
roi = np.power((budget + net/budget), (1./years)) 


# feedback to user
print("Avg earnings", avg)
print("Std", std)
print("Years invested", years)
print("Total earnings %.2lf%%" % (earnings) )
print("Yearly ROI ~ %.2lf%%" %(roi) )



# plot 
#plt.title("Random policy")
#subtitle = 'Buys are green, Sells are red, avg return on $1000   = %d over %d trading days' % (int(avg), len(last_plan))
#plt.suptitle(subtitle)
#plt.plot(prices, c='black')
#plt.show()
p1 = figure(plot_width=920, plot_height=402, x_axis_type="datetime")
p1.line(dates, prices, color="red", legend_label='Prediction')
#p1.line(df2[cut_off+forward_lag:]["Date"], df2[cut_off+forward_lag:]["Close_change"], color="blue", legend_label='Real')
#p1.line(df2[cut_off+forward_lag:]["Date"], df2[cut_off+forward_lag:]["SMA_10"], color="darkgreen" ,legend_label='Sma_10')
p1.title.text = f"timestocome-rl"
p1.legend.location = "top_left"
p1.grid.grid_line_alpha = 0.1
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'
window_size = 30
window = np.ones(window_size)/float(window_size)
chart_title = "timestocome-rl"
output_file(f"{chart_title}.html", title=chart_title)