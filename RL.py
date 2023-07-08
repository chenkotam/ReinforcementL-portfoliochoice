'''
File: RL.py
Project: Reinforcement Learning Examples

File Created: Tuesday, 4th Jul 2023 11:37:00 am

Author: Zhenhao Tan (zhenhao.tan@yale.edu)
-----
Last Modified: Tuesday, 8th Jul 2023 3:45:00 pm
-----
Description: This file shows a few examples of Q-learning and Deep Q-learning
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import gym
from IPython.display import clear_output
from time import sleep
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,InputLayer
from keras.layers import Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.initializers import he_normal
import statsmodels.api as sm
import seaborn as sns
from gym import error, spaces, utils
from gym.utils import seeding

#%%
'''
====================================================================
Taxi example via Q-Learning
====================================================================
'''

'''Set up environment'''
env = gym.make("Taxi-v3").env 
env.seed(34)
env_shape = [env.observation_space.n, env.action_space.n]


'''Training'''
# Q learning in a few lines of python
Q = np.zeros([env.observation_space.n, env.action_space.n])
r_avg_list_Qt = []
for ep in range(1000):
    clear_output(wait=True)
    print('Episode: %d' % (ep+1))
    state , done = env.reset(), False
    r_sum=0
    while not done:
        if np.random.uniform() < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        newstate, reward, done, info = env.step(action) 
        y = reward + 0.9 * np.max(Q[newstate])
        Q[state,action] -= 0.1 * (Q[state,action]-y)          
        state = newstate
        r_sum += reward
#         print(action)
    r_avg_list_Qt.append(r_sum)


'''Visualize Performance with Q-table'''
# trained agent
env.seed(56) # to make it screw up , change seed to 234
state , done = env.reset(), False
while not done:    
    # optimal action
    action = np.argmax(Q[state])
    
    # get new state and reward
    state, reward, done, _ = env.step(action)
    
    clear_output(wait=True)
    env.render()
    sleep(.5)


#%%
'''
====================================================================
Taxi example via Deep Q-Learning (Method 1)
====================================================================
'''
'''Build the Neural Network Model'''
hidden_layers = [10]

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env_shape[0])))
for layer in hidden_layers:
    model.add(Dense(layer, activation='relu'))
model.add(Dense(env_shape[1], activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()


'''Training with NN'''
seed_num = 34
env.seed(seed_num)
np.random.seed(seed_num)
tf.random.set_seed(seed_num)

lambda_ = 0.9
eps = 0.6
decay_factor = 0.999
id_matrix = np.identity(env_shape[0])
ls_record=[]
r_avg_list = []
for ep in range(1000):
    clear_output(wait=True)
    print('Episode: %d' % (ep+1))
    print(len(ls_record))
    state , done = env.reset(), False
    eps *= decay_factor
    r_sum=0
    while not done:
        if np.random.uniform() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(id_matrix[state:state + 1]))
        newstate, reward, done, info = env.step(action)
        # get the target based on Bellman
        y = reward + lambda_ * np.max(model.predict(id_matrix[newstate:newstate + 1]))
        # replace the selected action in Q values vector as the target
        y_vec = model.predict(id_matrix[state:state + 1])[0]
        y_vec[action] = y
        # conduct the update of the parameter
        model.fit(id_matrix[state:state + 1], y_vec.reshape(-1, env_shape[1]), epochs=1, verbose=1) 
        
        if state==1:
            ls_record.append(np.mean((model.predict(id_matrix[state:state + 1]) - y_vec.reshape(-1, env_shape[1]))**2))
        
        state = newstate
            
        r_sum += reward
#         print(action)
    r_avg_list.append(r_sum)


'''Visualize Performance with NN'''
## trained agent
env.seed(56) # to make it screw up , change seed to 234
state , done = env.reset(), False
while not done:    
    # optimal action
    action = np.argmax(model.predict(id_matrix[state:state + 1]))
    
    # get new state and reward
    state, reward, done, _ = env.step(action)
    
    clear_output(wait=True)
    env.render()
    sleep(.5)




#%%
'''
====================================================================
Taxi example via Deep Q-Learning (Method 2)
====================================================================
'''
# set random seed
RS = 42
np.random.seed(RS)
torch.manual_seed(RS)

class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)



net = Net(env.observation_space.n+1)
net_target = net
gamma = 0.9
decay_factor = 0.999
experience_num = 1
step = -1
C = 0
maxiter_num = 1e3
learning_rate = 0.05
epsilon_begin = 1
epsilon_later = 0.6

id_matrix = np.identity(env.observation_space.n)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
action_new_list = [i for i in range(env.action_space.n)] # get all possible actions to a list
r_avg_list_lec = []

# Q learning in a few lines of python
# Q = np.zeros([env.observation_space.n, env.action_space.n])
# memory=[]
for ep in range(1000):
    clear_output(wait=True)
    print('Episode: %d' % (ep+1))
    state , done = env.reset(), False
    maxiter = 0 # if exceed the max iteration that still not stopped, skip to next loop
#     memory=[]
    decay_factor *= decay_factor
    r_sum=0
    while not done:
        memory=[]
        # epsilon greedy policy
        if ep < 0:
            if np.random.uniform() < epsilon_begin:
                action = env.action_space.sample()
            else:
                act_m=[]
                for act in action_new_list:
                    input_m = np.concatenate((id_matrix[state:state + 1],np.array([[act]])),axis=1)
                    act_m.append(float(net.forward(torch.from_numpy(input_m).float()).detach().numpy()))
                action = np.argmax(act_m)
        else:
            if np.random.uniform() < epsilon_later*decay_factor:
                action = env.action_space.sample()
            else:
                act_m=[]
                for act in action_new_list:
                    input_m = np.concatenate((id_matrix[state:state + 1],np.array([[act]])),axis=1)
                    act_m.append(float(net.forward(torch.from_numpy(input_m).float()).detach().numpy()))
                action = np.argmax(act_m)

        newstate, reward, done, info = env.step(action)

        # save the experience in memory
        memory.append([state,action,reward,newstate])
        
        diff2 = 0
        for m in range(experience_num):
            exp_curr = random.choice(memory)
            state_curr = exp_curr[0]
            action_curr = exp_curr[1]
            reward_curr = exp_curr[2]
            newstate_curr = exp_curr[3]
            
            # get the prediction
            input_curr = np.concatenate((id_matrix[state_curr:state_curr + 1],np.array([[action_curr]])),axis=1)
            Q_state = net.forward(torch.from_numpy(input_curr).float())

            # get the reward for all actions with newstate
            act_max=[]
            for act in action_new_list:
                input_max = np.concatenate((id_matrix[newstate_curr:newstate_curr + 1],np.array([[act]])),axis=1)
                act_max.append(net_target.forward(torch.from_numpy(input_max).float()))

            # compute the true Q using the info in newstate based on Bellman
            y = reward_curr + gamma * max(act_max)
            
            diff2 += (y - Q_state)**2
        
        loss = diff2 / (2*experience_num)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = newstate
        
        # for every C step, update the target
        if C > step:
            net_target = net
            C = 0
        else:
            C += 1
#         print(action)
        if maxiter > maxiter_num:
            maxiter = 0
            break
        else:
            maxiter += 1
        
        r_sum += reward
    r_avg_list_lec.append(r_sum)


'''Visualize Performance with NN'''
env.seed(56) # to make it screw up , change seed to 234
action_new_list = [i for i in range(env.action_space.n)] # get all possible actions to a list
state , done = env.reset(), False
while not done:    
    # optimal action
    act_m=[]
    for act in action_new_list:
        input_m = np.concatenate((id_matrix[state:state + 1],np.array([[act]])),axis=1)
        act_m.append(float(net.forward(torch.from_numpy(input_m).float()).detach().numpy()))
    action = np.argmax(act_m)
    
    # get new state and reward
    state, reward, done, _ = env.step(action)
    
    clear_output(wait=True)
    env.render()
    sleep(.5)


#%%
'''
====================================================================
Preparation - Merton Portfolio Choice with RL
====================================================================
'''
df = pd.read_csv('src/mkt_rf_dp.csv')
df.head()

## Define functions to compute RSI
def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)

## Compute RSI
df['RSI_6D'] = df["Ret"].rolling(center=False,window=6).apply(rsi)
df['RSI_12D'] = df["Ret"].rolling(center=False,window=12).apply(rsi)

## Excess Return
df['RetEx'] = df['Ret'] - df['Rfree']

#------------------------------------------------------------------

'''Build Signal: 1 if lagged DP is above median in prepro sample, 0 otherwise'''
## Make lags of DP to select the best lag number as signal
lags = []
for k in range(1,80):
    df['D/P_L%d' %k] = df['D/P'].shift(k)
    lags.append('D/P_L%d' %k)

## look at a "prepro" sample of data
n_pre = 750
data = df.dropna().loc[0:n_pre]

## cochrane style regression on each lag to check performance of each lag
stats = []
for lag in lags:
    X = sm.add_constant(data[lag])
    y = data['RetEx']
    res = sm.OLS(y,X).fit()
    stats.append([res.params[1],res.tvalues[1],res.rsquared])
        
stats = pd.DataFrame(stats,index=range(1,len(stats)+1),columns=['beta','t','R2'])
f = stats.plot(subplots=True,figsize=(10,8))
l = f[2].set(xlabel='Lag in regression: Excess return = a + b(lagged dp) + error')

## Select the best lag and build the signal
medi = df.loc[0:n_pre,'D/P_L12'].median()
df['signal'] = (df['D/P_L12'] > medi).astype('int')

## does this have any hope out of sample? (Check performance OOS)
sns.pointplot(data=df[n_pre:],x='signal',y='RetEx',ci=90)

#------------------------------------------------------------------

'''Build Signal: RSI'''
## RSI signal: 
df['signal_RSI'] = (df['RSI_6D'] > 50).astype('int')

## does this have any hope out of sample? (Check performance OOS)
sns.pointplot(data=df[n_pre:],x='signal_RSI',y='RetEx',ci=90)

#------------------------------------------------------------------

'''Build Signal: MACD'''
def calculateEMA(period, closeArray, emaArray=[]):
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan],(nanCounter + period - 1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])    
        emaArray.append(firstema)    
        for i in range(nanCounter+period,length):
            ema=(2*closeArray[i]+(period-1)*emaArray[-1])/(period+1)
            emaArray.append(ema)        
    return np.array(emaArray)

def calculateMACD(closeArray,shortPeriod = 12 ,longPeriod = 26 ,signalPeriod =9):
    ema12 = calculateEMA(shortPeriod ,closeArray,[])
    #print ema12
    ema26 = calculateEMA(longPeriod ,closeArray,[])
    #print ema26
    diff = ema12-ema26
    
    dea= calculateEMA(signalPeriod ,diff,[])
    macd = 2*(diff-dea)
    return macd,diff,dea

## Build the MACD signal
macd = calculateMACD(df['RetEx'])[0]
df['macd'] = macd
df['signal_macd'] = (df['macd'] > 0).astype('int')

## does this have any hope out of sample? (Check performance OOS)
sns.pointplot(data=df[n_pre:],x='signal_macd',y='RetEx',ci=90)


## save data for the agent
data = df[['yyyymm','RetEx','signal','signal_RSI','signal_macd']]
data.to_csv('esttab/ret_signal_discrete.csv',index=False)


#%%
'''Gym environment (Build environment for RL)'''
class MertonDiscreteEnv(gym.Env):

    def __init__(self,crra=1,horizon=12,rf=0.002):
        
        # economic parameters 
        self.rf = rf # risk free rate
        self.crra = crra # coefficient of relative risk aversion
        self.horizon = horizon # number of investment periods
        
        # data
        df = pd.read_csv('esttab/ret_signal_discrete.csv')  # data with stock returns and 0/1 signal
        self.n_pre = 750 # number of periods we have burned for pre-processing
        self.lookback = 500 # number of periods for merton calculation
        
        # merton calculation
        df['logret'] = np.log(1+df['RetEx']+self.rf) # log returns
        roll = df['logret'].rolling(self.lookback) # rolling window
        mu, sigma2 = roll.mean(), roll.var() # mean and variance
        df['merton'] = (mu+0.5*sigma2-self.rf)/(self.crra*sigma2) # optimal portfolio
        
        # save data used by agent
        self.data = df[n_pre:].dropna().reset_index(drop=True)
        
        # gym setup
        self.observation_space=spaces.Discrete((self.horizon+1)*8) # states = (periods of life) x (signals)
        self.action_space=spaces.Discrete(3) # invest nothing, merton, or 2*merton
        self.reset()

    def step(self, action):
        
        # portfolio
        self.merton = self.data['merton'].iloc[self.date]
        self.port =  self.merton + (action-1) * 0.5
        
        # stock return from next period
        Re = self.data['RetEx'].iloc[self.date]

        # update wealth
        self.Rp = 1 + self.rf + self.port * Re 
        self.wealth *= self.Rp
        self.history.append(self.wealth)
        
        # housekeeping
        self.date += 1
        self.life -= 1
        self.dp = self.data['signal'].iloc[self.date]
        self.rsi = self.data['signal_RSI'].iloc[self.date]
        self.macd = self.data['signal_macd'].iloc[self.date]
        
        if self.life == 0: # stop investing and enjoy utility
            return self.state(), self.utility(), True, {} 
        else:
            return self.state(), 0, False, {} 

    def reset(self):
        # wealth
        self.wealth = 100
        self.history = [self.wealth]
        
        # time
        self.date = np.random.choice(len(self.data)-self.horizon) # random birthday 
        self.life = self.horizon # periods of life left
        
        # dp
        self.dp = self.data['signal'].iloc[self.date]
        
        # rsi
        self.rsi = self.data['signal_RSI'].iloc[self.date]
    
        # macd
        self.macd = self.data['signal_macd'].iloc[self.date]
        return self.state()

    def render(self):
        ym = str(self.data['yyyymm'].iloc[self.date])
        print('*** RoboMerton ***')
        print('It is %s/%s, RoboMerton has %d months left' % (ym[4:],ym[0:4],self.life))
        print('Signal: log(D/P) bin = %d' % self.dp)
        print('Signal_RSI: RSI bin = %d' % self.rsi)
        print('Signal_macd: macd bin = %d' % self.macd)
        print('Bank balance: $%.2f'% self.wealth)
        if self.life != self.horizon: 
            print('Robert Merton\'s portfolio: %.2f pct stocks' % (self.merton*100))
            print('RoboMerton\'s portfolio:    %.2f pct stocks' % (self.port*100))
        
    def utility(self):
        if self.crra == 1:
            return np.log(self.wealth)
        else:
            return (self.wealth**(1-self.crra)-1)/(1-self.crra)
    
    def state(self):
        # returns state index: sorted by period then signal
#         index = 2*self.life + self.dp
        index = 8*self.life + self.dp + self.rsi*2 + self.macd*4
        
        assert self.observation_space.contains(index)
        return index

#%%
'''Agent Testing Before Training'''
## look at data the agent sees
env = MertonDiscreteEnv()
print(env.data.head(10))
f = env.data[['RetEx','signal','signal_RSI','signal_macd','merton']].plot(subplots=True,figsize=(10,8))
plt.savefig('gph/stock_signals.pdf')


## watch a random agent
state , done = env.reset(), False
while not done:
    
    # random action
    action = env.action_space.sample() 
    
    # get new state and reward
    state, reward, done, info = env.step(action) 
    
    clear_output(wait=True)
    env.render()
    sleep(1)


## Test the performance before training
# benchmarking 
env = MertonDiscreteEnv()
env.seed(56)
np.random.seed(20235)
robo, robert = [], []

# run many episodes to get average utility
for ep in range(20000):
    
    # episode with random agent
    state , done = env.reset(), False
    while not done:
        action = env.action_space.sample() 
        state, reward, done, info = env.step(action)
    robo.append(reward)
    
    # episode with naive merton
    state , done = env.reset(), False
    while not done:
        action = 1 
        state, reward, done, info = env.step(action)
    robert.append(reward)
    
    if (ep+1) % 500 == 0:
        clear_output(wait=True)    
        print('*** Robo vs Robert ***')
        print('Episode %d' % (ep+1))
        roi_robo, roi_robert = np.exp(robo)/100-1-env.rf*env.horizon, np.exp(robert)/100-1-env.rf*env.horizon
        print('RoboMerton average: \t utility %.4f\t  Re %.4f' % \
            (np.mean(robo),np.mean(roi_robo)))
        print('Robert Merton average: \t utility %.4f\t  Re %.4f' % \
            (np.mean(robert),np.mean(roi_robert)))   

#%%
'''
====================================================================
Training - Merton Portfolio Choice with RL -> Q-Learning
====================================================================
'''
## Q learning
seed_num_pc = 34
env = MertonDiscreteEnv(horizon=10)
env.seed(seed_num_pc)
np.random.seed(seed_num_pc)

## we learn return on wealth (q) instead of full value (Q=q+log(w))
q = np.zeros([env.observation_space.n, env.action_space.n])
visits = np.zeros_like(q)
ys = np.zeros_like(q)
lrs = []
lrs0 = []
n=0
alpha = 0.1
for ep in range(100000):
    
    # initialise episode
    state , done = env.reset(), False
      
    # set learning rate and exploration parameters
    alpha = 1e-3 if ep < 10000 else 1e-4
    epsilon = 1 if ep < 10000 else 0.1
    
    # play 
    while not done:
        
        # action and step
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state])
        newstate, reward, done, info = env.step(action) 
        
        # q target
        y = np.log(env.Rp) + np.max(q[newstate])

        # q update
        q[state,action] -=  alpha * (q[state,action]-y)
        
        # track learning rates (for debugging)
        visits[state,action] += 1
        lr0 = 1/(visits[state,action])
        lrs0.append(lr0)
        lrs.append(alpha)
        
        # next state
        state = newstate
    
    # counter
    if (ep+1) % 1000 == 0:
        clear_output(wait=True)
        print('Experience: %d episodes' % (ep+1))
       

## visualise learned value and policy
plt.figure()
plt.plot(pd.Series(lrs).rolling(1000).mean(),label='actual')
plt.plot(pd.Series(lrs0).rolling(1000).mean(),label='ideal')
plt.yscale('log')
plt.title('learning rates')
plt.legend()

sig_col = ['signal=0,0,0','signal=1,0,0','signal=0,1,0','signal=1,1,0','signal=0,0,1','signal=1,0,1','signal=0,1,1','signal=1,1,1']
pd.DataFrame(np.max(q,axis=1).reshape(-1,8),columns=sig_col)[1:].plot()
plt.title('Max return on wealth q')
plt.xlabel('horizon')
plt.savefig('gph/stock_return.pdf')

pd.DataFrame(np.argmax(q,axis=1).reshape(-1,8),columns=sig_col)[1:].plot()
plt.title('Policy')
plt.xlabel('horizon')
plt.savefig('gph/stock_policy.pdf')






#%%
'''
====================================================================
Training - Merton Portfolio Choice with RL -> Deep Q-Learning
====================================================================
'''
## Build up the Neural Network Model
hidden_layers = [64,32,16]
learning_rate = 0.0003
model_pc = Sequential()
model_pc.add(InputLayer(batch_input_shape=(1, env_shape[0])))
for layer in hidden_layers:
    model_pc.add(Dense(layer, activation='relu',
                       kernel_initializer=he_normal()))
model_pc.add(Dense(env_shape[1], activation='linear',
                   kernel_initializer=he_normal()))
model_pc.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['mae'])
model_pc.summary()

## Training
seed_num_pc = 34
env = MertonDiscreteEnv(horizon=10)
env.seed(seed_num_pc)
np.random.seed(seed_num_pc)
#tf.set_random_seed(seed_num_pc)

# we learn return on wealth (q) instead of full value (Q=q+log(w))
visits = np.zeros([env.observation_space.n, env.action_space.n])
lrs = []
lrs0 = []
n=0
alpha = 0.1
id_matrix = np.identity(env_shape[0])
for ep in range(10000):
    
    # initialise episode
    state , done = env.reset(), False
      
    # set learning rate and exploration parameters
    alpha = 1e-3 if ep < 10000 else 1e-4
    epsilon = 1 if ep < 1000 else 0.1
    
    # play 
    while not done:
        
        # action and step
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model_pc.predict(id_matrix[state:state + 1]))
        newstate, reward, done, info = env.step(action) 
        
        # q target
        y = np.log(env.Rp) + np.max(model_pc.predict(id_matrix[newstate:newstate + 1]))

        # replace the selected action in Q values vector as the target
        y_vec = model_pc.predict(id_matrix[state:state + 1])[0]
        y_vec[action] = y
        
        # q update
        model_pc.fit(id_matrix[state:state + 1], y_vec.reshape(-1, env_shape[1]), epochs=1, verbose=0)
        
        # track learning rates (for debugging)
        visits[state,action] += 1
        lr0 = 1/(visits[state,action])
        lrs0.append(lr0)
        lrs.append(alpha)
        
        # next state
        state = newstate
    
    # counter
    if (ep+1) % 100 == 0:
        clear_output(wait=True)
        print('Experience: %d episodes' % (ep+1))
       

# visualise learned value and policy
plt.figure()
plt.plot(pd.Series(lrs).rolling(1000).mean(),label='actual')
plt.plot(pd.Series(lrs0).rolling(1000).mean(),label='ideal')
plt.yscale('log')
plt.title('learning rates')
plt.legend()

q_value = []
for i in range(env_shape[0]):
    q_value.append(model_pc.predict(id_matrix[i,:][None,:]).tolist()[0])
q_value = np.array(q_value)

sig_col = ['signal=0,0,0','signal=1,0,0','signal=0,1,0','signal=1,1,0','signal=0,0,1','signal=1,0,1','signal=0,1,1','signal=1,1,1']
pd.DataFrame(np.max(q_value,axis=1).reshape(-1,8),columns=sig_col)[1:].plot()
plt.title('Max return on wealth q')
plt.xlabel('horizon')
plt.savefig('stock_return_NN.pdf')

pd.DataFrame(np.argmax(q_value,axis=1).reshape(-1,8),columns=sig_col)[1:].plot()
plt.title('Policy')
plt.xlabel('horizon')
plt.savefig('stock_policy_NN.pdf')

#%%
'''
====================================================================
Checking Performance - Portfolio Choice with RL
====================================================================
'''
## benchmarking 
env = MertonDiscreteEnv(horizon=10)
env.seed(56)
np.random.seed(20235)
robo, robert, robo_NN = [], [], []

## run many episodes to get average utility
for ep in range(20000):
    
    # episode with Q-learning (Q table)
    state , done = env.reset(), False
    while not done:
        action = np.argmax(q[state])
        state, reward, done, info = env.step(action)
    robo.append(reward)
    
    # episode with naive merton
    state , done = env.reset(), False
    while not done:
        action = 1 
        state, reward, done, info = env.step(action)
    robert.append(reward)

    # episode with Q-learning (NN)
    state , done = env.reset(), False
    while not done:
        action = np.argmax(model_pc.predict(id_matrix[state:state + 1]))
        state, reward, done, info = env.step(action)
    robo_NN.append(reward)
    
    if (ep+1) % 500 == 0:
        clear_output(wait=True)    
        print('*** Robo vs Robert vs Robo_NN ***')
        print('Episode %d' % (ep+1))
        roi_robo, roi_robert, roi_robo_NN = np.exp(robo)/100-1-env.rf*env.horizon, np.exp(robert)/100-1-env.rf*env.horizon, np.exp(robo_NN)/100-1-env.rf*env.horizon
        print('RoboMerton average: \t utility %.4f\t  Re %.4f' % \
            (np.mean(robo),np.mean(roi_robo)))
        print('Robert Merton average: \t utility %.4f\t  Re %.4f' % \
            (np.mean(robert),np.mean(roi_robert)))
        print('RoboMerton_NN average: \t utility %.4f\t  Re %.4f' % \
            (np.mean(robo_NN),np.mean(roi_robo_NN)))











