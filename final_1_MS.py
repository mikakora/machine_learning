#Intelligent Systems Final Project
#Michael Sikora

import gym
from gym import wrappers, logger
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

#define learning rate

LR = 1e-3
#define the gym environment
env = gym.make('CartPole-v0')
#reset environment
env.reset()
#set goal steps
goal_steps = 500
score_requirement = 50
initial_games = 100000

#creating random moves for testing
def random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            #create a random action in game to generate more data
            #action = env.action_space.sample()
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if done:
                break
            
#random_games()


# Creating training data
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0 
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0], output])
                
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
            
    return training_data
initial_population()
    


## creating network model


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    
    
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')
    
    return model


#training the model
def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    model.fit({'input':X}, {'targets':y}, n_epoch=2, snapshot_step=500, show_metric=True,
              run_id='cart_model')
    return model

training_data = initial_population()
model = train_model(training_data)



#model.load(final_1_MS_1.model)


#Running the trained model to render
#accepted_scores = []
scores = []
choices = []
#outdir = '/tmp/random-agent-results'
#env = wrappers.Monitor(env, directory=outdir, force=True)
#env.seed(0)

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
        choices.append(action)
        
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)
    
print('Average Score:', sum(scores)/len(scores))


model.save('final_1_MS_2.model')        
            
            

