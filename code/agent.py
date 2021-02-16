import gym
import gym_snake
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import autokeras as ak


# Construct Environment
def makeModel():
    model = ak.AutoModel(inputs=[ak.StructuredDataInput(), ak.StructuredDataInput()],outputs=[ak.RegressionHead( metrics=['mse'])],overwrite=False,max_trials=1)
    #reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)

    return model
def createSnakeGame(env,grid_object):
    env.seed(123)
    gameFrames =np.ndarray(None)
    gameActions =np.ndarray(None)
    gameRewards =np.ndarray(None)
    model = makeModel()
    lastFood = 0
    for _ in range(500):
        
        
        gameCount = 1
        if True:
        
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #env.step(env.action_space.sample()) # take a random action
            frameData = grid_object.grid
            print(type(action)) # pandas.DataFrame
            print(str(reward))
           #gameFrames = gameFrames.insert(frameData)
            #gameActions = gameActions.insert(action)
            # = gameRewards.insert(reward)
            #model.fit([frameData,np.asarray(action, dtype=np.float32)], np.asarray(reward, dtype=np.int32), epochs=1,overwrite=True,batchSize=1)
            
            if lastFood == 0:
                lastFood = reward
            else:
                #print(model.fit([np.asarray(gameActions, dtype=np.int64),np.asarray(frameData, dtype=np.float32)], np.asarray(reward, dtype=np.int32), epochs=1,overwrite=False,batchSize=1))
                lastFood = lastFood + 1
                env.render()
                print("Food increased to " + str(lastFood))
                lastFood = 0
                #print(model.fit([np.asarray(gameActions, dtype=np.int8),np.asarray(frameData, dtype=np.float32)], np.asarray(reward, dtype=np.int32), epochs=1,overwrite=False,batchSize=1))
                
                
        if game_controller.snakes[0] == None:
            
            print("Game over Rewards " + str(reward) + " | Total Food Taken: " + str((lastFood)))
            print(action)
         
            env.reset()
    return gameFrames,gameActions,gameRewards
env = gym.make('snake-v0')
observation = env.reset()
# Controller
print(env)
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]
grid_object = game_controller.grid
grid_pixels = grid_object.grid



nb_actions = env.action_space.n
#create model 

isFirst = True
x,z,y = createSnakeGame(env,grid_object)
print(type(x)) # pandas.DataFrame
print(type(z)) # pandas.DataFrame
print(type(y)) # pandas.DataFrame

#x = np.asarray(x, dtype=np.float32)
#z = np.asarray(z, dtype=np.int32)
#y = np.asarray(y, dtype=np.float32)
print(str(x.shape))
env.reset()


print(str(x.shape))
# Fit the model with prepared data.


#model.fit([x,z], y, epochs=2)
print("done training")
env.reset()

for _ in range(10000):
    env.render()
    gamePred = model.predict(env.grid_object.grid)
    env.step(gamePred)

# Feed the tensorflow Dataset to the classifier.
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
#dqn.fit(env, nb_steps=500000, visualize=True, verbose=2)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format("snake-v0"), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=True)