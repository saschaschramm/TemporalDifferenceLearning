
# Environment
Catch is a simple environment to study reinforcement learning problems. 1 is the player. The goal is to
catch the enemy at the bottom right corner:

```
1 0 0 0
0 0 0 0
0 0 0 0
0 0 0 X
```

# Introduction
Temporal-difference learning is a method to compute the values of all states by sampling the environment. 
It approximates the current estimate of a state value based on previously learned estimates 
(bootstrapping).

The target for the TD update is

``` python
td_target = reward + discount_rate * state_values[next_state].
```

The ```td_target``` is an estimate of the state value. We receive the estimate by sampling 
a successor state  ```state_values[next_state]```. The value of the sampled 
successor state ```state_values[next_state]``` together with the reward is our ```td_target```.

The difference between the estimated state value ```state_values[state]``` and
the better estimate ```td_target``` is called ```td_error``` 

``` python
td_error = td_target - state_values[state].
```
Finally the ```td_error``` is used to update the state value 

``` python
state_values[state] = state_values[state] + learning_rate * td_error.
```

# Example
By applying temporal-difference learning on the catch environment we can approximate 
the state-value function for a random policy:
```
0.07 0.08 0.09 0.14 
0.08 0.11 0.15 0.22 
0.10 0.14 0.27 0.46 
0.11 0.20 0.52 0.00 
```