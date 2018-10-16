# David Silver DRL
Algorithms implementation for [David Silver DRL courses](https://www.youtube.com/watch?v=2pWv7GOvuf0) based on Book: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf).

## Iterative policy evaluation 

### environment
* grid world
* Assumption: The agent can choose "hit the wall" action. If our agent choose "hit the wall" action, then the agent stays the same grid for the next state.
* label information, please refer to `Grid_World_Label_Definition.jpeg`

### results
* value function for each state
* possibility of the four actions for each state
* the same as the result shown by David

## value iteration

### environment
* grid world 
* Assumption: The agent can choose "hit the wall" action. If our agent choose "hit the wall" action, then the agent stays the same grid for the next state.
* value iteration is a special case of policy iteration. (Update policy after k=1 step of evaluation.)
### results
* value function for each state
* the same as the result shown by David

## Monte Carlo Control

### environment
* Blackjack
* Game Rule: 
	1. First step : the dealer and player have two cards, one of the dealers is called "showing card"
	2. Second step : the player can choose "stick"(stop getting card) or "twist"(get one more card) until the player's sum is over 21 or choose to stick.
	3. Third step: the dealer's policy is fixed: if the dealer's sum is over 17, twist, otherwise,stick. 
	4. The person whose sum is over 21 loses.
	5. If no one busts, the person whose sum is bigger wins.
	6. Assumption: 1-9 stands for 1-9, 10-13 stands for 10, the probalibility of getting any card is the same for every round (get card with replacement).
	
### results
* policy result (when to twist and when to result)
* average value function 

## Sarsa

### environment
* cliffwalking
* Game Rule: Get to the goal using as less steps as possible
	
### results
* `Sarsa_qlearning_comparison`
* `Sarsa_Qlearning_Track_Comparison.png`
* `cliff_walk_state_transition_different_parameter`
* 


## Q learning

### environment
* cliffwalking
* Game Rule: Get to the goal using as less steps as possible
	
### results
* `Sarsa_qlearning_comparison`
* `Sarsa_Qlearning_Track_Comparison.png`


## Sarsa semi gradient approximator

### environment
* mountain car
* Use TileCoding.py 
* Game Rule: Get to the goal (the right spot of a hill) using as less steps as possible. State is based on position and velocity. Three actions (forward,zero,backward). Q value is approximated by a linear function (X*THETA), where X is a feature vector obtained using method `TileCoding` and THETA is the weight matrix.
	
### results
* `Sarsa_semi_gradient_car_104episode`

## Deep Q Net

### environment
* mountain car
* Use openAI Gym (install: `pip install gym`) and Sklearn
* Game Rule: Get to the goal (the right spot of a hill) using as less steps as possible. State is based on position and velocity. Three actions (forward,zero,backward). Q value is approximated by a one hidden layer neural network.(Input is `position,velocity,action`, output is `q value`)
	
### results
* Run the python file and a video will show you how the car moves

## Actor Critic

### environment
* mountain car
* Use openAI Gym (install: `pip install gym`) and Tensorflow
* Game Rule: Get to the goal (the right spot of a hill) using as less steps as possible. State is based on position and velocity. Three actions (forward,zero,backward). Q value and policy are approximated by two one-hidden-layer neural network respectively.
	
### results
* Run the python file and a video will show you how the car moves

## Actor_critic_with_config

### environment
* mountain car
* Configuration file is for parameter configurarion. To run it, use command like "python Actor_critic_tsf_config.py Actor_Critic.conf  -game=game0" under the project directory.
	
### results
* Run the python file and a video will show you how the car moves
* Uncomment "render()"
