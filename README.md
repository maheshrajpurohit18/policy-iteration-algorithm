# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
### Step1 :
we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

### Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.

## PROGRAM:
NAME : Mahesh Raj Purohit J

REGISTER NUMBER : 212222240058



## POLICY IMPROVEMENT FUNCTION

```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state,reward, done in P[s][a]:
          Q[s][a]+= prob*(reward+gamma*V[next_state]*(not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi

```


## POLICY ITERATION FUNCTION

```python
def policy_iteration(P, gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
  return V,pi

```


## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
</br>

![image](https://github.com/user-attachments/assets/7122d0c0-cd25-4aac-bc5a-78aebbcb9d53)


![image](https://github.com/user-attachments/assets/396d9969-f7a8-4a44-92b4-8c0531eba862)


</br>

### 2. Policy, Value function and success rate for the Improved Policy
</br>

![image](https://github.com/user-attachments/assets/1438a40b-b2c0-4a55-bb6c-45541f5ad9aa)



![image](https://github.com/user-attachments/assets/72b8130b-1419-4ce0-a026-734381441553)


</br>

### 3. Policy, Value function and success rate after policy iteration
</br>

![image](https://github.com/user-attachments/assets/a1153e0a-8e17-4efa-8a43-3ab8c203b6fe)

![image](https://github.com/user-attachments/assets/778864a6-bd9b-4982-830b-bc9054e95a85)


![image](https://github.com/user-attachments/assets/8d95bb7b-5ee8-4129-a5ec-7d6c460b4f3b)



</br>


## RESULT:

Thus, a program is developed to perform policy iteration for the given MDP.
