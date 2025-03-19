# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
Explain the problem statement.

## POLICY ITERATION ALGORITHM
Include the steps involved in policy iteration algorithm
</br>
</br>

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
</br>

### 2. Policy, Value function and success rate for the Improved Policy
</br>
</br>

### 3. Policy, Value function and success rate after policy iteration
</br>
</br>


## RESULT:

Thus, a program is developed to perform policy iteration for the given MDP.
