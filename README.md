# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>


## POLICY IMPROVEMENT FUNCTION
### Name : Shalini V
### Register Number : 212222240096
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
### Name : Shalini V
### Register Number :212222240096
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
![image](https://github.com/user-attachments/assets/7c85ce94-4ab5-4eeb-abe7-19e4299c0e71)

![image](https://github.com/user-attachments/assets/b4f59e54-4033-4a61-b0be-48db095fe3ee)

### 2. Policy, Value function and success rate for the Improved Policy
![image](https://github.com/user-attachments/assets/6939e2b7-95f1-4d32-8ea5-6e9dd74dcd56)
![image](https://github.com/user-attachments/assets/5022ffda-8fa7-4091-8790-de9aff5fd145)


### 3. Policy, Value function and success rate after policy iteration
![image](https://github.com/user-attachments/assets/5f4f2385-6d0e-40cf-be0e-9aa990b52dae)

![image](https://github.com/user-attachments/assets/2896c7c7-83c5-4103-9c97-bca3e3b7a1b1)



## RESULT:

Thus, the program to iterate Policy improvement and evaluation is implemented successfully
