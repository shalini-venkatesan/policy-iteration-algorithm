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
### Name : KARTHIKEYAN P
### Register Number : 212223230102
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
### Name : KARTHIKEYAN P
### Register Number :212223230102
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
![Screenshot 2025-04-25 135427](https://github.com/user-attachments/assets/e60973ad-b3b9-4559-b81e-01b61740d5fa)

![Screenshot 2025-04-25 135435](https://github.com/user-attachments/assets/ebfbad18-9eca-44f9-ae92-c821e113c496)

### 2. Policy, Value function and success rate for the Improved Policy
![Screenshot 2025-04-25 135544](https://github.com/user-attachments/assets/fbee3982-9ce3-4fc5-a0a0-9281759320b2)

![Screenshot 2025-04-25 135550](https://github.com/user-attachments/assets/ba2d3fd8-9cd4-44cf-a572-0637ab4338aa)


### 3. Policy, Value function and success rate after policy iteration
![Screenshot 2025-04-25 135730](https://github.com/user-attachments/assets/618a1170-f529-4113-965e-61a0f8ea15f2)

![Screenshot 2025-04-25 135735](https://github.com/user-attachments/assets/ecb09843-a82d-4a99-8402-fbc6696c1edc)



## RESULT:

Thus, the program to iterate Policy improvement and evaluation is implemented successfully
