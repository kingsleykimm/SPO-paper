# SPO-paper

Steps for implementing SPO paper: 
- Set up ACME environment and the actor-critics (use paper appendix)
- Set up interaction protocols for both continuous and discrete environments
- Implement SPO with actor-critics
- Implemented iterative reward modeling algorithm
- Run experiments

SAC for continuous control
PPO for discrete

Need to have an SPO and iterative RM framework
Structure: files for the PPO and SAC network separate from the SPO and iterative RM
- User can choose to run either framework based on command line arguments
- SPO and iterative RM should choose SAC or PPO depending on environment requirements

Offline Preference Function for RL applications:
Steps:
1. Train a preference model - what are the best choices for this? Contrastive Learning? State-action pair
    (s1, a1, s1'), (s2, a2, s2') -> should give a preference 1 over the other
    To create a preference model need these three things:
        Function to define how a trajectory is "better" -> return
        How to train it to prefer state action pairs and output a preference -> this is the difficult part, could use k step segments
    multiple rounds of training preference, then policy, outer loops?

2. After preference model is trained, 
How is this different from IPL? - avoids reward modeling, but also is no state action pairs with a preference, and we;re training a preference model
actions that have a better reward
