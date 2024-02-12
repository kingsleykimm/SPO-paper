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