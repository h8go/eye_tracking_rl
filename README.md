# Saliency maps

This is a framework to calculate the saliency maps using the Dopamine framework.
There are several branches in this project in which different calculations are made.
First, the contents of the branches are described
Then, the working of the algorithm is explained late.

# Description of the branches:
- master: Perturbation High and low resolution perturbation saliency with DQN for any game
- all_saliency_maps: High resolution perturbation saliency, gradient saliency and perturbation approximation saliency (non optimized) with DQN for Pong
- rainbow_perturbation_saliency: Full and low resolution perturbation saliency for rainbow agent for any game
- saliency_Enduro_dqn : Gradient and perturbation approximation saliency (optimized) for DQN and Enduro
- saliency_Freeway_dqn : Gradient and perturbation approximation saliency (optimized) for DQN and Freeway
- saliency_MsPacman_dqn : Gradient and perturbation approximation saliency (optimized) for DQN and MsPacman
- saliency_Pong_dqn : Gradient and perturbation approximation saliency (optimized) for DQN and Pong

Advices: - all_saliency_maps allows to understand how all saliency are computed in a simple way. It can be a good idea to start here
         - the master and rainbow_perturbation_saliency compute perturbation saliency maps. They are useful if you want to compare the saliency of dqn and rainbow agents
         - saliency_Enduro_dqn, saliency_Freeway_dqn, saliency_MsPacman_dqn, saliency_Pong_dqn have the optimized and last version of the approximation saliency perturbation

# Description of the saliency calculation:
All the saliency types are calculated in the _select_action in dqn_agent.py 
For the optimized gradient and approximation perturbation saliency saliencies, the placeholder and tensoflow operations are declared in the load_and_play_trained_agents.py
The operation related to the perturbation is in perturbation.py

# How to calculate those saliencies using trained agent ?
1) In example_viz.py fill 'dqn' or 'rainbow' for the agent
   Select the game name
   Let num_steps being equal to 1
   The root_dir doesn't have any importance
   Enter the path of the 3 files of the checkpoint of the agent
      You can find the checkpoint files here https://github.com/google/dopamine/tree/master/docs
   use_legacy_checkpoint must be True
2) In dqn_agent.py, in the _select_action function, fill True or False if you want to compute the saliency or not
   Same if you want to save the last frame of the state variable
   You need to specify for which step you want the saliency to be calculated and/or saved
      For example the following line "if step_number > 900 and step_number < 1000:" will save the saliency from the frame 901 to 999
   You need to specify as well the path where to save the saliencies
   You need to repeat these operations for saving the render. This is done in load_and_play.py in the _run_one_step function 

Warning: If a branch is explicitly coded for a specific game or an agent, you shall respect this choice.
         The program might work, but you will obtain wrong results

# How to use this framework on Google Colab
1. Make sure to follow the steps of the previous paragraph
2. In the master branch the script_main.ipynb notebook is provided. To make it work, change the path is the first cell to replace it by the path of this project in your google drive file system
3. Then download the Atari 2600 VCS ROM Collection  https://github.com/openai/atari-py#roms, unzip it in a folder called roms whose. This folder should be at the same level as the dopamine folder
4. Once you executed the two first cells, you can run the third cell to make sure you have a GPU
5. Running the fourth cell will launch the saliency computation and save them
