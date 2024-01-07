# MSc Dissertation - Exploration by Random Network Distillation in Sparse Reward MiniGrid Environments

Tested on Python 3.9, not compatible with Python 3.12 (tensorflow incompatability). 
Microsoft Visual Studio required for C++ build tools

The following instructions provide details of the code structure and guidance on how to run the experiments detailed in this dissertation.

### Code Structure

The dissertation_files folder contains 3 subfolders:

1. **agents**: This folder contains the code required for setting up the Random, DQN, PPO and RND agents, running training and evaluation, along with some utilities.
   1. **agent.py**: This file contains the classes for each agent, setting up all networks and required structures (such as buffers), based on the environment.
   2. **buffer.py**: This file contains buffer classes for each agent, as each needs to record and track different information during training.
   3. **config.py**: The hyperparameters and architectures for the experiments.
   4. **evaluation.py**: This file runs the evaluation of the agents during training, recording the results and saving the models for generalisation. It also contains the functionality for plotting the results.
   5. **training.py**: This file contains the main training loops for the agents.
   6. **utils.py**: This file contains some utility functions such as calculating discounted sums and creating neural networks.


2. **environments**: This folder contains the code for the MiniGrid environments used in the experiments.
   1. **minigrid_environments.py**: This file contains the classes for custom MiniGrid environments, including some not used in the experiments. It also contains wrappers for transforming observations from the environment with a CNN (`RGBImgPartialObsWrapper`).


3. **tests**: This folder contains the code for testing the agents on each individual environment. The sparse_reward, no_reward and generalisation subfolders are structured identically. Each environment in the respective subfolder has a "test_pipeline" file and a "plots" file. The test pipeline file runs the experiments and saves the results and videos to the test_data folder, the plots file loads the data and plots the results.
   1. **sparse_reward**: This folder contains the code for the _sequential rooms_, _multiroom_ and _key-corridor_ environments.
   2. **no_reward**: This folder contains the code for the _double spiral_ environment.
   3. **generalisation**: This folder contains the code for _key-corridor_ and _multiroom_ pretraining and tests. 
   4. **test_data**: This folder contains no code, but it used to store the output of the test pipelines, as well as videos and plots (initially these folders are empty)


### Installation and running the experiments

1. Extract the contents of the zip file to a directory of your choice.


2. Navigate to the directory and run `pip install -r requirements.txt`.


3. To run an experiment, navigate to the relevant part of the tests folder and run the test pipeline file for the environment you wish to test. For example, `multiroom_test_pipeline.py` will run the entire pipeline for all algorithms and save the results.
   1. To reduce the time taken to run the file, the `EVALUATION_PIPELINE_RUNS` parameter can be reduced
   2. In addition, as the algorithms are run separately, sections can be commented out to further reduce the time taken to obtain results. For example to run only the random agent, comment out all lines from "DQN" onwards.
   3. For generalisation environments key_corridor_S3R3_pretraining and multiroom_N4_pretraining must be run before their respective test pipelines as these save checkpoints for the RND agent to load.
   4. Videos of training are also saved in the relevant folder in test_data.


4. To plot the results of a test, navigate to the tests folder and run the plots file for the environment. For example, `multiroom_plots.py` will load the results, plot and save the relevant graphs.
   1. The `load_file_for_plot` date arguments must be updated to match the date the relevant test was conducted. ie. `random_rewards = load_file_for_plot('sparse_multiroom', 'random', 'rewards', '<DATE_HERE>')`


(NB. You may need to run sys.path.append to add the dissertation_files parent directory to the path before running the code)


