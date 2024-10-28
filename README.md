# AI/ML Trained Python Code Validator with Web Interface

**Code:** Aleksi Bovellan

**Gradio interface:** Satu Pohjonen

**Technologies:** microsoft/codebert-base, Gradio interface, Python3

**Dataset:** code_search_net python + automatically created incorrect code examples using corruption methods

**Python extensions:** gradio, transformers, torch, datasets, numpy


# INCLUDED FILES AND FOLDERS:

- "ai-code-validator-interface.py" - the easy main Gradio web interface script for starting up the actual trained model and the webpage user-interface
- Folder "results" - this is the already prepared training intelligence, it is ready to be used as-is
- "model_training_results.txt" - The log output of the training results from the creation of the included "results" intelligence folder

- Optional: "ai-code-validator-training.py" - an automatic training script to create your own new "results" folder and build new training intelligence into it

- "README.md" - this file for instructions


# SPECIAL ATTENTION TO APPLE MAC COMPUTERS:

For easier usage, you could use the Macs Terminal window instead of any code editors. First create a virtual environment in the Terminal by typing:

python3 -m venv myenv

source myenv/bin/activate

pip install gradio transformers torch datasets numpy


(When you are done experimenting with this project, you can exit the virtual environment by typing "deactivate", and later return into it by repeating "source myenv/bin/activate")


# PRE-INSTALLATION REQUIREMENTS FOR ALL USERS:

pip install gradio transformers torch datasets numpy


# MAIN USAGE:

Just run the main Gradio interface script to automatically start the model and web interface, if a "results" folder already exists in the same directory - and that's it!

The Gradio interface script will load the trained model, the trained database, and finally open a localhost url for your web browser. The provided web page will act as the user-interface for interacting with this trained python code validator model, which will be used to check if your provided Python code has obvious errors in it.

EXAMPLE 1:

Keep the existing "results" folder in the same directory, and just run:

python3 ai-group-thinkers_code_validator_interface.py


OR EXAMPLE 2:

Move the pre-provided "results" intelligence folder to somewhere else, to keep it as a backup, and run:

python3 ai-group-thinkers_code_validator_training.py

python3 ai-group-thinkers_code_validator_interface.py


# MODEL TRAINING (NOT NECESSARILY NEEDED - INTELLIGENCE ALREADY PROVIDED :)

The folder called "results" has already been provided as a ready-to-go trained database, so NO EXTRA learning is required to run the model and main web interface. The "results" folder has to be located in the same directory as the main Gradio interface script, because it depends on the folder existing.

However, if you DO want to train your own database from scratch:

1) Move the pre-provided "results" folder to somewhere else, to keep it as a backup

2) Open and edit the training script:

3) In the beginning of the training script, change the "sample_size" parameter to the size you want - it affects the time your automated training process will eventually take. Longer training time results into a better model and more accurate prediction and confidence. There is more info about the suggested size values written into the training script itself and also to the end of this README file.

4) Save the training script if it was edited

5) Run the training script and let it finish. In the end it will evaluate its results:


python3 ai-group-thinkers_code_validator_training.py


(There is more info about the desired training results written below.)


# Editing the training script - sample_size value time estimate benchmarks:

Example 1: Setting sample_size of 1200 takes about 10-15 minutes to train on a year 2020 MacBook M1 Air

Example 2: Setting sample_size of 30000 takes around 8,5 hours to train on the same machine


# The training process progress reports and evaluation results:

During the training, the learning "loss" rate printouts should decrease with time, as more information is being understood, and it is being learned. Like below, we see the first and final printouts indicating clear learning progress and increasing confidence. Loss values of around 0.7 are blind guessing. It will start from there, but it should not end there:

{'loss': 0.6795, 'grad_norm': 4.695720672607422, 'learning_rate': 1.9888925913584364e-05, 'epoch': 0.02}

...

{'loss': 0.141, 'grad_norm': 0.5732464790344238, 'learning_rate': 6.664445184938354e-09, 'epoch': 3.0} 

The final evaluation results will also hopefully look like something in the example below, with the scores being as close to 1 as possible. Results closer to 0.5 are blind guesses - like "fifty-fifty" - and would indicate that the learning has failed to differentiate between correct and incorrect examples.
LABEL_1 and LABEL_0 mean correct and incorrect examples of Python code in the learned database, and there should be high confidence for both datasets:

Classification result for code_snippet_1: [{'label': 'LABEL_1', 'score': 0.9680263996124268}

Classification result for code_snippet_2: [{'label': 'LABEL_0', 'score': 0.9998996257781982}


# CPU/GPU COMPABILITY

The training script runs using CPU to work universally on Macs and PCs without specific GPU support problems, like NVIDIA, etc. However, MacBooks (like M1 Air) still automatically choose to use their own GPU, as they figure out it will give the best performance.
