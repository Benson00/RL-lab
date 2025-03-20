# RL-Laboratory🤖

Code for the Reinforcement Learning lab of *Reinforcement Learning and Advanced programming for AI* course, MSc degree in Artificial Intelligence 2024/2025 at the University of Verona.

## First Set-Up (Conda)
1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your System.

2.  Install Miniconda
	- On Linux/Mac 
		- Use *./Miniconda3-latest-Linux-{version}.sh* to install.
		- *sudo apt-get install git* (may be required).
	- On Windows
		- Double click the installer to launch.
		- *NB: Ensure to install "Anaconda Prompt" and use it for the other steps.*

3.  Set-Up conda environment:
	- *git clone https://github.com/Isla-lab/RL-lab*
	- *conda env create -f RL-Lab/tools/rl-lab-environment.yml*

## First Set-Up (Python Virtual Environments)
Python virtual environments users (venv) can avoid the Miniconda installation. The following package should be installed:
  - scipy, numpy, gym
  - jupyter, matplotlib, tqdm
  - tensorflow, keras

## Assignments
Following the link to the code snippets for the lessons:

**First Semester**
- [x] Lesson 1: MDP and Gym Environments [Slides](slides/slides_lesson_1.pdf), [Code](lessons/lesson_1_code.py), [Results](results/lesson_1_results.txt)
- [x] Lesson 2: Multi-Armed Bandit [Slides](slides/slides_lesson_2.pdf), [Code](lessons/lesson_2_code.py), [Results](results/lesson_2_results.txt)
- [x] Lesson 3: Monte Carlo RL methods [Slides](slides/slides_lesson_3.pdf), [Code](lessons/lesson_3_code.py), [Results](results/lesson_3_results.txt)
- [x] Lesson 4: Temporal difference methods [Slides](slides/slides_lesson_4.pdf), [Code](lessons/lesson_4_code.py), [Results](results/lesson_4_results.txt)
- [x] Lesson 5: Dyna-Q [Slides](slides/slides_lesson_5.pdf), [Code](lessons/lesson_5_code.py), [Results](results/lesson_5_results.txt)

**Second Semester**
- [x] Lesson 6: Tensorflow-PyTorch and Deep Neural Networks [Slides](slides/slides_lesson_6.pdf), [Code](lessons/lesson_6_code.py), [Results](results/lesson_6_results.txt)
<!---  - [ ] Lesson 8: Deep Q-Network [Slides](slides/slides_lesson_8.pdf), [Code](lessons/lesson_8_code.py), [Results](results/lesson_8_result.png) --->
<!---  - [ ] Lesson 9: REINFORCE [Slides](slides/slides_lesson_9.pdf), [Code](lessons/lesson_9_code.py), [Results](results/lesson_9_result.png) --->
<!---  - [ ] Lesson 10: A2C [Slides](slides/slides_lesson_10.pdf), [Code](lessons/lesson_10_code.py), [Results](results/lesson_10_result.png) --->
<!---  - [ ] Lesson 11: DRL in Practice [Code!](lessons/lesson_11_code.py) [Results 1!](results/lesson_11_result.png) [Results 2!](results/lesson_11_results_TB3.png) [Slides!](slides/slides_lesson_11.pdf) --->

**Extra exercise**
- [x] Lesson Extra: Value/Policy Iteration [Slides](slides/slides_lesson_extra.pdf), [Code](lessons/lesson_extra_code.py), [Results](results/lesson_extra_results.txt)

## Tutorials
This repo includes a set of introductory tutorials to help accomplish the exercises. In detail, we provide the following Jupyter notebook that contains the basic instructions for the lab:
- **Tutorial 1 - Gym Environment:** [Here!](tutorials/tutorial_environment.ipynb)
- **Tutorial 2 - TensorFlow (Keras):** [Here!](tutorials/tutorial_tensorflow.ipynb)
- **Tutorial 3 - PyTorch:** [Here!](tutorials/tutorial_pytorch.ipynb)


## Contact information
*  Teaching assistant: **Gabriele Roncolato** - gabriele.roncolato@univr.it
*  Professor: **Alberto Castellini** - alberto.castellini@univr.it
