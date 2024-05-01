## Syntouch Biotac Robotic Finger: Deducing Force Direction

The Syntouch Biotac is a robotic finger with multiple electrode and pressure sensors that allow for tactile feedback. In this project, I created a neural network that takes in this sensor information and deduces the direction of a tactile force in real-time, mediated by ROS (ie, if the finger is touching something, it knows which direction the apparent force is coming from).<br>
![visual summary](imgs/gist_gif.gif)
[Run the Code!](#run-the-code)

### Table of Contents
- [Syntouch Biotac Robotic Finger: Deducing Force Direction](#syntouch-biotac-robotic-finger-deducing-force-direction)
  - [Table of Contents](#table-of-contents)
  - [Introdction](#introdction)
  - [Data Acquisition](#data-acquisition)
  - [Data processing](#data-processing)
  - [Machine Learning Approaches](#machine-learning-approaches)
    - [Framework for all approaches](#framework-for-all-approaches)
    - [Multi-Layered Perceptron(Neural Network)](#multi-layered-perceptronneural-network)
    - [Support Vector Regression (SVR) - Rejected](#support-vector-regression-svr---rejected)
    - [Gaussian Process Regression (GPR) - Abandoned](#gaussian-process-regression-gpr---abandoned)
  - [Thermal Noise \& Unwanted spurious effects](#thermal-noise--unwanted-spurious-effects)
  - [Run the code](#run-the-code)

### Introdction
---
The Syntouch Biotac contains a variety of sensors, of interest are the 

### Data Acquisition
---

![tracker frame](imgs/Capture.PNG.png)
![frame transform](imgs/frametransform.png)
![ROS data capture architecture](imgs/synchroniser.png)

### Data processing
---
![data pipeline](imgs/data_processing.png)
### Machine Learning Approaches
---
#### Framework for all approaches
The main metric used to determine the goodness of fit for regression was the coefficient of determination, $R^2$. The optimal model parameters were determined with a grid search. For a given gridpoint(a given permutation of hyperparameters), 5-fold cross validation was used to train and retrain a model, and extract the $R^2$ value for each fold in the 5-fold. The **median** $R^2$ value was chosen to represent the performance for that permutation of hyperparameters.<br>
AIC was used as a way to discriminate against more complex models in a given gridsearch. <br>
Given that the AIC criterion tends to favour performance at the expense of greater model complexity, this metric was only indicative for my experiments. I would look at the permutation of hyperparameters that minimised AIC and manually explored if better alternatives were being obscured. <br>
Ultimately I wanted the simplest model with the highest performance, so relying on AIC alone is not sufficient. $R^2$ generally took precedence. The summary statistics & metrics for each point(model) in the gridsearch were saved in an csv file, and inspected when the gridsearch finished. An example file for the neural network is found [here](source/4_Neural_Training/hyperparameter_grid_search_overview.csv). The other included statistics are largely arbitrary, and were included "just because I could". They came with no computation cost and they may have given some insight. <br>
Regression performance was important for this project(only models with $R>0.7$ were considered) so BIC was not considered as a discriminator.<br><br>
Given the datasets are composed of two trials, I decided to have a shuffled train/test/validation split of 64%/16%/20%. Tha train/test set were used exclusively for the 5-fold cross validation, and once the best model was found, it was retrained on the same training set and finally tested against the validation set. Below we visualise this process:  

![5-fold validation process](imgs/5_fold.png)

While this was conducted to the neural network and the SVR gridsearch permutations, the search of a simpler model required I change this approach for the neural network to make sure I was avoiding researcher overfitting. This will be discussed later.

#### Multi-Layered Perceptron(Neural Network)
![loss & R2 plots](imgs/loss_r2_plots.png)
#### Support Vector Regression (SVR) - Rejected
Since SVR was not the final implemented solution, the relevant code has been redacted in favour of the neural network. However, it's worth discussing the main insights, choices, and results.<br>
Similar to the neural network, I grid-searched over the hyperparameters to find the best ones. After experimentation, an `RBF` kernel was used.<br>
The following ranges were used:
- $\epsilon \in [0.05, 0.2]$ Tolerance for noise
- $C \in [1, 5000]$ Penalty for outliers
- $\gamma \in [10^{-5}, 10^{-4}] \Leftrightarrow \sigma \in [70, 223]$ RBF kernel width

**NOTE**: After discussion with colleagues and the rest of the lab, it was determined that the number of support vectors should **Not** exceed 20-30. The reasoning for this was based on past experience & experiments from others. There was no concrete mathematical reasoning given to me, so I took this range 'as-is'. <br>
The `AIC` for SVR is not directly applicable, thus the use of cross-validation and $R^2$ to determine the final model.<br> Never-the-less. I'd like to define an AIC-like metric that roughly   encapsulates both the complexity and performance of SVR. Therefore, I define my pseudo-AIC as $AIC_{pseudo} = -\alpha ln(1-R^2) + S_v N^2$ where:
- $R^2$ is the usually defined coefficient of determination
- $S_v$ is the number of support vectors of the model
- $N$ is the dimensionality of the dataset (19)
- $\alpha$ is a tunable parameter that can adjust the importance of predictive power and model complexity.

Tuning $\alpha$ is down to experimentation, however it should favour model performance when $S_v \in [20, 30]$. <br>
My $AIC_{psudeo}$ should **not** be used to compare the SVR models against the Neural Networks. In the latter case I used the appropriate formula for multi-layered perceptrons, but in the former case it's a guess/estimate of what it may look like. <br>
Ultimately the best models after the gridsearch and cross-validation did not yield satisfactory results.

![SVR performance plots](imgs/svr_performance.png)

We can see the $R^2$ is no where near satisfactory for a maximum of 30 support vectors. Ideally an $R^2 \gt 0.7$ would have merited further investigation. However for $R^2 = 0.68$ we need 51 support vectors in the x-direction, and 81 in the z-direction. Therefore an SVR model was rejected for this project.
#### Gaussian Process Regression (GPR) - Abandoned
I also attempted Gaussian Process Regression as a way to compare it against the performance of the other models. It became very apparent very quickly that this wasn't the way to go. <br>
The modelling was very sensitive to choices on the priors despite popular approaches. <br>
To get it to work required tightly constraining priors, however this significantly increases the likelihood of overfitting to get acceptable results. Rather than spending the computation time grid-searching for small islands of acceptable performance(which ultimately were very close to each other), this approach was abandoned. 

### Thermal Noise & Unwanted spurious effects

### Run the code
