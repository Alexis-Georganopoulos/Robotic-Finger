## Syntouch Biotac Robotic Finger: Deducing Force Direction

The Syntouch Biotac is a robotic finger with multiple electrode and pressure sensors that allow for tactile feedback. In this project, I created a neural network that takes in this sensor information and deduces the direction of a tactile force in real-time, mediated by ROS (ie, if the finger is touching something, it knows which direction the apparent force is coming from).<br>
![visual summary](imgs/gist_gif.gif)
[Run the Code!](#run-the-code)

### Table of Contents
- [Syntouch Biotac Robotic Finger: Deducing Force Direction](#syntouch-biotac-robotic-finger-deducing-force-direction)
  - [Table of Contents](#table-of-contents)
  - [Data Acquisition](#data-acquisition)
  - [Data processing](#data-processing)
  - [Machine Learning Approaches](#machine-learning-approaches)
    - [Multi-Layered Perceptron(Neural Network)](#multi-layered-perceptronneural-network)
    - [SVR (rejected)](#svr-rejected)
    - [GPR (abandoned)](#gpr-abandoned)
  - [Run the code](#run-the-code)

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

#### Multi-Layered Perceptron(Neural Network)
![loss & R2 plots](imgs/loss_r2_plots.png)
#### SVR (rejected)
#### GPR (abandoned)

### Run the code
