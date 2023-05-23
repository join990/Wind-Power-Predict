# Wind_Power_Predict
Visualization system for wind power forecasting

 ## Target

1. Real-time monitoring of key indicators and trends, and timely alarm when exceeding the standard.

2. Remote real-time monitoring to reduce the risk of information asymmetry.

3. The artificial intelligence model is integrated into the system to realize the intelligence of the system.

## Introduction

The figure below shows the overall framework of the library. The main part is a **CL Trainer** class, composed of a **CL Algorithm** class and a **Model Trainer** class, which interact with each other through five **APIs** of the CL Algorithm class to implement the curriculum.

<img src="./docs/img/framework.svg">

An general workflow of curriculum machine learning is illustrated below. 

<img src="./docs/img/flow.svg">



## Environment

1. python >= 3.6

2. pytorch >= 1.9.0

3. Unity == 2019.4.14

## Quick Start
``` bash
# 1. clone from the repository
git clone https://github.com/join990/Wind-Power-Predict

# 2. Import the visualization system into Unity3D.
Project->import Package

# 3. inport the windpower predict model
python examples/base.py

# 4. run the windpower predict model

# 5. run the scenes

```







