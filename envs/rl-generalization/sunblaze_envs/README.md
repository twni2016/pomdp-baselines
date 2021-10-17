## Overview of Generalization Environments

There are six environments, built on top of the corresponding OpenAI Gym and Roboschool implementations:
* CartPole
* MountainCar
* Acrobot
* Pendulum
* HalfCheetah
* Hopper

Each has three versions:
* **D**: Environment parameters are set to the default values in Gym and Roboschool. Access by Sunblaze*Environment*-v0, e.g. SunblazeCartPole-v0.
* **R**: Environment parameters are randomly sampled from intervals containing their default values. Access by Sunblaze*Environment*RandomNormal-v0.
* **E**: Environment parameters are randomly sampled from intervals outside those in **R**, containing more extreme values. Access by Sunblaze*Environment*RandomExtreme-v0.

## Environment details

Ranges of parameters for each version of each environment, using set notation.

| Environment  |  Parameter  |  D  |  R  |  E  | 
| --- | --- | --- | --- | --- | 
| CartPole  |  Force  |  10  |  [5,15] |  [1,5] U [15,20]    | 
| |  Length  |  0.5  |  [0.25,0.75]  |  [0.05,0.25] U [0.75,1.0]  | 
|   |  Mass  |  0.1  |  [0.05,0.5]  |  [0.01,0.05] U [0.5,1.0]  | 
|  MountainCar  |  Force  |  0.001  |  [0.0005,0.005]  |  [0.0001,0.0005] U [0.005,0.01]  | 
|   |  Mass  |  0.0025 |  [0.001,0.005]  |  [0.0005,0.001] U [0.005,0.01]  | 
|  Acrobot  |  Length  |  1  |  [0.75,1.25]  |  [0.5,0.75] U [1.25,1.5]  | 
|   |  Mass  |  1  |  [0.75,1.25]  |  [0.5,0.75] U [1.25,1.5]  | 
|   |  MOI  |  1  |  [0.75,1.25]  |  [0.5,0.75] U [1.25,1.5]  | 
|  Pendulum  |  Length | 1  |  [0.75,1.25]  | [0.5,0.75] U [1.25,1.5]  | 
|   |  Mass  |  1  |  [0.75,1.25]  |  [0.5,0.75] U [1.25,1.5]  | 
|  HalfCheetah  |  Power  |  0.90  |  [0.70,1.10]  |  [0.50,0.70] U [1.10,1.30]  | 
|   |  Density  |  1000  |  [750,1250]  |  [500,750] U [1250,1500]  | 
|   |  Friction |  0.8  |  [0.5,1.1]  |  [0.2,0.5] U [1.1,1.4]  | 
|  Hopper | Power | 0.75 | [0.60,0.90]  |  [0.40,0.60] U [0.90,1.10]  | 
|   |  Density  |  1000  |  [750,1250]  |  [500,750] U [1250,1500]  | 
|   |  Friction  |  0.8  |  [0.5,1.1]  |  [0.2,0.5] U [1.1,1.4]  | 
