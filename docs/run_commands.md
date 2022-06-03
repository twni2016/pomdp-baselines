# Run Experiments

## Difficulty Levels of Environments

Before starting running any experiments, we suggest having a good plan of *environment series* based on difficulty level. As it is hard to analyze and varies from algorithm to algorithm, we provide some rough estimates:

1. Extremely Simple as a Sanity Check: Pendulum-V (also shown in our minimal example jupyter notebook) and CartPole-V (for discrete action space)
2. Simple, Fast, yet Non-trivial: Wind (require precise inference and control), Semi-Circle (sparse reward). Both are continuous gridworlds, thus very fast.
3. Medium: Cheetah-Vel (1-dim stationary hidden state), `*`-Robust (2-dim stationary hidden state), `*`-P (could be roughly inferred by 2nd order MDP)
4. Hard: `*`-Dir (relatively complicated dynamics), `*`-V (long-term inference), `*`-Generalize (extrapolation)

## Best Configs / Variants
To run the best variant of our implementation, please refer to [our_details.md](our_details.md), and then change the corresponding hyperparameters in the config files.


## Specific Running Commands for Each Subarea
General form:
```
export PYTHONPATH=${PWD}:$PYTHONPATH
python3 policies/main.py --cfg configs/<subarea>/<env_name>/<algo_name>.yml \
  [--env <env_name>  --oracle
   --algo {td3,sac,sacd} --(no)automatic_entropy_tuning --target_entropy <float> --entropy_alpha <float>
   --debug --seed <int> --cuda <int>
  ]
```

### "Standard" POMDP
{Ant,Cheetah,Hopper,Walker}-{P,V} in the paper, corresponding to `configs/pomdp/<ant|cheetah|hopper|walker>_blt/<p|v>`, which requires PyBullet. We also provide Pendulum environments for sanity check.

Take Ant-P as an example:
```bash
# Run our implementation
python policies/main.py --cfg configs/pomdp/ant_blt/p/rnn.yml --algo sac
# Run Markovian
python policies/main.py --cfg configs/pomdp/ant_blt/p/mlp.yml --algo sac
# Oracle: we directly use Table 1 results (SAC w/ unstructured row) in https://arxiv.org/abs/2005.05719 as it is well-tuned

# Run A2C-GRU from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
python PPO/main.py --config configs/pomdp/ant_blt/p/a2c_rnn.yml \
    --algo a2c --lr 7e-4 --gae-lambda 1.0 --entropy-coef 0.0 \
    --num-steps 5 --recurrent-policy
# Run PPO-GRU from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
python PPO/main.py --config configs/pomdp/ant_blt/p/ppo_rnn.yml \
    --algo ppo --lr 2.5e-4 --use-gae --entropy-coef 0.01 \
    --num-steps 128 --recurrent-policy
# Run VRM from https://github.com/oist-cnru/Variational-Recurrent-Models
python VRM/run_experiment.py configs/pomdp/ant_blt/p/vrm.yml
``` 

Mar 2022: we support recurrent SAC-discrete for POMDPs with **discrete action space**. Take CartPole-V as an example:
```
python policies/main.py --cfg configs/pomdp/cartpole/v/rnn.yml --target_entropy 0.7
```
See [this PR for detailed instructions](https://github.com/twni2016/pomdp-baselines/pull/1) and [this PR for results on a long-term credit assignment benchmark](https://github.com/twni2016/pomdp-baselines/pull/2).

### Meta RL 

{Semi-Circle, Wind, Cheetah-Vel} in the paper, corresponding to `configs/meta/<point_robot|wind|cheetah_vel|ant_dir>`. Among them, Cheetah-Vel requires MuJoCo, and Semi-Circle can serve as a sanity check. Wind looks simple but is not very easy to solve.

Take Semi-Circle as an example:
```bash
# Run our implementation
python policies/main.py --cfg configs/meta/point_robot/rnn.yml --algo td3
# Run Markovian
python policies/main.py --cfg configs/meta/point_robot/mlp.yml --algo sac
# Run Oracle
python policies/main.py --cfg configs/meta/point_robot/mlp.yml --algo sac --oracle

# Run off-policy variBAD from https://github.com/Rondorf/BOReL
cd BOReL; python online_training.py --env-type point_robot_sparse
```

{Ant, Cheetah, Humanoid}-Dir in the paper, corresponding to `configs/meta/<ant_dir|cheetah_dir|humanoid_dir>`. They require MuJoCo and are hard to solve.
Take Ant-Dir as an example:
```bash
# Run our implementation
python policies/main.py --cfg configs/meta/ant_dir/rnn.yml --algo sac
# Run Markovian
python policies/main.py --cfg configs/meta/ant_dir/mlp.yml --algo sac
# Run Oracle
python policies/main.py --cfg configs/meta/ant_dir/mlp.yml --algo sac --oracle

# For on-policy variBAD and RL2, we use the data from https://github.com/lmzintgraf/varibad
```

### Robust RL
Use roboschool. {Hopper,Walker,Cheetah}-Robust in the paper, corresponding to `configs/rmdp/<hopper|walker|cheetah>`. First, activate the roboschool docker env as introduced in the installation section. 

Take Cheetah-Robust as an example:
```bash
## In the docker environment:
# Run our implementation
python3 policies/main.py --cfg configs/rmdp/cheetah/rnn.yml --algo td3
# Run Markovian
python3 policies/main.py --cfg configs/rmdp/cheetah/mlp.yml --algo sac
# Run Oracle
python3 policies/main.py --cfg configs/rmdp/cheetah/mlp.yml --algo sac --oracle

# Run PPO-GRU from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
python PPO/main.py --config configs/rmdp/cheetah/ppo_rnn.yml \
    --algo ppo --lr 2.5e-4 --use-gae --entropy-coef 0.01 \
    --num-steps 128 --recurrent-policy
# Run MRPO from http://proceedings.mlr.press/v139/jiang21c/jiang21c-supp.zip
python3 MRPO/examples/MRPO/train.py configs/rmdp/cheetah/MRPO.yml
```

### Generalization in RL
Use roboschool. {Hopper|Cheetah}-Generalize in the paper, corresponding to `configs/generalize/Sunblaze<Hopper|HalfCheetah>/<DD-DR-DE|RD-RR-RE>`. 
First, activate the roboschool docker env as introduced in the installation section. 

To train on Default environment and test on the all environments, use `*DD-DR-DE*.yml`; to train on Random environment and test on the all environments, use use `*RD-RR-RE*.yml`. Please see the [SunBlaze paper](https://arxiv.org/abs/1810.12282) for details. 

Take running on `SunblazeHalfCheetahRandomNormal-v0` as an example:
```bash
## In the docker environment:
# Run our implementation
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/rnn.yml --algo td3
# Run Markovian
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/mlp.yml --algo sac
# Run Oracle
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/mlp.yml --algo sac --oracle

# For PPO, A2C, EPOpt-PPO-FF, we use the figures from SunBlaze paper
```

### Temporal Credit Assignment
{Delayed-Catch, Key-to-Door} in the paper, corresponding to `configs/credit/<catch|keytodoor>`. Note that this is discrete control on pixel inputs, so the architecture is a bit different from the default one.

To reproduce our results, please run:
```bash
python3 policies/main.py --cfg configs/credit/catch/rnn.yml
python3 policies/main.py --cfg configs/credit/keytodoor/rnn.yml
```

### Atari
Although Atari environments are **not** this paper's focus, we provide an implementation to train on a game, following the Dreamerv2 setting. The hyperparameters are **not** well-tuned, so the results are not expected to be good.

```bash
# train on Pong (confirmed it can work on Pong)
python3 policies/main.py --cfg configs/atari/rnn.yml --env Pong
```
