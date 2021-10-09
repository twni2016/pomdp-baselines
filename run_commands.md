## Specific Running Commands for Each Subarea

### "Standard" POMDP
Use PyBullet. {Ant,Cheetah,Hopper,Walker}-{P,V} in the paper, correspond to `configs/pomdp/<ant|cheetah|hopper|walker>_blt/<p|v>`. 

Take Ant-P as example:
```bash
# Run our implemention
python policies/main.py configs/pomdp/ant_blt/p/sac_rnn.yml
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

### Meta RL 
Use MuJoCo in Cheetah-Vel. {Semi-Circle, Wind, Cheetah-Vel} in the paper, correspond to `configs/meta/<point_robot|wind|cheetah_vel>`. 

Take Semi-Circle as example:
```bash
# Run our implemention
python policies/main.py configs/meta/point_robot/td3_rnn.yml
# Run (off-policy version of) VariBAD from https://github.com/Rondorf/BOReL
python BOReL/main.py configs/meta/point_robot/varibad.yml
```

### Robust RL
Use roboschool. {Hopper,Walker,Cheetah}-Robust in the paper, correspond to `configs/rmdp/<hopper|walker|cheetah>`. First, activate the roboschool docker env as introduced above. 

Take Cheetah-Robust as example:
```bash
## In the docker environment:
# Run our implemention
python3 policies/main.py configs/rmdp/cheetah/td3_rnn.yml
# Run MRPO from http://proceedings.mlr.press/v139/jiang21c/jiang21c-supp.zip
python3 MRPO/examples/MRPO/train.py configs/rmdp/cheetah/MRPO.yml
```

### Generalization in RL
Use roboschool. {Hopper|Cheetah}-Generalize in the paper, correspond to `configs/generalize/Sunblaze<Hopper|HalfCheetah>/<DD-DR-DE|RD-RR-RE>`. 
First, activate the roboschool docker env as introduced above. To train on Default environment, use `*DD-DR-DE*.yml`; to train on Random environment, use use `*RD-RR-RE*.yml`. Please see the [SunBlaze paper](https://arxiv.org/abs/1810.12282) for details. 

Take running on `SunblazeHalfCheetahRandomNormal-v0` as example:
```bash
## In the docker environment:
# Run our implemention
python3 policies/main.py configs/generalize/SunblazeHalfCheetah/RD-RR-REtd3_rnn.yml
# We use the figures from SunBlaze paper for EPOpt-PPO-FF
```

To run the best variant of our implemention, please refer to [our_details.md](our_details.md). 
