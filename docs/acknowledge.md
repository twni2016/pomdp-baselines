## Acknowledgement
We acknowledge the following repositories that greatly shaped our implementation:
- https://github.com/pranz24/pytorch-soft-actor-critic for providing a soft actor-critic implementation in PyTorch
- https://github.com/Rondorf/BOReL for providing the off-policy varibad algorithm and environments
- http://proceedings.mlr.press/v139/jiang21c/jiang21c-supp.zip for providing the robust RL MRPO algorithm and environments
- https://github.com/sunblaze-ucb/rl-generalization for providing the SunBlaze roboschool benchmark
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail for providing the on-policy recurrent RL baselines
- https://github.com/oist-cnru/Variational-Recurrent-Models for providing the pomdp VRM algorithm and environments
- https://github.com/quantumiracle/Popular-RL-Algorithms for inspiring the recurrent policies design
- https://github.com/lmzintgraf/varibad for inspiring the recurrent policies design and providing learning curve data
- https://github.com/ku2482/sac-discrete.pytorch for providing the SAC-discrete code

Please cite their work if you also find their code useful to your project:
```
@article{dorfman2020offline,
  title={Offline Meta Learning of Exploration},
  author={Dorfman, Ron and Shenfeld, Idan and Tamar, Aviv},
  journal={arXiv preprint arXiv:2008.02598},
  year={2020}
}
@inproceedings{jiang2021monotonic,
  title={Monotonic Robust Policy Optimization with Model Discrepancy},
  author={Jiang, Yuankun and Li, Chenglin and Dai, Wenrui and Zou, Junni and Xiong, Hongkai},
  booktitle={International Conference on Machine Learning},
  pages={4951--4960},
  year={2021},
  organization={PMLR}
}
@misc{PackerGao:1810.12282,
  Author = {Charles Packer and Katelyn Gao and Jernej Kos and Philipp Kr\"ahenb\"uhl and Vladlen Koltun and Dawn Song},
  Title = {Assessing Generalization in Deep Reinforcement Learning},
  Year = {2018},
  Eprint = {arXiv:1810.12282},
}
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
}
@article{han2019variational,
  title={Variational recurrent models for solving partially observable control tasks},
  author={Han, Dongqi and Doya, Kenji and Tani, Jun},
  journal={arXiv preprint arXiv:1912.10703},
  year={2019}
}
@inproceedings{zintgraf2020varibad,
  title={VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning},
  author={Zintgraf, Luisa and Shiarlis, Kyriacos and Igl, Maximilian and Schulze, Sebastian and Gal, Yarin and Hofmann, Katja and Whiteson, Shimon},
  booktitle={International Conference on Learning Representation (ICLR)},
  year={2020}}
@book{deepRL-2020,
 title={Deep Reinforcement Learning: Fundamentals, Research, and Applications},
 editor={Hao Dong, Zihan Ding, Shanghang Zhang},
 author={Hao Dong, Zihan Ding, Shanghang Zhang, Hang Yuan, Hongming Zhang, Jingqing Zhang, Yanhua Huang, Tianyang Yu, Huaqing Zhang, Ruitong Huang},
 publisher={Springer Nature},
 note={\url{http://www.deepreinforcementlearningbook.org}},
 year={2020}
}
@article{christodoulou2019soft,
  title={Soft actor-critic for discrete action settings},
  author={Christodoulou, Petros},
  journal={arXiv preprint arXiv:1910.07207},
  year={2019}
}
```
