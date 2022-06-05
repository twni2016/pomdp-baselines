export PYTHONPATH=${PWD}:$PYTHONPATH

## Standard POMDP (occlusion benchmark)

# python scripts/merge_csv.py --base_path results/logs/pomdp/Pendulum/V \
#     --max_episode_len 200 --start_x 2000 --interval_x 1000
# python scripts/plot_csv.py --csv_path results/data/pomdp/Pendulum/V/final.csv \
#     --max_x 30000 --window_size 3 # don't forget uncomment "Arch"
# python scripts/plot_diagnose.py --csv_path results/data/pomdp/Pendulum/V/final.csv \
#     --max_x 50000 --window_size 3 --instances sac-lstm-200-oar-separate,sac-lstm-200-oar-shared

# python scripts/merge_csv.py --base_path results/logs/pomdp/AntBLT/P \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/AntBLT/P/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Ant-P
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/AntBLT/P/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/AntBLT/V \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/AntBLT/V/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Ant-V
# python scripts/plot_csv.py --csv_path results/data/pomdp/AntBLT/V/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/AntBLT/V/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/HalfCheetahBLT/P \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/HalfCheetahBLT/P/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Cheetah-P
# python scripts/plot_csv.py --csv_path results/data/pomdp/HalfCheetahBLT/P/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/HalfCheetahBLT/P/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/HalfCheetahBLT/V \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/HalfCheetahBLT/V/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Cheetah-V
# python scripts/plot_csv.py --csv_path results/data/pomdp/HalfCheetahBLT/V/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/HalfCheetahBLT/V/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/HopperBLT/P \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/HopperBLT/P/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Hopper-P
# python scripts/plot_csv.py --csv_path results/data/pomdp/HopperBLT/P/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/HopperBLT/P/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/HopperBLT/V \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/HopperBLT/V/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Hopper-V
# python scripts/plot_csv.py --csv_path results/data/pomdp/HopperBLT/V/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/HopperBLT/V/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/WalkerBLT/P \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/WalkerBLT/P/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Walker-P
# python scripts/plot_csv.py --csv_path results/data/pomdp/WalkerBLT/P/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/WalkerBLT/P/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/pomdp/WalkerBLT/V \
#     --max_episode_len 1000 --start_x 12000 --interval_x 4000
# python scripts/plot_csv.py --csv_path results/data/pomdp/WalkerBLT/V/final.csv \
#     --window_size 20 --max_x 1500000 --best_variant td3-gru-64-oa-separate --other_methods ppo_gru,a2c_gru,VRM \
#     --name Walker-V
# python scripts/plot_csv.py --csv_path results/data/pomdp/WalkerBLT/V/final.csv \
#     --window_size 20 --max_x 500000 --best_variant td3-gru-64-oa-separate
# python scripts/plot_single_factor.py --csv_path results/data/pomdp/WalkerBLT/V/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs




## Meta RL (off-policy varibad benchmark)

# python scripts/merge_csv.py --base_path results/logs/meta/HalfCheetahVel-v0 \
#     --max_episode_len 400 --start_x 240000 --interval_x 40000
# python scripts/plot_csv.py --csv_path results/data/meta/HalfCheetahVel-v0/final.csv  \
#     --window_size 10 --max_x 20000000 --best_variant td3-lstm-64-oar-separate --other_methods offpolicy-varibad \
#     --name Cheetah-Vel --loc "lower right"
# python scripts/merge_csv.py --base_path results/logs/meta/HalfCheetahVel-v0/oracle \
#     --max_episode_len 400 --start_x 240000 --interval_x 40000
# python scripts/plot_csv.py --csv_path results/data/meta/HalfCheetahVel-v0/oracle/final.csv  \
#     --window_size 10 --max_x 5000000
# python scripts/plot_single_factor.py --csv_path results/data/meta/HalfCheetahVel-v0/final.csv \
#     --window_size 10 --factors RL,Encoder,Len

# python scripts/merge_csv.py --base_path results/logs/meta/PointRobotSparse-v0 \
#     --max_episode_len 120 --start_x 60000 --interval_x 15000
# python scripts/plot_csv.py --csv_path results/data/meta/PointRobotSparse-v0/final.csv \
#     --window_size 10 --best_variant td3-lstm-64-or-separate --other_methods offpolicy-varibad \
#     --name Semi-Circle --loc "lower right"
# python scripts/merge_csv.py --base_path results/logs/meta/PointRobotSparse-v0/oracle \
#     --max_episode_len 120 --start_x 60000 --interval_x 15000
# python scripts/plot_csv.py --csv_path results/data/meta/PointRobotSparse-v0/oracle/final.csv \
#     --window_size 10 --max_x 1500000
# python scripts/plot_diagnose.py --csv_path results/data/meta/PointRobotSparse-v0/final.csv \
#     --window_size 10 --instances td3-lstm-64-or-separate,td3-lstm-64-or-shared
# python scripts/plot_single_factor.py --csv_path results/data/meta/PointRobotSparse-v0/final.csv \
#     --window_size 10 --factors Arch,RL,Encoder,Len

# python scripts/merge_csv.py --base_path results/logs/meta/Wind-v0 \
#     --max_episode_len 75 --start_x 9000 --interval_x 3000
# python scripts/plot_csv.py --csv_path results/data/meta/Wind-v0/final.csv \
#     --window_size 10 --best_variant td3-lstm-64-oa-separate --other_methods offpolicy-varibad \
#     --name Wind --loc "lower right"
# python scripts/merge_csv.py --base_path results/logs/meta/Wind-v0/oracle \
#     --max_episode_len 75 --start_x 9000 --interval_x 3000
# python scripts/plot_csv.py --csv_path results/data/meta/Wind-v0/oracle/final.csv \
#     --window_size 10
# python scripts/plot_single_factor.py --csv_path results/data/meta/Wind-v0/final.csv \
#     --window_size 10 --factors RL,Encoder,Len,Inputs





## Meta RL (on-policy varibad benchmark)

# python scripts/merge_csv.py --base_path results/logs/meta/AntDir-v0 \
#     --max_episode_len 400 --start_x 240000 --interval_x 40000
# python scripts/plot_csv.py --csv_path results/data/meta/AntDir-v0/final.csv  \
#     --window_size 10 --max_x 100000000 --other_methods onpolicy-varibad,rl2,oracle_ppo,oracle_sac,Markovian_td3 \
#     --best_variant sac-gru-400-oar-separate \
#     --loc "lower right" --name Ant-Dir  

# python scripts/merge_csv.py --base_path results/logs/meta/CheetahDir-v0 \
#     --max_episode_len 400 --start_x 240000 --interval_x 40000
# python scripts/plot_csv.py --csv_path results/data/meta/CheetahDir-v0/final.csv  \
#     --window_size 10 --max_x 100000000 --other_methods onpolicy-varibad,rl2,oracle_ppo,Markovian_sac,oracle_sac \
#     --best_variant sac-gru-400-oar-separate \
#     --loc "lower right" --name Cheetah-Dir 

# python scripts/merge_csv.py --base_path results/logs/meta/HumanoidDir-v0 \
#     --max_episode_len 400 --start_x 240000 --interval_x 40000
# python scripts/plot_csv.py --csv_path results/data/meta/HumanoidDir-v0/final.csv  \
#     --window_size 10 --max_x 100000000 --other_methods onpolicy-varibad,rl2,oracle_ppo,oracle_sac,Markovian_sac \
#     --best_variant sac-gru-400-oar-separate \
#     --loc "lower right" --name Humanoid-Dir  





## Robust RL

# python scripts/merge_csv.py --base_path results/logs/rmdp/MRPOHalfCheetahRandomNormal-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/rmdp/MRPOHalfCheetahRandomNormal-v0/final.csv \
#     --window_size 20 --max_x 15000000 --best_variant td3-lstm-64-o-separate --other_methods MRPO,ppo_gru,oracle_sac,Markovian_sac \
#     --name Cheetah-Robust --loc "upper right"
# python scripts/plot_single_factor.py --csv_path results/data/rmdp/MRPOHalfCheetahRandomNormal-v0/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/rmdp/MRPOHopperRandomNormal-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/rmdp/MRPOHopperRandomNormal-v0/final.csv \
#     --window_size 20 --max_x 20000000 --best_variant td3-lstm-64-o-separate --other_methods MRPO,ppo_gru,oracle_sac,Markovian_sac \
#     --name Hopper-Robust --loc "upper right"
# python scripts/plot_single_factor.py --csv_path results/data/rmdp/MRPOHopperRandomNormal-v0/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs

# python scripts/merge_csv.py --base_path results/logs/rmdp/MRPOWalker2dRandomNormal-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/rmdp/MRPOWalker2dRandomNormal-v0/final.csv \
#     --window_size 20 --max_x 20000000 --best_variant td3-lstm-64-o-separate --other_methods MRPO,ppo_gru,oracle_sac,Markovian_sac \
#     --name Walker-Robust --loc "upper right"
# python scripts/plot_single_factor.py --csv_path results/data/rmdp/MRPOWalker2dRandomNormal-v0/final.csv \
#     --window_size 20 --factors RL,Encoder,Len,Inputs




## Generalization in RL

# python scripts/merge_csv.py --base_path results/logs/generalize/SunblazeHalfCheetah-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/generalize/SunblazeHalfCheetah-v0/final.csv \
#     --window_size 20 --max_x 20000000 --best_variant td3-lstm-64-o-separate
# python scripts/merge_csv.py --base_path results/logs/generalize/SunblazeHalfCheetahRandomNormal-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/generalize/SunblazeHalfCheetahRandomNormal-v0/final.csv \
#     --window_size 20 --max_x 20000000 --best_variant td3-lstm-64-o-separate
# python scripts/plot_generalization.py --merged_path results/data/generalize/Cheetah-Generalize \
#     --csv_paths results/data/generalize/SunblazeHalfCheetah-v0/final.csv,results/data/generalize/SunblazeHalfCheetahRandomNormal-v0/final.csv \
#     --window_size 10 --name Cheetah-Generalize --factors RL,Len,Inputs --best_variant td3-lstm-64-o-separate \
#     --other_methods oracle_td3,Markovian_sac


# python scripts/merge_csv.py --base_path results/logs/generalize/SunblazeHopper-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/generalize/SunblazeHopper-v0/final.csv \
#     --window_size 20 --max_x 20000000 --best_variant td3-lstm-64-o-separate
# python scripts/merge_csv.py --base_path results/logs/generalize/SunblazeHopperRandomNormal-v0 \
#     --max_episode_len 1000 --start_x 50000 --interval_x 50000
# python scripts/plot_csv.py --csv_path results/data/generalize/SunblazeHopperRandomNormal-v0/final.csv \
#     --window_size 20 --max_x 20000000 --best_variant td3-lstm-64-o-separate
# python scripts/plot_generalization.py --merged_path results/data/generalize/Hopper-Generalize \
#     --csv_paths results/data/generalize/SunblazeHopper-v0/final.csv,results/data/generalize/SunblazeHopperRandomNormal-v0/final.csv \
#     --window_size 10 --name Hopper-Generalize --factors RL,Len,Inputs --best_variant td3-lstm-64-o-separate \
#     --other_methods oracle_sac,Markovian_sac




## Long-term credit assignment

# python scripts/merge_csv.py --base_path results/logs/credit/Catch/40 \
#     --max_episode_len 279 --start_x 13950 --interval_x 13950
# python scripts/plot_csv.py --csv_path results/data/credit/Catch/40/final.csv \
#     --window_size 10 --loc "right" --max_x 2500000 \
#     --name Delayed-Catch --best_variant sacd-lstm-279-o-separate

# python scripts/merge_csv.py --base_path results/logs/credit/KeytoDoor/SR \
#     --max_episode_len 85 --start_x 4250 --interval_x 4250
# python scripts/plot_csv.py --csv_path results/data/credit/KeytoDoor/SR/final.csv \
#     --window_size 10 --loc "lower right" --max_x 4000000 \
#     --best_variant sacd-lstm-85-o-separate --name Key-to-Door
