export PYTHONPATH=${PWD}:$PYTHONPATH
# python3 policies/main.py --cfg configs/pomdp/pendulum/v/rnn.yml --algo sac --seed 10 --cuda 0
# python3 policies/main.py --cfg configs/pomdp/cartpole/v/rnn.yml --target_entropy 0.5 2 >> logs/error.log 1 > /dev/null &
# python3 policies/main.py --cfg configs/pomdp/cartpole/f/mlp.yml --target_entropy 0.98 --cuda -1 2 >> logs/error.log 1 > /dev/null &
# python3 policies/main.py --cfg configs/pomdp/lunalander/v/rnn.yml --target_entropy 0.3 2 >> logs/error.log 1 > /dev/null &
# python3 policies/main.py --cfg configs/pomdp/lunalander/f/mlp.yml --target_entropy 0.9 --cuda -1 2 >> logs/error.log 1 > /dev/null &

# python3  policies/main.py --cfg configs/meta/ant_dir/rnn.yml
# python3 policies/main.py --cfg configs/pomdp/cartpole/v/rnn.yml --noautomatic_entropy_tuning --entropy_alpha 0.1 --cuda 0 --seed 1011
# python3 policies/main.py --cfg configs/credit/catch/rnn.yml --seed 11

# python3 policies/main.py --cfg configs/credit/keytodoor/rnn.yml --noautomatic_entropy_tuning --entropy_alpha 0.1 --cuda 0 --seed 1011

python3 policies/main.py --cfg configs/atari/rnn.yml --cuda -1 --seed 1011 --env Solaris
