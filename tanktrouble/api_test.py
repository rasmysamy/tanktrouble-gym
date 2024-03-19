from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils import parallel_to_aec

import env.tanktrouble_env as tanktrouble
env = tanktrouble.TankTrouble()
parallel_api_test(env, num_cycles=1000)
api_test(parallel_to_aec(env), num_cycles=1000)