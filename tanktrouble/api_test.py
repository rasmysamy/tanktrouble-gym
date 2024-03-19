from pettingzoo.test import api_test, parallel_api_test
import env.tanktrouble_env as tanktrouble
env = tanktrouble.TankTrouble()
parallel_api_test(env, num_cycles=1000)