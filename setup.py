from pkg_resources import parse_requirements
from setuptools import setup

setup(
   name='tanktrouble_env',
   version='1.0',
   description='TankTrouble RL environment for PettingZoo',
   author='Samy Rasmy, Vincent Quirion, Amine Obeid',
   author_email='samyrasmy0@gmail.com',
   packages=['tanktrouble_rl'],  #same as name
   install_requires = [str(r) for r in parse_requirements(open('requirements.txt'))]
)