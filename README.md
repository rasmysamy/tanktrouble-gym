# Baseline RL et environment pour IFT3710

Ce projet contient le code utilisé pour les baslines de RL, ainsi que notre environment TankTrouble.

## Installation

Veuillez installer toutes les dépendenaces du fichier requirements.txt

## Utilisation

Après installation, il sera  possible de jouer manuellement dans l'environment en éxécutant le fichier test.py

Pour démarrer l'entrainement, il suffit d'éxécuter le fichier tianshou_train.py.

Durant l'entraînement, il est possible de visualiser l'agent jouer contre lui-même en éxécutant le fichier tianshou_train.py (avec l'option --watch). Il est possible de jouer contre l'agent, en modifiant la variable `user_play`. 

Des configurations PyCharm sont incluses dans le projet pour l'éxécution de ces trois fonctions.