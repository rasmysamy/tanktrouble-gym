import jax
import muax
from muax import nn

import haiku as hk


class Representation(hk.Module):
    def __init__(self, embedding_dim, name='representation'):
        super().__init__(name=name)

        self.repr_func = hk.Sequential([
            hk.Linear(embedding_dim),
            # jax.nn.elu,
        ])

    def __call__(self, obs):
        s = self.repr_func(obs)
        s = nn.min_max_normalize(s)
        return s


class Prediction(hk.Module):
    def __init__(self, num_actions, full_support_size, name='prediction'):
        super().__init__(name=name)

        self.v_func = hk.Sequential([
            hk.Linear(64), jax.nn.elu,
            hk.Linear(64), jax.nn.elu,
            hk.Linear(16), jax.nn.elu,
            hk.Linear(full_support_size)
        ])
        self.pi_func = hk.Sequential([
            hk.Linear(64), jax.nn.elu,
            hk.Linear(64), jax.nn.elu,
            hk.Linear(16), jax.nn.elu,
            hk.Linear(num_actions)
        ])

    def __call__(self, s):
        v = self.v_func(s)
        logits = self.pi_func(s)
        # logits = jax.nn.softmax(logits, axis=-1)
        return v, logits


class Dynamic(hk.Module):
    def __init__(self, embedding_dim, num_actions, full_support_size, name='dynamic'):
        super().__init__(name=name)

        self.ns_func = hk.Sequential([
            hk.Linear(64), jax.nn.elu,
            hk.Linear(64), jax.nn.elu,
            hk.Linear(16), jax.nn.elu,
            hk.Linear(embedding_dim)
        ])
        self.r_func = hk.Sequential([
            hk.Linear(64), jax.nn.elu,
            hk.Linear(64), jax.nn.elu,
            hk.Linear(16), jax.nn.elu,
            hk.Linear(full_support_size)
        ])
        self.cat_func = jax.jit(lambda s, a:
                                jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                                axis=1)
                                )

    def __call__(self, s, a):
        sa = self.cat_func(s, a)
        r = self.r_func(sa)
        ns = self.ns_func(sa)
        ns = nn.min_max_normalize(ns)
        return r, ns


def init_representation_func(representation_module, embedding_dim):
    def representation_func(obs):
        repr_model = representation_module(embedding_dim)
        return repr_model(obs)

    return representation_func


def init_prediction_func(prediction_module, num_actions, full_support_size):
    def prediction_func(s):
        pred_model = prediction_module(num_actions, full_support_size)
        return pred_model(s)

    return prediction_func


def init_dynamic_func(dynamic_module, embedding_dim, num_actions, full_support_size):
    def dynamic_func(s, a):
        dy_model = dynamic_module(embedding_dim, num_actions, full_support_size)
        return dy_model(s, a)

    return dynamic_func


support_size = 10
embedding_size = 8
discount = 0.99
num_actions = 2
full_support_size = int(support_size * 2 + 1)

repr_fn = init_representation_func(nn.Representation, embedding_size)
pred_fn = init_prediction_func(nn.Prediction, num_actions, full_support_size)
dy_fn = init_dynamic_func(nn.Dynamic, embedding_size, num_actions, full_support_size)

tracer = muax.PNStep(10, discount, 0.5)
buffer = muax.TrajectoryReplayBuffer(500)

gradient_transform = muax.model.optimizer(init_value=0.02, peak_value=0.02, end_value=0.002, warmup_steps=5000, transition_steps=5000)

model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                    optimizer=gradient_transform, support_size=support_size)

model_path = muax.fit(model, 'CartPole-v1',
                    max_episodes=1000,
                    max_training_steps=10000,
                    tracer=tracer,
                    buffer=buffer,
                    k_steps=10,
                    sample_per_trajectory=1,
                    num_trajectory=32,
                    tensorboard_dir='/content/tensorboard/cartpole',
                    model_save_path='/content/models/cartpole',
                    save_name='cartpole_model_params',
                    random_seed=0,
                    log_all_metrics=True)