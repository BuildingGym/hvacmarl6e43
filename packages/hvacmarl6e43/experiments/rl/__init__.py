import dataclasses as _dataclasses_
from pathlib import Path

import numpy as _numpy_
import pandas as _pandas_
from ray import tune, air
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune.stopper import (
    CombinedStopper, 
    MaximumIterationStopper, 
    TrialPlateauStopper,
)
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from .._common import (
    CommonExperiment,
    CommonExperimentResult,
    DataFrameStore, 
)
from .envs import MultiAgentBuildingEnv, MonoAgentBuildingEnv


@_dataclasses_.dataclass(frozen=True)
class RLExperimentResult(CommonExperimentResult):
    checkpoint_train: Path
    r"""
    The path to the checkpoint of the best training result.
    """

    metrics_train: _pandas_.DataFrame
    r"""
    The training metrics of the best training result returned by rllib.
    """

    #metrics_eval: _pandas_.DataFrame


class CommonRLExperiment(CommonExperiment):
    def _train(self, algo_config: PPOConfig, dryrun: bool):
        tuner = tune.Tuner(
            PPO,
            param_space=algo_config,
            tune_config=tune.TuneConfig(
                #reuse_actors=True,
                # scheduler=PopulationBasedTraining(
                #     time_attr='training_iteration',
                #     #perturbation_interval=4,
                #     perturbation_interval=1,
                #     resample_probability=0.25,
                #     hyperparam_mutations={
                #         #'lr': tune.uniform(1e-5, 0.1),
                #         'lr': list(_numpy_.logspace(-2, -16, base=2, num=8)),
                #         'train_batch_size': [1_000],
                #         #'sgd_minibatch_size': [32, 64, 128, 256, 512],
                #         #'num_sgd_iter': [10, 20, 30],
                #         'clip_param': tune.uniform(.1, .3),
                #     },
                # ),
                scheduler=PB2(
                    time_attr='training_iteration',
                    perturbation_interval=1,
                    quantile_fraction=0.25,
                    hyperparam_bounds={
                        'lambda': [0.9, 1.0],
                        'clip_param': [0.1, 0.5],
                        'lr': [1e-5, 1e-3],
                        'train_batch_size': [5_000, 10_000]
                    },
                ),
                # TODO
                #num_samples=3,
                num_samples=4,
                #num_samples=1,
                metric='env_runners/episode_reward_mean',
                mode='max',
            ),
            run_config=air.RunConfig(
                stop=CombinedStopper(
                    MaximumIterationStopper(max_iter=300),
                    TrialPlateauStopper(
                        metric='episode_reward_mean', 
                        std=0.01, 
                        num_results=20,
                        grace_period=10,
                    ),
                ),
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=True
                ),
                verbose=2,
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()

        if not dryrun:
            if best_result.checkpoint is not None:
                best_result.checkpoint.to_directory(
                    self._base_storage / 'checkpoint:train'
                )
            DataFrameStore(
                self._base_storage / 'metrics:train'
            ).set({'best': best_result.metrics_dataframe})

    def _eval(self, algo_config: PPOConfig, dryrun: bool):
        config_eval = (
            algo_config
            .env_runners(
                num_env_runners=0,
                create_env_on_local_worker=True,
            )
            .evaluation(
                evaluation_duration=2,
                evaluation_duration_unit='episodes',
                #evaluation_interval=1,
                evaluation_num_env_runners=0,
            )
        )

        algo_eval = PPO(config_eval)
        algo_eval.restore(str(self._base_storage / 'checkpoint:train'))
        algo_eval.evaluate()

    def get_results(self):
        return RLExperimentResult(
            **_dataclasses_.asdict(
                CommonExperiment.get_results(self)
            ),
            checkpoint_train=self._base_storage / 'checkpoint:train',
            metrics_train=DataFrameStore(self._base_storage / 'metrics:train').get()['best'],
        )


class MonoAgentExperiment(CommonRLExperiment):
    def __init__(self, name: str | None = 'monoagent'):
        super().__init__(name=name)

    def run(self, train: bool = True, dryrun: bool = True):        
        if train:
            self._train(
                MonoAgentBuildingEnv.get_algo_config(
                    PPOConfig()
                    # .api_stack(enable_rl_module_and_learner=True,
                    #            enable_env_runner_and_connector_v2=True)
                    .rollouts(
                        batch_mode='complete_episodes',
                        sample_timeout_s=600,
                        #num_rollout_workers=4, 
                        rollout_fragment_length='auto',
                        # num_envs_per_env_runner = 4,
                        # rollout_fragment_length=2000,
                    )
                    .resources(num_gpus=1/4)
                ),
                dryrun=dryrun,
            )

        _collect_observations = self._collect_observations
        class EvaluatedMonoAgentBuildingEnv(MonoAgentBuildingEnv):
            def _make_building(self):
                building = super()._make_building()
                building.add('logging:progress')
                _collect_observations(building, dryrun=dryrun)

                return building
            
        self._eval(
            EvaluatedMonoAgentBuildingEnv.get_algo_config(
                PPOConfig()
                .rollouts(
                    num_rollout_workers=4, 
                    rollout_fragment_length='auto',
                ),
            ),
            dryrun=dryrun,
        )


class MultiAgentExperiment(CommonRLExperiment):
    def __init__(self, name: str | None = 'multiagent'):
        super().__init__(name=name)

    def run(self, train: bool = True, dryrun: bool = True):
        if train:
            self._train(
                MultiAgentBuildingEnv.get_algo_config(
                    PPOConfig()
                    .rollouts(
                        batch_mode='complete_episodes',
                        sample_timeout_s=600,
                        #num_rollout_workers=4, 
                        rollout_fragment_length='auto',
                        # rollout_fragment_length=200,
                    )
                    .resources(num_gpus=1/4)
                ),
                dryrun=dryrun,
            )

        _collect_observations = self._collect_observations
        class EvaluatedMultiAgentBuildingEnv(MultiAgentBuildingEnv):
            def _make_building(self):
                building = super()._make_building()
                building.add('logging:progress')
                _collect_observations(building, dryrun=dryrun)
                return building
            
        self._eval(
            EvaluatedMultiAgentBuildingEnv.get_algo_config(
                PPOConfig()
                .rollouts(
                    num_rollout_workers=3, 
                    rollout_fragment_length='auto',
                ),
            ),
            dryrun=dryrun,
        )
