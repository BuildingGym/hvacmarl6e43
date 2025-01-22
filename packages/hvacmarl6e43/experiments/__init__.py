import concurrent.futures as _concurrent_futures_
from ._common import BaseExperiment


def run_experiments(experiments: list[BaseExperiment], *args, **kwargs):
    with _concurrent_futures_.ThreadPoolExecutor() as pool:
        pool.map(lambda x: x.run(*args, **kwargs), experiments)


import numpy as _numpy_
from .baseline import BaselineExperiment
baseline_experiments = {
    str(setpoint) if setpoint is not None else 'default'
        : BaselineExperiment(setpoint=setpoint)
    for setpoint in _numpy_.arange(23., 25.)
}


from .rl import MonoAgentExperiment, MultiAgentExperiment
rl_experiments: dict[str, MonoAgentExperiment | MultiAgentExperiment] = {
    'monoagent': MonoAgentExperiment(),
    'multiagent': MultiAgentExperiment(),
}


__all__ = [
    'run_experiments',
    'baseline_experiments',
    'rl_experiments',
]