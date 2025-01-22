import functools as _functools_
from typing import Callable, Literal

import numpy as _numpy_

from controllables.energyplus import System
from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import DiscreteSpace, BoxSpace, DictSpace
from controllables.core.tools.gymnasium.spaces import map_spaces
from controllables.core.tools.rllib import MultiAgentEnv, Env
from controllables.energyplus import Actuator, OutputVariable, OutputMeter

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from ...buildings import OfficeBuilding
from .rewards import (
    ComfortElecSavingRewardFunction,
    ComfortElecSavingVectorRewardFunction,
)


def get_reward_fn(
    name_or_fn: Literal['elec-saving-diff', 'elec-saving-vector'] | Callable
):
    match name_or_fn:
        case 'elec-saving-diff':
            return ComfortElecSavingRewardFunction()
        case 'elec-saving-vector':
            return ComfortElecSavingVectorRewardFunction()
        case _:
            pass
    return name_or_fn


class MultiAgentBuildingEnv(MultiAgentEnv):
    config: MultiAgentEnv.Config = {
        'agents': {},
    }

    # room agents
    class RoomAgentRewardFunction:
        def __init__(self, name='elec-saving-vector'):
            self._base_fn = get_reward_fn(name)

        def __call__(self, agent) -> float:
            try:
                return self._base_fn({
                    'hvac_elec': agent.observation.value['energy-consumption'],
                    'office_occupancy': agent.observation.value['occupancy'],
                    'temperature_drybulb': agent.observation.value['temperature:drybulb'],
                    'temperature_radiant': agent.observation.value['temperature:radiant'],
                    'humidity': agent.observation.value['humidity'],
                })
            except TemporaryUnavailableError:
                return 0.

    room_agent_ids = {
        '1FWEST': '1FFIRSTFLOORWEST:OPENOFFICE',
        '1FEAST': '1FFIRSTFLOOREAST:OPENOFFICE',
        '0FWEST': '0FGROUNDFLOORWEST:OPENOFFICE',
        '0FEAST': '0FGROUNDFLOOREAST:OPENOFFICE',
        '1FWEST1': '1FFIRSTFLOORWEST1:OPENOFFICE',
        '1FEAST1': '1FFIRSTFLOOREAST1:OPENOFFICE',
        '0FWEST1': '0FGROUNDFLOORWEST1:OPENOFFICE',
        '0FEAST1': '0FGROUNDFLOOREAST1:OPENOFFICE',
    }

    for agent_id, var_key in room_agent_ids.items():
        config['agents'][agent_id] = {
            'action_space': DictSpace({
                # 'thermostat': BoxSpace(
                #     low=20., high=30.,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(
                #     Actuator.Ref(
                #         type='Zone Temperature Control',
                #         control_type='Cooling Setpoint',
                #         key=var_key,
                #     )
                # ),
                'thermostat': BoxSpace(
                    low=20, high=30,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    Actuator.Ref(
                        type='Schedule:Compact',
                        control_type='Schedule Value',
                        key=rf'{var_key} COOLING SETPOINT SCHEDULE',
                    )
                ),
                # 'air flow rate': BoxSpace(
                #     low=0., high=100.,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(Actuator.Ref(
                #     type='Fan',
                #     control_type='Fan Air Mass Flow Rate',
                #     key='AIR LOOP AHU SUPPLY FAN'
                # )),
                # 'air-flow-rate': BoxSpace(
                #     low=0., high=20.,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(
                #     Actuator.Ref(
                #         type='System Node Setpoint',
                #         control_type='Mass Flow Rate Setpoint',
                #         key=f'{var_key} SINGLE DUCT VAV NO REHEAT SUPPLY OUTLET',
                #     )
                # ),
            }),
            'observation_space': DictSpace({
                'temperature:drybulb': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda x, var_key=var_key:
                        x[OutputVariable.Ref(
                            type='Zone Mean Air Temperature',
                            key=var_key,
                        )]
                        .cast(_numpy_.array)
                ),
                'temperature:radiant': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda x, var_key=var_key:
                        x[OutputVariable.Ref(
                            type='Zone Mean Radiant Temperature',
                            key=var_key,
                        )]
                        .cast(_numpy_.array)
                ),
                'humidity': BoxSpace(
                    low=0., high=100.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda x, var_key=var_key:
                        x[OutputVariable.Ref(
                            type='Zone Air Relative Humidity',
                            key=var_key,
                        )]
                        .cast(_numpy_.array)
                ),
                # 'AHU COOLING COIL': BoxSpace(
                #         low=-_numpy_.inf, high=+_numpy_.inf,
                #         dtype=_numpy_.float32,
                #         shape=(),
                #     ).bind(OutputVariable.Ref(
                #         type='Cooling Coil Total Cooling Rate',
                #         key='AIR LOOP AHU COOLING COIL',
                #     )),
                # 'Fan Electricity Rate': BoxSpace(
                #     low=-_numpy_.inf, high=+_numpy_.inf,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(OutputVariable.Ref(
                #     type='Fan Electricity Rate',
                #     key='AIR LOOP AHU SUPPLY FAN',
                # )),
                'energy-consumption': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda x:
                        x[OutputMeter.Ref(
                            type='Electricity:HVAC',
                        )]
                        .cast(_numpy_.array)
                ),
                'occupancy': BoxSpace(
                    low=0., high=1.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda x:
                        x[OutputVariable.Ref(
                            type='Schedule Value',
                            key='Office_OpenOff_Occ',
                        )]
                        .cast(_numpy_.array)
                ),
            }),
            'reward': RoomAgentRewardFunction(),
        }

    # HVAC equipment agents
    # config['agents'].update({
    #     'CHILLER': {
    #         'participation': lambda agent: False,
    #         'action_space': DictSpace({
    #             'temperature': BoxSpace(
    #                 low=-_numpy_.inf, high=+_numpy_.inf,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(Actuator.Ref(
    #                 'System Node Setpoint',
    #                 'Temperature Setpoint',
    #                 'CHILLER CHW OUTLET NODE',
    #             )),
    #         }),
    #         'observation_space': DictSpace({
    #             'cooling-rate': BoxSpace(
    #                 low=-_numpy_.inf, high=+_numpy_.inf,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(OutputVariable.Ref(
    #                 'Chiller Evaporator Cooling Rate',
    #                 'CHILLER',
    #             )),
    #             'elec-rate': BoxSpace(
    #                 low=-_numpy_.inf, high=+_numpy_.inf,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(OutputVariable.Ref(
    #                 'Chiller Electricity Rate',
    #                 'CHILLER',
    #             )),
    #         }),
    #         'reward': (
    #             lambda agent: (
    #                 agent.observation.value['cooling-rate']
    #                 / agent.observation.value['elec-rate']
    #             ) if agent.observation.value['elec-rate'] != 0. else 0.
    #         ),
    #     },
    #     'AHU': {
    #         # TODO disabled
    #         'participation': lambda agent: False,
    #         'action_space': DictSpace({
    #             'air flow rate': BoxSpace(
    #                 low=0., high=10.,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(Actuator.Ref(
    #                 type='Fan',
    #                 control_type='Fan Air Mass Flow Rate',
    #                 key='AIR LOOP AHU SUPPLY FAN'
    #             )),
    #         }),
    #         'observation_space': DictSpace({
    #             'energy-consumption': BoxSpace(
    #                 low=-_numpy_.inf, high=+_numpy_.inf,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(
    #                 OutputMeter.Ref(
    #                     type='Electricity:HVAC',
    #                 )
    #             ),
    #             'AHU COOLING COIL': BoxSpace(
    #                 low=-_numpy_.inf, high=+_numpy_.inf,
    #                 dtype=_numpy_.float32,
    #                 shape=(),
    #             ).bind(OutputVariable.Ref(
    #                 type='Cooling Coil Total Cooling Rate',
    #                 key='AIR LOOP AHU COOLING COIL',
    #             )),
    #         }),
    #         'reward': (
    #             lambda agent: (
    #                 agent.observation.value['AHU COOLING COIL']
    #                 / agent.observation.value['energy-consumption']
    #             ) if agent.observation.value['energy-consumption'] != 0. else 0.
    #         ),
    #     },
    # })

    def __init__(self, config: dict = dict()):
        super().__init__({
            **self.__class__.config,
            **config,
        })
        self._building_init = config.get('building_init')

    def _make_building(self) -> OfficeBuilding:
        return OfficeBuilding()

    def run(self):
        system = self._make_building()
        if self._building_init is not None:
            self._building_init(system)
        # TODO
        # system.add('logging:progress')
        self.__attach__(system)
        self.schedule_episode(errors='warn')
        # TODO this is a hack to make sure the variables are requested
        for _, agent_config in self.config['agents'].items():
            map_spaces(
                lambda space: space.deref(system),
                agent_config['observation_space'],
            )
        while True:
            # system.relocate('__random__')
            system.start().wait()

    @classmethod
    def get_algo_config(
        cls,
        base_config: AlgorithmConfig,
        env_config: dict = dict(),
    ):
        return (
            base_config
            .environment(cls, env_config=env_config)
            .env_runners(
                # NOTE this env (an `ExternalEnv`) does not support connectors
                enable_connectors=False,
            )
            .multi_agent(
                policies={
                    policy_id: PolicySpec(
                        action_space=agent_config['action_space'],
                        observation_space=agent_config['observation_space'],
                    )
                    for policy_id, agent_config in cls.config['agents'].items()
                },
                policy_mapping_fn=lambda agent_id, * \
                args, **kwargs: str(agent_id),
            )
        )


# TODO
class MonoAgentBuildingEnv(Env):
    config: Env.Config = {
        entry: DictSpace({
            key: MultiAgentBuildingEnv.config['agents'][key][entry]
            for key in MultiAgentBuildingEnv.room_agent_ids
        })
        for entry in ('action_space', 'observation_space')
    }

    class EnvRewardFunction:
        def __init__(self, name='elec-saving-vector'):
            self._reward_fns = {
                key: get_reward_fn(name)
                for key in MultiAgentBuildingEnv.room_agent_ids
            }

        def __call__(self, agent):
            try:
                return _numpy_.sum([
                    reward_fn({
                        'hvac_elec': agent.observation.value[key]['energy-consumption'],
                        'office_occupancy': agent.observation.value[key]['occupancy'],
                        'temperature_drybulb': agent.observation.value[key]['temperature:drybulb'],
                        'temperature_radiant': agent.observation.value[key]['temperature:radiant'],
                        'humidity': agent.observation.value[key]['humidity'],
                    })
                    for key, reward_fn in self._reward_fns.items()
                ])
            except TemporaryUnavailableError:
                return 0.

    # reward
    config['reward'] = EnvRewardFunction()

    def __init__(self, config: dict = dict()):
        super().__init__({
            **self.__class__.config,
            **config,
        })
        self._building_init = config.get('building_init')

    def _make_building(self) -> OfficeBuilding:
        return OfficeBuilding(repeat=True)

    def run(self):
        system = self._make_building()
        if self._building_init is not None:
            self._building_init(system)
        # system.add('logging:progress')
        self.__attach__(system)
        self.schedule_episode(errors='warn')
        # TODO this is a hack to make sure the variables are requested
        map_spaces(
            lambda space: space.deref(system),
            self.config['observation_space'],
        )
        while True:
            # system.relocate('__random__')
            system.start().wait()

    @classmethod
    def get_algo_config(
        cls,
        base_config: AlgorithmConfig,
        env_config: dict = dict(),
    ):
        return (
            base_config
            .environment(cls, env_config=env_config)
            .env_runners(
                # NOTE this env (an `ExternalEnv`) does not support connectors
                enable_connectors=False,
            )
        )


__all__ = [
    'MultiAgentBuildingEnv',
    'MonoAgentBuildingEnv',
]
