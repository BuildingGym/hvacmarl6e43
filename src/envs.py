import numpy as _numpy_

from controllables.energyplus import System
from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import DiscreteSpace, BoxSpace, DictSpace
from controllables.core.tools.rllib import MultiAgentEnv, Env
from controllables.energyplus import Actuator, OutputVariable, OutputMeter

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

# TODO
from rewards import ComfortElecSavingRewardFunction, ComfortElecSavingVectorRewardFunction


def _resolve(x):
    import os as _os_

    return (
        x if _os_.path.isabs(x) else 
        _os_.path.join(_os_.path.dirname(__file__), x)
    )


def make_building(**kwargs):
    return System(
        building=_resolve('building_all_zone_hvac.idf'),
        weather=_resolve('weather_sgp.epw'),
        **kwargs,
    )


class MultiAgentBuildingEnv(MultiAgentEnv):
    config: MultiAgentEnv.Config = {
        'agents': {},
    }

    # room agents
    class RoomAgentRewardFunction:
        def __init__(self):
            self._comfort_elec_saving_reward_function = ComfortElecSavingRewardFunction()
            self._comfort_elec_saving_vector_reward_function = ComfortElecSavingVectorRewardFunction()

        def __call__(self, agent) -> float:
            try:
                return self._comfort_elec_saving_vector_reward_function({
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
                'thermostat': BoxSpace(
                    low=20., high=30.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    Actuator.Ref(
                        type='Zone Temperature Control',
                        control_type='Cooling Setpoint',
                        key=var_key,
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
                # 'temperature': BoxSpace(
                #     low=-_numpy_.inf, high=+_numpy_.inf,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(
                #     OutputVariable.Ref(
                #         type='Zone Air Temperature',
                #         key=var_key,
                #     )
                # ),
                'temperature:drybulb': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(OutputVariable.Ref(
                        type='Zone Mean Air Temperature',
                        key=var_key,
                    )),
                'temperature:radiant': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(OutputVariable.Ref(
                        type='Zone Mean Radiant Temperature',
                        key=var_key,
                    )),
                'humidity': BoxSpace(
                        low=-_numpy_.inf, high=+_numpy_.inf,
                        dtype=_numpy_.float32,
                        shape=(),
                    ).bind(OutputVariable.Ref(
                        type='Zone Air Relative Humidity',
                        key=var_key,
                    )),
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
                ).bind(OutputMeter.Ref(
                    type='Electricity:HVAC',
                )),
                'occupancy': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    type='Schedule Value',
                    key='Office_OpenOff_Occ',
                )),
            }),
            'reward': RoomAgentRewardFunction(),
        }

    # HVAC equipment agents
    config['agents'].update({
        'CHILLER': {
            'participation': lambda agent: False,
            'action_space': DictSpace({
                'temperature': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(Actuator.Ref(
                    'System Node Setpoint',
                    'Temperature Setpoint',
                    'CHILLER CHW OUTLET NODE',
                )),
            }),
            'observation_space': DictSpace({
                'cooling-rate': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    'Chiller Evaporator Cooling Rate',
                    'CHILLER',
                )),
                'elec-rate': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    'Chiller Electricity Rate',
                    'CHILLER',
                )),
            }),
            'reward': (
                lambda agent: (
                    agent.observation.value['cooling-rate'] 
                    / agent.observation.value['elec-rate']
                ) if agent.observation.value['elec-rate'] != 0. else 0.
            ),
        },
        'AHU': {
            # TODO disabled
            'participation': lambda agent: False,
            'action_space': DictSpace({
                'air flow rate': BoxSpace(
                    low=0., high=10.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(Actuator.Ref(
                    type='Fan',
                    control_type='Fan Air Mass Flow Rate',
                    key='AIR LOOP AHU SUPPLY FAN'
                )),
            }),
            'observation_space': DictSpace({
                'energy-consumption': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    OutputMeter.Ref(
                        type='Electricity:HVAC',
                    )
                ),
                'AHU COOLING COIL': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    type='Cooling Coil Total Cooling Rate',
                    key='AIR LOOP AHU COOLING COIL',
                )),
            }),
            'reward': (
                lambda agent: (
                    agent.observation.value['AHU COOLING COIL'] 
                    / agent.observation.value['energy-consumption']
                ) if agent.observation.value['energy-consumption'] != 0. else 0.
            ),
        },
    })

    def __init__(self, config: dict = dict()):
        super().__init__({
            **self.__class__.config,
            **config,
        })

    def run(self):
        system = make_building(
            report='backup/tmp/',
            repeat=True,
        )
        # system.add('logging:progress')
        self.__attach__(system).schedule_episode(errors='ignore')
        system.start().wait()

    @classmethod
    def get_algo_config(cls, base_config: AlgorithmConfig):
        return (
            base_config
            .environment(cls)
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
                policy_mapping_fn=lambda agent_id, *args, **kwargs: str(agent_id),
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
        def __init__(self):
            self._reward_fns = {
                key: ComfortElecSavingVectorRewardFunction()
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

    def run(self):
        system = make_building(
            # TODO
            #report='tmp/',
            # TODO rm
            #report='backup/tmp/',
            repeat=True,
        )
        # system.add('logging:progress')
        self.__attach__(system).schedule_episode(errors='ignore')
        system.start().wait()

    @classmethod
    def get_algo_config(cls, base_config: AlgorithmConfig):
        return (
            base_config
            .environment(cls)
            .env_runners(
                # NOTE this env (an `ExternalEnv`) does not support connectors
                enable_connectors=False,
            )
        )


__all__ = [
    'MultiAgentBuildingEnv',
    'MonoAgentBuildingEnv',
]