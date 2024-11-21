import energyplus.ooep as _ooep_
from energyplus.dataset.basic import dataset as _epds_

simulator = _ooep_.World(
    input=_ooep_.World.InputSpecs(
        input='tmp_2.idf',
        weather='SGP_Singapore_486980_IWEC.epw',
    ),
    output=_ooep_.World.OutputSpecs(
        report='./tmp',
    ),
    options={
        'recurring': True,
    },
)

from energyplus.ooep.addons.logging import ProgressLogger

# add progress provider
_ = simulator.add(ProgressLogger())
_ = simulator.awaitable.run()

import numpy as _numpy_
import gymnasium as _gymnasium_

from energyplus.ooep.addons.rl import (
    VariableBox,
    SimulatorEnv,
)

from energyplus.ooep import (
    Actuator,
    OutputVariable,
)
import pythermalcomfort as pytc
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.algorithms.callbacks import DefaultCallbacks

import logging

import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='training.log')

class CustomSimulatorEnv(SimulatorEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.skip_control_start = env_config.get("skip_control_start", 8)
        self.skip_control_end = env_config.get("skip_control_end", 19)

    def step(self, action):
        self.current_time = simulator.variables.getdefault('wallclock').value.hour

        if self.skip_control_start <= self.current_time < self.skip_control_end:
            pass
        observation, reward, done, info = super().step(action)
        return observation, reward, done, info

class RewardFunction:
    def __init__(self, metab_rate=1.5, clothing=.5, pmv_limit=.5):
        self._metab_rate = _numpy_.asarray(metab_rate)
        self._clothing = _numpy_.asarray(clothing)
        self._pmv_limit = _numpy_.asarray(pmv_limit)
    
   
    def __call__(self, observation):     
        AHU_energy = observation['AHU energy consumption']
        pmv = pytc.models.pmv_ppd(
            tdb=(tdb := observation['temperature:drybulb']), 
            tr=observation['temperature:radiant'], 
            # calculate relative air speed
            vr=pytc.utilities.v_relative(v=observation.get('airspeed', .1), met=self._metab_rate), 
            rh=observation['humidity'], 
            met=self._metab_rate, 
            # calculate dynamic clothing
            clo=pytc.utilities.clo_dynamic(clo=self._clothing, met=self._metab_rate),
        )['pmv']
        #print(((self._pmv_limit - _numpy_.abs(pmv)) / self._pmv_limit) - self.energy_efficiency)
        
        reward = (
            ((self._pmv_limit - _numpy_.abs(pmv)) / self._pmv_limit) -(AHU_energy/1800000)*0.5
        )

        if math.isnan(reward):
            reward = -1

        return reward


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        print('on_episode_start')

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        logging.info('Step')
        action = episode.last_action_for()
        reward = episode.last_reward_for()
        # print(reward)
        return reward


    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        print('on_episode_end')

config = (
    PPOConfig.from_dict({        
        'create_env_on_local_worker': True,
        'train_batch_size': 4000,
    })
    .resources(num_gpus=1)
    .environment(
        SimulatorEnv,
        env_config=SimulatorEnv.Config(
            action_space=_gymnasium_.spaces.Dict({
                'thermostat': VariableBox(
                    low=22., high=30.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(Actuator.Ref(
                    type='Schedule:Compact',
                    control_type='Schedule Value',
                    key='1FFIRSTFLOORWEST:OPENOFFICE COOLING SETPOINT SCHEDULE',
                )),
                'air flow rate':VariableBox(
                    low=3., high=10.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(Actuator.Ref(
                    type='Fan',
                    control_type='Fan Air Mass Flow Rate',
                    key='AIR LOOP AHU SUPPLY FAN'
                )),
                # 'thermostat_1': VariableBox(
                #     low=22., high=30.,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(Actuator.Ref(
                #     type='Schedule:Compact',
                #     control_type='Schedule Value',
                #     key='1FFIRSTFLOOREAST:OPENOFFICE COOLING SETPOINT SCHEDULE',
                # )),
            }),    
            observation_space=_gymnasium_.spaces.Dict({
                'temperature:drybulb': VariableBox(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    type='Zone Mean Air Temperature',
                    key='1FFIRSTFLOORWEST:OPENOFFICE',
                )),
                'temperature:radiant' : VariableBox(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    type='Zone Mean Radiant Temperature',
                    key='1FFIRSTFLOORWEST:OPENOFFICE',
                )),
                'humidity' : VariableBox(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    type='Zone Air Relative Humidity',
                    key='1FFIRSTFLOORWEST:OPENOFFICE',
                )),
                'AHU energy consumption' : VariableBox(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(OutputVariable.Ref(
                    type='Cooling Coil Total Cooling Rate',
                    key='AIR LOOP AHU COOLING COIL',
                )),
            }),
            reward_function=RewardFunction(),
            event_refs=[
                'begin_zone_timestep_after_init_heat_balance',
            ],
            simulator_factory=lambda simulator=simulator: simulator,
        ),
        
    )
    .rollouts(
        #num_rollout_workers=10,
        num_rollout_workers=0,
        enable_connectors=False,
        rollout_fragment_length='auto',
    )
    .framework("torch")
    .callbacks(MyCallbacks)
    # .training(
    #     model={"fcnet_hiddens": [128, 128]},
    #     lr=0.0001,
    # )
    #.train_batch_size(1000)
    .evaluation(
        #evaluation_interval=1,
        #evaluation_num_workers=0
    )
)


algo = config.build(use_copy=False)

def train():
    global algo
    for _ in range(200):
        print(algo.train())

import asyncio
async def run_train():
    asyncio.get_running_loop().run_in_executor(None, train)

    await asyncio.create_task(run_train())

save_result = algo.save()
path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{datasave}'."
)
