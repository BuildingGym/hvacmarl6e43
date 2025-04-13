from collections import deque
from typing import Optional, TypedDict

import numpy as _numpy_
import pythermalcomfort as _pytc_


class ComfortFunction:
    def __init__(self, metab_rate=1.5, clothing=.5, pmv_limit=.5):
        self._metab_rate = _numpy_.asarray(metab_rate)
        self._clothing = _numpy_.asarray(clothing)
        self._pmv_limit = _numpy_.asarray(pmv_limit)

    class Inputs(TypedDict):
        temperature_drybulb: float
        temperature_radiant: float
        humidity: float
        airspeed: Optional[float]

    def __call__(self, inputs: Inputs) -> float:
        pmv = _pytc_.models.pmv_ppd(
            tdb=inputs['temperature_drybulb'], 
            tr=inputs['temperature_radiant'], 
            # calculate relative air speed
            vr=_pytc_.utilities.v_relative(v=inputs.get('airspeed', .1), met=self._metab_rate), 
            rh=inputs['humidity'], 
            met=self._metab_rate, 
            # calculate dynamic clothing
            clo=_pytc_.utilities.clo_dynamic(clo=self._clothing, met=self._metab_rate),
            limit_inputs=False ,
        )['pmv']
        return pmv


class ComfortElecSavingRewardFunction:
    def __init__(self):
        self._comfort_history, self._elec_history = deque(maxlen=2), deque(maxlen=2)
        self._comfort_function = ComfortFunction()

    class Inputs(TypedDict):
        hvac_elec: float
        office_occupancy: float
        temperature_drybulb: float
        temperature_radiant: float
        humidity: float
        airspeed: Optional[float]
    
    def __call__(self, inputs: Inputs) -> float:
        hvac_elec = inputs['hvac_elec']
        office_occupancy = inputs['office_occupancy']
        comfort = self._comfort_function({
            'temperature_drybulb': inputs['temperature_drybulb'],
            'temperature_radiant': inputs['temperature_radiant'],
            'humidity': inputs['humidity'],
            'airspeed': inputs.get('airspeed', .1),
        })

        if office_occupancy != 0:
            self._comfort_history.append(_numpy_.array(comfort))
            self._elec_history.append(_numpy_.array(hvac_elec))

            if len(self._comfort_history) < 2 or len(self._elec_history) < 2:
                return 0.

            with _numpy_.errstate(divide='ignore', invalid='ignore'):
                comfort_diff = (self._comfort_history[1] - self._comfort_history[0]) / self._comfort_history[0]
                #elec_diff = _numpy_.array((self._elec_history[1] - self._elec_history[0]) / self._elec_history[0])
                elec_diff_saving = -(self._elec_history[0] - self._elec_history[1]) / self._elec_history[1]
                reward = comfort_diff / elec_diff_saving
                if _numpy_.isnan(reward):
                    reward = 0.
                
            return _numpy_.clip(reward, -10, 10)
        
        return 0.


class ComfortElecSavingVectorRewardFunction:
    def __init__(self):
        self._comfort_history, self._elec_history = deque(maxlen=2), deque(maxlen=2)
        self._comfort_function = ComfortFunction()

    class Inputs(TypedDict):
        hvac_elec: float
        office_occupancy: float
        temperature_drybulb: float
        temperature_radiant: float
        humidity: float
        airspeed: Optional[float]

    # def calculate_pmv_penalty(self, pmv):
    #     if _numpy_.abs(pmv) > self._penalty_limit:
    #         pmv_penalty = 10 * (_numpy_.abs(pmv) - self._penalty_limit) / self._pmv_limit
    #     else:
    #         pmv_penalty = _numpy_.abs(pmv) / self._pmv_limit
    #     return pmv_penalty

    def __call__(self, inputs: Inputs) -> float:     
        hvac_elec = inputs['hvac_elec']
        office_occupancy = inputs['office_occupancy']
        comfort = self._comfort_function({
            'temperature_drybulb': inputs['temperature_drybulb'],
            'temperature_radiant': inputs['temperature_radiant'],
            'humidity': inputs['humidity'],
            'airspeed': inputs.get('airspeed', .1),
        })
        reward = 0
        energy = hvac_elec/2600000
        print(f'energy: {energy}')
        if office_occupancy != 0:  
            # pmv_penalty = self.calculate_pmv_penalty(comfort)
            reward = - energy - _numpy_.abs(comfort)
        print(f'pmv: {comfort}, reward: {reward}, office_occupancy: {office_occupancy}' )
        return reward


__all__ = [
    'ComfortFunction',
    'ComfortElecSavingRewardFunction',
    'ComfortElecSavingVectorRewardFunction',
]