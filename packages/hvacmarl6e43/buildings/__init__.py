import random as _random_
from typing import Literal

import numpy as _numpy_
from controllables.core import BaseVariable
from controllables.energyplus import System

from ..utils import resolve_path


class OfficeBuilding(System):
    ZoneIDStr = str

    zone_keys: dict[ZoneIDStr, str] = {
        '1FWEST': '1FFIRSTFLOORWEST:OPENOFFICE',
        '1FEAST': '1FFIRSTFLOOREAST:OPENOFFICE',
        '0FWEST': '0FGROUNDFLOORWEST:OPENOFFICE',
        '0FEAST': '0FGROUNDFLOOREAST:OPENOFFICE',
        '1FWEST1': '1FFIRSTFLOORWEST1:OPENOFFICE',
        '1FEAST1': '1FFIRSTFLOOREAST1:OPENOFFICE',
        '0FWEST1': '0FGROUNDFLOORWEST1:OPENOFFICE',
        '0FEAST1': '0FGROUNDFLOOREAST1:OPENOFFICE',
    }

    def __init__(self, **kwargs):
        super().__init__(
            building=resolve_path('model.idf'),
            **kwargs,
        )
        self.relocate('sin')

    def relocate(self, region: Literal['drw', 'gum', 'sin', '__random__']):
        match region:
            case 'drw':
                self.config['weather'] = resolve_path('weather_drw.epw')
            case 'gum':
                self.config['weather'] = resolve_path('weather_gum.epw')            
            case 'sin':
                self.config['weather'] = resolve_path('weather_sin.epw')
            case '__random__':
                return self.relocate(_random_.choice(['drw', 'gum', 'sin']))
        return self

class PMVVariable(BaseVariable):
    def __init__(
        self, 
        tdb: BaseVariable,
        tr: BaseVariable,
        rh: BaseVariable,
        metab_rate=1.5, clothing=.5, pmv_limit=.5,
    ):
        self.tdb = tdb
        self.tr = tr
        self.rh = rh
        self._metab_rate = _numpy_.asarray(metab_rate)
        self._clothing = _numpy_.asarray(clothing)
        self._pmv_limit = _numpy_.asarray(pmv_limit)
    
    @property
    def value(self):
        from pythermalcomfort import pmv_ppd
        from pythermalcomfort.utilities import v_relative, clo_dynamic

        res = pmv_ppd(
            tdb=self.tdb.value, 
            tr=self.tr.value, 
            # calculate relative air speed
            vr=v_relative(v=0.1, met=self._metab_rate), 
            rh=self.rh.value, 
            met=self._metab_rate, 
            # calculate dynamic clothing
            clo=clo_dynamic(clo=self._clothing, met=self._metab_rate),
            limit_inputs=False ,
        )['pmv']

        return res