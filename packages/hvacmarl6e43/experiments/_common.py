import abc as _abc_
from pathlib import Path
from collections import defaultdict
from os import PathLike
from dataclasses import dataclass

import pandas as _pandas_
from controllables.core import *
from controllables.energyplus import *

from ..buildings import OfficeBuilding, PMVVariable
from ..utils import resolve_path


class BaseExperiment(_abc_.ABC):
    @_abc_.abstractmethod
    def run(self, *args, **kwargs):
        ...
    
    @_abc_.abstractmethod
    def get_results(self, *args, **kwargs):
        ...


experiment_storage_base = resolve_path('__datastore__')


class DataFrameStore:
    def __init__(self, path: PathLike | str, with_ext: bool = True):
        self._path = Path(path)
        if with_ext:
            self._path = self._path.with_suffix('.h5')
    
    def set(self, dataframes: dict[str, _pandas_.DataFrame]):
        self._path.parent.absolute() \
            .mkdir(parents=True, exist_ok=True)
        with _pandas_.HDFStore(self._path, mode='w') as store:
            for key, df in dataframes.items():
                store.put(f'/{key}', df)

    def get(self) -> dict[str, _pandas_.DataFrame]:
        with _pandas_.HDFStore(self._path, mode='r') as store:
            res = dict()
            for key in store.keys():
                res[key.strip('/')] = store[key]
            return res


class OfficeBuildingObserver:
    def __init__(self, storage: str | None = None):
        self.observations = defaultdict(list)
        self._storage = storage

    def watch(self, system: OfficeBuilding):
        pmv_vars = {
            room_id: PMVVariable(
                tdb=system[OutputVariable.Ref('Zone Mean Air Temperature', zone_key)],
                tr=system[OutputVariable.Ref('Zone Mean Radiant Temperature', zone_key)],
                rh=system[OutputVariable.Ref('Zone Air Relative Humidity', zone_key)],
            )
            for room_id, zone_key in system.zone_keys.items()
        }
        tdb_vars = {
            room_id: system[OutputVariable.Ref('Zone Mean Air Temperature', zone_key)]
            for room_id, zone_key in system.zone_keys.items() 
        }
        sp_vars = {
            room_id: system[
                # Actuator.Ref(
                #     type='Zone Temperature Control',
                #     control_type='Cooling Setpoint',
                #     key=zone_key,
                # ) 
                Actuator.Ref(
                    'Schedule:Compact',
                    'Schedule Value',
                    f'{zone_key} COOLING SETPOINT SCHEDULE',
                )
            ]
            for room_id, zone_key in system.zone_keys.items()
        }

        @system.events['begin'].on
        def begin_f(*args, **kwargs):
            system.events['begin'].off(begin_f)
            system[OutputVariable.Ref('Schedule Value', 'Office_OpenOff_Occ')]

        @system.events['timestep'].on
        def timestep_f(*args, **kwargs):
            for room_id, zone_key in system.zone_keys.items():
                try:
                    self.observations[room_id].append({
                        'occupancy': system[OutputVariable.Ref(
                            'Schedule Value', 'Office_OpenOff_Occ'
                        )].value,
                        'time': system['time'].value,
                        'elec': system[OutputMeter.Ref('Electricity:HVAC')].value,
                        'pmv': pmv_vars[room_id].value,
                        'temp': tdb_vars[room_id].value,
                        'temp:setpoint': sp_vars[room_id].value,
                    })
                except TemporaryUnavailableError:
                    pass

        @system.events['end'].on
        def end_f(*args, **kwargs):
            system.events['end'].off(end_f)
            system.events['timestep'].off(timestep_f)

        return self
    
    @property
    def observation_dfs(self) -> dict[
        OfficeBuilding.ZoneIDStr, 
        _pandas_.DataFrame,
    ]:
        res = dict()
        for room_id, observation in self.observations.items():
            df = _pandas_.DataFrame(observation)
            df.set_index('time', inplace=True)
            df['pmv_abs'] = df['pmv'].abs()
            res[room_id] = df
        return res
    


@dataclass(frozen=True)
class CommonExperimentResult:
    observations: dict[
        OfficeBuilding.ZoneIDStr, 
        _pandas_.DataFrame,
    ]
    r"""
    Observations during evaluation. 
    The keys are the room (zone) IDs and the values are the dataframes.

    ..seealso::
        - :var:`hvacmarl6e43.buildings.OfficeBuilding.zone_keys`:
        for all room (zone) IDs.
        - :class:`hvacmarl6e43.experiments._common.OfficeBuildingObserver`:
        for the observer that collects this observation (`observation_dfs`).
    """


class CommonExperiment(BaseExperiment, _abc_.ABC):
    def __init__(self, name: str | Path):
        self._base_storage = Path(experiment_storage_base) / name

    def _collect_observations(self, building: OfficeBuilding, dryrun: bool):
        observer = OfficeBuildingObserver().watch(building)

        @building.events['end'].on
        def end_f(*args, **kwargs):
            building.events['end'].off(end_f)
            if not dryrun:
                DataFrameStore(
                    self._base_storage / 'observations:eval'
                ).set(observer.observation_dfs)

    @_abc_.abstractmethod
    def run(self, *args, dryrun: bool = True, **kwargs):
        ...

    def get_results(self, *args, **kwargs):
        return CommonExperimentResult(
            observations=DataFrameStore(self._base_storage / 'observations:eval').get(),
        )

