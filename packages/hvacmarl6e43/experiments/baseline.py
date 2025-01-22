from pathlib import Path

from controllables.core import *
from controllables.energyplus import *

from ..buildings import OfficeBuilding
from ._common import CommonExperiment


class OfficeBuildingController:
    def __init__(self, setpoint: float):
        self.setpoint = setpoint

    def watch(self, system: OfficeBuilding):
        @system.events['begin'].on
        def begin_f(*args, **kwargs):
            system.events['begin'].off(begin_f)
            system[OutputVariable.Ref('Schedule Value', 'Office_OpenOff_Occ')]

        @system.events['timestep'].on
        def timestep_f(*args, **kwargs):
            for x in system.zone_keys.values():
                occupancy = system[
                    OutputVariable.Ref('Schedule Value', 'Office_OpenOff_Occ')
                ].value
                system[
                    Actuator.Ref(
                        'Schedule:Compact',
                        'Schedule Value',
                        f'{x} COOLING SETPOINT SCHEDULE',
                    )
                ].value = (
                    self.setpoint 
                    if occupancy != 0 else 
                    100
                )

        @system.events['end'].on
        def end_f(*args, **kwargs):
            system.events['end'].off(end_f)
            system.events['timestep'].off(timestep_f)


class BaselineExperiment(CommonExperiment):
    def __init__(
        self, 
        name: str | Path | None = None,
        setpoint: float | None = None,
    ):
        super().__init__(
            name=Path('baseline') / (
                name
                if name is not None else
                str(setpoint)
            )
        )
        self._setpoint = setpoint

    def run(self, dryrun: bool = True):
        system = OfficeBuilding()
        if self._setpoint is not None:
            OfficeBuildingController(self._setpoint).watch(system)

        self._collect_observations(system, dryrun=dryrun)
        system.start().wait()
