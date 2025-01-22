from controllables.energyplus import (
    World,
    #WeatherModel,
    #Report,
    Actuator,
    OutputVariable,
)
from controllables.energyplus.logging import ProgressLogger

# from energyplus.dataset.basic import dataset as _epds_

world = World(
    input=World.Specs.Input(
        world=(
            '../../TODOs/20240826/Small office-1A-Long.idf'
        ),
        weather=('SGP_Singapore_486980_IWEC.epw'),
    ),
    output=World.Specs.Output(
        #report=('tmp/ooep-report-9e1287d2-8e75-4cf5-bbc5-f76580b56a69'),
    ),
    runtime=World.Specs.Runtime(
        recurring=True,
        #design_day=True,
    ),
).add(ProgressLogger())

from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import (
    SpaceVariableContainer,
    VariableBox,
)


import gymnasium as _gymnasium_
import numpy as _numpy_


env = SpaceVariableContainer(
    action_space=_gymnasium_.spaces.Dict({
        'thermostat': VariableBox(
            low=15., high=30.,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(Actuator.Ref(
            type='Schedule:Compact',
            control_type='Schedule Value',
            key='Always 26',
        ))
    }),    
    observation_space=_gymnasium_.spaces.Dict({
        'temperature': VariableBox(
            low=-_numpy_.inf, high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type='Zone Mean Air Temperature',
            key='Perimeter_ZN_1 ZN',
        )),
    }),
).__attach__(world)
# import pandas as pd
# pd.DataFrame(world.variables.available_keys().group(type)[OutputVariable.Ref])
@world.on('end_zone_timestep_after_zone_reporting')
def _(_):
    try:
        #print(world['wallclock'].value)
        # print(world[
        #     OutputVariable.Ref(
        #         type='Zone Air Temperature',
        #         key='Perimeter_ZN_1 ZN',
        #     )
        # ].value)
        #env.observe()
        a = 1

    except TemporaryUnavailableError:
        pass

    try:
        print(env.observe())
        a = 1
    except TemporaryUnavailableError:
        pass


    # env.action.value = {
    #     'thermostat': 27,
    # }
    #env.act(...)

# world.on('end_zone_timestep_after_zone_reporting', ...)
# import asyncio
# async def main():
#     world.awaitable.run()
#     # while True:
#     for i in range(0):        
#         ctx = await world.events['end_zone_timestep_after_zone_reporting'].awaitable(deferred=True)
#         # print(world['wallclock'].value)
#         print(i)
#         # print(env.observe())
#         # env.act({'thermostat': 25})
#         ctx.ack()
# #asyncio.run(main())

world.run()