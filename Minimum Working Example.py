############################
############################

## Import packages
import carbontax.optimalcontrol
import carbontax.scenarios
import carbontax.plot

import pyopencl as cl
import numpy as np




## Create OpenCL kernel

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue


## Results

scenario = carbontax.scenarios.createScenarioFunctions(
    context, queue,
    inertia='no', evolution='default',
    #extraparams={'maxReduct': 0.05, 'progressRatio':0.7},
    minEmissions=-10
)

output = carbontax.optimalcontrol.findOptimalCarbonPath(
    scenarioFunctions=scenario,
    T=86, 
    carbonBudgetRelToBaseline=0.3,
    maxPrice=250,
    numPricePoints=5000,
    numPointsE=0,
    numPointsC=5000,
    mostNegative=0
)

print(output['costs'], output['proxyCosts'])

carbontax.plot.plotStateAndControl(output)