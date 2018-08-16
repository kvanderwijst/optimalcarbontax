###################
##
## Preliminaries
##
###################

## Import packages

import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
from tqdm import tqdm
import carbontax.scenarios

import pyopencl as cl  # Import the OpenCL GPU computing API


## Define the penalty for not reaching the carbon budget
def phi(xC, totalEmissions): return 5000*(xC - totalEmissions)**3 if xC > totalEmissions else 0



#############################
### Calculate boundaries
#############################

def calcBoundaries (T, maxU, budget, baseline, baselineCumulative, numPointsE, numPointsC, mostNegative=0):
    """
    Returns the minimum and maximum values of xE and xC for every
    time t, as well as the number of points used in its discretisation

    Since the cumulative emissions have a much smaller range in the
    beginning, the number num_xC will be smaller for small t

    Also returns the factor 1 / (n-1) * (max - min), which is often
    needed in calculations.

    """
    xC_limits = np.zeros((T+1,4),dtype='f4')
    xE_limits = np.zeros((T+1,4),dtype='f4')

    for t in range(T+1):

        min_xC = 0.0
        max_xC = np.maximum(2,baselineCumulative(t)*2.01)
        num_xC = int(numPointsC * (0.2 + 0.8 * t/T)  * (1 + 1.24 ** (t-T+8)))
        factor_xC = 1.0 / (num_xC - 1.0) * (max_xC - min_xC)

        min_xE = mostNegative
        max_xE = baseline(t+1)*1.01
        num_xE = np.maximum(2,int(numPointsE * (0.5 + 0.5 * t/T) ) )
        factor_xE = 1.0 / (num_xE - 1.0) * (max_xE - min_xE)

        xC_limits[t] = np.array([min_xC, max_xC, num_xC, factor_xC])
        xE_limits[t] = np.array([min_xE, max_xE, num_xE, factor_xE])

    return xE_limits, xC_limits



#############################
### Backward induction step
#############################

def backward_induction(context, queue, program, J, uStars, xE_limits, xC_limits, uValues, T):
    """
    Performs the backward induction: first, the value function is calculated 
    at time T-1. With this, the value function (as well as the optimal carbon taxes)
    are calculated for T-2, and so on.

    Returns the memory locations of the value function and carbon tax arrays on the GPU
    """

    ## Copy state variable limits to the GPU
    cl_xE_limits = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=xE_limits)
    cl_xC_limits = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=xC_limits)

    ## Get maximum possible carbon price and convert it to single precision float
    uMax = np.float32(uValues[-1])
    nU = np.int32(len(uValues))

    ## Empty array which will contain the GPU memory locations of J[t] and uStars[t]
    memory_locations = [[None,None] for _ in range(T)]

    ## For all t, starting at T-1:
    for t in tqdm(range(T-1, -1, -1)):

        # CPU location of J[t] (value function) and uStars[t] (corresponding optimal taxes)
        J_curr = J[t].astype('f4')
        uStars_curr = uStars[t].astype('f4')

        # In the first step (T-1), J[T] has to be copied to GPU
        # This is not necessary anymore for the next steps, since J[t+1]
        # was already calculated on GPU in step t+1
        if (t < T-1):
            cl_J_next = memory_locations[t+1][0]
        else:
            J_next = J[t+1].astype('f4')
            cl_J_next = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=J_next)

        # Create GPU write array where J[t] and uStars[t] will be saved to on GPU
        cl_J_curr = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, J_curr.nbytes)
        cl_uStars_curr = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, uStars_curr.nbytes)

        # Calculate J[t] and uStars[t] for every grid combination of xE and xC
        program.calc_uStar(queue, J_curr.shape, None, np.int32(J_curr.shape[1]), cl_J_curr, cl_uStars_curr, cl_J_next, cl_xE_limits, cl_xC_limits, uMax, nU, np.int32(t))
        queue.finish()

        # Copy uStars[t] back to CPU/RAM. This is purely for debugging purposes,
        # since we will only access its values directly on GPU
        #cl.enqueue_copy(queue, J_curr, cl_J_curr)
        cl.enqueue_copy(queue, uStars_curr, cl_uStars_curr)

        memory_locations[t] = [cl_J_curr, cl_uStars_curr]

        uStars[t] = uStars_curr.astype('f8')

    return memory_locations




#############################
### Forward sweep
#############################

def forward_sweep (context, queue, program, x0E, x0C, T, xE_limits, xC_limits, uStars, mem_locations):
    """
    Performs the forward sweep step: while the backward induction step calculated
    the optimal policy and corresponding costs at *any* possible xE, xC and any time,
    this function starts at a given xE (emission level) and xC (cumulative emission 
    level), looks up the optimal carbon tax for this point, and propagates this further
    to the next step. It hereby creates the full emission and carbon tax path.
    """

    ## Initiate empty arrays for the state variables (xE and xC) and control variable (price)
    xE_stars = np.zeros(T+1)
    xE_stars[0] = x0E
    
    xC_stars = np.zeros(T+1)
    xC_stars[0] = x0C
    
    pricePath = np.zeros(T)

    ## The total costs start at 0
    cost = 0

    ## Copy state variable limits to the GPU
    cl_xE_limits = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=xE_limits)
    cl_xC_limits = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=xC_limits)

    
    for t in range(T):
        
        # Retrieve GPU memory location of uStars for this timestep
        cl_uStars = mem_locations[t][1]

        # Calculate f on GPU
        # This might seem unneccessarily complicated (and it probably is),
        # but it is not possible to calculate the full emission path on the GPU,
        # since this requires all arrays uStar[t] for all t directly available 
        # to the GPU. This is not the case, since these are all different arrays
        # with different memory locations. Giving them all at once to the GPU is
        # too much of a challenge, hence this method.
        outputValues = np.zeros((1,4), dtype='f4')
        cl_outputValues = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, outputValues.nbytes)
        program.next_value(queue, [outputValues.shape[0]], None, cl_outputValues, np.int32(t), np.float32(xE_stars[t]), np.float32(xC_stars[t]), mem_locations[t][1], cl_xE_limits, cl_xC_limits)
        queue.finish()

        # Copy output back from GPU to RAM
        cl.enqueue_copy(queue, outputValues, cl_outputValues)

        pricePath[t] = outputValues[0,0]
        xE_stars[t+1] = outputValues[0,1]
        xC_stars[t+1] = outputValues[0,2]
        cost = cost + outputValues[0,3]


    return (xE_stars, xC_stars, pricePath, cost)




#############################
### Proxy Hotelling costs and path
#############################

def proxyhotelling (scenario, relativeBudget):
    context = scenario['context']
    queue = scenario['queue']
    program = scenario['program']
    # Create possible starting prices
    budgets = np.array([relativeBudget], dtype='f4')
    
    pricePaths = np.zeros((len(budgets), 86), dtype='f4')
    emissionPaths = np.zeros((len(budgets), 86+1), dtype='f4')
    costs = np.zeros(len(budgets), dtype='f4')
    
    # Copy starting prices to GPU
    cl_budgets = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=budgets)
    
    # Create write buffer for costs on GPU
    cl_pricePaths = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, pricePaths.nbytes)
    cl_emissionPaths = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, emissionPaths.nbytes)
    cl_costs = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, costs.nbytes)
    
    # Calculate the costs
    program.proxyhotellingcost(queue, budgets.shape, None, cl_pricePaths, cl_emissionPaths, cl_costs, cl_budgets, np.int32(86))
    queue.finish()
    
    # Copy everything back
    cl.enqueue_copy(queue, pricePaths, cl_pricePaths)
    cl.enqueue_copy(queue, emissionPaths, cl_emissionPaths)
    cl.enqueue_copy(queue, costs, cl_costs)
    
    return (budgets[0], pricePaths[0,:], emissionPaths[0,:], costs[0])





#############################
### Putting it all together
#############################

def findOptimalCarbonPath(scenarioFunctions, T, carbonBudgetRelToBaseline, maxPrice, numPricePoints, numPointsE, numPointsC, mostNegative=0, showPlot=True):
    
    # Get functions necessary here
    
    context = scenarioFunctions['context']
    queue = scenarioFunctions['queue']
    program = scenarioFunctions['program']

    # Create shorthand for baseline functions
    baseline = lambda t: carbontax.scenarios.baseline(scenarioFunctions, t)
    baselineCumulative = lambda t: carbontax.scenarios.baselineCumulative(scenarioFunctions, t)
        
    
    # Create time points
    tValues = np.arange(0,T+1,dtype='f8')
    tValuesDate = [datetime.datetime(int(2015+t),1,1) for t in tValues]
    
    baselineCumEmissions = baselineCumulative(T)
    carbonBudget = carbonBudgetRelToBaseline * baselineCumEmissions
    
    # Create possible price and state points
    uValues = np.linspace(0, maxPrice, num=numPricePoints)
    xE_limits, xC_limits = calcBoundaries(T, maxPrice, carbonBudget, baseline, baselineCumulative, numPointsE, numPointsC, mostNegative)
    
    # 1. Set up the system: empty coefficient matrices
    J = [np.zeros((int(xE_limits[t,2]), int(xC_limits[t,2]))) for t in range(T+1)]
    uStars = [np.zeros((int(xE_limits[t,2]), int(xC_limits[t,2]))) for t in range(T+1)]
    
    # 2. Initalisation step: time=T
    for xi in range(int(xC_limits[T,2])):
        x = xC_limits[T,0] + xi * (xC_limits[T,1] - xC_limits[T,0]) / (xC_limits[T,2] - 1)
        J[T][:,xi] = phi(x, carbonBudget)
        #print(x)
    
    # 3. Backward induction
    mem_locations = backward_induction(context, queue, program, J, uStars, xE_limits, xC_limits, uValues, T)
    
    # 4. Forward sweep
    (emissionsPath, statePath, pricePath, costs) = forward_sweep(context, queue, program, baseline(0), 0, T, xE_limits, xC_limits, uStars, mem_locations)
    
    # 5. Calculate the proxy-hotelling path
    achievedBudget = statePath[-1]/baselineCumulative(86);
    _, proxyPricePath, proxyEmissionsPath, proxyCosts = proxyhotelling(scenarioFunctions,achievedBudget)
    
    # 5b Clean up GPU
    for mem in mem_locations:
        mem[0].release()
        mem[1].release()

    return {
        'emissions': emissionsPath,
        'cumulativeEmissions': statePath,
        'price': pricePath,
        'costs': costs,
        'proxyEmissions': proxyEmissionsPath,
        'proxyPrice': proxyPricePath,
        'proxyCosts': proxyCosts,
        'meta': {
            'budget': achievedBudget,
            'baseline': baseline(tValues),
            'tValuesDate': tValuesDate
        }
    }

