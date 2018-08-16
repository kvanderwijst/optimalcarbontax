import pyopencl as cl
import numpy as np
import pkg_resources

def createScenarioFunctions(context, queue, shape='default', evolution='default', inertia='no', minEmissions=-10, extraparams={}):
    """
    Create the OpenCL kernel for the given scenario
    
    Parameters
    ----------
    
    - minEmissions: float (default: -10)
        Minimum level of emissions that is allowed. If < 0, net negative emissions are allowed.
    
    - shape : 'default', 'discontinuous', 'convex'
        If 'discontinuous', add extra parameter 'numDiscont' for number of blocks
        
    - evolution : 'default', 'learningOverTime', 'learningByDoing'
        If 'learningOverTime', add extra parameter 'learningRate'
        If 'learningByDoing', add extra parameter 'progressRatio'
        
    - inertia : no, absolute, relative
        If absolute or relative, add extra parameter 'maxReduct'
    """
    


    #############################
    ### Baseline
    #############################

    # Baseline here is concave growing function.
    # Can also be constant
    baseline_str = """
        #define baseline(t) (1.0f + t/50.0f - 0.2f * t / 50.0f * t / 50.0f)
        #define baselineCumulative(t) (t + t*t/100.0f - 0.2f/(3.0f*2500.0f) * t*t*t)
        //#define baseline(t) (1.0f)
        //#define baselineCumulative(t) (baseline(t)*t)
    """

    
    minEmissions_str = "#define minEmissions %ff" % minEmissions


    #############################
    ### Shape of the MAC curve
    #############################

    # Default is a linear MAC, where $100/tCO2 achieves 100% abatement
    # The 'factor' parameter will be learning factor, defined later on
    if shape is 'default':
        shape_str = """
            #define MAC_inv(u, t, xC, factor) u / (100.0f * factor)
            #define MAC_integral(mEnd, t, x, factor) (100.0f * mEnd * mEnd * 0.5f * factor)
        """
    
    # Discountinuous MAC, parametrised by the number of discontinuities
    # evenly spread between 0 and 100% abatement
    elif shape is 'discontinuous':
        shape_str = """
            #define numDiscont %i
            #define MAC_inv(u, t, xC, factor) floor(u / 100.0f * numDiscont + 0.5f) / (numDiscont * factor)
            #define MAC_integral(mEnd, t, x, factor) (100.0f * mEnd * mEnd * 0.5f * factor) 
        """ % extraparams['numDiscont']
    
    # Convex MAC is a cubic function (cubic root for inverse of the MAC)
    elif shape is 'convex':
        shape_str = """
            #define MAC_inv(u, t, xC, factor) cbrt(u / (200.0f * factor))
            #define MAC_integral(mEnd, t, x, factor) (200.0f * mEnd * mEnd * mEnd * mEnd * 0.25f * factor)    
        """
    
    else:
        raise NotImplementedError
    


    #############################
    ### Time evolution of the MAC
    #############################

    # Default is no learning at all, MAC is constant over time
    if evolution is 'default':
        evolution_str = "#define evolutionFactor(t, xC) (1.0f)"
    
    # Learning over time
    # Functional form defined in GPU_main.cl, this function
    # simply calls learningOverTime(...)
    elif evolution is 'learningOverTime':
        evolution_str = """
            #define learningRate %ff
            #define evolutionFactor(t, xC) (learningOverTime(t, learningRate))
        """ % extraparams['learningRate']
    
    # Learning by doing
    # Function form defined in GPU_main.cl. That function
    # depends on cumulative abatement: cum.baseline(t) - cum.emissions(t)
    # and the progress ratio
    elif evolution is 'learningByDoing':
        evolution_str = """
            #define progressRatio %ff
            #define evolutionFactor(t, xC) (learningByDoing(baselineCumulative(t) - xC, progressRatio))
        """ % extraparams['progressRatio']
    
    else:
        raise notImplementedError
    


    #############################
    ### Inertia
    #############################

    # Default is no extra constraint on the dynamics function f
    # except that the emissions shouldn't be lower than minEmissions
    # Note that f(...) returns an instantaneous (yearly) emission level E(t)
    if inertia is 'no':
        f_str = "#define f(t, xE, xC, u, mac_inv) (maximum((baseline(t) * (1-mac_inv)), minEmissions))"
        
    # Absolute inertia
    # Emission difference between two years can now not be lower than
    # maxReduct parameter
    elif inertia is 'absolute':
        f_str = """ 
            #define maxReduct %ff
            #define f(t, xE, xC, u, mac_inv) (maximum((maximum(baseline(t) * (1-mac_inv), xE - maxReduct)), minEmissions))
        """ % extraparams['maxReduct']
    
    # Relative inertia
    # Same as absolute, but now emission level can not be smaller than
    # a percentage of previous emission level
    elif inertia is 'relative':
        f_str = """
            #define maxReduct %ff
            #define f(t, xE, xC, u, mac_inv) (maximum((maximum(baseline(t) * (1-mac_inv), xE*(1-maxReduct))), minEmissions))
        """ % extraparams['maxReduct']

    else:
        raise notImplementedError

    


    #############################
    ### Create OpenCL kernel string
    #############################

    # Here, the above OpenCL snippets are combined with the
    # main OpenCL files to create the full OpenCL kernel
    
    kernel_str = ""

    kernel_str += pkg_resources.resource_string(__name__, 'GPU_main.cl').decode('utf-8')
    kernel_str += pkg_resources.resource_string(__name__, 'GPU_proxycosts.cl').decode('utf-8')
    #kernel_str += pkg_resources.resource_string(__name__, 'GPU_checkcosts.cl').decode('utf-8')


    kernel_str = kernel_str.replace('[*BASELINE_STRING*]', baseline_str)
    kernel_str = kernel_str.replace('[*MINIMUM_EMISSIONS_STRING*]', minEmissions_str)
    kernel_str = kernel_str.replace('[*SHAPE_STRING*]', shape_str)
    kernel_str = kernel_str.replace('[*EVOLUTION_STRING*]', evolution_str)
    kernel_str = kernel_str.replace('[*F_STRING*]', f_str)

    
    ### Build the OpenCL kernel
    program = cl.Program(context, kernel_str).build()
    
    
    return {
        'context': context,
        'queue': queue,
        'program': program
    }





def calcBaselineAndCumBaselineOnGPU (scenario, tValuesInput):
    context = scenario['context']
    queue = scenario['queue']
    program = scenario['program']

    # First transform the time values to numpy float32 array, if not already
    tValues = np.array([tValuesInput], dtype='f4').flatten()

    # Copy tValues to GPU
    cl_tValues = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=tValues)

    # Create write buffer for baseline and cumulative baseline values
    baselineValues = np.zeros_like(tValues)     # On CPU
    cumBaselineValues = np.zeros_like(tValues)  # On CPU

    cl_baselineValues = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, baselineValues.nbytes)      # On GPU
    cl_cumBaselineValues = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, cumBaselineValues.nbytes)

    # Calculate baseline values
    program.giveBaseline(queue, tValues.shape, None, cl_baselineValues, cl_cumBaselineValues, cl_tValues)
    queue.finish()
    
    # Copy everything back
    cl.enqueue_copy(queue, baselineValues, cl_baselineValues)
    cl.enqueue_copy(queue, cumBaselineValues, cl_cumBaselineValues)
    
    return (baselineValues, cumBaselineValues)

def baseline(scenario, t):
    return calcBaselineAndCumBaselineOnGPU(scenario, t)[0]

def baselineCumulative(scenario, t):
    return calcBaselineAndCumBaselineOnGPU(scenario, t)[1]