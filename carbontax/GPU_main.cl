#define minimum(a, b) (a < b ? a : b)
#define maximum(a, b) (a > b ? a : b)

///// Baseline emissions
[*BASELINE_STRING*]

///// Minimum emissions
[*MINIMUM_EMISSIONS_STRING*]

///// Shape
[*SHAPE_STRING*]

#define learningOverTime(t, rate) (1.0f / pown(1.0f + rate, t))
//#define learningOverTime(t, rate) (1.0f - rate * t / 100.0f)
#define learningByDoing(Q, progratio) (pow(maximum(Q, 0.0f) + 1.0f, log(progratio)/log(2.0f)))

///// Evolution
[*EVOLUTION_STRING*]

///// Inertia or not
[*F_STRING*]

#define R(t, xC, mac_inv) (baseline(t) * MAC_integral(mac_inv, t, xC, factor) / discountf)


float bilinearinterp(float xE, float xC, __global const float *inputArray, float xE_low, int nE, float xE_factor, float xC_low, int nC, float xC_factor)
{

    int xEi_1, xEi_2, xCi_1, xCi_2;
    float x, y, x1, x2, y1, y2, z11, z12, z21, z22;

    xEi_1 = (int)((xE - xE_low) / xE_factor);
    xEi_1 = minimum(maximum(xEi_1, 0), nE - 1);
    xEi_2 = xEi_1 + 1;
    xEi_2 = minimum(maximum(xEi_2, 0), nE - 1);
    
    xCi_1 = (int)((xC - xC_low) / xC_factor);
    xCi_1 = minimum(maximum(xCi_1, 0), nC - 1);
    xCi_2 = xCi_1 + 1;
    xCi_2 = minimum(maximum(xCi_2, 0), nC - 1);
    
    x1 = xEi_1 * xE_factor + xE_low;
    x2 = xEi_2 * xE_factor + xE_low;
    y1 = xCi_1 * xC_factor + xC_low;
    y2 = xCi_2 * xC_factor + xC_low;
    
    z11 = inputArray[xCi_1 + nC * xEi_1];
    z12 = inputArray[xCi_2 + nC * xEi_1];
    z21 = inputArray[xCi_1 + nC * xEi_2];
    z22 = inputArray[xCi_2 + nC * xEi_2];

    x = xE; y = xC;

    if (x1 != x2 && y1 != y2) {
        return (z11*(x2-x)*(y2-y) + z21*(x-x1)*(y2-y) + z12*(x2-x)*(y-y1) + z22*(x-x1)*(y-y1)) / ((x2 - x1)*(y2 - y1));
    } else if (x1 != x2) {
        // Only y1 and y2 are similar, perform linear interpolation in x-direction
        return z11 + (x - x1) * (z11 - z22) / (x2 - x1);
    } else if (y1 != y2) {
        // Only x1 and x2 are similar, perform linear interpolation in y-direction
        return z11 + (y - y1) * (z11 - z22) / (y2 - y1);
    } else {
        // All are equal, just use one corner point
        return z11;
    }
}


__kernel void calc_uStar(unsigned const int size, __global float *J_curr, __global float *uStars_curr, __global const float *J_next, __global const float *xE_limits, __global const float *xC_limits, const float uMax, const unsigned int nU, const unsigned int t)
{
    int xEi = get_global_id(0);
    int xCi = get_global_id(1); // Or the opposite
    
    float xE = xEi * xE_limits[4*t+3] + xE_limits[4*t+0],   // xEi * xE_factor + xE_low
          xC = xCi * xC_limits[4*t+3] + xC_limits[4*t+0];   // xCi * xC_factor + xC_low

    
    float uStar = 0.0f, minValue = 1e10f;
    float u, new_xE, new_xC, x1, x2, y1, y2, z11, z12, z21, z22, mac_inv, interpolatedJ, value;
    int xEi_1, xEi_2, xCi_1, xCi_2;
    
    float discountf = pown(1.0f+0.05f, t);
    float factor = evolutionFactor(t, xC);
    
    for (int u_i = 0; u_i < nU; u_i++) {
        u = u_i / (nU - 1.0f) * uMax;
        
        // First calculate the new xE and xC given this carbon tax
        mac_inv = MAC_inv(u, t, xC, factor);
        new_xE = f(t, xE, xC, u, mac_inv);
        new_xC = xC + new_xE;
        
        // With this, perform bilinear interpolation
        interpolatedJ = bilinearinterp(
            new_xE, new_xC,
            J_next,
            xE_limits[4*(t+1)+0], (int)xE_limits[4*(t+1)+2], xE_limits[4*(t+1)+3], // xE_next_low, nE_next, xE_next_factor
            xC_limits[4*(t+1)+0], (int)xC_limits[4*(t+1)+2], xC_limits[4*(t+1)+3]  // xC_next_low, nC_next, xC_next_factor
        );
        
        // Calculate value function for this carbon price
        value = R(t, xC, mac_inv) + interpolatedJ;
        
        // Finally, update the best carbon tax
        uStar = value < minValue ? u : uStar;
        minValue = value < minValue ? value : minValue;
    }
    
    J_curr[xEi * size + xCi] = minValue;
    uStars_curr[xEi * size + xCi] = uStar;
}


__kernel void next_value(__global float *outputValues, const unsigned int t, const float xE, const float xC, __global const float *uStars, __global const float *xE_limits, __global const float *xC_limits) {
    int i = get_global_id(0);
    
    // First, calculate the new value of the carbon tax, uStar
    float uStar;

    uStar = bilinearinterp(
        xE, xC,
        uStars,
        xE_limits[4*t+0], (int)xE_limits[4*t+2], xE_limits[4*t+3], // xE_low, nE, xE_factor
        xC_limits[4*t+0], (int)xC_limits[4*t+2], xC_limits[4*t+3]  // xC_low, nC, xC_factor
    );

    // With this, calculate the next value of the state variable xE and xC
    float factor = evolutionFactor(t, xC),
          mac_inv = MAC_inv(uStar, t, xC, factor),
          discountf = pown(1.05f, t),
          new_xE, new_xC, costs;
    new_xE = f(t, xE, xC, uStar, mac_inv);
    new_xC = xC + new_xE;
    costs = R(t, xC, mac_inv);

    // Save all the newly calculated variables
    outputValues[i * 4 + 0] = uStar;
    outputValues[i * 4 + 1] = new_xE;
    outputValues[i * 4 + 2] = new_xC;
    outputValues[i * 4 + 3] = costs;
    
}


__kernel void giveBaseline(__global float *baselineValues, __global float *cumBaselineValues, __global float *tValues) {
    int i = get_global_id(0);
    float t = tValues[i];

    baselineValues[i] = baseline(t);
    cumBaselineValues[i] = baselineCumulative(t);
}


