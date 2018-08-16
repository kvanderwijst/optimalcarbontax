inline float endPoint (float u0, float xE0, unsigned int T) {
    float mac_inv, u, factor, discountf;
    
    float xC = 0.0f, xE = xE0;
    
    for (int t = 0; t < T; t++) {
        // New hotelling price:
        u = u0 * pown(1.05f, t);
        discountf = pown(1.05f, t);
        
        factor = evolutionFactor(t, xC);
        
        // First calculate current costs
        mac_inv = MAC_inv(u, t, xC, factor);
        
        // Calculate next step
        xE = f(t, xE, xC, u, mac_inv);
        xC = xC + xE;
    }
    
    return xC;
}


__kernel void proxyhotellingcost(__global float *pricePaths, __global float *emissionPaths, __global float *costs, __global const float *carbonBudgets, const unsigned int T) {
    int i = get_global_id(0);
    
    float u0low = 0.0f, u0high = 100.0f;
    float mac_inv, u, factor, discountf;
    float budget = carbonBudgets[i] * baselineCumulative((float)T);
    float xE0 = baseline(0.0f);
    
    float xClow, xChigh;
    
    for (int a = 0; a < 50; a++) {
        if (endPoint(u0low + (u0high - u0low) * 0.5f, xE0, T) > budget) {
            u0low = u0low + (u0high - u0low) * 0.5f;
        } else {
            u0high = u0low + (u0high - u0low) * 0.5f;
        }
    }
    
    float u0 = 0.5f * (u0high + u0low);
    
    // Calculate full path
    float xC = 0.0f, xE = xE0, true_relative_abatement;
    
    emissionPaths[i * (T+1) + 0] = xE;
    
    float cost = 0.0;
    
    for (int t = 0; t < T; t++) {
        // New hotelling price:
        u = u0 * pown(1.05f, t);
        discountf = pown(1.05f, t);
        
        factor = evolutionFactor(t, xC);
        
        // First calculate maximum abatement using this carbon tax
        mac_inv = MAC_inv(u, t, xC, factor);
        
        // Calculate next step
        xE = f(t, xE, xC, u, mac_inv);
        xC = xC + xE;
        
        // Calculate costs of this step, keeping in mind that the abatement
        // might be different than mac_inv because of inertia
        true_relative_abatement = 1 - xE / baseline(t);
        cost += R(t, xC, true_relative_abatement);
        
        
        pricePaths[i * T + t] = u;
        emissionPaths[i * (T+1) + t + 1] = xE;
    }
    
    costs[i] = cost;
    
}


