__kernel void checkCosts(__global float *pricePaths, __global float *emissionPaths, __global float *costs, const unsigned int T) {
    int i = get_global_id(0);
    
    
    float xE = emissionPaths[i * (T+1) + 0], xC = 0.0f, u;
    float cost = 0.0f;
    float mac_inv, factor, discountf, true_relative_abatement;
    
    for (int t = 0; t < T; t++) {
        u = pricePaths[i * T + t];
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
    }
    
    costs[i] = cost;
    
}


