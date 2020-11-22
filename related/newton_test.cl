#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define NEWTON_G 6.67384e-11
// Replaced in Python while reading code from file for pyopencl
// #define TIMESTEP 1

/**
 * Kernel for calculating new velocities (for time TIMESTEP). This will compute a new acceleration by a simple newtonian
 * algorithm and the universal law of gravitation.
 */
__kernel void new_vel(__global double *a_g, __global const double *m_g, const unsigned int arrSize) {
    int gid = get_global_id(0);

    double total_acc[3];
    for (int i = 0; i < arrSize; ++i) {
        // Do not compute gravitation exerted by the body itself (leads to DivisionByZero or NaNs)
        if (gid == i) continue;

        // Computing distance vector (vector2 - vector1)
        double d[3];
        for (int p = 0; p < 3; ++p) {
            d[p] = a_g[i * 6 + p] - a_g[gid * 6 + p];
        }
        // Squared Norm of distance vector
        double r_2 = 0;
        for (int i = 0; i < 3; ++ i)
            r_2 += pown(d[i], 2);

        // a = Factor * Normalized distance Vector
        double factor = NEWTON_G * (m_g[i] / r_2);
        for (int a = 0; a < 3; ++a) {
            total_acc[a] += factor * (d[a] / sqrt(r_2));
        }
    }

    // Setting new velocity for body
    for (int i = 0; i < 3; ++i) {
	     a_g[gid * 6 + 3 + i] += total_acc[i] * TIMESTEP;
	}
}

/**
 * Kernel for calculating new positions for an updated velocity and time TIMESTEP
 */
__kernel void new_pos(__global double *a_g) {
    int gid = get_global_id(0);
    for (int i = 0; i < 3; ++i) {
        a_g[gid * 6 + i] = a_g[gid * 6 + i] + a_g[gid * 6 + 3 + i] * TIMESTEP;
    }
}