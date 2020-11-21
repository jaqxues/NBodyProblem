#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define NEWTON_G 6.67384e-11
#define TIMESTEP 1


__kernel void new_vel(__global double *a_g, __global const double *m_g, const unsigned int arrSize) {
    int gid = get_global_id(0);

    double total_acc[3];
    for (int i = 0; i < arrSize; ++i) {
        if (gid == i) continue;
        // Computing distance vector and normalized distance vector
        double d[3];
        for (int p = 0; p < 3; ++p) {
            d[p] = a_g[i * 6 + p] - a_g[gid * 6 + p];
        }
        double norm = 0;
        for (int i = 0; i < 3; ++ i)
            norm += pown(d[i], 2);

        double factor = NEWTON_G * (m_g[i] / norm);
        for (int a = 0; a < 3; ++a)
            total_acc[a] = factor * d[a];
    }
    for (int i = 0; i < 3; ++i) {
	     a_g[gid * 6 + 3 + i] = total_acc[i] * TIMESTEP;
	}
}
