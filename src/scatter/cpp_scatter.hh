
/* Header for cpp_scatter.cpp */

#ifndef __CPP_SCATTER_INCLUDED__
#define __CPP_SCATTER_INCLUDED__

void GPUScatter(int device_id_,
                int bpg_,

                // scattering q-vectors
                int    nQ_,
                float* h_qx_,
                float* h_qy_,
                float* h_qz_,

                // atomic positions, ids
                int    nAtoms_,
                float* h_rx_,
                float* h_ry_,
                float* h_rz_,
                int*   h_id_,

                // cromer-mann parameters
                int    nCM_,
                float* h_cm_,

                // random numbers for rotations
                int    nRot_,
                float* h_rand1_,
                float* h_rand2_,
                float* h_rand3_,

                // output
                float* h_outQ_ 
               );

void CPUScatter(int    nQ,
                float* h_qx,
                float* h_qy,
                float* h_qz,
    
                // atomic positions, ids
                int    nAtoms,
                float* h_rx,
                float* h_ry,
                float* h_rz,
                int*   h_id,

                // cromer-mann parameters
                int    nCM,
                float* h_cm,

                // random numbers for rotations
                int    nRot,
                float* h_rand1,
                float* h_rand2,
                float* h_rand3,

                // output
                float* h_outQ );

// __CPP_SCATTER_INCLUDED__
#endif 
