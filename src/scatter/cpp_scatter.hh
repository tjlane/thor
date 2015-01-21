
/* Header for cpp_scatter.cpp */

#ifndef __CPP_SCATTER_INCLUDED__
#define __CPP_SCATTER_INCLUDED__

// void gpuscatter(int device_id_,
//                 int bpg_,
// 
//                 // scattering q-vectors
//                 int    nQ_,
//                 float* h_qx_,
//                 float* h_qy_,
//                 float* h_qz_,
// 
//                 // atomic positions, ids
//                 int    nAtoms_,
//                 float* h_rx_,
//                 float* h_ry_,
//                 float* h_rz_,
//                 int*   h_id_,
// 
//                 // cromer-mann parameters
//                 int    nCM_,
//                 float* h_cm_,
// 
//                 // random numbers for rotations
//                 int    nRot_,
//                 float* h_rand1_,
//                 float* h_rand2_,
//                 float* h_rand3_,
// 
//                 // output
//                 float* h_outQ_ 
//                );

void cpuscatter(
                int   n_q,
                float * q_x, 
                float * q_y, 
                float * q_z, 

                int   n_atoms,
                float * r_x, 
                float * r_y, 
                float * r_z,

                int   n_atom_types,
                int   * atom_types,
                float * cromermann,

                int   n_rotations,
                float * randN1, 
                float * randN2, 
                float * randN3,

                float * q_out_real,
                float * q_out_imag
                );

// __CPP_SCATTER_INCLUDED__
#endif 
