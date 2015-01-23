
/* Header for cpp_scatter.cpp */

#ifndef __CPP_SCATTER_INCLUDED__
#define __CPP_SCATTER_INCLUDED__

void gpuscatter(int   device_id,

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

// guarenteed interface between cpp_scatter.cu and cpp_scatter.cpp
void _gpuscatter(int device_id,

                 // scattering q-vectors
                 int     n_q,
                 float * h_qx,
                 float * h_qy,
                 float * h_qz,

                 // atomic positions, ids
                 int     n_atoms,
                 float * h_rx,
                 float * h_ry,
                 float * h_rz,

                 // cromer-mann parameters
                 int     n_atom_types,
                 int   * h_atom_types,
                 float * h_cromermann,

                 // random numbers for rotations
                 int     n_rotations,
                 float * rand1,
                 float * rand2,
                 float * rand3,

                 // output
                 float * h_q_out_real,
                 float * h_q_out_imag);

// __CPP_SCATTER_INCLUDED__
#endif 
