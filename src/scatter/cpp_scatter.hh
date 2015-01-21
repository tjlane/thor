
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

// __CPP_SCATTER_INCLUDED__
#endif 
