 /*! 
 *  Code for computing the Thompson diffraction from an atomic structure,
 *  both on the GPU and CPU.
 *
 *  First version: YTZ 2012
 *  Updated TJL 2012, 2014
 *  Added Diffuse: 2017
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

#include "cpp_scatter.hh"


/******************************************************************************
 * Shared CPU/GPU Code
 ******************************************************************************/


#ifdef __CUDACC__
    __host__ __device__ 
#endif
void generate_random_quaternion(float r1, float r2, float r3,
                                float &q1, float &q2, float &q3, 
                                float &q4) {
                    
    /* Generate a quaterion randomly, drawn from a distribution such that
     * quaternion multiplication by that quaternion represents a random rotation
     * draw uniformly from SO(3).
     *
     * More simply, this function will give you a quaterion that represesnts a
     * random rotation in 3D.
     *
     * Arguments:
     * r1/r2/r3    : three random floats in the range [0,1)
     * q1/q2/q3/q4 : four floats representing the xijk compontents of the 
     *               quaternion (quaternions are "4D", just like complex numbers 
     *               are "2D")
     *
     * Citations:
     * http://planning.cs.uiuc.edu/node198.html
     */
     
    float s, sig1, sig2, theta1, theta2, w, x, y, z;
    
    s = r1;
    sig1 = sqrt(s);
    sig2 = sqrt(1.0 - s);
    
    theta1 = 2.0 * M_PI * r2;
    theta2 = 2.0 * M_PI * r3;
    
    w = cos(theta2) * sig2;
    x = sin(theta1) * sig1;
    y = cos(theta1) * sig1;
    z = sin(theta2) * sig2;
    
    q1 = w;
    q2 = x;
    q3 = y;
    q4 = z;
}


#ifdef __CUDACC__
    __host__ __device__ 
#endif
void rotate(float x, float y, float z,
            float b0, float b1, float b2, float b3,
            float &ox, float &oy, float &oz) {

    // x,y,z      -- float vector
    // b          -- quaternion for rotation
    // ox, oy, oz -- rotated float vector
    
    float a0 = 0;
    float a1 = x;
    float a2 = y;
    float a3 = z;

    float c0 = b0*a0 - b1*a1 - b2*a2 - b3*a3;
    float c1 = b0*a1 + b1*a0 + b2*a3 - b3*a2;
    float c2 = b0*a2 - b1*a3 + b2*a0 + b3*a1;
    float c3 = b0*a3 + b1*a2 - b2*a1 + b3*a0;   

    float bb0 = b0;
    float bb1 = -b1;
    float bb2 = -b2;
    float bb3 = -b3;

  //float cc0 = c0*bb0 - c1*bb1 - c2*bb2 - c3*bb3;
    float cc1 = c0*bb1 + c1*bb0 + c2*bb3 - c3*bb2;
    float cc2 = c0*bb2 - c1*bb3 + c2*bb0 + c3*bb1;
    float cc3 = c0*bb3 + c1*bb2 - c2*bb1 + c3*bb0;   

    ox = cc1;
    oy = cc2;
    oz = cc3;

}


#ifdef __CUDACC__
    __host__ __device__ 
#endif
void qVq_product(float const * const V,
				 int i, 
				 int j,
				 int n_atoms,
				 
				 float qx, 
				 float qy, 
				 float qz,
				 
				 float &o_qVq
                 
				 ){

    // compute < q | V_ij | q >

    float qVq;
	int ab_idx;
	
	ab_idx = n_atoms * i + j;
	qVq  =     qx * qx * V[9*ab_idx + 0];
	qVq +=     qy * qy * V[9*ab_idx + 4];
	qVq +=     qz * qz * V[9*ab_idx + 8];
	qVq += 2 * qx * qy * V[9*ab_idx + 1];
	qVq += 2 * qx * qz * V[9*ab_idx + 2];
	qVq += 2 * qy * qz * V[9*ab_idx + 5];

    o_qVq = qVq;

}

#ifdef __CUDACC__
    __host__ __device__
#endif
void qUq_product(float const * const U,
		 int i,
		 float qx,
		 float qy,
		 float qz,

		 float &o_qUq
		  
		 ){

      // compute < q | U_ii | q >

      float qUq;
      
      qUq  =     qx * qx * U[9*i + 0];
      qUq +=     qy * qy * U[9*i + 4];
      qUq +=     qz * qz * U[9*i + 8];
      qUq += 2 * qx * qy * U[9*i + 1];
      qUq += 2 * qx * qz * U[9*i + 2];
      qUq += 2 * qy * qz * U[9*i + 5];

      o_qUq = qUq;

}


/******************************************************************************
 * Hybrid CPU/GPU Code
 * decides whether to try and call GPU code or raise an exception, depending
 * on if this file was compiled with nvcc or not
 ******************************************************************************/


void gpuscatter (int device_id_,
            
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

		 // atomic displacement parameters
		 float * h_U,

                 // random numbers for rotations
                 int     n_rotations,
                 float * rand1,
                 float * rand2,
                 float * rand3,

                 // output
                 float * h_q_out_real,
                 float * h_q_out_imag
                ) {

    #ifdef __CUDACC__
        _gpuscatter( device_id_,
                     n_q, h_qx, h_qy, h_qz,
                     n_atoms, h_rx, h_ry, h_rz,
                     n_atom_types, h_atom_types, h_cromermann, h_U,
                     n_rotations, rand1, rand2, rand3,
                     h_q_out_real, h_q_out_imag);
    #else
        #warning ("Warning : gpuscatter DISABLED")
        throw std::runtime_error("gpuscatter called but cpp_scatter.cpp not compiled w/nvcc!");
    #endif

}


void gpudiffuse (int device_id_,
            
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

                 // covariance mtx
                 float * h_V,

                 // output
                 float * h_q_out_bragg,
                 float * h_q_out_diffuse
                ) {

    #ifdef __CUDACC__
        _gpudiffuse( device_id_,
                     n_q, h_qx, h_qy, h_qz,
                     n_atoms, h_rx, h_ry, h_rz,
                     n_atom_types, h_atom_types, h_cromermann, h_V,
                     h_q_out_bragg, h_q_out_diffuse);
    #else
        #warning ("Warning : gpudiffuse DISABLED")
        throw std::runtime_error("gpudiffuse called but cpp_scatter.cpp not compiled w/nvcc!");
    #endif

}

/******************************************************************************
 * CPU Only Code
 ******************************************************************************/

void cpu_kernel( int   const n_q,
                 float const * const __restrict__ q_x, 
                 float const * const __restrict__ q_y, 
                 float const * const __restrict__ q_z, 
                 
                 int   const n_atoms,
                 float const * const __restrict__ r_x, 
                 float const * const __restrict__ r_y, 
                 float const * const __restrict__ r_z,
                 
                 int   const n_atom_types,
                 int   const * const __restrict__ atom_types,
                 float const * const __restrict__ cromermann,
                 
		 float const * const __restrict__ U,

                 int   const n_rotations,
                 float const * const __restrict__ randN1, 
                 float const * const __restrict__ randN2, 
                 float const * const __restrict__ randN3,
                 
                 float * q_out_real, // <-- not const 
                 float * q_out_imag  // <-- not const 
                ) {
                      
    /* CPU code for computing a scattering simulation
     *
     * This code is designed to mirror the implementation of the GPU code above,
     * which should aid development and testing.
     *
     * Arguments
     * ---------
     * n_q / q_{x,y,z}     : the number and xyz positions of momentum q-vectors
     * n_atoms / r_{x,y,z} : the number and xyz positions of atomic positions
     * n_atom_types        : the number of unique atom types (formfactors)
     * atom_types          : the atom "type", which is an arbitrary index
     * cromermann          : 9 params specifying formfactor for each atom type 
     * U                   : 3d array of the atomic displacement parameters
     * n_rotations/        : The number of molecules to independently rotate and
     *  randN{1,2,3}         the random numbers used to perform those rotations
     *
     * Output
     * ------
     * q_out_real : the real part of the complex scattering amplitude
     * q_out_imag : the imaginary part of the scattering amplitude
     *
     */

    // private variables
    float qx, qy, qz;             // extracted q vector
    float ax, ay, az;             // rotated r vector
    float mq, qo, fi;             // mag of q, formfactor for atom i
    float q_sum_real, q_sum_imag; // partial sum of real and imaginary amplitude
    float qr;                     // dot product of q and r
    
    // we will use a small array to store form factors and Debye-Waller factors
    float * formfactors = (float *) malloc(n_atom_types * sizeof(float));
    float * qUq = (float *) malloc(n_atoms * sizeof(float));
    
    // pre-compute rotation quaternions    
    float * q0 = (float *) malloc(n_rotations * sizeof(float));
    float * q1 = (float *) malloc(n_rotations * sizeof(float));
    float * q2 = (float *) malloc(n_rotations * sizeof(float));
    float * q3 = (float *) malloc(n_rotations * sizeof(float));
    
    for( int im = 0; im < n_rotations; im++ ) {
        generate_random_quaternion(randN1[im], randN2[im], randN3[im],
                                   q0[im], q1[im], q2[im], q3[im]);
    }


    // ---> main loop (3 nested loops)
    // for each q vector (1st nested loop)
    for( int iq = 0; iq < n_q; iq++ ) {
        
        qx = q_x[iq];
        qy = q_y[iq];
        qz = q_z[iq];
        
        // Cromer-Mann computation, precompute for this value of q
        mq = qx*qx + qy*qy + qz*qz;
        qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2

        // accumulant: real and imaginary amplitudes for this q vector
        q_sum_real = 0;
        q_sum_imag = 0;
    
        // precompute atomic form factors for each atom type
        int tind;
        for (int type = 0; type < n_atom_types; type++) {
                        
            tind = type * 9;
            fi =  cromermann[tind]   * exp(-cromermann[tind+4]*qo);
            fi += cromermann[tind+1] * exp(-cromermann[tind+5]*qo);
            fi += cromermann[tind+2] * exp(-cromermann[tind+6]*qo);
            fi += cromermann[tind+3] * exp(-cromermann[tind+7]*qo);
            fi += cromermann[tind+8];
            
            formfactors[type] = fi;

        }

	// precompute Debye-Waller factors for each atom
	for( int a = 0; a < n_atoms; a++ ) {
	  qUq_product(U, a, qx, qy, qz, qUq[a]);
	}

        // for each molecule (2nd nested loop)
        for( int im = 0; im < n_rotations; im++ ) {
            
            int id;
            
            // for each atom in molecule (3rd nested loop)
            for( int a = 0; a < n_atoms; a++ ) {

                id = atom_types[a];
                fi = formfactors[id];

                rotate(r_x[a], r_y[a], r_z[a], 
                       q0[im], q1[im], q2[im], q3[im],
                       ax, ay, az);
                
                qr = ax*qx + ay*qy + az*qz;

                // FIXME :: swap cos and sin???? e^i*t = cos(t) + i sin(t)
                q_sum_real += fi * sinf(qr) * exp(- 0.5 * qUq[a]);
                q_sum_imag += fi * cosf(qr) * exp(- 0.5 * qUq[a]);
                
            } // finished one atom (3rd loop)
        } // finished one molecule (2nd loop)
        
        // add the output to the total intensity array
        q_out_real[iq] = q_sum_real;
        q_out_imag[iq] = q_sum_imag;
        
    } // finished one q vector (1st loop)
    
    free(formfactors);
    free(q0);
    free(q1);
    free(q2);
    free(q3);
    free(qUq);
    
}

void cpuscatter(  int  n_q,
                  float * q_x, 
                  float * q_y, 
                  float * q_z, 

                  int    n_atoms,
                  float * r_x, 
                  float * r_y, 
                  float * r_z,

                  int   n_atom_types,
                  int   * atom_types,
                  float * cromermann,

		  float * U,

                  int   n_rotations,
                  float * randN1, 
                  float * randN2, 
                  float * randN3,

                  float * q_out_real,
                  float * q_out_imag
                 ) {

    cpu_kernel( n_q, q_x, q_y, q_z, 
                n_atoms, r_x, r_y, r_z,
                n_atom_types, atom_types, cromermann, U,
                n_rotations, randN1, randN2, randN3,
                q_out_real, q_out_imag );
}


void cpudiffuse( int   n_q,
                 float * q_x,
                 float * q_y,
                 float * q_z,

                 int    n_atoms,
                 float  * r_x,
                 float  * r_y,
                 float  * r_z,

                 int    n_atom_types,
                 int    * atom_types,
                 float  * cromermann,

                 float * V,

                 float * q_out_bragg,   // <-- not const 
                 float * q_out_diffuse  // <-- not const 
                ) {
                      
    /* CPU code for computing a scattering simulation, including the possibility
     * of Gaussian diffuse scatter encoded in the MVN correlation matrix V_ij.
     *
     * ONLY for a single orientation/molecule
     *              
     * Arguments
     * ---------
     * n_q / q_{x,y,z}     : the number and xyz positions of momentum q-vectors
     * n_atoms / r_{x,y,z} : the number and xyz positions of atomic positions
     * n_atom_types        : the number of unique atom types (formfactors)
     * atom_types          : the atom "type", which is an arbitrary index
     * cromermann          : 9 params specifying formfactor for each atom type 
     * V                   : 4d array of the correlation between atom i and j 
     *                       in xyz [i,j,i_x,j_x]
     *
     * Output
     * ------
     * q_out_ints : the total scattering intensity
     *
     */

    // private variables
    float qx, qy, qz;             // extracted q vector
    float mq, qo, fi;             // mag of q, formfactor for atom i
    float dx, dy, dz;             // difference r_i - r_j for {x,y,z}
	
    float q_sum_bragg;			  // partial sum of ordered term
	float q_sum_diffuse;          // partial sum of disordered term
	
    float qr;                     // dot product of q and r
	float W;					  // intermediate result
	
    float qVabq;                  // matrix product qT * V_ab * q (atoms a & b)
	float * qViiq_cache = (float *) malloc(n_atoms * sizeof(float)); // V_aa's
	
    // we will use a small array to store form factors
    float * formfactors = (float *) malloc(n_atom_types * sizeof(float));
    
    // ---> main loop (3 nested loops)
    // for each q vector (1st nested loop)
    for( int iq = 0; iq < n_q; iq++ ) {
        
        qx = q_x[iq];
        qy = q_y[iq];
        qz = q_z[iq];
        
        // Cromer-Mann computation, precompute for this value of q
        mq = qx*qx + qy*qy + qz*qz;
        qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2

        // accumulant: real and imaginary amplitudes for this q vector
        q_sum_bragg   = 0;
        q_sum_diffuse = 0;
    
        // precompute atomic form factors for each atom type
        int tind;
        for (int type = 0; type < n_atom_types; type++) {
                        
            tind = type * 9;
            fi =  cromermann[tind]   * exp(-cromermann[tind+4]*qo);
            fi += cromermann[tind+1] * exp(-cromermann[tind+5]*qo);
            fi += cromermann[tind+2] * exp(-cromermann[tind+6]*qo);
            fi += cromermann[tind+3] * exp(-cromermann[tind+7]*qo);
            fi += cromermann[tind+8];
            
            formfactors[type] = fi;

        }

        // for each atom in molecule [again] (2nd nested loop)
        for( int a = 0; a < n_atoms; a++ ) {
            
            int id_a = atom_types[a];
            float fa = formfactors[id_a];

            // do diagonal elements (a == b)
            qVq_product(V, a, a, n_atoms, qx, qy, qz, qViiq_cache[a]);
            W = fa * fa;
            q_sum_bragg   += W * exp(-1 * qViiq_cache[a]);
            q_sum_diffuse += W * (1 - exp(-1 * qViiq_cache[a]));
            
            // for each atom in molecule [again], a != b (3rd nested loop)
            for( int b = 0; b < a; b++ ) {
                
                int id_b = atom_types[b];
                float fb = formfactors[id_b];

                // iqr [structure factor]      
                dx = r_x[a] - r_x[b];
                dy = r_y[a] - r_y[b];
                dz = r_z[a] - r_z[b];
                qr = dx*qx + dy*qy + dz*qz;

                // qVq [disorder factor]
				qVq_product(V, a, b, n_atoms, qx, qy, qz, qVabq);

                // accumulate (for atom pair a/b)
				W = 2 * fa * fb * cosf(qr) * exp(- 0.5 * qViiq_cache[a] - 0.5 * qViiq_cache[b]);
                q_sum_bragg   += W;
                q_sum_diffuse += W * ( exp( qVabq ) - 1 );
 
            } // finished one atom (3rd loop)

        } // finished 2nd atom (2nd loop)
        
        // add the output to the total intensity array
        q_out_bragg[iq]   = q_sum_bragg;  
        q_out_diffuse[iq] = q_sum_diffuse;
        
    } // finished one q vector (1st loop)
    
    free(formfactors);
    
}

// This is a meaningless test, for speed only...
#ifndef __CUDACC__
int main() {

    int nQ_ = 1000;
    int nAtoms_ = 1000;
    int n_atom_types_ = 10;
    int nRot_ = 1000;

    float * h_qx_ = new float[nQ_];
    float * h_qy_ = new float[nQ_];
    float * h_qz_ = new float[nQ_];

    float * h_rx_ = new float[nAtoms_];
    float * h_ry_ = new float[nAtoms_];
    float * h_rz_ = new float[nAtoms_];

    int   * atom_types_ = new int[nAtoms_];
    float * cromermann_ = new float[n_atom_types_ * 9];
    float * U = new float[nAtoms_ * 3 * 3];

    float * h_rand1_ = new float[nRot_];
    float * h_rand2_ = new float[nRot_];
    float * h_rand3_ = new float[nRot_];

    float * h_outQ_R = new float[nQ_];
    float * h_outQ_I = new float[nQ_];

    cpuscatter    ( // q vectors
                    nQ_,
                    h_qx_,
                    h_qy_,
                    h_qz_,

                    // atomic positions, ids
                    nAtoms_,
                    h_rx_,
                    h_ry_,
                    h_rz_,

                    // formfactor info
                    n_atom_types_,
                    atom_types_,
                    cromermann_,

		    // ADP info
		    U,

                    // random numbers for rotations
                    nRot_,
                    h_rand1_,
                    h_rand2_,
                    h_rand3_,

                    // output
                    h_outQ_R,
                    h_outQ_I );

    printf("CPP OUTPUT:\n");
    printf("%f\n", h_outQ_R[0]);
    printf("%f\n", h_outQ_I[0]);

    return 0;
}
#endif

// #ifndef __CUDACC__
// int main() {
// 
//     int nQ_ = 100;
//     int nAtoms_ = 1500;
//     
//     std::cout << nQ_ << " q-vectors :: " << nAtoms_ << " atoms\n";
//     std::cout << "remember: linear in q-vectors, quadratic in atoms\n";
// 
//     int n_atom_types_ = 10;
// 
//     float * h_qx_ = new float[nQ_];
//     float * h_qy_ = new float[nQ_];
//     float * h_qz_ = new float[nQ_];
// 
//     float * h_rx_ = new float[nAtoms_];
//     float * h_ry_ = new float[nAtoms_];
//     float * h_rz_ = new float[nAtoms_];
// 
//     int   * atom_types_ = new int[nAtoms_];
//     float * cromermann_ = new float[n_atom_types_ * 9];
// 
//     float * V = new float[nAtoms_ * nAtoms_ * 3 * 3];
// 
//     float * h_outQ_R = new float[nQ_];
//     float * h_outQ_I = new float[nQ_];
// 
//     cpudiffuse   ( // q vectors
//                     nQ_,
//                     h_qx_,
//                     h_qy_,
//                     h_qz_,
// 
//                     // atomic positions, ids
//                     nAtoms_,
//                     h_rx_,
//                     h_ry_,
//                     h_rz_,
// 
//                     // formfactor info
//                     n_atom_types_,
//                     atom_types_,
//                     cromermann_,
// 
//                     // correlation matrix
//                     V,
// 
//                     // output
//                     h_outQ_R,
//                     h_outQ_I );
// 
//     printf("CPP OUTPUT:\n");
//     printf("%f\n", h_outQ_R[0]);
//     printf("%f\n", h_outQ_I[0]);
// 
//     return 0;
// }
// #endif
