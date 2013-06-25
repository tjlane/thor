
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

#ifdef NO_OMP
   #define omp_get_thread_num() 0
#else
   #include <omp.h>
#endif

#include "cpuscatter.hh"

using namespace std;

/*! TJL 2012 */

#define MAX_NUM_TYPES 10

void generate_random_quaternion(float r1, float r2, float r3,
                float &q1, float &q2, float &q3, float &q4) {
    
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

    float cc1 = c0*bb1 + c1*bb0 + c2*bb3 - c3*bb2;
    float cc2 = c0*bb2 - c1*bb3 + c2*bb0 + c3*bb1;
    float cc3 = c0*bb3 + c1*bb2 - c2*bb1 + c3*bb0;   

    ox = cc1;
    oy = cc2;
    oz = cc3;

}

// "kernel" is the function that computes the scattering intensities
void kernel( float const * const __restrict__ q_x, 
             float const * const __restrict__ q_y, 
             float const * const __restrict__ q_z, 
             float *outQ, // <-- not const 
             int   const nQ,
             float const * const __restrict__ r_x, 
             float const * const __restrict__ r_y, 
             float const * const __restrict__ r_z,
             int   const * const __restrict__ r_id, 
             int   const numAtoms, 
             int   const numAtomTypes,
             float const * const __restrict__ cromermann,
             float const * const __restrict__ randN1, 
             float const * const __restrict__ randN2, 
             float const * const __restrict__ randN3,
             const int n_rotations ) {
            

    // private variables
    float rand1, rand2, rand3;
    float q0, q1, q2, q3;
    float qx, qy, qz;
    float Qsumx, Qsumy;
    float mq, qo, fi;
    int tind;
    float rx, ry, rz;
    int id;
    float ax, ay, az;
    float qr;

    // main loop
    // #pragma omp parallel for shared(outQ) private(rand1, rand2, rand3, q0, q1, q2, \
    //     q3, qx, qy, qz, Qsumx, Qsumy, mq, qo, fi, tind, rx, ry, rz, id, ax, ay, az, qr)
    for( int im = 0; im < n_rotations; im++ ) {
       
        // determine the rotated locations
        rand1 = randN1[im]; 
        rand2 = randN2[im]; 
        rand3 = randN3[im]; 

        // rotation quaternions
        generate_random_quaternion(rand1, rand2, rand3, q0, q1, q2, q3);

        // for each q vector
        for( int iq = 0; iq < nQ; iq++ ) {
            qx = q_x[iq];
            qy = q_y[iq];
            qz = q_z[iq];

            // workspace for cm calcs -- static size, but hopefully big enough
            float formfactors[MAX_NUM_TYPES];

            // accumulant
            Qsumx = 0;
            Qsumy = 0;
     
            // Cromer-Mann computation, precompute for this value of q
            mq = qx*qx + qy*qy + qz*qz;
            qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
            
            // for each atom type, compute the atomic form factor f_i(q)
            for (int type = 0; type < numAtomTypes; type++) {
            
                // scan through cromermann in blocks of 9 parameters
                tind = type * 9;
                fi =  cromermann[tind]   * exp(-cromermann[tind+4]*qo);
                fi += cromermann[tind+1] * exp(-cromermann[tind+5]*qo);
                fi += cromermann[tind+2] * exp(-cromermann[tind+6]*qo);
                fi += cromermann[tind+3] * exp(-cromermann[tind+7]*qo);
                fi += cromermann[tind+8];
                
                formfactors[type] = fi; // store for use in a second
            }

            // for each atom in molecule
            // #pragma omp parallel for private(rx, ry, rz, ax, ay, az, id, qr, fi) shared(formfactors, q0, q1, q2, q3, qx, qy, qz)
            for( int a = 0; a < numAtoms; a++ ) {

                // get the current positions
                rx = r_x[a];
                ry = r_y[a];
                rz = r_z[a];
                id = r_id[a];

                rotate(rx, ry, rz, q0, q1, q2, q3, ax, ay, az);
                qr = ax*qx + ay*qy + az*qz;

                fi = formfactors[id];
                
                Qsumx += fi*sinf(qr);
                Qsumy += fi*cosf(qr);
                
            } // finished one molecule.
                        
            // add the output to the total intensity array
            // #pragma omp critical
            outQ[iq] += (Qsumx*Qsumx + Qsumy*Qsumy); // / n_rotations;
            
        }
    }
}


CPUScatter::CPUScatter( int    nQ_,
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
                        float* h_outQ_ ) {
                                
    // unpack arguments
    n_rotations = nRot_;
    nQ = nQ_;
    h_qx = h_qx_;
    h_qy = h_qy_;
    h_qz = h_qz_;

    nAtoms = nAtoms_;
    int numAtomTypes = nCM_ / 9;
    h_rx = h_rx_;
    h_ry = h_ry_;
    h_rz = h_rz_;
    h_id = h_id_;

    h_cm = h_cm_;

    h_rand1 = h_rand1_;
    h_rand2 = h_rand2_;
    h_rand3 = h_rand3_;

    h_outQ = h_outQ_;
    

    // execute the kernel
    kernel(h_qx, h_qy, h_qz, h_outQ, nQ, h_rx, h_ry, h_rz, h_id, nAtoms, numAtomTypes, h_cm, h_rand1, h_rand2, h_rand3, n_rotations);
}

CPUScatter::~CPUScatter() {
    // destroy the class
}
