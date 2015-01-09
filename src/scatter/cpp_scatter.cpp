/*! 
 *  Code for computing the Thompson diffraction from an atomic structure,
 *  in parallel, both on the GPU and CPU.
 *
 *  First version: YTZ 2012
 *  Updated TJL 2012, 2014
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "cpp_scatter.hh"

using namespace std;


#define GBLTPB 32         // threads per block
#define MAX_NUM_TYPES 10  // maximum number of atom types


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




/******************************************************************************
 * GPU Only Code
 ******************************************************************************/

// ---- DEVICE CODE
#ifdef __CUDACC__

double __device__ atomicAdd(double* address, double val) {
     double old = *address, assumed;
     do{
         assumed = old;
         old =__longlong_as_double(atomicCAS((unsigned long long int*)address,
             __double_as_longlong(assumed),
             __double_as_longlong(val + assumed)));
     }
     while(assumed != old);
     return old;
}


template<unsigned int blockSize>
void __global__ gpu_kernel(int   const n_q,
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
             
                           int   const n_rotations,
                           float const * const __restrict__ q0,
                           float const * const __restrict__ q1,
                           float const * const __restrict__ q2,
                           float const * const __restrict__ q3,
             
                           float * q_out_real, // <-- not const 
                           float * q_out_imag  // <-- not const 
                          ) {
                              
    /* On-device kernel for scattering simulation
     * 
     */
    
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // blank-out reduction buffer. 
    sdata[tid] = 0;
    __syncthreads();
    
    // private variables (for each thread)
    float qx, qy, qz;             // extracted q vector
    float ax, ay, az;             // rotated r vector
    float mq, qo, fi;             // mag of q, formfactor for atom i
    float qr;                     // dot product of q and r
    
    
    while(gid < n_q) {
       
        // workspace for cm calcs -- static size, but hopefully big enough
       float formfactors[MAX_NUM_TYPES];
       
        // determine the rotated locations
        qx = q_x[iq];
        qy = q_y[iq];
        qz = q_z[iq];
        
        // Cromer-Mann computation, precompute for this value of q
        mq = qx*qx + qy*qy + qz*qz;
        qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
        
        // accumulant: real and imaginary amplitudes for this q vector
        float2 q_sum;
        q_sum.real = 0;
        q_sum.imag = 0;

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
                
                q_sum.real += fi*__sinf(qr);
                q_sum.imag += fi*__cosf(qr);
            } // finished one atom (3rd loop)
        } // finished one molecule (2nd loop)
        
        // put q 
        q_out_real[gid] = q_sum.real;
        q_out_imag[gid] = q_sum.imag;

        // syncthreads are important here!
        __syncthreads();

        // offset by total working threads across all blocks. 
        gid += gridDim.x * blockDim.x;
    }

}


// ---- HOST CODE

void deviceMalloc( void ** ptr, int bytes ) {
    cudaError_t err = cudaMalloc(ptr, (size_t) bytes);
    assert(err == 0);
}


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

                 // random numbers for rotations
                 int     n_rotations,
                 float * rand1,
                 float * rand2,
                 float * rand3,

                 // output
                 float * h_q_out_real,
                 float * h_q_out_imag,
                ) {
    
    /* All arguments consist of 
     *   (1) a float pointer to the beginning of the array to be passed
     *   (2) ints representing the size of each array
     */


    // set GPU size parameters
    static const int tpb = GBLTPB;     // threads per block
    bpg = n_q / GBLTPB;                // blocks per grid (TODO: +1?)
    unsigned int total_q = tpb * bpg;  // total q positions to compute
    
    
    // set the device
    cudaError_t err;
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        printf("Error setting device ID. CUDA error: %s\n", cudaGetErrorString(err));
        printf("Tried to set device to: %d\n", device_id);
        exit(-1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error synching device. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    
    // compute the memory necessary to hold input/output
    const unsigned int q_size           = total_q * sizeof(float);
    const unsigned int r_size           = n_atoms * sizeof(float);
    const unsigned int a_size           = n_atom_types * sizeof(int);
    const unsigned int cm_size          = 9 * n_atom_types * sizeof(float);
    const unsigned int quat_size        = n_rotations * sizeof(float);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error before device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // allocate memory on the board
    float *d_qx;         deviceMalloc( (void **) &d_qx, q_size);
    float *d_qy;         deviceMalloc( (void **) &d_qy, q_size);
    float *d_qz;         deviceMalloc( (void **) &d_qz, q_size);

    float *d_rx;         deviceMalloc( (void **) &d_rx, r_size);
    float *d_ry;         deviceMalloc( (void **) &d_ry, r_size);
    float *d_rz;         deviceMalloc( (void **) &d_rz, r_size);
    
    int   *d_id;         deviceMalloc( (void **) &d_id, a_size);
    float *d_cm;         deviceMalloc( (void **) &d_cm, cm_size);
    
    float *d_q0;         deviceMalloc( (void **) &d_q0, quat_size);
    float *d_q1;         deviceMalloc( (void **) &d_q1, quat_size);
    float *d_q2;         deviceMalloc( (void **) &d_q2, quat_size);
    float *d_q3;         deviceMalloc( (void **) &d_q3, quat_size);
    
    float *d_q_out_real; deviceMalloc( (void **) &d_q_out_real, q_size);
    float *d_q_out_imag; deviceMalloc( (void **) &d_q_out_imag, q_size);
    
    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // pre-compute quaternions from random numbers
    float * h_q0 = (float *) malloc(n_rotations * sizeof(float));
    float * h_q1 = (float *) malloc(n_rotations * sizeof(float));
    float * h_q2 = (float *) malloc(n_rotations * sizeof(float));
    float * h_q3 = (float *) malloc(n_rotations * sizeof(float));
    
    for( int im = 0; im < n_rotations; im++ ) {
        generate_random_quaternion(rand1[im], rand2[im], rand3[im],
                                   h_q0[im], h_q1[im], h_q2[im], h_q3[im]);
    }

    // copy input/output arrays to board memory
    cudaMemcpy(d_qx, &h_qx[0], q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &h_qy[0], q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &h_qz[0], q_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_rx, &h_rx[0], r_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, &h_ry[0], r_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, &h_rz[0], r_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_id, &h_atom_types[0], a_size,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_cm, &h_cromermann[0], cm_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_q0, &h_q0[0], quat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q1, &h_q1[0], quat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q2, &h_q2[0], quat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q3, &h_q3[0], quat_size, cudaMemcpyHostToDevice);

    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in cuda memcpy. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // execute the kernel
    gpu_kernel<tpb> <<<bpg, tpb>>> (n_q, d_qx, d_qy, d_qz, 
                                    n_atoms, d_rx, d_ry, d_rz,
                                    n_atom_types, d_id, d_cm,
                                    n_rotations, d_q0, d_q1, d_q2, d_q3,
                                    d_q_out_real, d_q_out_imag)
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // retrieve the output off the board and back into CPU memory
    // copys the array to the output array passed as input
    const unsigned int wanted_q_size = n_q * sizeof(float);
    cudaMemcpy(&h_q_out_real[0], d_q_out_real, wanted_q_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_q_out_imag[0], d_q_out_imag, wanted_q_size, cudaMemcpyDeviceToHost);
    
    cudaThreadSynchronize();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in memcpy from device --> host. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // free memory
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_qz);
    
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    
    cudaFree(d_id);
    cudaFree(d_cm);
    
    cudaFree(d_q0);
    cudaFree(d_q1);
    cudaFree(d_q2);
    cudaFree(d_q3);
    
    cudaFree(d_q_out_real);
    cudaFree(d_q_out_imag);
    
    free(q0);
    free(q1);
    free(q2);
    free(q3);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error freeing memory. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaThreadSynchronize();
    cudaDeviceSynchronize();
    cudaThreadExit();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error resetting device. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// end of GPU enabled code <---
#endif

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
    
    // we will use a small array to store form factors
    float * formfactors = (float *) malloc(n_atom_types * sizeof(float));
    
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
                
                q_sum_real += fi * sinf(qr);
                q_sum_imag += fi * cosf(qr);
                
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

                  int   n_rotations,
                  float * randN1, 
                  float * randN2, 
                  float * randN3,

                  float * q_out_real,
                  float * q_out_imag
                 ) {

    cpu_kernel( n_q, q_x, q_y, q_z, 
                n_atoms, r_x, r_y, r_z,
                n_atom_types, atom_types, cromermann,
                n_rotations, randN1, randN2, randN3,
                q_out_real, q_out_imag );

}