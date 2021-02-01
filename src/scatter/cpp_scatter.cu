/*! 
 *  Code for computing the Thompson diffraction from an atomic structure,
 *  both on the GPU and CPU.
 *
 *  First version: YTZ 2012
 *  Updated TJL 2012, 2014
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

// usually bad form, but here it makes sense to include a cpp file -- this
// facilitaties compilation on GPU enabled and disabled platforms
#include "cpp_scatter.hh"
#include "cpp_scatter.cpp"


#define GBLTPB 256        // global threads per block
#define MAX_NUM_TYPES 10  // maximum number of atom types

using namespace std;

/******************************************************************************
 * GPU Only Code
 ******************************************************************************/

// ---- DEVICE CODE


// this fxn written by yutong, no longer used, retained "just in case"
// -- TJL March 2017
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
			   float const * const __restrict__ U,
             
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
    
    // int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // blank-out reduction buffer. 
    // sdata[tid] = 0;
    __syncthreads();
    
    // private variables (for each thread)
    float qx, qy, qz;             // extracted q vector
    float ax, ay, az;             // rotated r vector
    float mq, qo, fi;             // mag of q, formfactor for atom i
    float qr;                     // dot product of q and r
    float qUq;                    // matrix product of qT * U_ii * q
    
    while(gid < n_q) {
       
        // workspace for cm calcs -- static size, but hopefully big enough
        float formfactors[MAX_NUM_TYPES];
       
        // determine the rotated locations
        qx = q_x[gid];
        qy = q_y[gid];
        qz = q_z[gid];
        
        // Cromer-Mann computation, precompute for this value of q
        mq = qx*qx + qy*qy + qz*qz;
        qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
        
        // accumulant: real and imaginary amplitudes for this q vector
        float2 q_sum;
        q_sum.x = 0; // x=real
        q_sum.y = 0; // y=imag

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
                
		qUq_product(U, a, qx, qy, qz, qUq);

                q_sum.x += fi*__sinf(qr) * exp(- 0.5 * qUq);
                q_sum.y += fi*__cosf(qr) * exp(- 0.5 * qUq);
                
            } // finished one atom (3rd loop)
        } // finished one molecule (2nd loop)
        
        // put q 
        q_out_real[gid] = q_sum.x;
        q_out_imag[gid] = q_sum.y;

        // syncthreads are important here!
        __syncthreads();

        // offset by total working threads across all blocks. 
        gid += gridDim.x * blockDim.x;
    } // finished all pixels
}


template<unsigned int blockSize>
void __global__ gpu_diffuse_kernel(int   const n_q,
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
             
                                   float const * const __restrict__ V,
             
                                   float * q_out_bragg,   // <-- not const 
                                   float * q_out_diffuse  // <-- not const 
                                  ) {
                              
    /* On-device kernel for scattering simulation
     * 
     */
    
    // int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // blank-out reduction buffer. 
    // sdata[tid] = 0;
    __syncthreads();
    
    // private variables (for each thread)
    float qx, qy, qz;             // extracted q vector
    float mq, qo, fi;             // mag of q, formfactor for atom i
    float dx, dy, dz;             // difference r_i - r_j for {x,y,z}
    
    float qr;                     // dot product of q and r
    float W;                      // intermediate result
    
    float qVabq, qVaaq, qVbbq;    // matrix product qT * V_ab * q (atoms a & b)
    
    // ---> main loop (3 nested loops)
    // for each q vector (1st nested loop)
    while(gid < n_q) {
       
        // workspace for cm calcs -- static size, but hopefully big enough
        float formfactors[MAX_NUM_TYPES];
        
        // -- cache for V_ii's
        // NOT using cache for GPU... too much memory for each thread
        // to have it's own cache
        //float qViiq_cache[n_atoms];
       
        // determine the rotated locations
        qx = q_x[gid];
        qy = q_y[gid];
        qz = q_z[gid];
        
        // Cromer-Mann computation, precompute for this value of q
        mq = qx*qx + qy*qy + qz*qz;
        qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
        
        // accumulant: real and imaginary amplitudes for this q vector
        float2 q_sum;
        q_sum.x = 0; // x=bragg
        q_sum.y = 0; // y=diffuse
        
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
        
        int id_a, id_b;
        float fa, fb;

        // for each atom in molecule [again] (2nd nested loop)
        for( int a = 0; a < n_atoms; a++ ) {
            
            id_a = atom_types[a];
            fa   = formfactors[id_a];

            // do diagonal elements (a == b)
            qVq_product(V, a, a, n_atoms, qx, qy, qz, qVaaq);
            W = fa * fa;
            q_sum.x += W * exp(-1 * qVaaq);
            q_sum.y += W * (1 - exp(-1 * qVaaq));
    
            // for each atom in molecule [again], a != b (3rd nested loop)
            for( int b = 0; b < a; b++ ) {

                id_b = atom_types[b];
                fb   = formfactors[id_b];
                
                // iqr [structure factor]      
                dx = r_x[a] - r_x[b];
                dy = r_y[a] - r_y[b];
                dz = r_z[a] - r_z[b];
                qr = dx*qx + dy*qy + dz*qz;
                
                // qVq [disorder factor]
                qVq_product(V, a, b, n_atoms, qx, qy, qz, qVabq);
                qVq_product(V, b, b, n_atoms, qx, qy, qz, qVbbq);

                // accumulate (for atom pair a/b)
                W = 2 * fa * fb * cosf(qr) * exp(- 0.5 * qVaaq - 0.5 * qVbbq);
                q_sum.x += W;
                q_sum.y += W * ( exp( qVabq ) - 1 );
                
            } // finished one atom (3rd loop)
        } // finished 2nd atom (2nd loop)
        
        // put q 
        q_out_bragg[gid]   = q_sum.x;
        q_out_diffuse[gid] = q_sum.y;

        // syncthreads are important here!
        __syncthreads();

        // offset by total working threads across all blocks. 
        gid += gridDim.x * blockDim.x;
    } // finished all pixels
}


// ---- HOST CODE

void deviceMalloc( void ** ptr, int bytes ) {
    cudaError_t err = cudaMalloc(ptr, (size_t) bytes);
    // assert(err == 0);
}


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
    
    /* This is the code to be called if nvcc is found, a GPU is around, etc
     *
     *  All arguments consist of 
     *   (1) a float pointer to the beginning of the array to be passed
     *   (2) ints representing the size of each array
     */


    // set GPU size parameters
    static const int tpb = GBLTPB;     // threads per block
    int bpg = n_q / GBLTPB + 1;        // blocks per grid
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
    const unsigned int id_size          = n_atoms * sizeof(int);
    const unsigned int cm_size          = 9 * n_atom_types * sizeof(float);
    const unsigned int quat_size        = n_rotations * sizeof(float);
    const unsigned int U_size           = n_atoms * 9 * sizeof(float);

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
    
    int   *d_id;         deviceMalloc( (void **) &d_id, id_size);
    float *d_cm;         deviceMalloc( (void **) &d_cm, cm_size);
    float *d_U;          deviceMalloc( (void **) &d_U, U_size);

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
    
    cudaMemcpy(d_id, &h_atom_types[0], id_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cm, &h_cromermann[0], cm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, &h_U[0], U_size, cudaMemcpyHostToDevice);    

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
                                    n_atom_types, d_id, d_cm, d_U,
                                    n_rotations, d_q0, d_q1, d_q2, d_q3,
                                    d_q_out_real, d_q_out_imag);
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
    cudaFree(d_U);
    
    cudaFree(d_q0);
    cudaFree(d_q1);
    cudaFree(d_q2);
    cudaFree(d_q3);
    
    cudaFree(d_q_out_real);
    cudaFree(d_q_out_imag);
    
    free(h_q0);
    free(h_q1);
    free(h_q2);
    free(h_q3);

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




void _gpudiffuse(int device_id,
            
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

                 // correlation matrix
                 float * h_V,

                 // output
                 float * h_q_out_bragg,
                 float * h_q_out_diffuse
                ) {
    
    /* This is the code to be called if nvcc is found, a GPU is around, etc
     *
     *  All arguments consist of 
     *   (1) a float pointer to the beginning of the array to be passed
     *   (2) ints representing the size of each array
     */


    // set GPU size parameters
    static const int tpb = GBLTPB;     // threads per block
    int bpg = n_q / GBLTPB + 1;        // blocks per grid
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
    const unsigned int id_size          = n_atoms * sizeof(int);
    const unsigned int cm_size          = 9 * n_atom_types * sizeof(float);
    const unsigned int V_size           = n_atoms * n_atoms * 9 * sizeof(float);


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
    
    int   *d_id;         deviceMalloc( (void **) &d_id, id_size);
    float *d_cm;         deviceMalloc( (void **) &d_cm, cm_size);
    
    float *d_V;          deviceMalloc( (void **) &d_V, V_size);
    float *d_q_out_bragg;   deviceMalloc( (void **) &d_q_out_bragg,   q_size);
    float *d_q_out_diffuse; deviceMalloc( (void **) &d_q_out_diffuse, q_size);
    
    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy input/output arrays to board memory
    cudaMemcpy(d_qx, &h_qx[0], q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &h_qy[0], q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &h_qz[0], q_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_rx, &h_rx[0], r_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, &h_ry[0], r_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, &h_rz[0], r_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_id, &h_atom_types[0], id_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cm, &h_cromermann[0], cm_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_V, &h_V[0], V_size, cudaMemcpyHostToDevice);

    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in cuda memcpy. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // execute the kernel
    gpu_diffuse_kernel<tpb> <<<bpg, tpb>>> (n_q, d_qx, d_qy, d_qz, 
                                            n_atoms, d_rx, d_ry, d_rz,
                                            n_atom_types, d_id, d_cm, d_V,
                                            d_q_out_bragg, d_q_out_diffuse);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // retrieve the output off the board and back into CPU memory
    // copys the array to the output array passed as input
    const unsigned int wanted_q_size = n_q * sizeof(float);
    cudaMemcpy(&h_q_out_bragg[0],   d_q_out_bragg,   wanted_q_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_q_out_diffuse[0], d_q_out_diffuse, wanted_q_size, cudaMemcpyDeviceToHost);
    
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
    
    cudaFree(d_V);
    
    cudaFree(d_q_out_bragg);
    cudaFree(d_q_out_diffuse);

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

// int main() {
//
//     int nQ_ = 1000;
//     int nAtoms_ = 1000;
//     int n_atom_types_ = 10;
//     int nRot_ = 1000;
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
//     float * h_rand1_ = new float[nRot_];
//     float * h_rand2_ = new float[nRot_];
//     float * h_rand3_ = new float[nRot_];
//
//     float * h_outQ_R = new float[nQ_];
//     float * h_outQ_I = new float[nQ_];
//
//     gpuscatter    ( 0, // device ID
//
//                     // q vectors
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
//                     // random numbers for rotations
//                     nRot_,
//                     h_rand1_,
//                     h_rand2_,
//                     h_rand3_,
//
//                     // output
//                     h_outQ_R,
//                     h_outQ_I );
//
//     cout << h_outQ_R[0] << endl;
//     cout << h_outQ_I[0] << endl;
//
//     return 0;
// }

int main() {

    int nQ_ = 100000;
    int nAtoms_ = 1500;
    int n_atom_types_ = 10;

    float * h_qx_ = new float[nQ_];
    float * h_qy_ = new float[nQ_];
    float * h_qz_ = new float[nQ_];

    float * h_rx_ = new float[nAtoms_];
    float * h_ry_ = new float[nAtoms_];
    float * h_rz_ = new float[nAtoms_];

    int   * atom_types_ = new int[nAtoms_];
    float * cromermann_ = new float[n_atom_types_ * 9];

    float * h_V_ = new float[nAtoms_ * nAtoms_ * 3 * 3];

    float * h_outQ_R = new float[nQ_];
    float * h_outQ_I = new float[nQ_];

    gpudiffuse    ( 0, // device ID

                    // q vectors
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

                    // random numbers for rotations
                    h_V_,

                    // output
                    h_outQ_R,
                    h_outQ_I );

    cout << h_outQ_R[0] << endl;
    cout << h_outQ_I[0] << endl;

    return 0;
}
