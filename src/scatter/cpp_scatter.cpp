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

#include <iostream>
#include <fstream>
#include <sstream>

#include "cpp_scatter.hh"

using namespace std;



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


// blockSize = tpb, templated in case we need to use a faster reduction
// method later. 
template<unsigned int blockSize>
void __global__ gpu_kernel(float const * const __restrict__ q_x, 
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
                       unsigned int numRotations) {

    // shared array for block-wise reduction
    __shared__ float sdata[blockSize];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // blank-out reduction buffer. 
    sdata[tid] = 0;
    __syncthreads();
    
    while(gid < numRotations) {
       
        // determine the rotated locations
        float rand1 = randN1[gid]; 
        float rand2 = randN2[gid]; 
        float rand3 = randN3[gid]; 

        // rotation quaternions
        float q0, q1, q2, q3;
        generate_random_quaternion(rand1, rand2, rand3, q0, q1, q2, q3);

        // for each q vector
        for(int iq = 0; iq < nQ; iq++) {
            float qx = q_x[iq];
            float qy = q_y[iq];
            float qz = q_z[iq];

            // workspace for cm calcs -- static size, but hopefully big enough
            float formfactors[MAX_NUM_TYPES];

            // accumulant
            float2 Qsum;
            Qsum.x = 0;
            Qsum.y = 0;
     
            // Cromer-Mann computation, precompute for this value of q
            float mq = qx*qx + qy*qy + qz*qz;
            float qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
            float fi;
            
            // for each atom type, compute the atomic form factor f_i(q)
            int tind;
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
            // bottle-necked by this currently. 
            for(int a = 0; a < numAtoms; a++) {

                // get the current positions
                float rx = r_x[a];
                float ry = r_y[a];
                float rz = r_z[a];
                int   id = r_id[a];
                float ax, ay, az;

                rotate(rx, ry, rz, q0, q1, q2, q3, ax, ay, az);
                float qr = ax*qx + ay*qy + az*qz;

                fi = formfactors[id];
                Qsum.x += fi*__sinf(qr);
                Qsum.y += fi*__cosf(qr);
            } // finished one molecule.
            
            float fQ = (Qsum.x*Qsum.x + Qsum.y*Qsum.y); // / numRotations;  
            sdata[tid] = fQ;
            __syncthreads();

            // Todo: quite slow reduction but correct, speed up reduction later if becomes bottleneck!
            for(unsigned int s=1; s < blockDim.x; s *= 2) {
                if(tid % (2*s) == 0) {
                    sdata[tid] += sdata[tid+s];
                }
                __syncthreads();
            }
            if(tid == 0) {
                atomicAdd(outQ+iq, sdata[0]); 
            } 
        }

        // blank out reduction buffer.
        sdata[tid] = 0;

        // syncthreads are important here!
        __syncthreads();

        // offset by total working threads across all blocks. 
        gid += gridDim.x * blockDim.x;
    }

}


// ---- HOST CODE

void deviceMalloc( void ** ptr, int bytes) {
    cudaError_t err = cudaMalloc(ptr, (size_t) bytes);
    assert(err == 0);
}


void GPUScatter (int device_id_,
            
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
                 ) {
    
    /* All arguments consist of 
     *   (1) a float pointer to the beginning of the array to be passed
     *   (2) ints representing the size of each array
     */


    bpg = nRot_ / 512;
    
    // unpack arguments
    device_id = device_id_;
    //cout << "device id: " << device_id << endl;
    bpg = bpg_;

    nQ = nQ_;
    h_qx = h_qx_;
    h_qy = h_qy_;
    h_qz = h_qz_;

    nAtoms = nAtoms_;
    numAtomTypes = nCM_ / 9;
    h_rx = h_rx_;
    h_ry = h_ry_;
    h_rz = h_rz_;
    h_id = h_id_;

    h_cm = h_cm_;

    h_rand1 = h_rand1_;
    h_rand2 = h_rand2_;
    h_rand3 = h_rand3_;

    h_outQ = h_outQ_;
    
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
    
    // set some size parameters
    static const int tpb = 512;
    unsigned int nRotations = tpb*bpg;
    
    // compute the memory necessary to hold input/output
    const unsigned int nQ_size = nQ*sizeof(float);
    const unsigned int nAtoms_size = nAtoms*sizeof(float);
    const unsigned int nAtoms_idsize = nAtoms*sizeof(int);
    const unsigned int nRotations_size = nRotations*sizeof(float);
    const unsigned int cm_size = 9*numAtomTypes*sizeof(float);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error before device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // allocate memory on the board
    float *d_qx;        deviceMalloc( (void **) &d_qx, nQ_size);
    float *d_qy;        deviceMalloc( (void **) &d_qy, nQ_size);
    float *d_qz;        deviceMalloc( (void **) &d_qz, nQ_size);
    float *d_outQ;      deviceMalloc( (void **) &d_outQ, nQ_size);
    float *d_rx;        deviceMalloc( (void **) &d_rx, nAtoms_size);
    float *d_ry;        deviceMalloc( (void **) &d_ry, nAtoms_size);
    float *d_rz;        deviceMalloc( (void **) &d_rz, nAtoms_size);
    int   *d_id;        deviceMalloc( (void **) &d_id, nAtoms_idsize);
    float *d_cm;        deviceMalloc( (void **) &d_cm, cm_size);
    float *d_rand1;     deviceMalloc( (void **) &d_rand1, nRotations_size);
    float *d_rand2;     deviceMalloc( (void **) &d_rand2, nRotations_size);
    float *d_rand3;     deviceMalloc( (void **) &d_rand3, nRotations_size);
    
    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy input/output arrays to board memory
    cudaMemcpy(d_qx, &h_qx[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &h_qy[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &h_qz[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outQ, &h_outQ[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx, &h_rx[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, &h_ry[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, &h_rz[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_id, &h_id[0], nAtoms_idsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cm, &h_cm[0], cm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand1, &h_rand1[0], nRotations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand2, &h_rand2[0], nRotations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand3, &h_rand3[0], nRotations_size, cudaMemcpyHostToDevice);

    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in cuda memcpy. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // execute the kernel
    kernel<tpb> <<<bpg, tpb>>> (d_qx, d_qy, d_qz, d_outQ, nQ, d_rx, d_ry, d_rz, d_id, nAtoms, numAtomTypes, d_cm, d_rand1, d_rand2, d_rand3, nRotations);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // retrieve the output off the board and back into CPU memory
    // copys the array to the output array passed as input
    cudaMemcpy(&h_outQ[0], d_outQ, nQ_size, cudaMemcpyDeviceToHost);
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
    cudaFree(d_rand1);
    cudaFree(d_rand2);
    cudaFree(d_rand3);
    cudaFree(d_outQ);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error freeing memory. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    //err = cudaDeviceReset();
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