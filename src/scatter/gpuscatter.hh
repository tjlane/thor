
/* Header file for gpuscatter_mgr.cu, GPUScatter class */

class GPUScatter {
    
    // declare variables
    // in CPU memory
    int device_id;
    int bpg;      // <-- defines the number of rotations
    static const int tpb; // always set in cu file
    unsigned int nQ_size;

    int nQ;
    float* h_qx;    // size: nQ
    float* h_qy;    // size: nQ
    float* h_qz;    // size: nQ

    int nAtoms;
    int numAtomTypes;
    float* h_rx;    // size: nAtoms
    float* h_ry;    // size: nAtoms
    float* h_rz;    // size: nAtoms
    int*   h_id;
    float* h_cm;    // size: numAtomTypes*9

    float* h_rand1; // size: nRotations
    float* h_rand2; // size: nRotations
    float* h_rand3; // size: nRotations
    
    int    finite_photons;
    int*   h_n_photons; // size: nRotations
    int*   h_photon_rands;

    float* h_outQ;  // size: nQ (OUTPUT)

    // on device
    float* d_qx;    // size: nQ
    float* d_qy;    // size: nQ
    float* d_qz;    // size: nQ

    float* d_rx;    // size: nAtoms
    float* d_ry;    // size: nAtoms
    float* d_rz;    // size: nAtoms
    int*   d_id;
    float* d_cm;    // size: numAtomTypes*9

    float* d_rand1; // size: nRotations
    float* d_rand2; // size: nRotations
    float* d_rand3; // size: nRotations
    
    int*   d_n_photons; // size: nRotations
    float* d_photon_rands;

    float* d_outQ;  // size: nQ (OUTPUT)


public:
    GPUScatter( int device_id_,
                int bpg_,      // <-- defines the number of rotations

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
                
                // finite photon parameters
                int    finite_photons_,
                int*   h_n_photons_,
                float* h_photon_rands,

                // output
                float* h_outQ_ 
               );
           
  ~GPUScatter();                           // destructor
};
