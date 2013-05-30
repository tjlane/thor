#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include "cpuscatter.cpp"

using namespace std;

// This is a meaningless test, for speed only...

int main() {
    
    int nQ_ = 1000;
    int nAtoms_ = 100;
    int nRot_ = 1000;
    
    
    
    float * h_qx_ = new float[nQ_];
    float * h_qy_ = new float[nQ_];
    float * h_qz_ = new float[nQ_];
    
    float * h_rx_ = new float[nAtoms_];
    float * h_ry_ = new float[nAtoms_];
    float * h_rz_ = new float[nAtoms_];
    int * h_id_ = new int[nAtoms_];
    
    float * h_rand1_ = new float[nRot_];
    float * h_rand2_ = new float[nRot_];
    float * h_rand3_ = new float[nRot_];
    
    float * h_outQ_ = new float[nQ_];
    
    int nCM_ = 1;
    float * h_cm_ = new float[9];
    
    
    CPUScatter sc ( nQ_,
                    h_qx_,
                    h_qy_,
                    h_qz_,

                    // atomic positions, ids
                    nAtoms_,
                    h_rx_,
                    h_ry_,
                    h_rz_,
                    h_id_,

                    // cromer-mann parameters
                    nCM_,
                    h_cm_,

                    // random numbers for rotations
                    nRot_,
                    h_rand1_,
                    h_rand2_,
                    h_rand3_,

                    // output
                    h_outQ_ );
          
    cout << h_outQ_[0] << endl;

    return 0;
}
