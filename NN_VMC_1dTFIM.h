#ifndef FFNet_H
#define FFNet_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "PsiLayer.cpp"
#include <complex>
#include "ConvLayer_1d.cpp"

using namespace std;
using namespace Eigen;

class VariationalMC {
    
    public:
        
        //Network Parameters
        int epochs;
        int L;
        double learning_rate; 
        int MCS;
        double J;
        double h;

        VectorXcd spins;

        MatrixXcd dPSI_dZ_cj_e;          
        MatrixXcd dPSI_dZ_e_cj;
        MatrixXcd dPSI_dZ;          
        MatrixXcd dPSI_dZ_cj;
        MatrixXcd dPSI_dW_cj_e;          
        MatrixXcd dPSI_dW_e_cj;
        MatrixXcd dPSI_dW;          
        MatrixXcd dPSI_dW_cj; 
        complex<double> E;

        double Energy;
    
        // Constructor
        VariationalMC(MTRand & random, int L_,int MC_steps);

        // Core Functions
        void reset(PsiLayer& PL, ConvLayer& CL);
        complex<double> getLocalEnergy(PsiLayer& PL, ConvLayer& CL);
        void equilibrate(MTRand& random,PsiLayer& PL, ConvLayer& CL);
        void MetropolisUpdate(MTRand & random,PsiLayer& PL, ConvLayer& CL);
        void updateObservables(PsiLayer& PL, ConvLayer& CL);
        complex<double> getPSI(PsiLayer& PL, ConvLayer& CL);
        void train(MTRand & random,PsiLayer& PL, ConvLayer& CL);
 
        // Utilities
        void saveParameters(string& modelName);
        void loadParameters(string& modelName);
        void printNetwork(PsiLayer& PL, ConvLayer& CL);

};

#endif
