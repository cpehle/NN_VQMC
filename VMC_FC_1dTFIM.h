#ifndef VMCFC1DTFIM_H
#define VMCFC1DTFIM_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "PsiFC.cpp"
#include <complex>
#include "HiddenFC.cpp"

using namespace std;
using namespace Eigen;

class VariationalMC {
    
    public:
        
        //Network Parameters
        int epochs;
        int L;
        double lr; 
        int MCS;
        int eq;
        double J;
        double h;

        vector<int> spins;

        double E;
        
        vector<vector<double> > dP_dW;
        vector<vector<double> > dP_dW_e;
        vector<double> dP_db;
        vector<double> dP_db_e;
        
        vector<double> dP_dZ;
        vector<double> dP_dZ_e;
        double dP_dc;
        double dP_dc_e;

        double Energy;
    
        // Constructor
        VariationalMC(MTRand & random, int L_,double h_,int MCS_, double lr_,
                PsiLayer& PL, HiddenLayer& HL);

        // Core Functions
        void reset(PsiLayer& PL, HiddenLayer& HL);
        double getLocalEnergy(PsiLayer& PL, HiddenLayer& HL);
        double getPSI(PsiLayer& PL, HiddenLayer& HL);
        void updateObservables(PsiLayer& PL, HiddenLayer& HL);
        void MC_run(MTRand & random,PsiLayer& PL, HiddenLayer& HL);
        void updateParameters(PsiLayer& PL, HiddenLayer& HL);
        void train(MTRand & random,PsiLayer& PL, HiddenLayer& HL);
 
        //// Utilities
        //void saveParameters(string& modelName);
        //void loadParameters(string& modelName);
        //void printNetwork(PsiLayer& PL, ConvLayer& CL);

};

#endif
