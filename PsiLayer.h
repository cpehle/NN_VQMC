#ifndef PSILAYER_H
#define PSILAYER_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MersenneTwister.h"
#include <Eigen/Core>
#include <map>
#include <complex>

using namespace std;
using namespace Eigen;

class PsiLayer {
    
    public:
        
        //Network Parameters
        int n_f;
        int n_in;

        MatrixXcd Z;
         
        // Constructor
        PsiLayer(MTRand & random, int nIn, int nF);
        
        // Functions
        complex<double> forward_pass(MatrixXcd & input);
        
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);

        VectorXcd sigmoid(VectorXcd vector); 
        //VectorXd hyperTan(VectorXd vector);
        
};

#endif
