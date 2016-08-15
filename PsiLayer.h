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
        int n_in;

        VectorXcd Z;
         
        // Constructor
        PsiLayer(MTRand & random, int nIn);
        
        // Functions
        VectorXcd flattenHiddenMatrix(MatrixXcd & h_matrix);
        complex<double> forward_pass(VectorXcd & h_vector);
        
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);

        VectorXcd sigmoid(VectorXcd vector); 
        //VectorXd hyperTan(VectorXd vector);
        
};

#endif
