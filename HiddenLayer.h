#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MersenneTwister.h"

using namespace std;
using namespace Eigen;

class HiddenLayer {
    
    public:
        
        //Network Parameters
        int n_h;
        int n_in;

        double initial_width;

        MatrixXd W;
         
        // Constructor
        FFNet(MTRand & random, int nIn, int nH);
        
        // Functions
        VectorXd forward_pass(VectorXd input);
        
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);

        VectorXd sigmoid(VectorXd vector); 
        VectorXd hyperTan(VectorXd vector);
        
};

#endif
