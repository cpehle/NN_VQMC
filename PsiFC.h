#ifndef PSIFC_H
#define PSIFC_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MersenneTwister.h"
//#include <Eigen/Core>
//#include <map>
//#include <complex>
#include "utilities.cpp"

using namespace std;
//using namespace Eigen;

class PsiLayer {
    
    public:
        
        //Network Parameters
        int n_in;
        double bound;
        
        vector<double> Z;
        double c;

        // Constructor
        PsiLayer(MTRand & random, int nIN, double B);
        
        // Forward the input signal through the layer
        double forward_pass(vector<double> & input);
            
        //Parameters Saving and Loading
        void loadParameters(ifstream & file);
        void saveParameters(ofstream & file);
        
};

#endif
