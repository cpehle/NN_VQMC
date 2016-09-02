#ifndef PSICONV_H
#define PSICONV_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MersenneTwister.h"
//#include <Eigen/Core>
//#include <map>
#include <complex>
#include <vector>
#include "utilities.cpp"

using namespace std;
//using namespace Eigen;

class PsiLayer {
    
    public:
        
        //Network Parameters
        int n_f;        //Number of Filters of the previous hidden layer
        int n_c;       //Number of possible configuration of filters
        double bound;   //Initial width of weigth distribution
        
        vector<vector<double> > Z;      //Weigths
        double c;                       //Biase

        // Constructor
        PsiLayer(MTRand & random, int nC, int nF, double B);
        
        // Forward the input signal through the layer
        double forward_pass(vector<vector<double> > & input);

        //Parameters Saving and Loading
        void loadParameters(ofstream & file);
        void saveParameters(ofstream & file);

        
};

#endif
