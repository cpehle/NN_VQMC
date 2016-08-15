#ifndef CONVLAYER1D_H
#define CONVLAYER1D_H

#include "PsiLayer.h"

using namespace std;
using namespace Eigen;

class ConvLayer {
    
    public:
        
        //Network Parameters
        int n_f;          //M
        int f_size;        //l
        int n_in;
        int n_h;

        MatrixXcd W;
        
        MatrixXi LocalityTable; 

        // Constructor
        ConvLayer(MTRand & random, int nIn, int nF, int F_size);
         
        // Functions
        MatrixXcd forward_pass(VectorXcd input);
        
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);

        MatrixXcd sigmoid(MatrixXcd & matrix); 
        //VectorXd hyperTan(VectorXd vector);
        
};

#endif
