#include "ConvLayer_1d.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

ConvLayer::ConvLayer(MTRand& random, int nIn, int nF, int F_size) 
{
    
    n_in = nIn-1;
    n_f = nF;
    f_size = F_size;
    n_h = n_f * n_in;

    W.setZero(n_f,f_size);
    
    double bound = 4.0 * sqrt(6.0/(f_size+n_f));
    double r;

    for (int i=0; i<n_f; i++) {
        
        for (int j=0; j<f_size;j++) {

            r = random.rand(bound);
            W.real()(i,j) = 2.0 * r - bound;
            
            r = random.rand(bound);
            W.imag()(i,j) = 2.0 * r - bound;

        }
    }
    
    LocalityTable.setZero(n_in,2);
    
    for (int i=0; i<n_in; i++) {
        
        LocalityTable(i,0) = i;
        LocalityTable(i,1) = i+1;
    }

}

//*****************************************************************************
// Forward Pass 
//*****************************************************************************

MatrixXcd ConvLayer::forward_pass(VectorXcd input) {
    
    MatrixXcd pre_activation;
    MatrixXcd activation;
    
    pre_activation.setZero(n_in,n_f);
    activation.setZero(n_in,n_f);     
    
    for (int k=0; k<n_in; k++) {
        
        for (int i=0; i<n_f; i++) {
        
            for (int j=0; j<f_size; j++) {
                
                pre_activation(k,i) += W(i,j) * input(LocalityTable(k,j));
            }
        }
    }
    
    activation = sigmoid(pre_activation);
    
    return activation;
}


////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void ConvLayer::loadParameters(string& modelName) 
//{
//        
//    ifstream file(modelName);
//
//    for (int i=0; i<n_h;i++) {
//        for (int j=0; j<n_in; j++) {
//            file >> W.real()(i,j);
//            file >> W.imag()(i,j);
//        }
//    }
//}
//
//
////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void ConvLayer::saveParameters(string& modelName) 
//{
//
//    ofstream file(modelName);
//
//    for (int i=0; i<n_h;i++) {
//        for (int j=0; j<n_in; j++) {
//            file << W.real()(i,j) << " ";
//            file << W.imag()(i,j) << " ";
// 
//        }
//        file << endl;
//    }
//    
//    file.close(); 
//
//}
//
//
////*****************************************************************************
//// Sigmoid function
////*****************************************************************************

MatrixXcd ConvLayer::sigmoid(MatrixXcd & matrix) 

{

    MatrixXcd X(matrix.rows(),matrix.cols());
    
    for (int i=0; i< X.rows(); i++) {
        for(int j=0; j<X.cols();j++) {
            X(i,j) = 1.0/(1.0+exp(-matrix(i,j)));
        }
    }

    return X;

}

