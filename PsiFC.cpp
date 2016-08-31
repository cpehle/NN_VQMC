#include "PsiFC.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

PsiLayer::PsiLayer(MTRand& random, int nH, double B) 
{
    
    n_h = nH;
    bound = B;

    double r;

    for (int i=0; i<n_h; i++) {
        r = random.rand(bound);
        Z.push_back(bound*(2.0 * r - 1.0));
    }

    c = 0.0;
}


//*****************************************************************************
// Forward Pass 
//*****************************************************************************

double PsiLayer::forward_pass(vector<double> & input) {
    
    double activation= 0.0;
    double psi;
    
    for (int i=0; i<n_h; i++) {
        activation += Z[i] * input[i] + c; 
    }
    psi = 1.0/(1.0+exp(-activation));

    return psi;
}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::loadParameters(ofstream & file) 
{
        
    for (int i=0; i<n_h; i++) {
        file >> Z[i];
    }
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::saveParameters(ofstream & file) 
{

    for (int i=0; i<n_h; i++) {
        file << Z[i] << "  ";
    }
    file << endl << endl;
}


////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void PsiLayer::loadParameters(string& modelName) 
//{
//        
//    ifstream file(modelName);
//
//    for (int j=0; j<n_in; j++) {
//        file >> Z.real()(j);
//        file >> Z.imag()(j);
//    }
//}
//
//
////*****************************************************************************
//// Save the Network Parameters
////*****************************************************************************
//
//void PsiLayer::saveParameters(string& modelName) 
//{
//
//    ofstream file(modelName);
//
//    for (int j=0; j<n_in; j++) {
//        file << Z.real()(j) << " ";
//        file << Z.imag()(j) << " ";
// 
//    }
//    
//    file.close(); 
//
//}


//*****************************************************************************
// Sigmoid function
//*****************************************************************************

//complex<double> PsiLayer::sigmoid(complex<double> pre_act) {
////
//    //VectorXcd X(vector.rows());
//    
//    //for (int i=0; i< X.rows(); i++) {
//    //        X(i) = 1.0/(1.0+exp(-vector(i)));
//    //}
//
//    //return X;
//    return 1.0/(1.0+exp(-pre_act));
//}
//
