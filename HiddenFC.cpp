#include "HiddenFC.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

HiddenLayer::HiddenLayer(MTRand& random, int nIn, int nH, double B) 
{
    
    n_in = nIn;
    n_h = nH;
    bound = B;
    
    vector<double> temp;
    
    double r;
    
    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h;i++) {
            r = random.rand();
            temp.push_back(bound*(2.0 * r - 1.0));
        }
        W.push_back(temp);
        temp.clear();
    }
    for (int i=0; i<n_h;i++) {
            r = random.rand();
            b.push_back(bound*(2.0 * r - 1.0));
        }

    //b.assign(n_h,0.0);
    
}

//*****************************************************************************
// Forward Pass 
//*****************************************************************************

vector<double> HiddenLayer::forward_pass(vector<int> & input) {
    
    vector<double> h;
    double activation;

    h.assign(n_h,0.0); 
    
    for (int i=0; i<n_h;i++) {
        activation = 0.0;
        for (int k=0; k<n_in; k++) {
            activation += W[k][i] * input[k] + b[i];
        }
        h[i] = 1.0/(1.0+exp(-activation));
    }
     
    return h;
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void HiddenLayer::loadParameters(ifstream & file) 
{
        
    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h;i++) {
            file >> W[k][i];
        }
    }
    
    for (int i=0; i<n_h;i++) {
        file >> b[i];
    }

}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void HiddenLayer::saveParameters(ofstream & file) 
{

    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h;i++) {
            file << W[k][i] << " ";
        }
        file << endl;
    }
    
    file << endl << endl;

    for (int i=0; i<n_h;i++) {
        file << b[i] << " ";
    }
    file << endl << endl;

}

