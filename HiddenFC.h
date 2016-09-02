#ifndef HIDDENFC_H
#define HIDDENFC_H

#include "PsiFC.cpp"

using namespace std;

class HiddenLayer {
    
    public:
        
        //Network Parameters
        int n_in;
        int n_h;
        double bound;

        vector<vector<double> > W;
        
        vector<double> b;

        // Constructor
        HiddenLayer(MTRand & random, int nIN, int nH, double B);
         
        // Functions
        vector<double> forward_pass(vector<int> & input);
        
        void loadParameters(ifstream & file);
        void saveParameters(ofstream & file);
        
};

#endif
