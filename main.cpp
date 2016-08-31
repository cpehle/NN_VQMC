#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "VMC_FC_1dTFIM.cpp"
//#include "HiddenFC.cpp"
//#include "PsiFC.cpp"

int main(int argc, char* argv[]) {
   
    //setNbThreads(8);

    //map<string,string> Helper;
    //map<string,float> Parameters;
    //string CD_id;
    //string Reg_id;

    MTRand random(1234);
    
    //int N_FILTERS = 4;
    //int L_ = 6;
    //int FILTER_SIZE = 2;
    //int HIDDEN_UNITS = (L_-1) * N_FILTERS;
    //double bound = 0.01;
    
    double bound = 1.0;
    int nH = 8;
    int L_ = 8; 
    double h_ = 2.0;

    HiddenLayer   HL(random,L_,nH,bound);
    PsiLayer      PL(random,nH,bound);
    VariationalMC VMC(random,L_,h_,1000,0.1,PL,HL); 
    
    //cout << PL.Z << endl << endl;
    //cout << CL.W << endl; 
    VMC.train(random,PL,HL);

}
