#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "NN_VMC_1dTFIM.cpp"

int main(int argc, char* argv[]) {
   
    //setNbThreads(8);

    map<string,string> Helper;
    map<string,float> Parameters;
    //string CD_id;
    //string Reg_id;

    MTRand random(1234);
    
    int N_FILTERS = 3;
    int L_ = 6;
    int FILTER_SIZE = 2;
    int HIDDEN_UNITS = (L_-1) * N_FILTERS;

    ConvLayer CL(random,L_,N_FILTERS,FILTER_SIZE);
    PsiLayer PL(random,L_-1,N_FILTERS);
    cout << PL.Z << endl << endl;
    cout << CL.W << endl; 

    //CL.generateTable_1dNN();
    VariationalMC VMC(random,L_,1000);
    
    VMC.printNetwork(PL,CL); 
    VMC.train(random,PL,CL);


}
