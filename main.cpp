#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "VMC_FC_1dTFIM.cpp"

int main(int argc, char* argv[]) {
   
    
    map<string,string> Helper;
    map<string,float> Parameters;
    
    initializeParameters(Parameters);
    
    string model = argv[1];
    string network = "FCNet";

    get_option("h","Magnetic Field",argc,argv,Parameters,Helper);
    get_option("L","System Size",argc,argv,Parameters,Helper);
    get_option("lr","Learning Rate",argc,argv,Parameters,Helper);
    //get_option("w","Initial parameters width distribution",argc,argv,Parameters,Helper);
    get_option("nH","Number of Hidden Units",argc,argv,Parameters,Helper);
    
    Parameters["w"]   = 1.0;
    Parameters["ep"]  = 10000;
    Parameters["MCS"] = 100;

    MTRand random(1234);
    
    HiddenLayer   HL(random,Parameters["L"],Parameters["nH"],Parameters["w"]);
    
    PsiLayer      PL(random,Parameters["nH"],Parameters["w"]);
    
    VariationalMC VMC(random,Parameters["L"],Parameters["h"],
                             Parameters["MCS"],Parameters["lr"],
                             Parameters["ep"],PL,HL); 
    
    string rawFileName = buildOutputNameRaw(network,model,Parameters);
    ofstream fout(rawFileName);
    
    VMC.printNetwork(PL,HL); 
    VMC.train(random,fout,PL,HL);

}
