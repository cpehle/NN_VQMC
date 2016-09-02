//#include <Eigen/Core>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <map>
#include <boost/format.hpp>

using namespace std;

//*****************************************************************************
// Get command line options
//*****************************************************************************

void initializeParameters(map<string,float>& par) 
{
    par["h"]   = 0.0;
    par["nH"]  = 0;
    par["nF"]  = 0;
    par["lr"]  = 0.0;
    par["ep"]  = 0;
    par["L"]   = 0;
    par["w"]   = 0.0;
    par["MCS"] = 0;

}

//*****************************************************************************
// Get command line options
//*****************************************************************************

void get_option(const string& arg, const string& description,
                int argc, char** argv, 
                map<string,float>& par, map<string,string>& helper)
{
    string flag = "--" + arg ;
    for (int i=2; i<argc; ++i) {
            
        if (flag.compare(argv[i]) ==0) {

            par[arg] = atof(argv[i+1]);
            break;
        }
    }

    helper[arg] = description;
}

//*****************************************************************************
// Generate the Name of the output file
//*****************************************************************************

string buildOutputNameRaw(const string& network, const string& model,
                     map<string,float>& par)
{
    
    string Name = "data/";
    Name += model;
    Name += "/";
    Name += network;
    Name += "_";
    Name += model;
    Name += "_L";
    Name += boost::str(boost::format("%d") % par["L"]);
    Name += "_nH";
    Name += boost::str(boost::format("%d") % par["nH"]);
    Name += "_lr";
    Name += boost::str(boost::format("%.3f") % par["lr"]);
    Name += "_w";
    Name += boost::str(boost::format("%.1f") % par["w"]);
    Name += "_MCS";
    Name += boost::str(boost::format("%d") % par["MCS"]);
    Name += "_h";
    Name += boost::str(boost::format("%.1f") % par["h"]);
    Name += "_raw.txt";
 
    return Name;
}


