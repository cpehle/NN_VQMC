#include "VMC_FC_1dTFIM.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

VariationalMC::VariationalMC(MTRand& random, int L_, double h_,
                             int MCS_, double lr_) 
{
    
    //n_in          = int(parameters["nIn"]); 
    //epochs        = int(parameters["ep"]);
    //learning_rate = parameters["lr"];
    lr = lr_;
    MCS = MCS_;
    L = L_;
    epochs = 100;

    J = 1.0;
    h = h_;
    
    for (int i=0; i<L; i++) {
        if (random.rand() > 0,5)
            spins.push_back(1);
        else
            spins.push_back(-1);
    }

}

void VariationalMC::reset(PsiLayer& PL, HiddenLayer& HL) {

   
    E = 0.0;
    
    for (int i=0; i<HL.n_h; i++) {
        for (int k=0; k< HL.n_in; k++) {
            dP_dW[k][i]   = 0.0;
            dP_dW_e[k][i] = 0.0;
        }
        dP_db[i]   = 0.0;
        dP_db_e[i] = 0.0;
    }
    
    for (int i=0; i<PL.n_in; i++) {
        dP_dZ[i]   = 0.0;
        dP_dZ_e[i] = 0.0;
    }

    dP_dc   = 0.0;
    dP_dc_e = 0.0;
     
}

double VariationalMC::getPSI(PsiLayer& PL, HiddenLayer& HL) {
    
    vector<double> h;
    double psi;
    
    h = HL.forward_pass(spins);
    psi = PL.forward_pass(h);

    return psi;
}


double VariationalMC::getLocalEnergy(PsiLayer& PL, HiddenLayer& HL) {
    
    double e = 0.0;
    double psi;
    double psi_flip;
    
    psi = getPSI(PL,HL);

    for (int i=0; i<L-2; i++) {
        e += -J * spins[i]*spins[i+1];
        spins[i] *= -1;
        psi_flip = getPSI(PL,HL);
        e += -h*psi_flip/psi;
        spins[i] *= -1;
    }
    spins[L-1] *= -1;
    psi_flip = getPSI(PL,HL);
    e += -h*psi_flip/psi;
    spins[L-1] *= -1;
 
    return e;
}


void VariationalMC::updateObservables(PsiLayer& PL, HiddenLayer& HL) {

    vector<double> h;
    double psi;
    double eL;
    
    h   = HL.forward_pass(spins);
    psi = PL.forward_pass(h);
    eL  = getLocalEnergy(PL,HL);

    for (int i=0;i<PL.n_in; i++) {
        dP_dZ[i]   += (1.0-psi)*h[i];
        dP_dZ_e[i] += (1.0-psi)*h[i]*eL;
        
    }

    dP_dc   += (1.0-psi);
    dP_dc_e += (1.0-psi)*eL;

    for (int k=0; k<HL.n_in; k++) {
        for (int i=0; i<HL.n_h; i++) {
            dP_dW[k][i]   += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i]*spins[k];
            dP_dW_e[k][i] += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i]*spins[k]*eL;
        }
    }
    
    for (int i=0; i<HL.n_h; i++) {
        dP_db[i]   += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i];
        dP_db_e[i] += (1.0-psi)*h[i]*(1.0-h[i])*PL.Z[i]*eL;
    }

    E += eL;

}
//void VariationalMC::equilibrate(MTRand& random,PsiLayer& PL, ConvLayer& CL){
//    
//    int site;
//    double ran_num;
//    double ratio;
//   
//    for (int k = 0; k< L; k++) {
//
//        site = random.randInt(L-1);
//
//        complex<double> PSI;
//        complex<double> PSI_prime;
//
//        PSI = getPSI(PL,CL);
//        
//        spins(site) *= -1;
//
//        PSI_prime = getPSI(PL,CL);
//        
//        ratio = norm(PSI_prime) / norm(PSI);
//        
//        if (ratio < random.rand()) {
//            spins(site) *= -1;
//        }
//    }
//}
//
//
//void VariationalMC::MetropolisUpdate(MTRand& random,PsiLayer& PL, ConvLayer& CL){
//    
//    int site;
//    double ran_num;
//    double ratio;
//   
//    for (int k = 0; k< L; k++) {
//
//        site = random.randInt(L-1);
//
//        complex<double> PSI;
//        complex<double> PSI_prime;
//
//        PSI = getPSI(PL,CL);
//        
//        spins(site) *= -1;
//
//        PSI_prime = getPSI(PL,CL);
//        
//        ratio = norm(PSI_prime) / norm(PSI);
//        
//        updateObservables(PL,CL);
// 
//        if (ratio < random.rand()) {
//            spins(site) *= -1;
//        }
//    }
//}
//
//void VariationalMC::train(MTRand & random, PsiLayer& PL, ConvLayer& CL) 
//{
//    
//    int eq_steps = 500;
//    double lr = learning_rate;
//
//    for (int k=0; k<epochs; k++) {
//
//        reset(PL,CL);
//        
//        for (int i=0; i<eq_steps; i++) {
//            equilibrate(random,PL,CL);
//        }
//        
//        for (int i=0; i<MCS; i++) {
//            MetropolisUpdate(random,PL,CL);
//        }
//
//        for (int k=0; k<PL.n_in; k++) {
//            for (int i=0; i<PL.n_f; i++) {
//                PL.Z(i,k) += -(lr/1.0*MCS)*(dPSI_dZ_cj_e(i,k)-dPSI_dZ_cj(i,k)*E + dPSI_dZ_e_cj(i,k)-dPSI_dZ(i,k)*E);
//            }
//        }
//        for (int i=0; i<CL.n_f; i++) {
//            for (int j=0; j<CL.f_size; j++) {
//                CL.W(i,j) += -(lr/1.0*MCS)*(dPSI_dW_cj_e(i,j)-dPSI_dW_cj(i,j)*E + dPSI_dW_e_cj(i,j)-dPSI_dW(i,j)*E);
//            }
//        }
//
//        //Energy = E.real() / 1.0*MCS;
//        //cout << E.real()/MCS << endl;
//        cout << "Epoch: " << k << "   Ground State Energy: " << E.real()/MCS << endl; 
//
//    }
//    
//
//
//}
//
////*****************************************************************************
//// Print Network Informations
////*****************************************************************************
//
//void VariationalMC::printNetwork(PsiLayer& PL, ConvLayer& CL) 
//{
//
//    cout << "\n\n******************************\n\n" << endl;
//    cout << "NEURAL NETWORK VARIATIONAL QUANTUM MONTE CARLO\n\n";
//    cout << "Machine Parameter\n\n";
//    cout << "\tNumber of Inputs Units: " << L << "\n";
//    //cout << "\tNumber of Output Units: " << n_out << "\n";
//    cout << "\nNumber of filters: " << CL.n_f << "  ("<< PL.n_f <<")";
//    cout << "\n W matrix size (" << CL.W.rows() << "x" << CL.W.cols() << ")" ;
//    cout << "\n Z matrix size (" << PL.Z.rows() << "x" << PL.Z.cols() << ")" ;
//    cout << "\nFilter Size: " << CL.f_size;
//    cout << "\nHyper-parameters\n\n";
//    cout << "\tLearning Rate: " << learning_rate << "\n";
//    cout << "\tEpochs: " << epochs << "\n";
//
//    
// 
//}
