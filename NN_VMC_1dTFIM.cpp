#include "NN_VMC_1dTFIM.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

VariationalMC::VariationalMC(MTRand& random, int L_, int MC_steps) 
{
    
    //n_in          = int(parameters["nIn"]); 
    //epochs        = int(parameters["ep"]);
    //learning_rate = parameters["lr"];
    L = L_;
    learning_rate = 0.1;
    epochs = 100;
    MCS = MC_steps;

    J = 1.0;
    h = 0.5;
    
    spins.setZero(L);
    
    for (int i=0; i<L; i++) {
        if (random.rand() > 0,5)
            spins.real()(i) = 1.0;
        else 
            spins.real()(i) = -1.0;
        spins.imag()(i) = 0.0;
    }

}

void VariationalMC::reset(PsiLayer& PL, ConvLayer& CL) {

    dPSI_dZ_cj_e.setZero(PL.n_f,PL.n_in);          
    dPSI_dZ_e_cj.setZero(PL.n_f,PL.n_in);
    dPSI_dZ.setZero(PL.n_f,PL.n_in);          
    dPSI_dZ_cj.setZero(PL.n_f,PL.n_in);
    dPSI_dW_cj_e.setZero(CL.n_f,CL.f_size);          
    dPSI_dW_e_cj.setZero(CL.n_f,CL.f_size);
    dPSI_dW.setZero(CL.n_f,CL.f_size);          
    dPSI_dW_cj.setZero(CL.n_f,CL.f_size);
    E = 0.0;

}

complex<double> VariationalMC::getPSI(PsiLayer& PL, ConvLayer& CL) {
    

    MatrixXcd ConvState(CL.n_in,CL.n_f);
    complex<double> psi;
    
    ConvState = CL.forward_pass(spins);
    psi = PL.forward_pass(ConvState);

    return psi;
}


void VariationalMC::equilibrate(MTRand& random,PsiLayer& PL, ConvLayer& CL){
    
    int site;
    double ran_num;
    double ratio;
   
    for (int k = 0; k< L; k++) {

        site = random.randInt(L-1);

        complex<double> PSI;
        complex<double> PSI_prime;

        PSI = getPSI(PL,CL);
        
        spins(site) *= -1;

        PSI_prime = getPSI(PL,CL);
        
        ratio = norm(PSI_prime) / norm(PSI);
        
        if (ratio < random.rand()) {
            spins(site) *= -1;
        }
    }
}


void VariationalMC::MetropolisUpdate(MTRand& random,PsiLayer& PL, ConvLayer& CL){
    
    int site;
    double ran_num;
    double ratio;
   
    for (int k = 0; k< L; k++) {

        site = random.randInt(L-1);

        complex<double> PSI;
        complex<double> PSI_prime;

        PSI = getPSI(PL,CL);
        
        spins(site) *= -1;

        PSI_prime = getPSI(PL,CL);
        
        ratio = norm(PSI_prime) / norm(PSI);
        
        if (ratio > random.rand()) {
            updateObservables(PL,CL);
        }
        else {
            spins(site) *= -1;
        }
    }
}

complex<double> VariationalMC::getLocalEnergy(PsiLayer& PL, ConvLayer& CL) {
    
    complex<double> e;
    complex<double> psi;
    complex<double> psi0;
    
    psi0 = getPSI(PL,CL);

    for (int i=0; i<L-1; i++) {
        
        e += -J * spins(i)*spins(i+1);
    }
    
    for (int i=0; i<L-1; i++) {
        
        spins(i) *= -1;
        psi = getPSI(PL,CL);
        e += -h*psi/psi0;
        spins(i) *= -1;
    }

    return e;

}

void VariationalMC::updateObservables(PsiLayer& PL, ConvLayer& CL) {

   
    MatrixXcd delta_dPSI_dZ_e_cj(PL.n_f,PL.n_in);                        
    MatrixXcd delta_dPSI_dZ_cj_e(PL.n_f,PL.n_in);
    MatrixXcd delta_dPSI_dZ(PL.n_f,PL.n_in);          
    MatrixXcd delta_dPSI_dZ_cj(PL.n_f,PL.n_in);
    MatrixXcd delta_dPSI_dW_e_cj(CL.n_f,CL.f_size);          
    MatrixXcd delta_dPSI_dW_cj_e(CL.n_f,CL.f_size);
    MatrixXcd delta_dPSI_dW(CL.n_f,CL.f_size);          
    MatrixXcd delta_dPSI_dW_cj(CL.n_f,CL.f_size);
    
    complex<double> psi;
    complex<double> localEnergy;
    
    MatrixXcd ConvState(CL.n_in,CL.n_f);
    
    ConvState = CL.forward_pass(spins);
    
    psi = PL.forward_pass(ConvState);
    
    localEnergy = getLocalEnergy(PL,CL);
       
    for (int k=0; k<PL.n_in; k++) {
        for (int i=0; i<PL.n_f; i++) {
            delta_dPSI_dZ(i,k) = (1.0-psi)*ConvState(k,i);
            delta_dPSI_dZ_cj(i,k) = (1.0-conj(psi))*ConvState.conjugate()(k,i);
            delta_dPSI_dZ_cj_e(i,k) = (1.0-conj(psi))*ConvState.conjugate()(k,i)*localEnergy;
            delta_dPSI_dZ_e_cj(i,k) = (1.0-psi)*ConvState(k,i)*conj(localEnergy);

        }
    }
    
    
    for (int i=0; i<CL.n_f; i++) {

        for (int j=0; j<CL.f_size; j++) {

            for (int k=0; k<L-1; k++) {
                delta_dPSI_dW(i,j) += (1.0-psi)*ConvState(k,i)*(1.0-ConvState(k,i))*PL.Z(i,k)*spins(CL.LocalityTable(k,j));
                delta_dPSI_dW_cj(i,j) += (1.0-conj(psi))*ConvState.conjugate()(k,i)*(1.0-ConvState.conjugate()(k,i))*PL.Z.conjugate()(i,k)*spins(CL.LocalityTable(k,j));
                delta_dPSI_dW_cj_e(i,j) += (1.0-conj(psi))*ConvState.conjugate()(k,i)*(1.0-ConvState.conjugate()(k,i))*PL.Z.conjugate()(i,k)*spins(CL.LocalityTable(k,j))*localEnergy;
                delta_dPSI_dW_e_cj(i,j) += (1.0-psi)*ConvState(k,i)*(1.0-ConvState(k,i))*PL.Z(i,k)*spins(CL.LocalityTable(k,j))*conj(localEnergy);

            }
        }
    }

    delta_dPSI_dZ      += dPSI_dZ;
    delta_dPSI_dZ_cj   += dPSI_dZ_cj;
    delta_dPSI_dZ_e_cj += dPSI_dZ_e_cj;
    delta_dPSI_dZ_cj_e += dPSI_dZ_cj_e;
    delta_dPSI_dW      += dPSI_dW;
    delta_dPSI_dW_cj   += dPSI_dW_cj;
    delta_dPSI_dW_e_cj += dPSI_dW_e_cj;
    delta_dPSI_dW_cj_e += dPSI_dW_cj_e;
    E                  += localEnergy;
}


void VariationalMC::train(MTRand & random, PsiLayer& PL, ConvLayer& CL) 
{
    
    int eq_steps = 500;
    double lr = learning_rate;

    for (int k=0; k<epochs; k++) {

        reset(PL,CL);
        
        for (int i=0; i<eq_steps; i++) {
            equilibrate(random,PL,CL);
        }
        
        for (int i=0; i<MCS; i++) {
            MetropolisUpdate(random,PL,CL);
        }

        for (int k=0; k<PL.n_in; k++) {
            for (int i=0; i<PL.n_f; i++) {
                PL.Z(i,k) += -(lr/1.0*MCS)*(dPSI_dZ_cj_e(i,k)-dPSI_dZ_cj(i,k)*E + dPSI_dZ_e_cj(i,k)-dPSI_dZ(i,k)*E);
            }
        }
        for (int i=0; i<CL.n_f; i++) {
            for (int j=0; j<CL.f_size; j++) {
                CL.W(i,j) += -(lr/1.0*MCS)*(dPSI_dW_cj_e(i,j)-dPSI_dW_cj(i,j)*E + dPSI_dW_e_cj(i,j)-dPSI_dW(i,j)*E);
            }
        }

        //Energy = E.real() / 1.0*MCS;
        //cout << E.real()/MCS << endl;
        cout << "Epoch: " << k << "   Ground State Energy: " << E.real()/MCS << endl; 

    }
    


}

//*****************************************************************************
// Print Network Informations
//*****************************************************************************

void VariationalMC::printNetwork(PsiLayer& PL, ConvLayer& CL) 
{

    cout << "\n\n******************************\n\n" << endl;
    cout << "NEURAL NETWORK VARIATIONAL QUANTUM MONTE CARLO\n\n";
    cout << "Machine Parameter\n\n";
    cout << "\tNumber of Inputs Units: " << L << "\n";
    //cout << "\tNumber of Output Units: " << n_out << "\n";
    cout << "\nNumber of filters: " << CL.n_f << "  ("<< PL.n_f <<")";
    cout << "\n W matrix size (" << CL.W.rows() << "x" << CL.W.cols() << ")" ;
    cout << "\n Z matrix size (" << PL.Z.rows() << "x" << PL.Z.cols() << ")" ;
    cout << "\nFilter Size: " << CL.f_size;
    cout << "\nHyper-parameters\n\n";
    cout << "\tLearning Rate: " << learning_rate << "\n";
    cout << "\tEpochs: " << epochs << "\n";

    
 
}
