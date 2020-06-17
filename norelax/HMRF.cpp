#include <cmath>
#include <random>
#include <iomanip>
//#include <opencv2/imgproc/imgproc.hpp>
#include "HMRF.hpp"
#include "lemmas.hpp"

using namespace std;
using namespace Eigen;


HMRF::HMRF()
{

}



void HMRF::getUnarySolution(Eigen::MatrixXd &_X){
    _X = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    for(size_t c = 0; c < factors.size(); c++){
        if(factors[c].size() != 1)
            continue;
        size_t i = factors[c][0];
        double p_min = potentials[c][0];
        size_t l_min = 0;
        for(size_t idx = 0; idx < potentials[c].size(); idx++){
            double p = potentials[c][idx];
            if(p < p_min){
                p_min = p;
                l_min = idx;
            }
        }
        _X(l_min, i) = 1.0;
    }
}



void HMRF::inference(const std::string &method)
{
    visitor.iteration.resize(0);
    visitor.energies.resize(0);
    visitor.bounds.resize(0);
    visitor.residuals.resize(0);
    visitor.times.resize(0);

    //cout<<"V = "<<numberOfNodes<<endl;
    //cout<<"F = "<<numberOfFactors<<endl;
    //cout<<"L = "<<numberOfLabels[0]<<endl;

    /// Initialization
    if(max_degree < 2){
        this->getUnarySolution(X);
        return;
    }
    if(method == "cqp" && max_degree > 2){
        cout<<"CQP (convex QP relaxation) is only valid for pairwise MRFs!"<<endl;
        return;
    }

    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)

    MatrixXd X_best;
    double energy, energy_best = DBL_MAX;
    for(int idx = 0; idx < parameters.numInit; idx++){
        if(idx == 0){
            if(method == "admm"){
                X = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes).array() + 1.0/(double)numberOfLabels[0];
            } else{
                this->getUnarySolution(X);
            }
            //this->getUnarySolution(X);
        }else{
            // Generate random initial solution
            X = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
            for(size_t i = 0; i < numberOfNodes; i++){
                std::uniform_int_distribution<int> uni(0,numberOfLabels[i]-1); // guaranteed unbiased
                auto random_label = uni(rng);
                X(random_label,i) = 1.0;
            }
        }
        // Inference with the current initial solution
        if(method == "admm"){
            // If the input penalty value is negative then compute it from the model dimensions
            if(parameters.rho_min < 0)
                parameters.rho_min = (double)numberOfNodes*numberOfLabels[0]*max_degree/1e8;
            //this->BCD(parameters.MAX_ITER, true, X_TYPE_DENSE);
            this->ADMM();
            parameters.precision = 1e-10;
            parameters.lineSearch = true;
            this->FW();
            energy = this->BCD(parameters.MAX_ITER, true, X_TYPE_SPARSE);
        } else if(method == "pgd"){
            this->PGD();
            energy = this->BCD(1, true, X_TYPE_SPARSE);
        } else if(method == "fw"){
            this->FW();
            energy = this->BCD(1, true, X_TYPE_SPARSE);
        } else if(method == "cqp"){
            this->CQP();
            energy = this->BCD(1, true, X_TYPE_SPARSE);
        } else if(method == "bcd"){
            energy = this->BCD(parameters.MAX_ITER, true, X_TYPE_DISCRETE);
        } else{
            cout<<"Invalid method: "<<method<<endl;
            return;
        }

        if(energy < energy_best){
            energy_best = energy;
            X_best = X;
        }
    }

    X = X_best;

    // Add the best energy to the end of the visitor (useful for later evaluation)
    visitor.energies.push_back(energy_best);
    visitor.iteration.push_back(visitor.iteration.back() + 1);
    visitor.times.push_back(visitor.times.back());
    //visitor.residuals.push_back(visitor.residuals.back());
    visitor.bounds = VD(visitor.energies.size(), -DBL_MAX);
    this->saveResults(output);

    //this->saveResults(output + ".ADMM." + to_string2(parameters.rho_min));
    //string outputmatrix = output + ".ADMM." + to_string2(parameters.rho_min) + ".X";
    //if(parameters.verbose)
    //    cout<<"Saving the continuous assignment matrix to " << outputmatrix <<endl;
    //WriteMatrix2File(X, outputmatrix.c_str());

    //this->FW();
    //this->PGD();


}



void HMRF::saveResults(const std::string &outputfile)
{
    if(parameters.verbose)
        cout<<"Saving the following data to "<< outputfile <<":"<<endl<<"bounds, states, iterations, energies, residuals, times"<<endl;
    H5::H5File* file = new H5::H5File( outputfile, H5F_ACC_TRUNC );
    write2hdf(file, visitor.bounds, "bounds", visitor.bounds.size(), 1);
    write2hdf(file, visitor.iteration, "iteration", visitor.iteration.size(), 1);
    write2hdf(file, visitor.residuals, "residuals", visitor.residuals.size(), 1);
    write2hdf(file, states, "states", states.size(), 1);
    write2hdf(file, visitor.times, "times", visitor.times.size(), 1);
    write2hdf(file, visitor.energies, "values", visitor.energies.size(), 1);

    delete file;
}



double HMRF::energy()
// Continuous energy. Compute based on X.
{
    double energy = 0.0;
    for(size_t c = 0; c < factors.size(); c++){
        size_t d = factors[c].size() - 1;

        for(size_t idx = 0; idx < potentials[c].size(); idx++){
            double prod = potentials[c][idx];
            bool iszero = false;
            if(prod == 0)
                iszero = true;
            if(prod != 0){
                for(size_t s = 0; s < factors[c].size(); s++){
                    if(X(TensorIndicesOfAllRanks[d][idx][s], factors[c][s]) == 0){
                        iszero = true;
                        break;
                    }
                }
            }

            if(!iszero){
                for(size_t s = 0; s < factors[c].size(); s++){
                    prod *= X(TensorIndicesOfAllRanks[d][idx][s], factors[c][s]);
                }
                energy += prod;
            }
        }
    }
    return energy;
}


double HMRF::energy(const MatrixXd &_X)
{
    double energy = 0.0;
    for(size_t c = 0; c < factors.size(); c++){
        size_t d = factors[c].size() - 1;

        for(size_t idx = 0; idx < potentials[c].size(); idx++){
            double prod = potentials[c][idx];
            bool iszero = false;
            if(prod == 0)
                iszero = true;
            if(prod != 0){
                for(size_t s = 0; s < factors[c].size(); s++){
                    if(_X(TensorIndicesOfAllRanks[d][idx][s], factors[c][s])  == 0){
                        iszero = true;
                        break;
                    }
                }
            }

            if(!iszero){
                for(size_t s = 0; s < factors[c].size(); s++){
                    prod *= _X(TensorIndicesOfAllRanks[d][idx][s], factors[c][s]);
                }
                energy += prod;
            }
        }
    }
    return energy;
}


double HMRF::energy_discrete()
// Compute based on states
{
    double energy = 0.0;
    for(size_t c = 0; c < factors.size(); c++){
        // size_t d = factors[c].size() - 1;
        // Suppose that the nodes of the current factor are i0, i1,..., id
        // The corresponding numbers of labels is L0, L1,..., Ld
        // The corresponding labels is l0 = states[i0], l1 = states[i1],...
        // Then the linear index of the corresponding potential value is
        // idx = l0*L1*L2*...*Ld + l1*L2*L3*... + l(d-1)*Ld + ld
        // In this version, the numbers of labels are equal (= L)
        size_t idx = 0;
        size_t coeff = 1;
        for(size_t s = factors[c].size(); s --> 0 ;){// This loops from (factors[c].size()-1) to 0
            idx += coeff*states[factors[c][s]];
            coeff *= numberOfLabels[factors[c][s]];
        }
        energy += potentials[c][idx];
    }
    return energy;
}




void HMRF::gradient(MatrixXd &G, size_t i, int type)
/// Partial derivative of the energy over x_i (node i)
/// TODO: Need to implement speed up version for X_TYPE_DISCRETE and X_TYPE_SPARSE
{
    VectorXd Gi = VectorXd::Zero(numberOfLabels[i]);
    if(type == X_TYPE_DISCRETE){
        // Iterate over the factors that contain i
        for(size_t idx = 0; idx < containing_factors[i].size(); idx++){
            size_t c = containing_factors[i][idx]; // Index of the factor
            // For each factor, check the position of the node i
            // And contract the tensor at that position
            for(size_t s = 0; s < factors[c].size(); s++){
                if(factors[c][s] == i){
                    VectorXd p;
                    TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
                    Gi += p;
                }
            }
        }
    } else if(type == X_TYPE_SPARSE){
        // Iterate over the factors that contain i
        for(size_t idx = 0; idx < containing_factors[i].size(); idx++){
            size_t c = containing_factors[i][idx]; // Index of the factor
            // For each factor, check the position of the node i
            // And contract the tensor at that position
            for(size_t s = 0; s < factors[c].size(); s++){
                if(factors[c][s] == i){
                    VectorXd p;
                    TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
                    Gi += p;
                }
            }
        }
    } else{
        // Iterate over the factors that contain i
        for(size_t idx = 0; idx < containing_factors[i].size(); idx++){
            size_t c = containing_factors[i][idx]; // Index of the factor
            // For each factor, check the position of the node i
            // And contract the tensor at that position
            for(size_t s = 0; s < factors[c].size(); s++){
                if(factors[c][s] == i){
                    VectorXd p;
                    TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
                    Gi += p;
                }
            }
        }
    }
    G.col(i) = Gi;
}


//void HMRF::gradient_discrete(MatrixXd &G, size_t i)
//// Partial derivative of the energy over x_i (node i)
//{
//    VectorXd Gi = VectorXd::Zero(numberOfLabels[i]);
//    // Iterate over the factors that contain i
//    for(size_t idx = 0; idx < nodes[i].size(); idx++){
//        size_t c = nodes[i][idx]; // Index of the factor
//        // For each factor, check the position of the node i
//        // And contract the tensor at that position
//        for(size_t s = 0; s < factors[c].size(); s++){
//            if(factors[c][s] == i){
//                VectorXd p;
//                TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
//                Gi += p;
//            }
//        }
//    }
//    G.col(i) = Gi;
//}

// TODO: The above discrete function is actually the same as the continuous one. Need to implement a faster discrete version.
//void HMRF::gradient_discrete(MatrixXd &G, size_t i)
// FIXME: this is slow, even slower than the continuous version.
//// Partial derivative of the energy over x_i (node i)
//{
//    VectorXd Gi = VectorXd::Zero(numberOfLabels[i]);
//    // Iterate over the factors that contain i
//    for(size_t idx = 0; idx < nodes[i].size(); idx++){
//        size_t c = nodes[i][idx]; // Index of the factor
//        // For each factor, check the position of the node i
//        // And contract the tensor at that position
//        for(size_t s = 0; s < factors[c].size(); s++){
//            if(factors[c][s] == i){
//                // VectorXd p;
//                // TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
//                VectorXd p = VectorXd::Zero(numberOfLabels[i]);
//                for(size_t l = 0; l < numberOfLabels[i]; l++){
//                    VS states_temp(states);
//                    states_temp[i] = l;
//                    size_t idx = 0;
//                    size_t coeff = 1;
//                    for(size_t t = factors[c].size(); t --> 0 ;){// This loops from (factors[c].size()-1) to 0
//                        idx += coeff*states_temp[factors[c][t]];
//                        coeff *= numberOfLabels[factors[c][t]];
//                    }
//                    p(l) = potentials[c][idx];
//                }
//                Gi += p;
//            }
//        }
//    }
//    G.col(i) = Gi;
//}




void HMRF::gradient(MatrixXd &G, int type)
/// TODO: Need to implement speed up version for X_TYPE_DISCRETE and X_TYPE_SPARSE
{
    G = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    if(type == X_TYPE_DISCRETE){
        for(size_t c = 0; c < factors.size(); c++){
            size_t S = factors[c].size();
            for(size_t s = 0; s < S; s++){
                VectorXd p;
                TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
                G.col(factors[c][s]) += p;
            }
        }
    } else if(type == X_TYPE_SPARSE){
        for(size_t c = 0; c < factors.size(); c++){
            size_t S = factors[c].size();
            for(size_t s = 0; s < S; s++){
                VectorXd p;
                TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
                G.col(factors[c][s]) += p;
            }
        }
    } else{
        for(size_t c = 0; c < factors.size(); c++){
            size_t S = factors[c].size();
            for(size_t s = 0; s < S; s++){
                VectorXd p;
                TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
                G.col(factors[c][s]) += p;
            }
        }
    }
}


//void HMRF::gradient_discrete(MatrixXd &G)
//{
//    G = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
//    for(size_t c = 0; c < factors.size(); c++){
//        size_t S = factors[c].size();
//        for(size_t s = 0; s < S; s++){
//            VectorXd p;
//            TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
//            G.col(factors[c][s]) += p;
//        }
//    }
//}


// TODO: The above discrete function is actually the same as the continuous one. Need to implement a faster discrete version.
//void HMRF::gradient_discrete(MatrixXd &G)
// FIXME: this is slow, even slower than the continuous version.
//{
//    G = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
//    for(size_t c = 0; c < factors.size(); c++){
//        // size_t d = factors[c].size() - 1;
//        // Suppose that the nodes of the current factor are i0, i1,..., id
//        // The corresponding numbers of labels is L0, L1,..., Ld
//        // The corresponding labels is l0 = states[i0], l1 = states[i1],...
//        // Then the linear index of the corresponding potential value is
//        // idx = l0*L1*L2*...*Ld + l1*L2*L3*... + l(d-1)*Ld + ld
//        // In this version, the numbers of labels are equal (= L)

//        for(size_t s = 0; s < factors[c].size(); s++){
//            // VectorXd p;
//            // TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], X, s);
//            // Compute the discrete tensor contraction between the tensor potentials[c] and all neighbors of factors[c][s] in factors[c]
//            size_t i = factors[c][s];
//            VectorXd p = VectorXd::Zero(numberOfLabels[i]);
//            for(size_t l = 0; l < numberOfLabels[i]; l++){
//                VS states_temp(states);
//                states_temp[i] = l;
//                size_t idx = 0;
//                size_t coeff = 1;
//                for(size_t t = factors[c].size(); t --> 0 ;){// This loops from (factors[c].size()-1) to 0
//                    idx += coeff*states_temp[factors[c][t]];
//                    coeff *= numberOfLabels[factors[c][t]];
//                }
//                p(l) = potentials[c][idx];
//            }
//            G.col(factors[c][s]) += p;
//        }
//    }
//}




void HMRF::gradient2(MatrixXd &G)
{
    G = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    for(size_t i = 0; i < numberOfNodes; i++){
        gradient(G, i);
    }
}



//void PGD_backup()
///// Projected gradient descent
///// Update x_{k+1} = [x_k - alpha_k*gradient(x)]_X
///// where alpha_k = C/sqrt(k) and []_X denotes the projection onto X
//{
//    double energy = this->energy();
//    double residual;

//    int reached_count = 0;

//    // Step size
//    double alpha;

//    double C = 10.0;
//    double beta = 0.99;
//    double sigma = 0.9;
//    int m = -1;

//    if(parameters.verbose){
//        cout<<"Starting Projected Gradient Descent..."<<endl;
//        cout<<"Energy = "<<energy<<endl;
//    }

//    MatrixXd G;
//    MatrixXd X_temp;
//    for(size_t k = 0; k < parameters.MAX_ITER; k++)
//    {
//        gradient(G);

//        if(parameters.lineSearch){
//            bool lineSearch_success = false;
//            for(m = 0; m < INT_MAX; m++){
//                SimplexProjection(X_temp, X - (std::pow(beta,m)*C)*G);
//                double energy_temp = this->energy(X_temp);
//                double right_hand_side =  (G.cwiseProduct(X-X_temp)).sum();
//                if(energy - energy_temp >= sigma*right_hand_side){
//                    residual = (X-X_temp).norm();
//                    X = X_temp;
//                    energy = energy_temp;
//                    lineSearch_success = true;
//                    break;
//                }
//            }
//            assert(lineSearch_success);
//        } else{
//            alpha = C/(k + 1.0);
//            X_temp = X;
//            SimplexProjection(X, X - alpha*G);
//            residual = (X-X_temp).norm();
//        }

//        if(parameters.verbose && k%parameters.protocolFrequence == 0){
//            if(!parameters.lineSearch)
//                energy = this->energy();
//            cout<<k<<") m_best = "<<m<<", Residual = "<<residual<<"\tPGD Energy = "<<energy<<endl;
//            if(residual <= parameters.precision){
//                reached_count++;
//                if(reached_count >= 3)
//                    break;
//            }
//        }
//    }
//}



double HMRF::CQP()
/// Convex QP relaxation (only valid for pairwise MRFs
{
    return -1;
}


double HMRF::PGD()
/// Projected gradient descent
/// Update x_{k+1} = [x_k - alpha_k*gradient(x)]_X
/// where alpha_k = C/sqrt(k) and []_X denotes the projection onto X
{
    double timeout = parameters.timeout;
    double total_time;
    if(visitor.times.empty())
        total_time = 0.0;
    else
        total_time = visitor.times.back();

    size_t last_iter;
    if(visitor.iteration.empty())
        last_iter = 0;
    else
        last_iter = visitor.iteration.back();

    myclock::time_point begin, end;
    begin = myclock::now();

    double energy = this->energy();
    double residual;

    int reached_count = 0;
    int num_step_before_line_search = 0;

    // Step size
    double alpha;
    double energy_best = energy;
    double alpha_best;
    if(parameters.verbose){
        cout<<"Initial energy = "<<energy<<". Starting Projected Gradient Descent..."<<endl;

    }

    MatrixXd G;
    MatrixXd R, X_temp, X_best = X;;
    for(int k = 0; k < parameters.MAX_ITER; k++)
    {
        this->gradient(G, X_TYPE_DENSE);

        if(k >= num_step_before_line_search && parameters.lineSearch){
            SimplexProjection(R, X - G);
            R = R - X;

//            alpha = lineSearchStep;
//            alpha_best = 0.0;
//            while(alpha <= 1.0){
//                // Compute E(x + gamma*d)
//                X_temp = X + alpha*D;
//                double energy_temp = this->energy(X_temp);
//                if(energy_temp < energy_best){
//                    X_best = X_temp;
//                    energy_best = energy_temp;
//                    alpha_best = alpha;
//                }
//                alpha += lineSearchStep;
//            }
//            residual = (X-X_best).norm();
//            X = X_best;
//            energy = energy_best;

            this->lineSearch(alpha_best, energy_best, energy, R);
            X_best = X + alpha_best*R;
            residual = (X-X_best).norm();
            X = X_best;
            // FIXME: the returned energy_best seems to be incorrect.
            //energy = energy_best;
            energy = this->energy();
        } else{
            alpha = 10.0/(k + 1.0);
            X_best = X;
            SimplexProjection(X, X - alpha*G);
            residual = (X-X_best).norm();
            energy = this->energy();
        }

        if(residual <= parameters.precision){
            reached_count++;
            if(reached_count >= 3){
                if(parameters.verbose)
                    cout<<k+1<<")\tResidual: "<<setw(10)<<residual<<"\tPDG energy: "<<setw(10)<<energy<<endl;

                end = myclock::now();
                total_time += measure_time(begin, end);
                visitor.times.push_back(total_time);
                visitor.iteration.push_back(k + last_iter + 1);
                visitor.energies.push_back(energy);
                visitor.residuals.push_back(residual);
                break;
            }
        }

        if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
            if(parameters.verbose)
                cout<<k+1<<")\tResidual: "<<setw(10)<<residual<<"\tPDG energy: "<<setw(10)<<energy<<endl;

            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);
            visitor.iteration.push_back(k + last_iter + 1);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(residual);
            begin = myclock::now();

            if(total_time > timeout){
                if(parameters.verbose)
                    cout<<"--- TIMEOUT ---"<<endl;
                break;
            }
        }
    }
    return energy;
}




double HMRF::FW()
/// Frank–Wolfe algorithm (a.k.a conditional gradient descent)
{
    double timeout = parameters.timeout;
    double total_time;
    if(visitor.times.empty())
        total_time = 0.0;
    else
        total_time = visitor.times.back();

    size_t last_iter;
    if(visitor.iteration.empty())
        last_iter = 0;
    else
        last_iter = visitor.iteration.back();

    myclock::time_point begin, end;
    begin = myclock::now();

    double energy = this->energy();
    double fwGap = 0.0, fwGap_old;
    int reached_count = 0;
    int noimprovement_count = 0;
    int num_step_before_line_search = 0; // Run a few iterations with diminishing step size before doing line search

    // Step size
    double alpha;
    double energy_best = energy;
    double alpha_best;

    if(parameters.verbose){
        cout<<"Initial energy = "<<energy<<". Starting Frank-Wolfe algorithm..."<<endl;
    }

    MatrixXd G;
    MatrixXd R, X_temp, X_best = X;
    for(int k = 0; k < parameters.MAX_ITER; k++)
    {
        // Step 1: compute min_{S \in \cX} <S, G>
        this->gradient(G, X_TYPE_DENSE);
        R.setZero(G.rows(), G.cols());
        for(size_t i = 0; i < (size_t)G.cols(); i++){
            size_t l_min;
            G.col(i).minCoeff(&l_min);
            R(l_min, i) = 1.0;
        }

        // Step 2: compute the FW update direction R = S - X and the FW gap
        R = R - X;

        fwGap_old = fwGap;
        fwGap = -(G.cwiseProduct(R)).sum()/absMax;
        if(fwGap <= parameters.precision){
            reached_count++;
            if(reached_count >= 3){
                if(parameters.verbose)
                    cout<<k+1<<")\tGap: "<<setw(10)<<fwGap<<"\tFW energy: "<<setw(10)<<energy<<endl;

                end = myclock::now();
                total_time += measure_time(begin, end);
                visitor.times.push_back(total_time);
                visitor.iteration.push_back(k + last_iter + 1);
                visitor.energies.push_back(energy);
                visitor.residuals.push_back(fwGap);
                break;
            }
        }

        if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
            if(parameters.verbose)
                cout<<k+1<<")\tGap: "<<setw(10)<<fwGap<<"\tFW energy: "<<setw(10)<<energy<<endl;

            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);
            visitor.iteration.push_back(k + last_iter + 1);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(fwGap);
            begin = myclock::now();

            if(abs(fwGap - fwGap_old) < 1e-6)
                noimprovement_count++;

            if(total_time > timeout){
                if(parameters.verbose)
                    cout<<"--- TIMEOUT ---"<<endl;
                break;
            }

            if(noimprovement_count > 2){
                if(parameters.verbose)
                    cout<<"--- NO FURTHER IMPROVEMENT ---"<<endl;
                break;
            }
        }


        if(k >= num_step_before_line_search && parameters.lineSearch){
//            alpha = lineSearchStep;
//            alpha_best = 0.0;
//            while(alpha <= 1.0){
//                // Compute E(x + alpha*d)
//                X_temp = X + alpha*D;
//                double energy_temp = this->energy(X_temp);
//                if(energy_temp < energy_best){
//                    X_best = X_temp;
//                    energy_best = energy_temp;
//                    alpha_best = alpha;
//                }
//                alpha += lineSearchStep;
//            }
//            X = X_best;
//            energy = energy_best;

            this->lineSearch(alpha_best, energy_best, energy, R);
            X += alpha_best*R;
            // FIXME: the returned energy_best seems to be incorrect.
            //energy = energy_best;
            //double energy_check = this->energy();
            //cout<<"energy - energy_check = "<<energy - energy_check<<endl;
            energy = this->energy();
        } else{
            alpha = 2.0/(double)(k + 2.0);
            X += alpha*R;
            energy = this->energy();
        }

    }
    return energy;
}


void HMRF::lineSearch(double &alpha_best, double &p_best, double energy, const MatrixXd &R)
// Line search for Projected Gradient Descent and Frank-Wolfe algorithm
// Find alpha such that E(X + alpha*D) is minimum. Return the value of E at this minimum to p_best.
{

    double p0, p1, alpha;
    double lineSearchStep = 0.0001; // Compare the values in [0, 0.2, 0.4,..., 1.0]
    p0 = energy;
    // E(X + alpha*D) is a polynomial p of degree n, where n is the degree of the MRF
    // This polynomial has (n + 1) coefficients, one of them is the constant term
    // which is equal to E(X)
    // The following vector (VectorXd c) contains the remaining n coefficients.
    // We need n equations for finding these n coefficients
    // A*c = e
    MatrixXd A(max_degree,max_degree);
    VectorXd e(max_degree);

    A.row(0).setOnes();
    e(0) = this->energy(X + R);
    p1 = e(0);

    p_best = min(p0,p1);
    alpha_best = (p0 < p1)?0.0:1.0;

    for(size_t j = 1; j < max_degree; j++){
        alpha = 1.0/(double)(j+1.0);
        A(j,0) = alpha;
        for(size_t idx = 1; idx < max_degree; idx++){
            A(j,idx) = A(j,idx-1)*alpha;
        }
        e(j) = this->energy(X + alpha*R);
    }
    e = e.array() - energy;
    // Solve the linear system to find the coefficients
    VectorXd c = A.colPivHouseholderQr().solve(e);
    // Now as we have found the polynomial wrt alpha
    // We need to minimize it over [0,1]
    if(max_degree == 3){
        // The derivative of this polynomial = A*alpha^2 + B*alpha + C
        // Roots: (-B +- sqrt(B^2-4AC))/(2A) if A!= 0
        // Compute only the value of the polynomial at the roots between [0,1],
        // and compare these values with p0, p1 to find the minimum
        double A = 3.0*c(2);
        double B = 2.0*c(1);
        double C = c(0);

        if(A == 0){
            if(B != 0){
                alpha = -C/B;
                if(alpha >= 0 && alpha <= 1){
                    double p = p0 + c(0)*alpha + c(1)*alpha*alpha + c(2)*alpha*alpha*alpha;
                    if(p < p_best){
                        p_best = p;
                        alpha_best = alpha;
                    }
                }
            }
        } else{
            double delta = B*B - 4.0*A*C;
            if(delta == 0){
                alpha = -B/(2.0*A);
                if(alpha >= 0 && alpha <= 1){
                    double p = p0 + c(0)*alpha + c(1)*alpha*alpha + c(2)*alpha*alpha*alpha;
                    if(p < p_best){
                        p_best = p;
                        alpha_best = alpha;
                    }
                }
            } else if(delta > 0){
                alpha = (-B - sqrt(delta))/(2.0*A);
                if(alpha >= 0 && alpha <= 1){
                    double p = p0 + c(0)*alpha + c(1)*alpha*alpha + c(2)*alpha*alpha*alpha;
                    if(p < p_best){
                        p_best = p;
                        alpha_best = alpha;
                    }
                }
                alpha = (-B + sqrt(delta))/(2.0*A);
                if(alpha >= 0 && alpha <= 1){
                    double p = p0 + c(0)*alpha + c(1)*alpha*alpha + c(2)*alpha*alpha*alpha;
                    if(p < p_best){
                        p_best = p;
                        alpha_best = alpha;
                    }
                }
            }
        }
    } else{ // If the degree of the polynomial is higher than 3. TODO: add code for degree 2 also.
        alpha = lineSearchStep;
        while(alpha < 1.0){
            //double p = p0 + c(0)*alpha + c(1)*alpha*alpha + c(2)*alpha*alpha*alpha;
            double p = p0;
            double AA = alpha;
            for(size_t idx = 0; idx < (size_t)c.size(); idx++){
                p += c(idx)*AA;
                AA *= alpha;
            }
            if(p < p_best){
                p_best = p;
                alpha_best = alpha;
            }
            alpha += lineSearchStep;
        }
    }
}



double HMRF::BCD(int numIter, bool random, int type)
/// Rounding: Block coordinate descent
/// If 'random = true' then randomly shuffling the processing order of the nodes
/// at each (outer) iteration
/// 'type' is a prior on the sparsity of the labeling matrix X
/// This is for speed up when computing the first BCD pass
/// It takes values among "dense", "sparse", "discrete"
{
    if(numIter < 1)
        return -1;
    auto engine = std::default_random_engine{};
    std::vector<size_t> random_indices(numberOfNodes);

    double timeout = parameters.timeout;
    double total_time;
    if(visitor.times.empty())
        total_time = 0.0;
    else
        total_time = visitor.times.back();

    size_t last_iter;
    if(visitor.iteration.empty())
        last_iter = 0;
    else
        last_iter = visitor.iteration.back();

    myclock::time_point begin, end;
    begin = myclock::now();

    // 'NeighborChanged[i] = true' means one of the neighbors of the node 'i'
    // has changed value since the last iteration
    // If 'NeighborChanged[i] = true' then the gradient of E(x) at 'i' has also changed
    vector<bool> NeighborChanged(numberOfNodes, false);

    /// We really need a first pass over the continuous solution to get a discrete solution

    if(random){
        std::iota(random_indices.begin(), random_indices.end(), 0);
        std::shuffle(std::begin(random_indices), std::end(random_indices), engine);
    }

    MatrixXd G;
    states.resize(numberOfNodes);

    if(type == X_TYPE_DISCRETE){
        for(size_t i = 0; i < numberOfNodes; i++){
            size_t label_max;
            X.col(i).maxCoeff( &label_max );
            states[i] = label_max;
        }
    }

    this->gradient(G, type);

    for(size_t idx = 0; idx < numberOfNodes; idx++){
        size_t i = random_indices[idx];
        // Need to re-compute the gradient only if one of the neighbors has changed its value
        if(NeighborChanged[i]){
            this->gradient(G, i, type);
            NeighborChanged[i] = false;
        }
        size_t l_min;
        G.col(i).minCoeff(&l_min);
        X.col(i).setZero();
        X(l_min, i) = 1.0;
        states[i] = l_min;

        for (set<size_t>::iterator it=neighbors[i].begin(); it!=neighbors[i].end(); ++it){
            size_t j = *it;
            NeighborChanged[j] = true;
        }
    }

    /// Now all the variables are discrete, we can use the discrete computation
    double energy = this->energy_discrete();
    if(parameters.verbose){
        cout<<"BCD pass 1: \tEnergy = "<<energy<<endl;
    }
    end = myclock::now();
    total_time += measure_time(begin, end);
    visitor.times.push_back(total_time);
    visitor.iteration.push_back(last_iter + 1);
    visitor.energies.push_back(energy);
    begin = myclock::now();

    if(numIter == 1)
        return energy;

    // Next we continue to further improve the solution
    // Pre-compute the discrete gradient
    //this->gradient_discrete(G);
    this->gradient(G, X_TYPE_DISCRETE);
    //MatrixXd G2; this->gradient(G2);
    //cout<<"norm(G_discrete,G_continuous) = "<<(G - G2).norm()<<endl;
    std::fill(NeighborChanged.begin(), NeighborChanged.end(), false);

    /// Now all the variables are discrete, we can use the discrete computation

//    double energy = this->energy_discrete();
//    double energy_continuous = this->energy();
//    double energy_previous = energy;

//    if(parameters.verbose){
//        cout<<"Continuous energy = "<<energy_continuous<<", discrete energy = "<<energy<<endl;
//        cout<<"Starting Block Coordinate Descent..."<<endl;
//    }

    for(int k = 1; k < numIter; k++)
    {
        if(random){
            std::iota(random_indices.begin(), random_indices.end(), 0);
            std::shuffle(std::begin(random_indices), std::end(random_indices), engine);
        }

        for(size_t idx = 0; idx < numberOfNodes; idx++){
            size_t i = random_indices[idx];
            // Need to re-compute the gradient only if one of the neighbors has changed its value
            if(NeighborChanged[i]){
                this->gradient(G, i, X_TYPE_DISCRETE);
                NeighborChanged[i] = false;
            }
            size_t l_min;
            G.col(i).minCoeff(&l_min);
            // If the node i has changed its label then the gradients of all its neighbors have also changed
            if(states[i] != l_min){
                states[i] = l_min;
                X.col(i).setZero();
                X(l_min, i) = 1.0;
                for (set<size_t>::iterator it=neighbors[i].begin(); it!=neighbors[i].end(); ++it){
                    size_t j = *it;
                    NeighborChanged[j] = true;
                }
            }
            // Otherwise, do nothing.
        }
        double energy_previous = energy;
        energy = this->energy_discrete();

        end = myclock::now();
        total_time += measure_time(begin, end);
        visitor.times.push_back(total_time);
        visitor.iteration.push_back(k + last_iter + 1);
        visitor.energies.push_back(energy);
        //visitor.bounds.push_back(-DBL_MAX);

        if(parameters.verbose){
            cout<<"BCD pass "<<k+1<<": \tEnergy = "<<energy<<endl;
        }
        if(energy >= energy_previous)
            break;

        /// If timeout reached
        if(total_time > timeout){
            if(parameters.verbose){
                cout<<"--- TIMEOUT ---"<<endl;
            }
            break;
        }

        begin = myclock::now();
    }
    return energy;
}



//void HMRF::BCD_backup(bool random)
///// Rounding: Block coordinate descent
///// If 'random = true' then randomly shuffling the processing order of the nodes
///// at each (outer) iteration
//{
//    auto engine = std::default_random_engine{};
//    std::vector<size_t> random_indices(numberOfNodes);

//    double timeout = parameters.timeout;
//    double total_time;
//    if(visitor.times.empty())
//        total_time = 0.0;
//    else
//        total_time = visitor.times.back();

//    size_t last_iter;
//    if(visitor.iteration.empty())
//        last_iter = 0;
//    else
//        last_iter = visitor.iteration.back();

//    myclock::time_point begin, end;
//    begin = myclock::now();





//    double energy = this->energy();
//    double energy_previous;
//    if(parameters.verbose){
//        cout<<"Computing initial energy and gradient..."<<endl;
//    }

//    MatrixXd G;
//    this->gradient(G);

//    // 'NeighborChanged[i] = true' means one of the neighbors of the node 'i'
//    // has changed value since the last iteration
//    // If 'NeighborChanged[i] = true' then the gradient of E(x) at 'i' has also changed
//    vector<bool> NeighborChanged(numberOfNodes, false);


//    if(parameters.verbose){
//        cout<<"Done. Initial energy = "<<energy<<endl;
//        cout<<"Starting Block Coordinate Descent..."<<endl;
//        //cout<<"Energy = "<<energy<<endl;
//    }


//    /// First pass to get discrete solution
//    if(random){
//        std::iota(random_indices.begin(), random_indices.end(), 0);
//        std::shuffle(std::begin(random_indices), std::end(random_indices), engine);
//    }
//    for(size_t idx = 0; idx < numberOfNodes; idx++){
//        size_t i = random_indices[idx];
//        // Need to re-compute the gradient only if one of the neighbors has changed its value
//        if(NeighborChanged[i]){
//            gradient(G, i);
//            NeighborChanged[i] = false;
//        }
//        size_t l_min;
//        G.col(i).minCoeff(&l_min);
//        X.col(i).setZero();
//        X(l_min, i) = 1.0;
//        states[i] = l_min;

//        for (set<size_t>::iterator it=neighbors[i].begin(); it!=neighbors[i].end(); ++it){
//            size_t j = *it;
//            NeighborChanged[j] = true;
//        }
//    }
//    energy_previous = energy;
//    energy = this->energy_discrete();
//    if(parameters.verbose){
//        cout<<"BCD pass 0: \tEnergy = "<<energy<<endl;
//    }

//    /// Now all the variables are discrete, we can using the discrete computation

//    // Pre-compute the discrete gradient
//    this->gradient_discrete(G);
//    std::fill(NeighborChanged.begin(), NeighborChanged.end(), false);

//    for(int k = 0; k < parameters.MAX_ITER; k++)
//    {
//        if(random){
//            std::iota(random_indices.begin(), random_indices.end(), 0);
//            std::shuffle(std::begin(random_indices), std::end(random_indices), engine);
//        }

//        for(size_t idx = 0; idx < numberOfNodes; idx++){
//            size_t i = random_indices[idx];
//            // Need to re-compute the gradient only if one of the neighbors has changed its value
//            if(NeighborChanged[i]){
//                gradient_discrete(G, i);
//                NeighborChanged[i] = false;
//            }
//            size_t l_min;
//            G.col(i).minCoeff(&l_min);
//            // If the node i has changed its label then the gradients of all its neighbors have also changed
//            if(states[i] != l_min){
//                states[i] = l_min;
//                for (set<size_t>::iterator it=neighbors[i].begin(); it!=neighbors[i].end(); ++it){
//                    size_t j = *it;
//                    NeighborChanged[j] = true;
//                }
//            }
//            // Otherwise, do nothing.
//        }
//        energy_previous = energy;
//        energy = this->energy_discrete();

//        end = myclock::now();
//        total_time += measure_time(begin, end);
//        visitor.times.push_back(total_time);
//        visitor.iteration.push_back(k + last_iter + 1);
//        visitor.energies.push_back(energy);
//        //visitor.bounds.push_back(-DBL_MAX);

//        if(parameters.verbose){
//            cout<<"BCD pass "<<k+1<<": \tEnergy = "<<energy<<endl;
//        }
//        if(k >= 1 && energy >= energy_previous)
//            break;

//        /// If timeout reached
//        if(total_time > timeout){
//            if(parameters.verbose){
//                cout<<"TIMEOUT!!!! Return the solution..."<<endl;
//            }
//            break;
//        }

//        begin = myclock::now();
//    }


//    for(size_t i = 0; i < numberOfNodes; i++){
//        X.col(i).setZero();
//        X(states[i], i) = 1.0;
//    }

//}




//void HMRF::BCD_slow(bool random)
///// Rounding: Block coordinate descent
///// If 'random = true' then randomly shuffling the processing order of the nodes
///// at each (outer) iteration
//{

//    double timeout = parameters.timeout;
//    double total_time;
//    if(visitor.times.empty())
//        total_time = 0.0;
//    else
//        total_time = visitor.times.back();

//    size_t last_iter;
//    if(visitor.iteration.empty())
//        last_iter = 0;
//    else
//        last_iter = visitor.iteration.back();

//    myclock::time_point begin, end;
//    begin = myclock::now();


//    std::vector<size_t> random_indices(numberOfNodes);
//    std::iota(random_indices.begin(), random_indices.end(), 0);


//    if(parameters.verbose){
//        cout<<"Starting Block Coordinate Descent..."<<endl;
//        //cout<<"Energy = "<<energy<<endl;
//    }

//    double energy = this->energy();
//    double energy_previous;

//    MatrixXd G = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
//    for(int k = 0; k < parameters.MAX_ITER; k++)
//    {
//        if(random){
//            auto engine = std::default_random_engine{};
//            std::shuffle(std::begin(random_indices), std::end(random_indices), engine);
//        }

//        for(size_t idx = 0; idx < numberOfNodes; idx++){
//            //if(idx%10 == 0)
//            //    cout<<idx<<".";
//            size_t i = random_indices[idx];
//            gradient(G, i);
//            size_t l_min;
//            G.col(i).minCoeff(&l_min);
//            X.col(i).setZero();
//            X(l_min, i) = 1.0;
//        }
//        energy_previous = energy;
//        energy = this->energy();

//        end = myclock::now();
//        total_time += measure_time(begin, end);
//        visitor.times.push_back(total_time);
//        visitor.iteration.push_back(k + last_iter + 1);
//        visitor.energies.push_back(energy);
//        //visitor.bounds.push_back(-DBL_MAX);

//        if(parameters.verbose){
//            cout<<"BCD pass "<<k+1<<": \tEnergy = "<<energy<<endl;
//        }
//        if(k >= 1 && energy >= energy_previous)
//            break;

//        /// If timeout reached
//        if(total_time > timeout){
//            if(parameters.verbose){
//                cout<<"TIMEOUT!!!! Return the solution..."<<endl;
//            }
//            break;
//        }

//        begin = myclock::now();
//    }

//    states.resize(numberOfNodes);
//    for(size_t i = 0; i < numberOfNodes; i++){
//        size_t label_max;
//        X.col(i).maxCoeff( &label_max );
//        states[i] = label_max;
//    }

//}



void HMRF::computeP(size_t d, const vector<MatrixXd> &XD)
{
    for(size_t i = 0; i < numberOfNodes; i++){
        if(hasChangedP[d][i])
            P[d].col(i).setZero();
    }

    for(size_t c = 0; c < factors.size(); c++){
        if(factors[c].size() > d){
            if(hasChangedRC[c][d]){
                VectorXd p;
                TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], XD, d);
                RC[c].col(d) = p;
            }
            if(hasChangedP[d][factors[c][d]])
                P[d].col(factors[c][d]) += RC[c].col(d);
        }
    }
}


double HMRF::ADMM()
{
    assert(factors.size() == potentials.size());

    if(parameters.verbose){
        cout<<"**********************************************************"<<endl;
        cout<<"ADMM parameters:"<<endl;
        cout<<"rho_min = "<<parameters.rho_min<<endl<<"rho_max = "<<parameters.rho_max<<endl<<"step = "<<parameters.step<<endl<<"maxIt = "<<parameters.MAX_ITER<<endl;
        cout<<"iter1 = "<<parameters.iter1<<endl<<"iter2 = "<<parameters.iter2<<endl<<"protocol = "<<parameters.protocolFrequence<<endl<<"precision = "<<parameters.precision<<endl;
        cout<<"timeout = "<<parameters.timeout<<endl;
        cout<<"**********************************************************"<<endl;
    }

    // Instead of normalizing the potentials to [-1,1] by dividing them by max_potential_value
    // (and storing the normalized potentials) we can save the memory by scaling rho (and scaling
    // the initial dual variables y as well, unless it is set to 0)
    /// Instead of normalizing the potentials to [-1,1], we equivalently scale
    /// the penalty parameter rho and the initial dual variable Y
//    double absmax = 0;
//    for(size_t c = 0; c < factors.size(); c++){
//        for(size_t idx = 0; idx < potentials[c].size(); idx++){
//            double pmax = std::abs(potentials[c][idx]);
//            if(absmax < pmax)
//                absmax = pmax;
//        }
//    }
//    if(verbose)
//        cout<<"Coeff max = "<<absmax<<endl;

    double rho_min = this->absMax*parameters.rho_min;
    double rho_max = this->absMax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t L = numberOfLabels[0];

    for(size_t i = 1; i < V; i++){
        if(L != numberOfLabels[i]){
            cerr<<"The current version supports only equal number of labels!"<<endl;
        }
        assert(L == numberOfLabels[i]);
    }

    /// The maximum size of the cliques (i.e. (D-1)-th order MRFs)
    size_t D = max_degree;
//    if(verbose){
//        cout<<"Compute the maximum factor size: D = "<<D<<endl;
//    }

    /// Initialization
    vector<MatrixXd> XD(D, X);
    vector<MatrixXd> Y(D, MatrixXd::Zero(L,V));

    vector<MatrixXd> XD_old;


    vector< vector<bool> > hasChangedX(D, vector<bool>(V, true));
    vector< vector<bool> > hasChangedY(D, vector<bool>(V, true));
    bool hasChangedRho = true;

    /// Each factor C has a contracted vector at a position d, d <= size(C) (see paper)
    /// Define for each factor c a matrix RC[c] that has dimension L x |c| where each column
    /// is the contraction at that position
    /// Only when updating X[d] that we need to compute the reduction of all factors at the position d
    RC.resize(factors.size());
    hasChangedRC.resize(factors.size());
    for(size_t c = 0; c < factors.size(); c++){
        if(factors[c].size() == 1){
            RC[c] = MatrixXd::Map(potentials[c].data(), L, 1);
            hasChangedRC[c].resize(1);
            hasChangedRC[c][0] = false;
        }else{
            RC[c] = MatrixXd::Zero(L, factors[c].size());
            hasChangedRC[c] = vector<bool>(factors[c].size(), true);
        }
    }


    P = vector<MatrixXd>(D, MatrixXd::Zero(L,V));
    hasChangedP = vector< vector<bool> >(D, vector<bool>(V, true));


    double energy;
    double energy_best = DBL_MAX;
    double rho = rho_min;
    int iter1_cumulated = parameters.iter1;
    double res_best_so_far = DBL_MAX;
    int counter = 0;
    double residual = DBL_MAX, residual_old;

    myclock::time_point begin, end;
    myclock::time_point begin2, end2;
    double T1 = 0.0, T2 = 0.0;


    double total_time;
    if(visitor.times.empty())
        total_time = 0.0;
    else
        total_time = visitor.times.back();

    size_t last_iter;
    if(visitor.iteration.empty())
        last_iter = 0;
    else
        last_iter = visitor.iteration.back();



    double r, s;

    begin = myclock::now();

    for(int k = 0; k < parameters.MAX_ITER; k++)
    {
        /// Step 1: Update x
        // Check which node in X has changed
//        if(k > 0){
//            std::fill(hasChangedX.begin(), hasChangedX.end(), vector<bool>(V, true));
//            for(size_t d = 0; d < D; d++){
//                for(size_t i = 0; i < V; i++){
//                    hasChangedX[d][i] = !(XD[d].col(i).isApprox(XD_old[d].col(i)));
//                }
//            }
//        }

        XD_old = XD;


        for(size_t d = 0; d < D; d++){
            // Compute pd (cf. paper for definition)
            begin2 = myclock::now();

            hasChangedP[d] = vector<bool>(V, false);
            for(size_t c = 0; c < factors.size(); c++){
                if(factors[c].size() > d){
                    hasChangedRC[c][d] = false;
                    for(size_t idx = 0; idx < factors[c].size(); idx++){
                        if(idx != d){
                            size_t i = factors[c][idx];
                            if(hasChangedX[idx][i]){
                                hasChangedRC[c][d] = true;
                                break;
                            }
                        }
                    }

                    if(hasChangedRC[c][d]){
                        hasChangedP[d][factors[c][d]] = true;
                    }

                }
            }

            if(k < 1)
                std::fill(hasChangedP[d].begin(), hasChangedP[d].end(), true);

            // Compute P[D]
            //computeP(d, XD);
            for(size_t i = 0; i < V; i++){
                if(hasChangedP[d][i])
                    P[d].col(i).setZero();
            }

            for(size_t c = 0; c < factors.size(); c++){
                if(factors[c].size() > d){
                    if(hasChangedRC[c][d]){
                        VectorXd p;
                        TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], XD, d);
                        RC[c].col(d) = p;
                    }
                    if(hasChangedP[d][factors[c][d]])
                        P[d].col(factors[c][d]) += RC[c].col(d);
                }
            }


            end2= myclock::now();
            T1 += measure_time(begin2, end2);


            // Solve the corresponding quadratic program
            begin2 = myclock::now();

            std::fill(hasChangedX[d].begin(), hasChangedX[d].end(), false);

            if(d == 0){
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[0][i] || hasChangedY[1][i] || hasChangedX[1][i] || hasChangedRho){
                        hasChangedX[d][i] = true;
                        VectorXd c = XD[1].col(i) - (Y[1].col(i) + P[d].col(i))/rho;
                        VectorXd x;
                        SimplexProjection(x, c);
                        XD[d].col(i) = x;
                    }
                }
            }
            if( d > 0 && d < D - 1){
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[d][i] || hasChangedY[d][i] || hasChangedY[d+1][i] || hasChangedX[d-1][i] || hasChangedX[d+1][i] ||hasChangedRho){
                        hasChangedX[d][i] = true;
                        VectorXd c = 0.5*(XD[d-1].col(i) + XD[d+1].col(i)) + (Y[d].col(i) - Y[d+1].col(i) - P[d].col(i))/(2.0*rho);
                        XD[d].col(i) = c.cwiseMax(0.0);
                    }
                }
            }
            if(d == D-1){
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[d][i] || hasChangedY[d][i] || hasChangedX[d-1][i] || hasChangedRho){
                        hasChangedX[d][i] = true;
                        VectorXd c = XD[D-2].col(i) + (Y[D-1].col(i) - P[d].col(i))/rho;
                        XD[d].col(i) = c.cwiseMax(0.0);
                    }
                }
            }


            end2 = myclock::now();
            T2 += measure_time(begin2, end2);

            // Add additional unchanged Xi to the list
            for(size_t i = 0; i < V; i++){
                if(hasChangedX[d][i])
                    hasChangedX[d][i] = !(XD[d].col(i).isApprox(XD_old[d].col(i)));
            }
        }

        /// Step 2: Update y
        for(size_t d = 1; d < D; d++){
            Y[d] += rho*(XD[d-1] - XD[d]);
        }

        for(size_t d = 1; d < D; d++){
            for(size_t i = 0; i < V; i++){
                hasChangedY[d][i] = !(XD[d-1].col(i).isApprox(XD[d].col(i)));
                //hasChangedY[d][i] = !(X[d-1].col(i) - X[d].col(i)).isMuchSmallerThan(1e-10);
            }
        }


        if(hasChangedRho)
            hasChangedRho = false;

        /// Step 3: compute the residuals and update rho
        if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(k + last_iter + 1);

            r = 0.0;
            s = 0.0;
            for(size_t d = 1; d < D; d++){
                r += (XD[d-1] - XD[d]).squaredNorm();
            }

            for(size_t d = 0; d < D; d++){
                s += (XD_old[d] - XD[d]).squaredNorm();
            }
            residual_old = residual;
            residual = r + s;

            energy = this->energy(XD[0]);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(residual);
            //visitor.bounds.push_back(-DBL_MAX);


            if(energy < energy_best){
                energy_best = energy;
                X = XD[0];
            }

            if(parameters.verbose)
                cout<<k+1<<")\tresidual: "<<setw(10)<<residual<<"\tenergy: "<<setw(10)<<energy<<endl;

            /// If convergence
            if(residual <= parameters.precision && residual_old <= parameters.precision){
                break;
            }

            /// If timeout reached
            if(total_time > parameters.timeout){
                if(parameters.verbose){
                    cout<<"--- TIMEOUT ---"<<endl;
                }
                break;
            }

            begin = myclock::now();
        }


        /// Only after iter1 iterations that we start to track the best residuals
        if(k >= iter1_cumulated){
            if(residual < res_best_so_far){
                res_best_so_far = residual;
                counter = 0;
            }else{
                counter++;
            }
            // If the best_so_far residual has not changed during iter2 iterations, then update rho
            if(counter >= parameters.iter2){
                if(rho < rho_max){
                    rho = std::min(rho*parameters.step, rho_max);
                    hasChangedRho = true;
                    if(parameters.verbose){
                        cout<<k+1<<") --- UPDATE rho = "<<rho/absMax<<endl;
                    }
                    counter = 0;
                    iter1_cumulated = k + parameters.iter1;
                }else{
                    break;
                }
            }
        }
    }



    /// If having reached the maxium number of iterations without convergence,
    /// then return the current solution and the residuals


    if(parameters.verbose){
        cout<<"Coefficient computation time:\t "<<T1<<endl;
        cout<<"Simplex projection time:\t "<<T2<<endl;
        cout<<"Total ADMM time:\t "<<total_time<<endl;
    }

    // Convert to labels
    states.resize(V);
    for(size_t i = 0; i < V; i++){
        size_t label_max;
        X.col(i).maxCoeff( &label_max );
        //if(x_max < 1.0) cout<<x_max<<", ";
        states[i] = label_max;
    }
    return energy_best;
}


double HMRF::ADMM_old()
{
    assert(factors.size() == potentials.size());
    double step = parameters.step;
    int iter1 = parameters.iter1;
    int iter2 = parameters.iter2;
    int MAX_ITER = parameters.MAX_ITER;
    bool verbose = parameters.verbose;
    int protocolFrequence = parameters.protocolFrequence;
    double precision = parameters.precision;
    double timeout = parameters.timeout;

    if(verbose){
        cout<<"**********************************************************"<<endl;
        cout<<"ADMM parameters:"<<endl;
        cout<<"rho_min = "<<parameters.rho_min<<endl<<"rho_max = "<<parameters.rho_max<<endl<<"step = "<<step<<endl<<"maxIt = "<<MAX_ITER<<endl;
        cout<<"iter1 = "<<iter1<<endl<<"iter2 = "<<iter2<<endl<<"protocol = "<<protocolFrequence<<endl<<"precision = "<<precision<<endl;
        cout<<"timeout = "<<timeout<<endl;
        cout<<"**********************************************************"<<endl;
    }

    // Instead of normalizing the potentials to [-1,1] by dividing them by max_potential_value
    // (and storing the normalized potentials) we can save the memory by scaling rho (and scaling
    // the initial dual variables y as well, unless it is set to 0)
    /// Instead of normalizing the potentials to [-1,1], we equivalently scale
    /// the penalty parameter rho and the initial dual variable Y
//    double absmax = 0;
//    for(size_t c = 0; c < factors.size(); c++){
//        for(size_t idx = 0; idx < potentials[c].size(); idx++){
//            double pmax = std::abs(potentials[c][idx]);
//            if(absmax < pmax)
//                absmax = pmax;
//        }
//    }
//    if(verbose)
//        cout<<"Coeff max = "<<absmax<<endl;

    double rho_min = this->absMax*parameters.rho_min;
    double rho_max = this->absMax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t L = numberOfLabels[0];

    for(size_t i = 1; i < V; i++){
        if(L != numberOfLabels[i]){
            cerr<<"The current version supports only equal number of labels!"<<endl;
        }
        assert(L == numberOfLabels[i]);
    }

    /// The maximum size of the cliques (i.e. (D-1)-th order MRFs)
    size_t D = max_degree;
//    if(verbose){
//        cout<<"Compute the maximum factor size: D = "<<D<<endl;
//    }

    /// Initialization
    vector<MatrixXd> XD(D, X);
    vector<MatrixXd> Y(D, MatrixXd::Zero(L,V));

    vector<MatrixXd> XD_old;


    vector< vector<bool> > hasChangedX(D, vector<bool>(V, true));
    vector< vector<bool> > hasChangedY(D, vector<bool>(V, true));
    bool hasChangedRho = true;

    /// Each factor C has a contracted vector at a position d, d <= size(C) (see paper)
    /// Define for each factor c a matrix RC[c] that has dimension L x |c| where each column
    /// is the contraction at that position
    /// Only when updating X[d] that we need to compute the reduction of all factors at the position d
    vector<MatrixXd> RC(factors.size());
    vector< vector<bool> > hasChangedRC(factors.size());
    for(size_t c = 0; c < factors.size(); c++){
        if(factors[c].size() == 1){
            RC[c] = MatrixXd::Map(potentials[c].data(), L, 1);
            hasChangedRC[c].resize(1);
            hasChangedRC[c][0] = false;
        }else{
            RC[c] = MatrixXd::Zero(L, factors[c].size());
            hasChangedRC[c] = vector<bool>(factors[c].size(), true);
        }
    }


    vector<MatrixXd> P(D, MatrixXd::Zero(L,V));
    vector< vector<bool> > hasChangedP(D, vector<bool>(V, true));




    double energy;
    double energy_best = DBL_MAX;
    double rho = rho_min;
    int iter1_cumulated = iter1;
    double res_best_so_far = DBL_MAX;
    int counter = 0;
    double R = DBL_MAX, R_old;

    myclock::time_point begin, end;
    myclock::time_point begin2, end2;
    double T1 = 0.0, T2 = 0.0;


    double total_time;
    if(visitor.times.empty())
        total_time = 0.0;
    else
        total_time = visitor.times.back();

    size_t last_iter;
    if(visitor.iteration.empty())
        last_iter = 0;
    else
        last_iter = visitor.iteration.back();



    double r, s;

    if(verbose){
        cout<<" Done!"<<endl;
        cout<<"Start iterating..."<<endl;
    }

    begin = myclock::now();

    for(int iter = 0; iter <= MAX_ITER; iter++)
    {
        /// Step 1: Update x
        // Check which node in X has changed
        if(iter > 0){
            std::fill(hasChangedX.begin(), hasChangedX.end(), vector<bool>(V, true));
            for(size_t d = 0; d < D; d++){
                for(size_t i = 0; i < V; i++){
                    hasChangedX[d][i] = !(XD[d].col(i).isApprox(XD_old[d].col(i)));
                }
            }
        }

        XD_old = XD;


        for(size_t d = 0; d < D; d++){
            // Compute pd (cf. paper for definition)
            begin2 = myclock::now();

            hasChangedP[d] = vector<bool>(V, false);
            for(size_t c = 0; c < factors.size(); c++){
                if(factors[c].size() > d){
                    hasChangedRC[c][d] = false;
                    for(size_t idx = 0; idx < factors[c].size(); idx++){
                        if(idx != d){
                            size_t i = factors[c][idx];
                            if(hasChangedX[idx][i]){
                                hasChangedRC[c][d] = true;
                                break;
                            }
                        }
                    }

                    if(hasChangedRC[c][d]){
                        hasChangedP[d][factors[c][d]] = true;
                    }

                }
            }

            if(iter < 1)
                std::fill(hasChangedP[d].begin(), hasChangedP[d].end(), true);

            for(size_t i = 0; i < V; i++){
                if(hasChangedP[d][i])
                    P[d].col(i).setZero();
            }

            for(size_t c = 0; c < factors.size(); c++){
                if(factors[c].size() > d){
                    if(hasChangedRC[c][d]){
                        VectorXd p;
                        TensorContraction(p, factors[c], potentials[c], TensorIndicesOfAllRanks[factors[c].size() -1], XD, d);
                        RC[c].col(d) = p;
                    }
                    if(hasChangedP[d][factors[c][d]])
                        P[d].col(factors[c][d]) += RC[c].col(d);
                }
            }


            end2= myclock::now();
            T1 += measure_time(begin2, end2);


            // Solve the corresponding quadratic program
            begin2 = myclock::now();

            std::fill(hasChangedX[d].begin(), hasChangedX[d].end(), false);

            if(d == 0){
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[0][i] || hasChangedY[1][i] || hasChangedX[1][i] || hasChangedRho){
                        hasChangedX[d][i] = true;
                        VectorXd c = XD[1].col(i) - (Y[1].col(i) + P[d].col(i))/rho;
                        VectorXd x;
                        SimplexProjection(x, c);
                        XD[d].col(i) = x;
                    }
                }
            }
            if( d > 0 && d < D - 1){
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[d][i] || hasChangedY[d][i] || hasChangedY[d+1][i] || hasChangedX[d-1][i] || hasChangedX[d+1][i] ||hasChangedRho){
                        hasChangedX[d][i] = true;
                        VectorXd c = 0.5*(XD[d-1].col(i) + XD[d+1].col(i)) + (Y[d].col(i) - Y[d+1].col(i) - P[d].col(i))/(2.0*rho);
                        XD[d].col(i) = c.cwiseMax(0.0);
                    }
                }
            }
            if(d == D-1){
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[d][i] || hasChangedY[d][i] || hasChangedX[d-1][i] || hasChangedRho){
                        hasChangedX[d][i] = true;
                        VectorXd c = XD[D-2].col(i) + (Y[D-1].col(i) - P[d].col(i))/rho;
                        XD[d].col(i) = c.cwiseMax(0.0);
                    }
                }
            }


            end2 = myclock::now();
            T2 += measure_time(begin2, end2);

            // Add additional unchanged Xi to the list
            for(size_t i = 0; i < V; i++){
                if(hasChangedX[d][i])
                    hasChangedX[d][i] = !(XD[d].col(i).isApprox(XD_old[d].col(i)));
            }



        }

        /// Step 2: Update y
        for(size_t d = 1; d < D; d++){
            Y[d] += rho*(XD[d-1] - XD[d]);
        }

        for(size_t d = 1; d < D; d++){
            for(size_t i = 0; i < V; i++){
                hasChangedY[d][i] = !(XD[d-1].col(i).isApprox(XD[d].col(i)));
                //hasChangedY[d][i] = !(X[d-1].col(i) - X[d].col(i)).isMuchSmallerThan(1e-10);
            }
        }


        if(hasChangedRho)
            hasChangedRho = false;

        /// Step 3: compute the residuals and update rho
        if(iter%protocolFrequence == 0){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(iter + last_iter + 1);

            r = 0.0;
            s = 0.0;
            for(size_t d = 1; d < D; d++){
                r += (XD[d-1] - XD[d]).squaredNorm();
            }

            for(size_t d = 0; d < D; d++){
                s += (XD_old[d] - XD[d]).squaredNorm();
            }
            R_old = R;
            R = r + s;

            energy = this->energy(XD[0]);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(R);
            //visitor.bounds.push_back(-DBL_MAX);

            if(energy < energy_best){
                energy_best = energy;
                X = XD[0];
            }

            if(verbose){
                cout<<iter<<") "<<"residual = "<<R<<"\t energy = "<<energy<<endl;
            }
            /// If convergence
            if(R <= precision && R_old <= precision){
                if(verbose){
                    cout<<"CONVERGED!!!! Return the solution..."<<endl;
                }
                break;
            }

            /// If timeout reached
            if(total_time > timeout){
                if(verbose){
                    cout<<"TIMEOUT!!!! Return the solution..."<<endl;
                }
                break;
            }

            begin = myclock::now();
        }


        /// Only after iter1 iterations that we start to track the best residuals
        if(iter >= iter1_cumulated){
            if(R < res_best_so_far){
                res_best_so_far = R;
                counter = 0;
            }else{
                counter++;
            }
            // If the best_so_far residual has not changed during iter2 iterations, then update rho
            if(counter > iter2){
                if(verbose){
                    cout<<iter<<") No improvements. Update rho or stop."<<endl;
                }
                if(rho < rho_max){
                    rho = std::min(rho*step, rho_max);
                    hasChangedRho = true;
                    if(verbose){
                        cout<<"-------- Update rho = "<<rho/absMax<<endl;
                    }
                    counter = 0;
                    iter1_cumulated = iter + iter1;
                }else{
                    break;
                }
            }
        }
    }



    /// If having reached the maxium number of iterations without convergence,
    /// then return the current solution and the residuals


    if(parameters.verbose){
        cout<<"Coefficient computation time:\t "<<T1<<endl;
        cout<<"Simplex projection time:\t "<<T2<<endl;
        cout<<"Total ADMM time:\t "<<total_time<<endl;
    }

    // Convert to labels
    states.resize(V);
    for(size_t i = 0; i < V; i++){
        size_t label_max;
        X.col(i).maxCoeff( &label_max );
        //if(x_max < 1.0) cout<<x_max<<", ";
        states[i] = label_max;
    }
    return energy_best;
}


/*
void HMRF::importStereoModel(float sigma, float smoothScale, const string &proposalsFile, const string &unariesFile, const string &horizontalWeightsFile, const string &verticalWeightsFile)
/// numLabels[i]: the number of labels of the node i
/// factors
{
    double scale = 1.0;

    vector<cv::Mat> proposals, unaries;
    cout << "Reading proposals..."<<endl;
    ReadProposals_CV(proposals, proposalsFile);
    cout << "Number of proposals = "<<proposals.size()<<endl;

    cout << "Reading unaries..."<<endl;
    ReadProposals_CV(unaries, unariesFile);

    assert(proposals.size() == unaries.size());

    std::cout << "Reading CRF weights...\n";
    cv::Mat crf_weights_horizontal, crf_weights_vertical;
    int rows1, cols1;
    ReadMatrixFile_CV(horizontalWeightsFile.c_str(), crf_weights_horizontal, rows1, cols1);
    ReadMatrixFile_CV(verticalWeightsFile.c_str(), crf_weights_vertical, rows1, cols1);

    if(scale != 1.0){
        for(size_t i = 0; i < proposals.size(); i++){
            cv::resize(proposals[i], proposals[i], cv::Size(), scale, scale, CV_INTER_AREA);
            proposals[i] *= scale;
            resize(unaries[i], unaries[i], cv::Size(), scale, scale, CV_INTER_AREA);
        }
        resize(crf_weights_horizontal, crf_weights_horizontal, cv::Size(), scale, scale, CV_INTER_AREA);
        resize(crf_weights_vertical, crf_weights_vertical, cv::Size(), scale, scale, CV_INTER_AREA);
    }

    size_t cols = proposals[0].cols;
    size_t rows = proposals[0].rows;
    size_t L = proposals.size();

    //assert(rows == (size_t)rows1 && cols == (size_t)cols1);

    numberOfNodes = cols*rows;
    numberOfLabels = vector<size_t>(numberOfNodes, L);

    numberOfFactors = numberOfNodes + rows*(cols - 2) + cols*(rows - 2);

    factors.clear();
    potentials.clear();
    nodes.clear();
    factors.reserve(numberOfFactors);
    potentials.reserve(numberOfFactors);
    nodes.resize(numberOfNodes);
    max_degree = 3;

    size_t factor_idx = 0;
    // Add single-node factors and the corresponding unary terms
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            size_t n = i*cols + j;
            VS factor{ n };
            factors.push_back(factor);
            VD potential(L);
            for (size_t l = 0; l < L; ++l)
                potential[l] = unaries[l].at<float>(i,j);
            potentials.push_back(potential);
            nodes[n].push_back(n);
            factor_idx++;
        }
    }

    // For each 1x3 patch, add in a StereoClique
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols - 2; ++j) {
            // Add the factor
            VS factor{ size_t(i*cols+j), size_t(i*cols+j+1), size_t(i*cols+j+2) };
            factors.push_back(factor);

            // Add the potential tensor
            float weight = crf_weights_horizontal.at<float>(i,j+1);
            VD potential;
            potential.reserve(L*L*L);
            size_t labels[3] = { 0, 0, 0 };
            for (labels[0] = 0; labels[0] < L; ++labels[0]) {
                for (labels[1] = 0; labels[1] < L; ++labels[1]) {
                    for (labels[2] = 0; labels[2] < L; ++labels[2]) {
                        float disparity[3];
                        for (int k = 0; k < 3; ++k)
                            disparity[k] = proposals[labels[k]].at<float>(factor[k]);
                        float curvature = std::abs(disparity[0] - 2*disparity[1] + disparity[2]);
                        double energy = weight*smoothScale*std::min(curvature, sigma);
                        potential.push_back(energy);
                    }
                }
            }
            potentials.push_back(potential);
            nodes[factor[0]].push_back(factor_idx);
            nodes[factor[1]].push_back(factor_idx);
            nodes[factor[2]].push_back(factor_idx);
            factor_idx++;
        }
    }
    // For each 3x1 patch, add in a StereoClique
    for (size_t i = 0; i < rows-2; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Add the factor
            VS factor{ size_t(i*cols+j), size_t((i+1)*cols+j), size_t((i+2)*cols+j) };
            factors.push_back(factor);

            // Add the potential tensor
            float weight = crf_weights_vertical.at<float>(i+1,j);
            VD potential;
            potential.reserve(L*L*L);
            size_t labels[3] = { 0, 0, 0 };
            for (labels[0] = 0; labels[0] < L; ++labels[0]) {
                for (labels[1] = 0; labels[1] < L; ++labels[1]) {
                    for (labels[2] = 0; labels[2] < L; ++labels[2]) {
                        float disparity[3];
                        for (int k = 0; k < 3; ++k)
                            disparity[k] = proposals[labels[k]].at<float>(factor[k]);
                        float curvature = std::abs(disparity[0] - 2*disparity[1] + disparity[2]);
                        double energy = weight*smoothScale*std::min(curvature, sigma);
                        potential.push_back(energy);
                    }
                }
            }
            potentials.push_back(potential);
            nodes[factor[0]].push_back(factor_idx);
            nodes[factor[1]].push_back(factor_idx);
            nodes[factor[2]].push_back(factor_idx);
            factor_idx++;
        }
    }

    TensorIndicesOfAllRanks.clear();
    TensorIndicesOfAllRanks.resize(max_degree);
    for(size_t d = 0; d < max_degree; d++){
        getTensorIndices(TensorIndicesOfAllRanks[d], numberOfLabels[0], d+1);
    }


    X = MatrixXd::Zero(numberOfLabels[0], numberOfNodes).array() + 1.0/(double)numberOfLabels[0];
}

*/




double measure_time(myclock::time_point begin, myclock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/(double)1e6;
}

void getTensorIndices_Loop(VVS &TensorIndices, VS &currentIndex, const size_t depth, const size_t L, const size_t S)
/// currentIndex: pre-allocation S elements
// A looping function used to create the indices of the tensor of dimension L x L x ... x L (S times, i.e. rank S)
{
    if(depth < S){
        for(size_t l = 0; l < L; l++){
            currentIndex[depth] = l;
            getTensorIndices_Loop(TensorIndices, currentIndex, depth + 1, L, S);
        }
    }else{
        TensorIndices.push_back(currentIndex);
    }
}



void getTensorIndices(VVS &TensorIndices, const size_t L, const size_t S)
// Basically do this:
// for(size_t l0 = 0; l0 < L; l0++)
// for (size_t l1 = 0; l1 < L; l1++)
//    ...
//    for (size_t l(S-1) = 0; l(S-1) < L; l(S-1)++)
//    {
//        TensorIndices.push_back(l1,l2,...,l(S-1));
//    }
{
    TensorIndices.resize(0);
    VS currentIndex(S);
    getTensorIndices_Loop(TensorIndices, currentIndex, 0, L, S);
}


void TensorContraction(VectorXd &p, const VS &factor, const VD &potential, const VVS &TensorIndices, const vector<MatrixXd> &XD, const size_t d)
/// factor: composed of the nodes (i0, i1,..., i(s-1)) where s is the size of factor, s <= D the maximum size of all factors
/// potential: a vector representing the tensor F_{i0i1...i(s-1)}
/// X: a vector of assignment matrix X0, X1,..., X(D-1)
/// 0 <= d <= s-1:
/// This function compute p = F_{i0i1...i(s-1)} \otimes_0 X0.col(i0) \otimes_1 X_1.col(i1) ...
///                         \otimes_{d-1} X_{d-1}.col(i_{d-1}) \otimes_{d+1} X_{d+1}.col(i_{d+1}) ... \otimes_{s-1} X_{s-1}.col(i_{s-1})
/// Example of tensor indices (i0,i1,i2) of a potential vector for a factor of 3 nodes, each has 4 labels:
/// i0  i1  i2  potential
/// 0   0   0   0.5
/// 0   0   1   0.7
/// 0   0   2   0.7
/// 0   0   3   0.7
/// 0   1   0   0.5
/// 0   1   1   0.7
/// 0   1   2   0.7
/// 0   1   3   0.7
/// 0   2   0   0.5
/// 0   2   1   0.7
/// 0   2   2   0.7
/// 0   2   3   0.7
/// 0   3   0   0.5
/// 0   3   1   0.7
/// 0   3   2   0.7
/// 0   3   3   0.7
/// 1   0   0   0.5
/// ...etc...
/// The 'potential' vector is the column-wise replica of the above matrix
///
{
    size_t S = factor.size();
    size_t L = XD[0].rows();
    //assert(std::pow(L, S) == potential.size());
    //assert(d < S);
    p = VectorXd::Zero(L);

    if(S == 1){
        p = VectorXd::Map(potential.data(), L);
        return;
    }

    //assert(TensorIndices.size() == potential.size());

    for(size_t idx = 0; idx < potential.size(); idx++){
        // The corresponding index of the d-th column
        size_t ld = TensorIndices[idx][d];
        double prod = potential[idx];
        bool iszero = false;
        if(prod == 0)
            iszero = true;
        if(prod != 0){
            for(size_t s = 0; s < S; s++){
                if(s != d){
                    if(XD[s](TensorIndices[idx][s], factor[s]) == 0){
                        iszero = true;
                        break;
                    }
                }
            }
        }

        if(!iszero){
            for(size_t s = 0; s < S; s++){
                if(s != d){
                    prod *= XD[s](TensorIndices[idx][s], factor[s]);
                }
            }
            p[ld] += prod;
        }
    }
}




void TensorContraction(VectorXd &p, const VS &factor, const VD &potential, const VVS &TensorIndices, const MatrixXd &X, const size_t d)
/// factor: composed of the nodes (i0, i1,..., i(s-1)) where s is the size of factor, s <= D the maximum size of all factors
/// potential: a vector representing the tensor F_{i0i1...i(s-1)}
/// X: assignment matrix
/// 0 <= d <= s-1:
/// This function compute the contraction of the tensor 'factor' at all mode except at mode d (contracted by the vector X.col(i) where i != d)
///  p = F_{i0i1...i(s-1)} \otimes_0 X.col(i0) \otimes_1 X.col(i1) ...
///                         \otimes_{d-1} X.col(i_{d-1}) \otimes_{d+1} X.col(i_{d+1}) ... \otimes_{s-1} X.col(i_{s-1})
{
    size_t S = factor.size();
    size_t L = X.rows();
    assert(std::pow(L, S) == potential.size());
    assert(d < S);
    p = VectorXd::Zero(L);

    if(S == 1){
        p = VectorXd::Map(potential.data(), L);
        return;
    }

    assert(TensorIndices.size() == potential.size());

    for(size_t idx = 0; idx < potential.size(); idx++){
        // The corresponding index of the d-th column
        size_t ld = TensorIndices[idx][d];
        double prod = potential[idx];
        bool iszero = false;
        if(prod == 0)
            iszero = true;
        if(prod != 0){
            for(size_t s = 0; s < S; s++){
                if(s != d){
                    if(X(TensorIndices[idx][s], factor[s]) == 0){
                        iszero = true;
                        break;
                    }
                }
            }
        }

        if(!iszero){
            for(size_t s = 0; s < S; s++){
                if(s != d){
                    prod *= X(TensorIndices[idx][s], factor[s]);
                }
            }
            p[ld] += prod;
        }
    }
}
