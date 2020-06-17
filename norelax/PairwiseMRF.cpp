#include <iomanip>
#include "PairwiseMRF.hpp"
#include "lemmas.hpp"

using namespace std;
using namespace Eigen;

PairwiseMRF::PairwiseMRF()
{

}


PairwiseSharedMRF::PairwiseSharedMRF()
{

}


void PairwiseMRF::getUnarySolution(Eigen::MatrixXd &_X){
    _X = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    for(size_t i = 0; i < numberOfNodes; i++){
        size_t l_min;
        nodePotentials.col(i).minCoeff(&l_min);
        _X(l_min, i) = 1.0;
    }
}


double PairwiseMRF::energy()
{
    double energy = 0.0;

//    size_t L = nodePotentials.rows();
//    for(size_t i = 0; i < numberOfNodes; i++){
//        for(size_t l = 0; l < L; l++){
//            if(X(l,i) > 0 && nodePotentials(l,i) != 0)
//                energy += nodePotentials(l,i)*X(l,i);
//        }
//    }

//    for(size_t e = 0; e < numberOfEdges; e++){
//        size_t i = edges[e][0];
//        size_t j = edges[e][1];
//        for(size_t li = 0; li < L; li++){
//            for(size_t lj = 0; lj < L; lj++){
//                if(X(li,i) > 0 && X(lj,j) > 0 && edgePotentials[e](li,lj) != 0)
//                    energy += edgePotentials[e](li,lj)*X(li,i)*X(lj,j);
//            }
//        }
//    }

    energy += (X.cwiseProduct(nodePotentials)).sum();

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        double ee = X.col(i).transpose()*edgePotentials[e]*X.col(j);
        energy += ee;
    }

    return energy;
}


double PairwiseMRF::energy(const MatrixXd &_X)
{
    double energy = 0.0;

    energy += (_X.cwiseProduct(nodePotentials)).sum();

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        double ee = _X.col(i).transpose()*edgePotentials[e]*_X.col(j);
        energy += ee;
    }

    return energy;
}


double PairwiseSharedMRF::energy()
{
    double energy = 0.0;

    energy += (X.cwiseProduct(nodePotentials)).sum();

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        double ee = X.col(i).transpose()*edgeSharedPotentials*X.col(j);
        energy += edgeWeights[e]*ee;
    }

    return energy;
}


double PairwiseSharedMRF::energy(const MatrixXd &_X)
{
    double energy = 0.0;

//    size_t L = nodePotentials.rows();
//    for(size_t i = 0; i < numberOfNodes; i++){
//        for(size_t l = 0; l < L; l++){
//            if(_X(l,i) > 0 && nodePotentials(l,i) != 0)
//                energy += nodePotentials(l,i)*_X(l,i);
//        }
//    }

//    for(size_t e = 0; e < numberOfEdges; e++){
//        size_t i = edges[e][0];
//        size_t j = edges[e][1];
//        for(size_t li = 0; li < L; li++){
//            for(size_t lj = 0; lj < L; lj++){
//                if(_X(li,i) > 0 && _X(lj,j) > 0 && edgeSharedPotentials(li,lj) != 0)
//                    energy += edgeWeights[e]*edgeSharedPotentials(li,lj)*_X(li,i)*_X(lj,j);
//            }
//        }
//    }

    energy += (_X.cwiseProduct(nodePotentials)).sum();

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        double ee = _X.col(i).transpose()*edgeSharedPotentials*_X.col(j);
        energy += edgeWeights[e]*ee;
    }

    return energy;
}



double PairwiseMRF::energy_discrete()
{
    double energy = 0.0;

    for(size_t i = 0; i < numberOfNodes; i++){
        energy += nodePotentials(states[i],i);
    }

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        double ee = edgePotentials[e](states[i], states[j]);
        energy += ee;
    }

    return energy;
}



double PairwiseSharedMRF::energy_discrete()
{
    double energy = 0.0;

    for(size_t i = 0; i < numberOfNodes; i++){
        energy += nodePotentials(states[i],i);
    }

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        double ee = edgeWeights[e]*edgeSharedPotentials(states[i], states[j]);
        energy += ee;
    }

    return energy;
}



void PairwiseMRF::gradient(Eigen::MatrixXd &G, int type)
// Partial derivative of the energy over x_i (node i)
{
    G = nodePotentials;
    // Iterate over the edges that contain i
    if(type == X_TYPE_DISCRETE){
        for(size_t e = 0; e < numberOfEdges; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            G.col(i) += edgePotentials[e].col(states[j]);
            G.col(j) += edgePotentials[e].row(states[i]).transpose();
        }
    } else if(type == X_TYPE_SPARSE){
        // TODO: Implement and check if SPARSE version is faster than DENSE version
        for(size_t e = 0; e < numberOfEdges; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            G.col(i) += edgePotentials[e]*X.col(j);
            G.col(j) += edgePotentials[e].transpose()*X.col(i);
        }
    } else{
        for(size_t e = 0; e < numberOfEdges; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            G.col(i) += edgePotentials[e]*X.col(j);
            G.col(j) += edgePotentials[e].transpose()*X.col(i);
        }
    }

}


void PairwiseSharedMRF::gradient(Eigen::MatrixXd &G, int type)
// Partial derivative of the energy over x_i (node i)
{
    G = nodePotentials;
    // Iterate over the edges that contain i    
    if(type == X_TYPE_DISCRETE){
        for(size_t e = 0; e < numberOfEdges; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            G.col(i) += edgeWeights[e]*edgeSharedPotentials.col(states[j]);
            G.col(j) += edgeWeights[e]*edgeSharedPotentials.col(states[i]);
        }
    } else if(type == X_TYPE_SPARSE){
        // TODO: Implement and check if SPARSE version is faster than DENSE version
        MatrixXd H = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
        for(size_t e = 0; e < numberOfEdges; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            H.col(i) += edgeWeights[e]*X.col(j);
            H.col(j) += edgeWeights[e]*X.col(i);
        }
        G += edgeSharedPotentials*H;
    } else{
        MatrixXd H = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
        for(size_t e = 0; e < numberOfEdges; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            H.col(i) += edgeWeights[e]*X.col(j);
            H.col(j) += edgeWeights[e]*X.col(i);
        }
        G += edgeSharedPotentials*H;
    }
}



void PairwiseMRF::gradient(MatrixXd &G, size_t i, int type)
// Partial derivative of the energy over x_i (node i)
{
    VectorXd Gi = nodePotentials.col(i);
    // Iterate over the edges that contain i
    // IMPORTANT: iterating over containing_edges is much faster than over edges
    if(type == X_TYPE_DISCRETE){
        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
            size_t e = containing_edges[i][idx]; // Index of the edge
            if(edges[e][0] == i){
                Gi += edgePotentials[e].col(states[edges[e][1]]);
            }else{
                Gi += edgePotentials[e].row(states[edges[e][0]]).transpose();
            }
        }
    } else if(type == X_TYPE_SPARSE){
        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
            size_t e = containing_edges[i][idx]; // Index of the edge
            if(edges[e][0] == i){
                // Gi += edgePotentials[e]*X.col(edges[e][1]);
                size_t j = edges[e][1];
                for(size_t li = 0; li < numberOfLabels[i]; li++){
                    for(size_t lj = 0; lj < numberOfLabels[j]; lj++){
                        if(edgePotentials[e](li,lj) != 0 && X(lj, j) != 0)
                            Gi(li) += edgePotentials[e](li,lj)*X(lj, j);
                    }
                }
            }else{
                // Gi += edgePotentials[e].transpose()*X.col(edges[e][0]);
                size_t j = edges[e][0];
                for(size_t li = 0; li < numberOfLabels[i]; li++){
                    for(size_t lj = 0; lj < numberOfLabels[j]; lj++){
                        if(edgePotentials[e](lj,li) != 0 && X(lj, j) != 0)
                            Gi(li) += edgePotentials[e](lj,li)*X(lj, j);
                    }
                }
            }
        }
    } else{ // type == X_TYPE_DENSE
        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
            size_t e = containing_edges[i][idx]; // Index of the edge
            if(edges[e][0] == i){
                Gi += edgePotentials[e]*X.col(edges[e][1]);
            }else{
                Gi += edgePotentials[e].transpose()*X.col(edges[e][0]);
            }
        }
    }
    G.col(i) = Gi;
}



void PairwiseSharedMRF::gradient(MatrixXd &G, size_t i, int type)
// Partial derivative of the energy over x_i (node i)
{
    VectorXd Gi = nodePotentials.col(i);
    // Iterate over the edges that contain i
    if(type == X_TYPE_DISCRETE){
        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
            size_t e = containing_edges[i][idx]; // Index of the edge
            if(edges[e][0] == i){
                Gi += edgeWeights[e]*edgeSharedPotentials.col(states[edges[e][1]]);
            }else{
                Gi += edgeWeights[e]*edgeSharedPotentials.col(states[edges[e][0]]);
            }
        }
    } else if(type == X_TYPE_SPARSE){
//        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
//            size_t e = containing_edges[i][idx]; // Index of the edge
//            VectorXd ge = VectorXd::Zero(numberOfLabels[i]);
//            if(edges[e][0] == i){
//                //Gi += edgeWeights[e]*edgeSharedPotentials*X.col(edges[e][1]);
//                size_t j = edges[e][1];
//                for(size_t li = 0; li < numberOfLabels[i]; li++){
//                    for(size_t lj = 0; lj < numberOfLabels[j]; lj++){
//                        if(edgeSharedPotentials(li,lj) != 0 && X(lj, j) != 0)
//                            ge(li) += edgeSharedPotentials(li,lj)*X(lj, j);
//                    }
//                }
//            }else{
//                //Gi += edgeWeights[e]*edgeSharedPotentials*X.col(edges[e][0]);
//                size_t j = edges[e][0];
//                for(size_t li = 0; li < numberOfLabels[i]; li++){
//                    for(size_t lj = 0; lj < numberOfLabels[j]; lj++){
//                        if(edgeSharedPotentials(lj,li) != 0 && X(lj, j) != 0)
//                            ge(li) += edgeSharedPotentials(lj,li)*X(lj, j);
//                    }
//                }
//            }
//            ge *= edgeWeights[e];
//            Gi += ge;
//        }
        VectorXd Hi = VectorXd::Zero(numberOfLabels[i]);
        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
            size_t e = containing_edges[i][idx]; // Index of the edge
            if(edges[e][0] == i){
                Hi += edgeWeights[e]*X.col(edges[e][1]);
            }else{
                Hi += edgeWeights[e]*X.col(edges[e][0]);
            }
        }
        //Gi += edgeSharedPotentials*Hi;
        for(size_t li = 0; li < numberOfLabels[i]; li++){
            for(size_t lj = 0; lj < numberOfLabels[i]; lj++){
                if(edgeSharedPotentials(li,lj) != 0 && Hi(lj) != 0)
                    Gi(li) += edgeSharedPotentials(li,lj)*Hi(lj);
            }
        }

    } else{
        VectorXd Hi = VectorXd::Zero(numberOfLabels[i]);
        for(size_t idx = 0; idx < containing_edges[i].size(); idx++){
            size_t e = containing_edges[i][idx]; // Index of the edge
            if(edges[e][0] == i){
                Hi += edgeWeights[e]*X.col(edges[e][1]);
            }else{
                Hi += edgeWeights[e]*X.col(edges[e][0]);
            }
        }
        Gi += edgeSharedPotentials*Hi;
    }

    G.col(i) = Gi;
}



void PairwiseMRF::lineSearch(double &alpha_best, double &p_best, double energy, const MatrixXd &R)
// Line search for Projected Gradient Descent and Frank-Wolfe algorithm
// Find alpha such that E(X + alpha*D) is minimum where 0 <= alpha <= 1. Return the value of E at this minimum to p_best.
// E(X + alpha*D) is a polynomial of degree n where n is the degree of the MRF
// For higher-order MRFs, this polynomial can be determined by evaluating E(X + alpha*D) at n different
// values of alpha.
// For pairwise MRFs, this polynomial is quaratic where the coefficients and the minimum are given by closed formed formulas.
{
    // Exact line search
    // E(X + alpha*D) = A*alpha^2 + B*alpha + C
    double A = 0.0;
    double B = 0.0;
    double C = energy;

    B += (R.cwiseProduct(nodePotentials)).sum();

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        VectorXd P = R.col(i).transpose()*edgePotentials[e];
        double a = P.dot(R.col(j));
        A += a;
        double b = X.col(i).transpose()*edgePotentials[e]*R.col(j) + P.dot(X.col(j));
        B += b;
    }

    double p0 = C;
    double p1 = A + B + C;
    p_best = min(p0,p1);
    alpha_best = (p0 < p1)?0.0:1.0;

    if(A > 0){
        double alpha = -B/(2.0*A);
        if(alpha > 0 && alpha < 1){
            p_best = -B*B/(4.0*A) + C;
            alpha_best = alpha;
        }
    }
}



void PairwiseSharedMRF::lineSearch(double &alpha_best, double &p_best, double energy, const MatrixXd &R)
// Line search for Projected Gradient Descent and Frank-Wolfe algorithm
// Find alpha such that E(X + alpha*D) is minimum where 0 <= alpha <= 1. Return the value of E at this minimum to p_best.
// E(X + alpha*D) is a polynomial of degree n where n is the degree of the MRF
// For higher-order MRFs, this polynomial can be determined by evaluating E(X + alpha*D) at n different
// values of alpha.
// For pairwise MRFs, this polynomial is quaratic where the coefficients and the minimum are given by closed formed formulas.
{
    // Exact line search
    // E(X + alpha*D) = A*alpha^2 + B*alpha + C
    double A = 0.0;
    double B = 0.0;
    double C = energy;

    B += (R.cwiseProduct(nodePotentials)).sum();

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        VectorXd P = R.col(i).transpose()*edgeSharedPotentials;
        double a = P.dot(R.col(j));
        A += a*edgeWeights[e];
        double b = X.col(i).transpose()*edgeWeights[e]*edgeSharedPotentials*R.col(j) + P.dot(X.col(j));
        B += b;
    }

    double p0 = C;
    double p1 = A + B + C;
    p_best = min(p0,p1);
    alpha_best = (p0 < p1)?0.0:1.0;

    if(A > 0){
        double alpha = -B/(2.0*A);
        if(alpha > 0 && alpha < 1){
            p_best = -B*B/(4.0*A) + C;
            alpha_best = alpha;
        }
    }
}



double PairwiseMRF::CQP()
// Solving the Convex QP relaxation using Frank-Wolfe algorithm
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
    //int num_step_before_line_search = 20; // Run a few iterations with diminishing step size before doing line search

    // Step size
    double alpha;
    double energy_best = energy;
    double alpha_best;

    if(parameters.verbose){
        cout<<"Initial energy = "<<energy<<". Starting Frank-Wolfe algorithm for convex QP relaxation..."<<endl;
    }

    // Construct the convex relaxation
    MatrixXd D = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        MatrixXd E = 0.5*edgePotentials[e].cwiseAbs();
        D.col(i) += E.rowwise().sum();
        D.col(j) += E.transpose().rowwise().sum();
    }
    MatrixXd DX = D.cwiseProduct(X);
    double energy_CQP = energy - DX.sum() + (DX.cwiseProduct(X)).sum();

    MatrixXd G;
    MatrixXd R, X_temp, X_best = X;
    for(int k = 0; k < parameters.MAX_ITER; k++)
    {
        DX = D.cwiseProduct(X);
        double eplus = (-DX + X.cwiseProduct(DX)).sum();
        // FIXME: the returned energy_best seems to be incorrect.
//        if(!parameters.lineSearch){
//            energy_CQP = this->energy() + eplus;
//        }
        energy_CQP = this->energy() + eplus;
        energy = energy_CQP - eplus;

        // Step 1: compute min_{S \in \cX} <S, G>
        this->gradient(G, X_TYPE_DENSE);
        G += (-D + DX*2.0);
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
                    cout<<k+1<<")\tGap: "<<setw(10)<<fwGap<<"\tCQP energy: "<<setw(10)<<energy_CQP<<"\tMRF energy: "<<setw(10)<<energy<<endl;

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
                cout<<k+1<<")\tGap: "<<setw(10)<<fwGap<<"\tCQP energy: "<<setw(10)<<energy_CQP<<"\tMRF energy: "<<setw(10)<<energy<<endl;

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


        //if(k >= num_step_before_line_search && parameters.lineSearch){
        if(parameters.lineSearch){
            //this->lineSearchCQP(alpha_best, energy_best, energy, R);
            // Line search
            // Exact line search
            // E(X + alpha*D) = A*alpha^2 + B*alpha + C
            double A = 0.0;
            double B = 0.0;
            double C = energy_CQP;

            B += (R.cwiseProduct(nodePotentials)).sum();

            for(size_t e = 0; e < numberOfEdges; e++){
                size_t i = edges[e][0];
                size_t j = edges[e][1];
                VectorXd P = R.col(i).transpose()*edgePotentials[e];
                double a = P.dot(R.col(j));
                A += a;
                double b = X.col(i).transpose()*edgePotentials[e]*R.col(j) + P.dot(X.col(j));
                B += b;
            }

            // The above A, B, C is for the original energy. Now add terms for the convex relaxation.
            MatrixXd DR = D.cwiseProduct(R);

            A += (R.cwiseProduct(DR)).sum();
            B += (-DR + X.cwiseProduct(DR)*2.0).sum();

            double p0 = C;
            double p1 = A + B + C;
            energy_best = min(p0,p1);
            alpha_best = (p0 < p1)?0.0:1.0;

            if(A > 0){
                double alpha = -B/(2.0*A);
                if(alpha > 0 && alpha < 1){
                    energy_best = -B*B/(4.0*A) + C;
                    alpha_best = alpha;
                }
            }

            X += alpha_best*R;
            // FIXME: the returned energy_best seems to be incorrect.
            energy_CQP = energy_best;
        } else{
            alpha = 2.0/(double)(k + 2.0);
            X += alpha*R;
        }

    }
    return energy;
}



double PairwiseSharedMRF::CQP()
// Solving the Convex QP relaxation using Frank-Wolfe algorithm
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
    //int num_step_before_line_search = 20; // Run a few iterations with diminishing step size before doing line search

    // Step size
    double alpha;
    double energy_best = energy;
    double alpha_best;

    if(parameters.verbose){
        cout<<"Initial energy = "<<energy<<". Starting Frank-Wolfe algorithm for convex QP relaxation..."<<endl;
    }

    // Construct the convex relaxation
    MatrixXd D = MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        MatrixXd E = 0.5*edgeWeights[e]*edgeSharedPotentials.cwiseAbs();
        D.col(i) += E.rowwise().sum();
        D.col(j) += E.transpose().rowwise().sum();
    }
    MatrixXd DX = D.cwiseProduct(X);
    double energy_CQP = energy - DX.sum() + (DX.cwiseProduct(X)).sum();

    MatrixXd G;
    MatrixXd R, X_temp, X_best = X;
    for(int k = 0; k < parameters.MAX_ITER; k++)
    {
        DX = D.cwiseProduct(X);
        double eplus = (-DX + X.cwiseProduct(DX)).sum();
//        if(!parameters.lineSearch){
//            energy_CQP = this->energy() + eplus;
//        }
        energy_CQP = this->energy() + eplus;
        energy = energy_CQP - eplus;

        // Step 1: compute min_{S \in \cX} <S, G>
        this->gradient(G, X_TYPE_DENSE);
        G += (-D + DX*2.0);
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
                    cout<<k+1<<")\tGap: "<<setw(10)<<fwGap<<"\tCQP energy: "<<setw(10)<<energy_CQP<<"\tMRF energy: "<<setw(10)<<energy<<endl;

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
                cout<<k+1<<")\tGap: "<<setw(10)<<fwGap<<"\tCQP energy: "<<setw(10)<<energy_CQP<<"\tMRF energy: "<<setw(10)<<energy<<endl;

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


        //if(k >= num_step_before_line_search && parameters.lineSearch){
        if(parameters.lineSearch){
            //this->lineSearchCQP(alpha_best, energy_best, energy, R);
            // Line search
            // Exact line search
            // E(X + alpha*D) = A*alpha^2 + B*alpha + C
            double A = 0.0;
            double B = 0.0;
            double C = energy_CQP;

            B += (R.cwiseProduct(nodePotentials)).sum();

            for(size_t e = 0; e < numberOfEdges; e++){
                size_t i = edges[e][0];
                size_t j = edges[e][1];
                VectorXd P = edgeWeights[e]*R.col(i).transpose()*edgeSharedPotentials;
                double a = P.dot(R.col(j));
                A += a;
                double b = edgeWeights[e]*X.col(i).transpose()*edgeSharedPotentials*R.col(j) + P.dot(X.col(j));
                B += b;
            }

            // The above A, B, C is for the original energy. Now add terms for the convex relaxation.
            MatrixXd DR = D.cwiseProduct(R);

            A += (R.cwiseProduct(DR)).sum();
            B += (-DR + X.cwiseProduct(DR)*2.0).sum();

            double p0 = C;
            double p1 = A + B + C;
            energy_best = min(p0,p1);
            alpha_best = (p0 < p1)?0.0:1.0;

            if(A > 0){
                double alpha = -B/(2.0*A);
                if(alpha > 0 && alpha < 1){
                    energy_best = -B*B/(4.0*A) + C;
                    alpha_best = alpha;
                }
            }

            X += alpha_best*R;
            energy_CQP = energy_best;
        } else{
            alpha = 2.0/(double)(k + 2.0);
            X += alpha*R;
        }

    }
    return energy;
}




void PairwiseMRF::
computeP(size_t d, const vector<MatrixXd> &XD, const std::vector<size_t> &random_node_indices, const std::vector<size_t> &random_edge_indices)
{
    for(size_t i = 0; i < numberOfNodes; i++){
        if(hasChangedP[d][i])
            P[d].col(i).setZero();
    }

    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        if(hasChangedRCa[e][d]){
            RCa[e].col(d) = 0.5*edgePotentials[e]*XD[1-d].col(j);
        }
        if(hasChangedRCb[e][d]){
            RCb[e].col(d) = 0.5*edgePotentials[e].transpose()*XD[1-d].col(i);
        }

        if(hasChangedP[d][i])
            P[d].col(i) += RCa[e].col(d);
        if(hasChangedP[d][j])
            P[d].col(j) += RCb[e].col(d);
    }
}


// OLD CODE: no parallelization
//void PairwiseSharedMRF::computeP(size_t d, const vector<MatrixXd> &XD)
//{
//    for(size_t i = 0; i < numberOfNodes; i++){
//        if(hasChangedP[d][i])
//            P[d].col(i).setZero();
//    }

//    for(size_t e = 0; e < numberOfEdges; e++){
//        size_t i = edges[e][0];
//        size_t j = edges[e][1];
//        if(hasChangedRCa[e][d]){
//            RCa[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(j);
//        }
//        if(hasChangedRCb[e][d]){
//            RCb[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(i);
//        }

//        if(hasChangedP[d][i])
//            P[d].col(i) += RCa[e].col(d);
//        if(hasChangedP[d][j])
//            P[d].col(j) += RCb[e].col(d);
//    }

//    for(size_t i = 0; i < numberOfNodes; i++){
//        if(hasChangedP[d][i])
//            P[d].col(i) = edgeSharedPotentials*P[d].col(i);
//    }

//}



//void PairwiseSharedMRF::computeP(size_t d, const vector<MatrixXd> &XD)
//{
//    #pragma omp parallel for
//    for(size_t e = 0; e < numberOfEdges; e++){
//        size_t i = edges[e][0];
//        size_t j = edges[e][1];
//        if(hasChangedRCa[e][d]){
//            RCa[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(j);
//        }
//        if(hasChangedRCb[e][d]){
//            RCb[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(i);
//        }
//    }

//    #pragma omp parallel for
//    for(size_t i = 0; i < numberOfNodes; i++){
//        if(hasChangedP[d][i]){
//            P[d].col(i).setZero();
//            for(size_t edx = 0; edx < containing_tails[i].size(); edx++){
//                size_t e = containing_tails[i][edx];
//                P[d].col(i) += RCa[e].col(d);
//            }
//            for(size_t edx = 0; edx < containing_heads[i].size(); edx++){
//                size_t e = containing_heads[i][edx];
//                P[d].col(i) += RCb[e].col(d);
//            }
//            P[d].col(i) = edgeSharedPotentials*P[d].col(i);
//        }
//    }

//}



//void PairwiseSharedMRF::computeP(size_t d, const vector<MatrixXd> &XD)
//{
//    size_t threads = 8;
//    size_t p = floor(numberOfEdges/threads);

//    #pragma omp parallel for
//    for(size_t b = 0; b < threads; b++){
//        size_t first = b*p;
//        size_t last = (b+1)*p - 1;
//        if(b >= threads - 1){
//            last = numberOfEdges - 1;
//        }
//        for(size_t e = first; e <= last; e++){
//            size_t i = edges[e][0];
//            size_t j = edges[e][1];
//            if(hasChangedRCa[e][d]){
//                RCa[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(j);
//            }
//            if(hasChangedRCb[e][d]){
//                RCb[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(i);
//            }
//        }
//    }

//    size_t q = floor(numberOfNodes/threads);
//    #pragma omp parallel for
//    for(size_t b = 0; b < threads; b++){
//        size_t first = b*q;
//        size_t last = (b+1)*q - 1;
//        if(b >= threads - 1){
//            last = numberOfNodes - 1;
//        }
//        for(size_t i = first; i <= last; i++){
//            if(hasChangedP[d][i]){
//                P[d].col(i).setZero();
//                for(size_t edx = 0; edx < containing_tails[i].size(); edx++){
//                    size_t e = containing_tails[i][edx];
//                    P[d].col(i) += RCa[e].col(d);
//                }
//                for(size_t edx = 0; edx < containing_heads[i].size(); edx++){
//                    size_t e = containing_heads[i][edx];
//                    P[d].col(i) += RCb[e].col(d);
//                }
//                P[d].col(i) = edgeSharedPotentials*P[d].col(i);
//            }
//        }
//    }

//}




void PairwiseSharedMRF::computeP(size_t d, const vector<MatrixXd> &XD, const std::vector<size_t> &random_node_indices, const std::vector<size_t> &random_edge_indices)
{
    #pragma omp parallel for
    for(size_t e = 0; e < numberOfEdges; e++){
    //for(size_t edx = 0; edx < numberOfEdges; edx++){
    //    size_t e = random_edge_indices[edx];
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        if(hasChangedRCa[e][d]){
            RCa[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(j);
        }
        if(hasChangedRCb[e][d]){
            RCb[e].col(d) = (0.5*edgeWeights[e])*XD[1-d].col(i);
        }
    }

    #pragma omp parallel for
    for(size_t i = 0; i < numberOfNodes; i++){
    //for(size_t idx = 0; idx < numberOfNodes; idx++){
    //    size_t i = random_node_indices[idx];
        if(hasChangedP[d][i]){
            P[d].col(i).setZero();
            for(size_t edx = 0; edx < containing_tails[i].size(); edx++){
                size_t e = containing_tails[i][edx];
                P[d].col(i) += RCa[e].col(d);
            }
            for(size_t edx = 0; edx < containing_heads[i].size(); edx++){
                size_t e = containing_heads[i][edx];
                P[d].col(i) += RCb[e].col(d);
            }
            P[d].col(i) = edgeSharedPotentials*P[d].col(i);
        }
    }

}



double PairwiseMRF::ADMM()
{
    if(parameters.verbose){
        cout<<"**********************************************************"<<endl;
        cout<<"Number of nodes: "<<numberOfNodes<<endl;
        cout<<"Number of edges: "<<numberOfEdges<<endl;
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
//    double absmax = nodePotentials.cwiseAbs().maxCoeff();
//    for(size_t e = 0; e < numberOfEdges; e++){
//        double cmax = edgePotentials[e].cwiseAbs().maxCoeff();
//        if(cmax > absmax)
//            absmax = cmax;
//    }

    double rho_min = absMax*parameters.rho_min;
    double rho_max = absMax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t E = numberOfEdges;
    size_t L = numberOfLabels[0];


    for(size_t i = 1; i < V; i++){
        if(L != numberOfLabels[i]){
            cerr<<"The current version supports only equal number of labels!"<<endl;
        }
        assert(L == numberOfLabels[i]);
    }

    size_t D = max_degree;

    /// Initialization
    vector<MatrixXd> XD(D, X);
    vector<MatrixXd> XD_old;

    MatrixXd Y = MatrixXd::Zero(L,V);

    vector< vector<bool> > hasChangedX(D, vector<bool>(V, true));

    vector<bool> hasChangedY(V, true);
    bool hasChangedRho = true;

    P = vector<MatrixXd>(D, MatrixXd::Zero(L,V));
    RCa = std::vector<Eigen::MatrixXd>(numberOfEdges, MatrixXd::Zero(L, 2));
    RCb = std::vector<Eigen::MatrixXd>(numberOfEdges, MatrixXd::Zero(L, 2));

    hasChangedP = vector< vector<bool> >(D, vector<bool>(V, true));
    hasChangedRCa = std::vector< std::vector<bool> >(numberOfEdges,  vector<bool>(2, true));
    hasChangedRCb = std::vector< std::vector<bool> >(numberOfEdges,  vector<bool>(2, true));


    auto engine = std::default_random_engine{};
    std::vector<size_t> random_node_indices(numberOfNodes);
    std::iota(random_node_indices.begin(), random_node_indices.end(), 0);
    std::shuffle(std::begin(random_node_indices), std::end(random_node_indices), engine);

    std::vector<size_t> random_edge_indices(numberOfEdges);
    std::iota(random_edge_indices.begin(), random_edge_indices.end(), 0);
    std::shuffle(std::begin(random_edge_indices), std::end(random_edge_indices), engine);


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

    begin = myclock::now();


    double r, s;
    for(int k = 0; k <= parameters.MAX_ITER; k++)
    {

        /// Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        /// c1 = x2 - (d + M*x2 + y)/rho
        ///

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

            std::fill(hasChangedP[d].begin(), hasChangedP[d].end(), false);
            for(size_t e = 0; e < E; e++){
                size_t i = edges[e][0];
                size_t j = edges[e][1];
                hasChangedRCa[e][d] = false;
                hasChangedRCb[e][d] = false;
                if(hasChangedX[1-d][j]){
                    hasChangedRCa[e][d] = true;
                    hasChangedP[d][i] = true;
                }
                if(hasChangedX[1-d][i]){
                    hasChangedRCb[e][d] = true;
                    hasChangedP[d][j] = true;
                }
            }

            /*if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
                int rca = 0;
                int rcb = 0;
                int pp = 0;
                for(size_t e = 0; e < E; e++){
                    if(hasChangedRCa[e][d])
                        rca++;
                    if(hasChangedRCb[e][d])
                        rcb++;
                }
                for(size_t i = 0; i < V; i++){
                    if(hasChangedP[d][i])
                        pp++;
                }

                cout<<"Changed RCa"<<d<<": "<<rca<<"/"<<E<<endl;
                cout<<"Changed RCb"<<d<<": "<<rcb<<"/"<<E<<endl;
                cout<<"Changed Pa"<<d<<": "<<pp<<"/"<<V<<endl;
            }*/


            if(k < 1)
                std::fill(hasChangedP[d].begin(), hasChangedP[d].end(), true);

            // Compute P[D]
            //computeP(d, XD);
            computeP(d, XD, random_node_indices, random_edge_indices);


//            for(size_t e = 0; e < E; e++){
//                size_t i = edges[e][0];
//                size_t j = edges[e][1];
//                if(hasChangedP[d][i])
//                    P[d].col(i) += 0.5*edgePotentials[e]*XD[1-d].col(j);
//                if(hasChangedP[d][j])
//                    P[d].col(j) += 0.5*edgePotentials[e].transpose()*XD[1-d].col(i);
//            }


            end2= myclock::now();
            T1 += measure_time(begin2, end2);

            // Solve the corresponding quadratic program
            begin2 = myclock::now();

            std::fill(hasChangedX[d].begin(), hasChangedX[d].end(), false);

            size_t threads = 8;
            size_t q = floor(numberOfNodes/threads);
            if(d == 0){
                #pragma omp parallel for
                for(size_t b = 0; b < threads; b++){
                    size_t first = b*q;
                    size_t last = (b+1)*q - 1;
                    if(b >= threads - 1){
                        last = numberOfNodes - 1;
                    }
                    for(size_t i = first; i <= last; i++){
                        if(hasChangedP[0][i] || hasChangedY[i] || hasChangedX[1][i] || hasChangedRho){
                            hasChangedX[d][i] = true;
                            VectorXd c = XD[1].col(i) - (nodePotentials.col(i) + Y.col(i) + P[d].col(i))/rho;
                            VectorXd x;

                            //SimplexProjection(x, c);
                            //XD[d].col(i) = x;

                            // Bregman (KL)
                            c = c.array() - c.maxCoeff();
                            c = c.array().exp();
                            x = XD[1].col(i).cwiseProduct(c);
                            x = x/x.sum();
                            XD[d].col(i) = x.cwiseMax(1e-10);
                        }
                    }
                }
            }else{
                #pragma omp parallel for
                for(size_t b = 0; b < threads; b++){
                    size_t first = b*q;
                    size_t last = (b+1)*q - 1;
                    if(b >= threads - 1){
                        last = numberOfNodes - 1;
                    }
                    for(size_t i = first; i <= last; i++){
                        if(hasChangedP[d][i] || hasChangedY[i] || hasChangedX[d-1][i] || hasChangedRho){
                            hasChangedX[d][i] = true;
                            VectorXd c = XD[0].col(i) + (Y.col(i) - P[d].col(i))/rho;
                            XD[d].col(i) = c.cwiseMax(0.0);
                            //XD[d].col(i) = c;
                        }
                    }
                }
            }


            end2 = myclock::now();
            T2 += measure_time(begin2, end2);

            // Add additional unchanged Xi to the list
            //#pragma omp parallel for
            for(size_t i = 0; i < V; i++){
                if(hasChangedX[d][i])
                    hasChangedX[d][i] = !(XD[d].col(i).isApprox(XD_old[d].col(i)));
            }
        }


        /// Step 3: update y
        //if(iter >= 10)
        Y += rho*(XD[0] - XD[1]);
        //#pragma omp parallel for
        for(size_t i = 0; i < V; i++){
            hasChangedY[i] = !(XD[0].col(i).isApprox(XD[1].col(i)));
            //hasChangedY[i] = !(X.col(i) - Z.col(i)).isMuchSmallerThan(1e-10);
        }


        if(hasChangedRho)
            hasChangedRho = false;

        /// Step 4: compute the residuals and update rho

        if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(k + last_iter + 1);

            r = (XD[0] - XD[1]).squaredNorm();
            s = (XD[0] - XD_old[0]).squaredNorm() + (XD[1] - XD_old[1]).squaredNorm();
            residual_old = residual;
            residual = r + s;

            energy =  this->energy(XD[0]);
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
            if(residual <= parameters.precision && residual_old <= parameters.precision)
                break;


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




double PairwiseMRF::ADMM_old()
{
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
//    double absmax = nodePotentials.cwiseAbs().maxCoeff();
//    for(size_t e = 0; e < numberOfEdges; e++){
//        double cmax = edgePotentials[e].cwiseAbs().maxCoeff();
//        if(cmax > absmax)
//            absmax = cmax;
//    }

    double rho_min = absMax*parameters.rho_min;
    double rho_max = absMax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t E = numberOfEdges;
    size_t L = numberOfLabels[0];

    for(size_t i = 1; i < V; i++){
        if(L != numberOfLabels[i]){
            cerr<<"The current version supports only equal number of labels!"<<endl;
        }
        assert(L == numberOfLabels[i]);
    }


    /// Initialization
    MatrixXd X1 = X;
    MatrixXd X2 = X;

    MatrixXd Y = MatrixXd::Zero(L,V);

    vector<bool> hasChangedX1(V, true);
    vector<bool> hasChangedX2(V, true);
    vector<bool> hasChangedY(V, true);
    vector<bool> hasChangedP1(V, true);
    vector<bool> hasChangedP2(V, true);
    bool hasChangedRho = true;



    MatrixXd A1, A2, X1_old, X2_old, P1, P2;
    A1 = MatrixXd::Zero(L,V);
    A2 = MatrixXd::Zero(L,V);
    P1 = MatrixXd::Zero(L,V);
    P2 = MatrixXd::Zero(L,V);


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

    begin = myclock::now();


    double r, s;
    for(int k = 0; k <= parameters.MAX_ITER; k++)
    {

        /// Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        /// c1 = x2 - (d + M*x2 + y)/rho
        X1_old = X1;

        begin2 = myclock::now();

        std::fill(hasChangedP1.begin(), hasChangedP1.end(), false);
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedX2[j])
                hasChangedP1[i] = true;
            if(hasChangedX2[i])
                hasChangedP1[j] = true;
        }
        std::fill(hasChangedX1.begin(), hasChangedX1.end(), false);
        for(size_t i = 0; i < V; i++){
            if(hasChangedP1[i] || hasChangedY[i] || hasChangedX2[i] || hasChangedRho)
                hasChangedX1[i] = true;
        }


        // Compute P
        for(size_t i = 0; i < V; i++){
            if(hasChangedP1[i])
                P1.col(i).setZero();
        }
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedP1[i])
                P1.col(i) += 0.5*edgePotentials[e]*X2.col(j);
            if(hasChangedP1[j])
                P1.col(j) += 0.5*edgePotentials[e].transpose()*X2.col(i);
        }

        // Compute A = Z - (D + Y + C*P)/rho
        for(size_t i = 0; i < V; i++){
            if(hasChangedX1[i])
                A1.col(i) = X2.col(i) - (nodePotentials.col(i) + Y.col(i) + P1.col(i))/rho;
        }
        //A = Z - (D + Y + C*P2)/rho;
        //A = A*rho/(rho + 2.0*lambda);
        end2 = myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
        for(size_t i = 0; i < V; i++){
            if(hasChangedX1[i]){
                VectorXd x;
                SimplexProjection(x, A1.col(i));
                X1.col(i) = x;
            }
        }
        // Add additional unchanged Xi to the list
        for(size_t i = 0; i < V; i++){
            if(hasChangedX1[i])
                hasChangedX1[i] = !(X1.col(i).isApprox(X1_old.col(i)));
        }
        end2= myclock::now();
        T2 += measure_time(begin2, end2);


        /// Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        /// c2 = x1 - (M^T*x1 - y)/rho
        X2_old = X2;

        begin2 = myclock::now();

        std::fill(hasChangedP2.begin(), hasChangedP2.end(), false);
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedX1[i])
                hasChangedP2[j] = true;
            if(hasChangedX1[j])
                hasChangedP2[i] = true;
        }

        /// IMPORTANT: COMPUTE THE INITIAL Q
        if(k < 1)
            std::fill(hasChangedP2.begin(), hasChangedP2.end(), true);

        std::fill(hasChangedX2.begin(), hasChangedX2.end(), false);
        for(size_t i = 0; i < V; i++){
            if(hasChangedP2[i] || hasChangedY[i] || hasChangedX1[i] || hasChangedRho)
                hasChangedX2[i] = true;
        }

        // Compute Q
        for(size_t i = 0; i < V; i++){
            if(hasChangedP2[i])
                P2.col(i).setZero();
        }
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedP2[j])
                P2.col(j) += 0.5*edgePotentials[e].transpose()*X1.col(i);
            if(hasChangedP2[i])
                P2.col(i) += 0.5*edgePotentials[e]*X1.col(j);
        }

        // Compute B = X - (C^T*Q - Y)/rho
        for(size_t i = 0; i < V; i++){
            if(hasChangedX2[i])
                A2.col(i) = X1.col(i) - (P2.col(i) - Y.col(i))/rho;
        }
        //B = X - (C*Q2 - Y)/rho;
        end2 = myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
//        for(size_t i = 0; i < V; i++){
//            VectorXd z;
//            sublemma_EQUALITY_EIGEN(z, B.col(i));
//            Z.col(i) = z;
//        }
        X2 = A2.cwiseMax(0.0);
        end2 = myclock::now();
        T2 += measure_time(begin2, end2);

        // Add additional unchanged Zi to the list
        for(size_t i = 0; i < V; i++){
            if(hasChangedX2[i])
                hasChangedX2[i] = !(X2.col(i).isApprox(X2_old.col(i)));
        }



        /// Step 3: update y
        //if(iter >= 10)
        Y += rho*(X1 - X2);

        for(size_t i = 0; i < V; i++){
            hasChangedY[i] = !(X1.col(i).isApprox(X2.col(i)));
            //hasChangedY[i] = !(X.col(i) - Z.col(i)).isMuchSmallerThan(1e-10);
        }


        if(hasChangedRho)
            hasChangedRho = false;

        /// Step 4: compute the residuals and update rho

        if(k%parameters.protocolFrequence == 0 || k >= parameters.MAX_ITER - 1){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(k + last_iter + 1);

            r = (X1 - X2).squaredNorm();
            s = (X1 - X1_old).squaredNorm() + (X2 - X2_old).squaredNorm();
            residual_old = residual;
            residual = r + s;

            energy =  this->energy(X1);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(residual);
            //visitor.bounds.push_back(-DBL_MAX);

            if(energy < energy_best){
                energy_best = energy;
                X = X1;
            }

            if(parameters.verbose)
                cout<<k+1<<")\tresidual: "<<setw(10)<<residual<<"\tenergy: "<<setw(10)<<energy<<endl;

            /// If convergence
            if(residual <= parameters.precision && residual_old <= parameters.precision)
                break;


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




double PairwiseSharedMRF::ADMM_old()
{
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
//    double absmax = nodePotentials.cwiseAbs().maxCoeff();
//    double cmax = edgeSharedPotentials.cwiseAbs().maxCoeff();
//    for(size_t e = 0; e < numberOfEdges; e++){
//        double mmax = std::abs(edgeWeights[e])*cmax;
//        if(mmax > absmax)
//            absmax = mmax;
//    }

    double rho_min = absMax*parameters.rho_min;
    double rho_max = absMax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t E = numberOfEdges;
    size_t L = numberOfLabels[0];

    for(size_t i = 1; i < V; i++){
        if(L != numberOfLabels[i]){
            cerr<<"The current version supports only equal number of labels!"<<endl;
        }
        assert(L == numberOfLabels[i]);
    }


    /// Initialization

    for(size_t i = 0; i < V; i++){
        size_t label;
        nodePotentials.col(i).minCoeff( &label );
        //if(x_max < 1.0) cout<<x_max<<", ";
        X.col(i).setZero();
        X(label,i) = 1.0;
    }

    MatrixXd X1 = X;
    MatrixXd X2 = X;

    MatrixXd Y = MatrixXd::Zero(L,V);

    vector<bool> hasChangedX1(V, true);
    vector<bool> hasChangedX2(V, true);
    vector<bool> hasChangedY(V, true);
    vector<bool> hasChangedP1(V, true);
    vector<bool> hasChangedP2(V, true);
    bool hasChangedRho = true;



    MatrixXd A1, A2, X1_old, X2_old, P1, P2;
    A1 = MatrixXd::Zero(L,V);
    A2 = MatrixXd::Zero(L,V);
    P1 = MatrixXd::Zero(L,V);
    P2 = MatrixXd::Zero(L,V);


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


    if(verbose){
        cout<<"Starting ADMM..."<<endl;
    }

    begin = myclock::now();


    double r, s;
    for(int iter = 0; iter <= MAX_ITER; iter++)
    {

        /// Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        /// c1 = x2 - (d + M*x2 + y)/rho
        X1_old = X1;

        begin2 = myclock::now();

        std::fill(hasChangedP1.begin(), hasChangedP1.end(), false);
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedX2[j])
                hasChangedP1[i] = true;
            if(hasChangedX2[i])
                hasChangedP1[j] = true;
        }
        std::fill(hasChangedX1.begin(), hasChangedX1.end(), false);
        for(size_t i = 0; i < V; i++){
            if(hasChangedP1[i] || hasChangedY[i] || hasChangedX2[i] || hasChangedRho)
                hasChangedX1[i] = true;
        }


        // Compute P
        for(size_t i = 0; i < V; i++){
            if(hasChangedP1[i])
                P1.col(i).setZero();
        }
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            assert(i < numberOfNodes && j < numberOfNodes);
            if(hasChangedP1[i])
                P1.col(i) += (0.5*edgeWeights[e])*X2.col(j);
            if(hasChangedP1[j])
                P1.col(j) += (0.5*edgeWeights[e])*X2.col(i);
        }

        // Compute A = Z - (D + Y + C*P)/rho
        for(size_t i = 0; i < V; i++){
            if(hasChangedX1[i])
                A1.col(i) = X2.col(i) - (nodePotentials.col(i) + Y.col(i) + edgeSharedPotentials*P1.col(i))/rho;
        }
        //A = Z - (D + Y + C*P2)/rho;
        //A = A*rho/(rho + 2.0*lambda);
        end2 = myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
        for(size_t i = 0; i < V; i++){
            if(hasChangedX1[i]){
                VectorXd x;
                SimplexProjection(x, A1.col(i));
                X1.col(i) = x;
            }
        }
        // Add additional unchanged Xi to the list
        for(size_t i = 0; i < V; i++){
            if(hasChangedX1[i])
                hasChangedX1[i] = !(X1.col(i).isApprox(X1_old.col(i)));
        }
        end2= myclock::now();
        T2 += measure_time(begin2, end2);


        /// Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        /// c2 = x1 - (M^T*x1 - y)/rho
        X2_old = X2;

        begin2 = myclock::now();

        std::fill(hasChangedP2.begin(), hasChangedP2.end(), false);
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedX1[i])
                hasChangedP2[j] = true;
            if(hasChangedX1[j])
                hasChangedP2[i] = true;
        }

        /// IMPORTANT: COMPUTE THE INITIAL Q
        if(iter < 1)
            std::fill(hasChangedP2.begin(), hasChangedP2.end(), true);

        std::fill(hasChangedX2.begin(), hasChangedX2.end(), false);
        for(size_t i = 0; i < V; i++){
            if(hasChangedP2[i] || hasChangedY[i] || hasChangedX1[i] || hasChangedRho)
                hasChangedX2[i] = true;
        }

        // Compute Q
        for(size_t i = 0; i < V; i++){
            if(hasChangedP2[i])
                P2.col(i).setZero();
        }
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedP2[j])
                P2.col(j) += (0.5*edgeWeights[e])*X1.col(i);
            if(hasChangedP2[i])
                P2.col(i) += (0.5*edgeWeights[e])*X1.col(j);
        }

        // Compute B = X - (C^T*Q - Y)/rho
        for(size_t i = 0; i < V; i++){
            if(hasChangedX2[i])
                A2.col(i) = X1.col(i) - (edgeSharedPotentials.transpose()*P2.col(i) - Y.col(i))/rho;
        }
        //B = X - (C*Q2 - Y)/rho;
        end2 = myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
//        for(size_t i = 0; i < V; i++){
//            VectorXd z;
//            sublemma_EQUALITY_EIGEN(z, B.col(i));
//            Z.col(i) = z;
//        }
        X2 = A2.cwiseMax(0.0);
        // Add additional unchanged Zi to the list
        for(size_t i = 0; i < V; i++){
            if(hasChangedX2[i])
                hasChangedX2[i] = !(X2.col(i).isApprox(X2_old.col(i)));
        }

        end2 = myclock::now();
        T2 += measure_time(begin2, end2);

        /// Step 3: update y
        //if(iter >= 10)
        Y += rho*(X1 - X2);

        for(size_t i = 0; i < V; i++){
            hasChangedY[i] = !(X1.col(i).isApprox(X2.col(i)));
            //hasChangedY[i] = !(X.col(i) - Z.col(i)).isMuchSmallerThan(1e-10);
        }


        if(hasChangedRho)
            hasChangedRho = false;

        /// Step 4: compute the residuals and update rho

        if(iter%protocolFrequence == 0){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(iter + last_iter + 1);

            r = (X1 - X2).squaredNorm();
            s = (X1 - X1_old).squaredNorm() + (X2 - X2_old).squaredNorm();
            R_old = R;
            R = r + s;

            energy =  this->energy(X1);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(R);
            visitor.bounds.push_back(-DBL_MAX);
            if(energy < energy_best){
                energy_best = energy;
                X = X1;
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
            if(energy < res_best_so_far){
                res_best_so_far = energy;
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

    //this->ADMM();
    return energy_best;
}




void PairwiseSharedMRF::ADMM_Asymmetric()
{
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
    double absmax = nodePotentials.cwiseAbs().maxCoeff();
    double cmax = edgeSharedPotentials.cwiseAbs().maxCoeff();
    for(size_t e = 0; e < numberOfEdges; e++){
        double mmax = std::abs(edgeWeights[e])*cmax;
        if(mmax > absmax)
            absmax = mmax;
    }

    double rho_min = absmax*parameters.rho_min;
    double rho_max = absmax*parameters.rho_max;


    // Output

    size_t V = numberOfNodes;
    size_t E = numberOfEdges;
    size_t L = numberOfLabels[0];

    for(size_t i = 1; i < V; i++){
        if(L != numberOfLabels[i]){
            cerr<<"The current version supports only equal number of labels!"<<endl;
        }
        assert(L == numberOfLabels[i]);
    }


    /// Initialization
    MatrixXd X1 = X;
    MatrixXd X2 = X;

    MatrixXd Y = MatrixXd::Zero(L,V);

    vector<bool> hasChangedX(V, true);
    vector<bool> hasChangedZ(V, true);
    vector<bool> hasChangedY(V, true);
    vector<bool> hasChangedP(V, true);
    vector<bool> hasChangedQ(V, true);
    bool hasChangedRho = true;



    MatrixXd A, B, X1_old, X2_old, P, Q;
    A = MatrixXd::Zero(L,V);
    B = MatrixXd::Zero(L,V);
    P = MatrixXd::Zero(L,V);
    Q = MatrixXd::Zero(L,V);


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


    if(verbose){
        cout<<"Starting ADMM..."<<endl;
    }

    begin = myclock::now();


    double r, s;
    for(int iter = 0; iter <= MAX_ITER; iter++)
    {

        /// Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
        /// c1 = x2 - (d + M*x2 + y)/rho
        X1_old = X1;

        begin2 = myclock::now();

        std::fill(hasChangedP.begin(), hasChangedP.end(), false);
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedZ[j])
                hasChangedP[i] = true;
        }
        std::fill(hasChangedX.begin(), hasChangedX.end(), false);
        for(size_t i = 0; i < V; i++){
            if(hasChangedP[i] || hasChangedY[i] || hasChangedZ[i] || hasChangedRho)
                hasChangedX[i] = true;
        }


        // Compute P
        for(size_t i = 0; i < V; i++){
            if(hasChangedP[i])
                P.col(i).setZero();
        }
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedP[i])
                P.col(i) += edgeWeights[e]*X2.col(j);
        }

        // Compute A = Z - (D + Y + C*P)/rho
        for(size_t i = 0; i < V; i++){
            if(hasChangedX[i])
                A.col(i) = X2.col(i) - (nodePotentials.col(i) + Y.col(i) + edgeSharedPotentials*P.col(i))/rho;
        }
        //A = Z - (D + Y + C*P2)/rho;
        //A = A*rho/(rho + 2.0*lambda);
        end2 = myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
        for(size_t i = 0; i < V; i++){
            if(hasChangedX[i]){
                VectorXd x;
                SimplexProjection(x, A.col(i));
                X1.col(i) = x;
            }
        }
        // Add additional unchanged Xi to the list
        for(size_t i = 0; i < V; i++){
            if(hasChangedX[i])
                hasChangedX[i] = !(X1.col(i).isApprox(X1_old.col(i)));
        }
        end2= myclock::now();
        T2 += measure_time(begin2, end2);


        /// Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
        /// c2 = x1 - (M^T*x1 - y)/rho
        X2_old = X2;

        begin2 = myclock::now();

        std::fill(hasChangedQ.begin(), hasChangedQ.end(), false);
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedX[i])
                hasChangedQ[j] = true;
        }

        /// IMPORTANT: COMPUTE THE INITIAL Q
        if(iter < 1)
            std::fill(hasChangedQ.begin(), hasChangedQ.end(), true);

        std::fill(hasChangedZ.begin(), hasChangedZ.end(), false);
        for(size_t i = 0; i < V; i++){
            if(hasChangedQ[i] || hasChangedY[i] || hasChangedX[i] || hasChangedRho)
                hasChangedZ[i] = true;
        }

        // Compute Q
        for(size_t i = 0; i < V; i++){
            if(hasChangedQ[i])
                Q.col(i).setZero();
        }
        for(size_t e = 0; e < E; e++){
            size_t i = edges[e][0];
            size_t j = edges[e][1];
            if(hasChangedQ[j])
                Q.col(j) += edgeWeights[e]*X1.col(i);
        }

        // Compute B = X - (C^T*Q - Y)/rho
        for(size_t i = 0; i < V; i++){
            if(hasChangedZ[i])
                B.col(i) = X1.col(i) - (edgeSharedPotentials.transpose()*Q.col(i) - Y.col(i))/rho;
        }
        //B = X - (C*Q2 - Y)/rho;
        end2 = myclock::now();
        T1 += measure_time(begin2, end2);

        begin2 = myclock::now();
//        for(size_t i = 0; i < V; i++){
//            VectorXd z;
//            sublemma_EQUALITY_EIGEN(z, B.col(i));
//            Z.col(i) = z;
//        }
        X2 = B.cwiseMax(0.0);
        // Add additional unchanged Zi to the list
        for(size_t i = 0; i < V; i++){
            if(hasChangedZ[i])
                hasChangedZ[i] = !(X2.col(i).isApprox(X2_old.col(i)));
        }

        end2 = myclock::now();
        T2 += measure_time(begin2, end2);

        /// Step 3: update y
        //if(iter >= 10)
        Y += rho*(X1 - X2);

        for(size_t i = 0; i < V; i++){
            hasChangedY[i] = !(X1.col(i).isApprox(X2.col(i)));
            //hasChangedY[i] = !(X.col(i) - Z.col(i)).isMuchSmallerThan(1e-10);
        }


        if(hasChangedRho)
            hasChangedRho = false;

        /// Step 4: compute the residuals and update rho

        if(iter%protocolFrequence == 0){
            end = myclock::now();
            total_time += measure_time(begin, end);
            visitor.times.push_back(total_time);

            visitor.iteration.push_back(iter + last_iter + 1);

            r = (X1 - X2).squaredNorm();
            s = (X1 - X1_old).squaredNorm() + (X2 - X2_old).squaredNorm();
            R_old = R;
            R = r + s;

            energy =  this->energy(X1);
            visitor.energies.push_back(energy);
            visitor.residuals.push_back(R);
            visitor.bounds.push_back(-DBL_MAX);
            if(energy < energy_best){
                energy_best = energy;
                X = X1;
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
                        cout<<"-------- Update rho = "<<rho/absmax<<endl;
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
}

