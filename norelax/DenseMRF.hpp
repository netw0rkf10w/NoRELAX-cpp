#ifndef DENSEMRF_H
#define DENSEMRF_H

#include <Eigen/Sparse>

#include "HMRF.hpp"

class DenseMRF : public HMRF
{
public:
    DenseMRF();
    template <typename GraphicalModelType>
    void importModel_Dense(const GraphicalModelType &gm);
    virtual double ADMM() override;
private:
    Eigen::SparseMatrix<double,Eigen::RowMajor> M;  // (V*L)*(V*L) pairwise potential matrix
    Eigen::VectorXd d;                              // V*L dimensional unary potential vector
};


template <typename GraphicalModelType>
void DenseMRF::importModel_Dense(const GraphicalModelType &gm)
/// numLabels[i]: the number of labels of the node i
/// factors
{
    assert(gm.maxFactorOrder() <= 2);

    numberOfNodes = gm.numberOfVariables();
    numberOfLabels.resize(numberOfNodes);
    for(size_t i = 0; i < numberOfNodes; i++){
        numberOfLabels[i] = gm.numberOfLabels(i);
        // In the current version, only models with equal number of labels are supported
        assert(numberOfLabels[i] == numberOfLabels[0]);
    }
    size_t L = numberOfLabels[0];
    size_t A = numberOfNodes*L; // Number of possible assignments

    size_t numberOfFactors = gm.numberOfFactors();


    // Import unary potentials and compute the number of edges.
    // WARNING: numberOfEdges may not be equal to (numberOfFactors - numberOfNodes)
    // because some nodes may not be considered as a factor in our factor graph
    Eigen::MatrixXd nodePotentials = Eigen::MatrixXd::Zero(L, numberOfNodes);
    size_t numberOfEdges = 0;
    for(size_t c = 0; c < numberOfFactors; ++c){
        size_t S = gm[c].numberOfVariables();
        if(S == 1){
            VD potential(gm[c].size());
            gm[c].copyValuesSwitchedOrder(potential.begin());
            Eigen::VectorXd p = Eigen::VectorXd::Map(potential.data(), numberOfLabels[0]);
            nodePotentials.col(gm[c].variableIndex(0)) = p;
        }else if(S==2){
            numberOfEdges++;
        }
    }
    // Convert to the potential vector
    this->d = Eigen::Map<Eigen::VectorXd>(nodePotentials.data(), A);

    absMax = d.cwiseAbs().maxCoeff();


    std::vector<Eigen::Triplet<double>> abMab;
    abMab.reserve(2*numberOfEdges*L*L); // Symmetric matrix
    double absMax = 0;
    // Get the edges and the corresponding potential matrices
    for(size_t c = 0; c < numberOfFactors; ++c){
        size_t S = gm[c].numberOfVariables();
        if(S != 2)
            continue;

        size_t i = gm[c].variableIndex(0);
        size_t j = gm[c].variableIndex(1);

        VD potential(gm[c].size());
        gm[c].copyValuesSwitchedOrder(potential.begin());

        Eigen::MatrixXd pwPot = Eigen::MatrixXd::Map(potential.data(), L, L);
        double cmax = pwPot.cwiseAbs().maxCoeff();
        if(cmax > absMax)
            absMax = cmax;

        for(size_t l = 0; l < L; l++){
            size_t a = i*L + l;
            for(size_t k = 0; k < L; k++){
                size_t b = j*L + k;
                if(pwPot(l,k) != 0){
                    abMab.push_back(Eigen::Triplet<double> (a, b, 0.5*pwPot(l,k)));
                    abMab.push_back(Eigen::Triplet<double> (b, a, 0.5*pwPot(k,l)));
                }
            }
        }


    }

    M = Eigen::SparseMatrix<double,Eigen::RowMajor>(A, A);
    M.setFromTriplets(abMab.begin(), abMab.end());

    max_degree = 2;
    X = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes).array() + 1.0/(double)numberOfLabels[0];
}


#endif // DENSEMRF_H
