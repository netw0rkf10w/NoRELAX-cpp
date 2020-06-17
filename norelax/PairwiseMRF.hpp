#ifndef PAIRWISEMRF_H
#define PAIRWISEMRF_H

#include "HMRF.hpp"


class PairwiseMRF : public HMRF
{
private:
    std::vector<Eigen::MatrixXd> edgePotentials;   // Each element is a L*L matrix corresponding to the potentials of each edge    
protected:
    Eigen::MatrixXd nodePotentials;         // Each column corresponds to a node
    VVS edges;
    VVS containing_edges;                 // containing_edges[i] is the list of edges that contain the node i
    VVS containing_tails;              // containing_tails[i] is the list of (directed) edges that start by i (i.e. e = (i,j) with some j)
    VVS containing_heads;              // containing_heads[i] is the list of (directed) edges that end by i (i.e. e = (j,i) with some j)
    size_t numberOfEdges;
    std::vector<Eigen::MatrixXd> P;
    std::vector<Eigen::MatrixXd> RCa, RCb;
    std::vector< std::vector<bool> > hasChangedP;
    std::vector< std::vector<bool> > hasChangedRCa, hasChangedRCb;
    virtual void computeP(size_t d, const std::vector<Eigen::MatrixXd> &XD, const std::vector<size_t> &random_node_indices, const std::vector<size_t> &random_edge_indices);
public:
    PairwiseMRF();

    template <typename GraphicalModelType>
    void importModel_Pairwise(const GraphicalModelType &gm);

    virtual double ADMM() override;    // ADMM
    virtual double ADMM_old();    // ADMM
    virtual double CQP();    // Convex quadratic programming relaxation
    virtual void lineSearch(double &alpha_best, double &p_best, double energy, const Eigen::MatrixXd &R) override;
    virtual double energy() override;
    virtual double energy(const Eigen::MatrixXd &_X) override;
    virtual double energy_discrete() override;

    virtual void gradient(Eigen::MatrixXd &G, int type = X_TYPE_DENSE) override;
    // The partial derivative over node i only
    virtual void gradient(Eigen::MatrixXd &Gi, size_t i, int type = X_TYPE_DENSE) override;

    // virtual void gradient_discrete(Eigen::MatrixXd &G) override;
    // virtual void gradient_discrete(Eigen::MatrixXd &G, size_t i) override;

    virtual void getUnarySolution(Eigen::MatrixXd &_X) override;
};




class PairwiseSharedMRF : public PairwiseMRF
{
private:
    std::vector<double> edgeWeights;
    Eigen::MatrixXd edgeSharedPotentials;   // L*L matrix
    virtual void computeP(size_t d, const std::vector<Eigen::MatrixXd> &XD, const std::vector<size_t> &random_node_indices, const std::vector<size_t> &random_edge_indices) override;
public:
    PairwiseSharedMRF();

    template <typename GraphicalModelType>
    void importModel_Pairwise_Metric(const GraphicalModelType &gm);

    //virtual double ADMM();    // ADMM
    virtual double ADMM_old();
    virtual double CQP();    // Convex quadratic programming relaxation
    //virtual void CQP() override;    // Convex quadratic programming relaxation
    virtual void lineSearch(double &alpha_best, double &p_best, double energy, const Eigen::MatrixXd &D) override;
    void ADMM_Asymmetric();
    virtual double energy() override;
    virtual double energy(const Eigen::MatrixXd &_X) override;
    virtual double energy_discrete() override;

    virtual void gradient(Eigen::MatrixXd &G, int type = X_TYPE_DENSE) override;
    // The partial derivative over node i only
    virtual void gradient(Eigen::MatrixXd &Gi, size_t i, int type = X_TYPE_DENSE) override;

    //virtual void gradient_discrete(Eigen::MatrixXd &G) override;
    //virtual void gradient_discrete(Eigen::MatrixXd &G, size_t i) override;
};




template <typename GraphicalModelType>
void PairwiseMRF::importModel_Pairwise(const GraphicalModelType &gm)
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

    size_t numberOfFactors = gm.numberOfFactors();


    // Import unary potentials and compute the number of edges.
    // WARNING: numberOfEdges may not be equal to (numberOfFactors - numberOfNodes)
    // because some nodes may not be considered as a factor in our factor graph
    nodePotentials = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    numberOfEdges = 0;
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

    edges.clear();
    edges.reserve(numberOfEdges);
    edgePotentials.clear();
    edgePotentials.reserve(numberOfEdges);

    // Get the edges and the corresponding potential matrices
    for(size_t c = 0; c < numberOfFactors; ++c){
        size_t S = gm[c].numberOfVariables();
        if(S != 2)
            continue;

        size_t i = gm[c].variableIndex(0);
        size_t j = gm[c].variableIndex(1);

        VD potential(gm[c].size());
        gm[c].copyValuesSwitchedOrder(potential.begin());

        Eigen::MatrixXd pwPot = Eigen::MatrixXd::Map(potential.data(), numberOfLabels[i], numberOfLabels[j]);


        /// Make sure the edge is listed as (i,j) where i < j, for later convenience
        /// This is not necessary if the edge potential matrix is symmetric
        VS edge(2);
        if(i < j){
            edge[0] = i;
            edge[1] = j;
        }else{
            edge[0] = j;
            edge[1] = i;
            pwPot = pwPot.transpose();
        }

        edges.push_back(edge);
        edgePotentials.push_back(pwPot);

    }

    // Get the set of neighbors for each node
    neighbors.resize(numberOfNodes);
    containing_edges.resize(numberOfNodes);
    containing_tails.resize(numberOfNodes);
    containing_heads.resize(numberOfNodes);
    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        neighbors[i].insert(j);
        neighbors[j].insert(i);
        containing_edges[i].push_back(e);
        containing_edges[j].push_back(e);
        containing_tails[i].push_back(e);
        containing_heads[j].push_back(e);
    }

    absMax = nodePotentials.cwiseAbs().maxCoeff();
    for(size_t e = 0; e < numberOfEdges; e++){
        double cmax = edgePotentials[e].cwiseAbs().maxCoeff();
        if(cmax > absMax)
            absMax = cmax;
    }

    max_degree = 2;

    X = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes).array() + 1.0/(double)numberOfLabels[0];
}




template <typename OpenGMType>
void PairwiseSharedMRF::importModel_Pairwise_Metric(const OpenGMType &gm)
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

    size_t numberOfFactors = gm.numberOfFactors();


    // Import unary potentials and compute the number of edges.
    // WARNING: numberOfEdges may not be equal to (numberOfFactors - numberOfNodes)
    // because some nodes may not be considered as a factor in our factor graph
    nodePotentials = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes);
    numberOfEdges = 0;
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


    size_t L = numberOfLabels[0];

    /// Set (metric) shared potentials
    edgeSharedPotentials = Eigen::MatrixXd::Zero(L, L);
    for(size_t i = 0; i < numberOfFactors; i++) {
       if(gm.numberOfVariables(i) == 2) {
          for(size_t k = 0; k < L; k++) {
             for(size_t l = 0; l < L; l++) {
                size_t index[] = {k, l};
                edgeSharedPotentials(k,l) = gm[i](index);
             }
          }
          break;
       }
    }


    /// Set edge weigths
    edges.clear();
    edges.reserve(numberOfEdges);
    edgeWeights = VD(numberOfEdges, 0.0);
    size_t currentPair = 0;
    for(size_t c = 0; c < numberOfFactors; c++) {
       if(gm.numberOfVariables(c) == 2) {
          assert(currentPair < numberOfEdges);
          VS edge(2);
          edge[0] = gm[c].variableIndex(0);
          edge[1] = gm[c].variableIndex(1);
          edges.push_back(edge);
          size_t k;
          for(k = 0; k < gm.numberOfLabels(0); k++) {
             size_t l;
             for(l = 0; l < gm.numberOfLabels(0); l++) {
                size_t index[] = {k, l};
                if((gm[c](index) != 0) && (edgeSharedPotentials(k,l) != 0)) {
                   double currentWeight = static_cast<double>(gm[c](index)) / static_cast<double>(edgeSharedPotentials(k,l));
                   edgeWeights[currentPair] = static_cast<double>(currentWeight);
                   if(fabs(currentWeight - static_cast<double>(edgeWeights[currentPair])) > OPENGM_FLOAT_TOL) {
                      throw(opengm::RuntimeError("Function not supported"));
                   }
                   currentPair++;
                   break;
                }
             }
             if(l != gm.numberOfLabels(0)) {
                break;
             }
          }
          if(k == gm.numberOfLabels(0)) {
             edgeWeights[currentPair] = 0;
             currentPair++;
          }
       }
    }

    // Get the set of neighbors for each node
    neighbors.resize(numberOfNodes);
    containing_edges.resize(numberOfNodes);
    containing_tails.resize(numberOfNodes);
    containing_heads.resize(numberOfNodes);
    for(size_t e = 0; e < numberOfEdges; e++){
        size_t i = edges[e][0];
        size_t j = edges[e][1];
        neighbors[i].insert(j);
        neighbors[j].insert(i);
        containing_edges[i].push_back(e);
        containing_edges[j].push_back(e);
        containing_tails[i].push_back(e);
        containing_heads[j].push_back(e);
    }

    absMax = nodePotentials.cwiseAbs().maxCoeff();
    double cmax = edgeSharedPotentials.cwiseAbs().maxCoeff();
    for(size_t e = 0; e < numberOfEdges; e++){
        double mmax = std::abs(edgeWeights[e])*cmax;
        if(mmax > absMax)
            absMax = mmax;
    }

    max_degree = 2;

    X = Eigen::MatrixXd::Zero(numberOfLabels[0], numberOfNodes).array() + 1.0/(double)numberOfLabels[0];

    if(parameters.verbose){
        std::cout<<"Number of nodes = "<<numberOfNodes<<std::endl;
        std::cout<<"Number of edges = "<<numberOfEdges<<std::endl;
    }
}




#endif // PMRF_H
