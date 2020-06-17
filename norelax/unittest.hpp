#ifndef UNITTEST_H
#define UNITTEST_H

#include "HMRF.hpp"
#include "outils_io.hpp"

using namespace std;
using namespace Eigen;

void checkProposalReading(){
    string datapath("/home/khue/Research/Stereo/data/");
    string sequence("teddy");
    string pfile = datapath + sequence + "/proposals.txt";
    //string ufile = datapath + sequence + "/unaries.txt";
    //double scale = 1.0;

//    string outfile;
//    if(scale == 1.0)
//        outfile = datapath + sequence + "/" + sequence + ".h5";
//    else
//        outfile = datapath + sequence + "/" + sequence + "_small.h5";

    vector<MatrixXd> proposals, unaries;
    cout << "Reading proposals..."<<endl;
    ReadProposals_EIGEN(proposals, pfile);
    cout << "Number of proposals = "<<proposals.size()<<endl;
    //cout << "Reading unaries..."<<endl;
    //ReadProposals(unaries, ufile);

//    assert(proposals.size() == unaries.size());

//    if(scale != 1.0){
//        for(size_t i = 0; i < proposals.size(); i++){
//            cv::resize(proposals[i], proposals[i], Size(), scale, scale, CV_INTER_AREA);
//            proposals[i] *= scale;
//            resize(unaries[i], unaries[i], Size(), scale, scale, CV_INTER_AREA);
//        }
//    }
}



void checkLabelIndices(){
    size_t L = 4, S = 3;
    VVS LabelIndices;
    getTensorIndices(LabelIndices, L, S);
    printVectorVector(LabelIndices);
}


void checkTensorIndices()
/// Example
/// (0) * * * * (1) * * * * (2)
///                *       *
///                  *   *
///                   (3)
/// Number of labels (for the nodes 0, 1, 2, 3): 2 3 2 4
{
    VS _numLabels;
    _numLabels.push_back(2);
    _numLabels.push_back(3);
    _numLabels.push_back(2);
    _numLabels.push_back(4);

    //size_t V = _numLabels.size();
    VVS _factors;
    VS factor0; factor0.push_back(0); _factors.push_back(factor0);
    VS factor1; factor1.push_back(1); _factors.push_back(factor1);
    VS factor2; factor2.push_back(2); _factors.push_back(factor2);
    VS factor3; factor3.push_back(3); _factors.push_back(factor3);
    VS factor01; factor01.push_back(0); factor01.push_back(1); _factors.push_back(factor01);
    VS factor123; factor123.push_back(1); factor123.push_back(2); factor123.push_back(3); _factors.push_back(factor123);

    /// Set the potentials
    VVD _potentials(_factors.size());
    for(size_t c = 0; c < _factors.size(); c++){
        if(_factors[c].size() == 1){
            size_t i = _factors[c][0]; // The node i
            for(size_t l = 0; l < _numLabels[i]; l++){
                _potentials[c].push_back(((double)c + 1.0)*l);
            }
        }
        if(_factors[c].size() == 2){
            // The edge ij
            size_t i = _factors[c][0];
            size_t j = _factors[c][1];
            for(size_t k = 0; k < _numLabels[i]; k++){
                for(size_t l = 0; l < _numLabels[j]; l++){
                    _potentials[c].push_back(std::abs(2.0*k - 1.5*l));
                }
            }
        }
        if(_factors[c].size() == 3){
            // The edge ij
            size_t i = _factors[c][0];
            size_t j = _factors[c][1];
            size_t k = _factors[c][2];
            for(size_t li = 0; li < _numLabels[i]; li++){
                for(size_t lj = 0; lj < _numLabels[j]; lj++){
                    for(size_t lk = 0; lk < _numLabels[k]; lk++){
                        _potentials[c].push_back(std::abs(2.0*li - 1.0*lj + 1.0*lk));
                    }
                }
            }
        }
    }

    /// Export to OpenGM model
    SpaceType space;
    for(size_t i = 0; i < _numLabels.size(); i++){
        space.addVariable(_numLabels[i]);
    }
    ModelExplicit gm(space);

    /// Add factors
    for(size_t c = 0; c < _factors.size(); c++){
        if(_factors[c].size() == 1){
            size_t i = _factors[c][0]; // The node i
            size_t L = _numLabels[i]; // Number of labels of node i
            const size_t shape[] = {L};
            ExplFunction f(shape, shape + 1);
            for(size_t l = 0; l < L; l++){
                f(l) = _potentials[c][l];
            }
            ModelExplicit::FunctionIdentifier fid = gm.addFunction(f);
            size_t variableIndices[] = {i};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }
        if(_factors[c].size() == 2){
            // The edge ij
            size_t i = _factors[c][0];
            size_t j = _factors[c][1];
            size_t K = _numLabels[i];
            size_t L = _numLabels[j];
            size_t shape[] = {K, L};
            ExplFunction f(shape, shape + 2);
            for(size_t k = 0; k < K; k++){
                for(size_t l = 0; l < L; l++){
                    f(k,l) = _potentials[c][k*L+l];
                }
            }
            ModelExplicit::FunctionIdentifier fid = gm.addFunction(f);
            IndexType vars[] = {i,j};
            gm.addFactor(fid, vars, vars + 2);
        }
        if(_factors[c].size() == 3){
            // The edge ij
            size_t i = _factors[c][0];
            size_t j = _factors[c][1];
            size_t k = _factors[c][2];
            size_t Li = _numLabels[i];
            size_t Lj = _numLabels[j];
            size_t Lk = _numLabels[k];
            size_t shape[] = {Li, Lj, Lk};
            ExplFunction f(shape, shape + 3);
            for(size_t li = 0; li < Li; li++){
                for(size_t lj = 0; lj < Lj; lj++){
                    for(size_t lk = 0; lk < Lk; lk++){
                        f(li,lj,lk) = _potentials[c][(li*Lj+lj)*Lk + lk];
                    }
                }
            }
            ModelExplicit::FunctionIdentifier fid = gm.addFunction(f);
            IndexType vars[] = {i,j,k};
            gm.addFactor(fid, vars, vars + 3);
        }
    }


    /// Displaying
    cout<<"Numbers of labels:"<<endl;
    printVector(_numLabels);
    cout<<"Factors:"<<endl;
    printVectorVector(_factors);
    cout<<"Potentials:"<<endl;
    printVectorVector(_potentials);


//    /// Import to our model to check
//    VS numLabels;
//    VVS factors;
//    VVD potentials;

//    importFromOpenGM(gm, numLabels, factors, potentials);

//    cout<<"Numbers of labels:"<<endl;
//    printVector(numLabels);
//    cout<<"Factors:"<<endl;
//    printVectorVector(factors);
//    cout<<"Potentials:"<<endl;
//    printVectorVector(potentials);


    /// Convert to HMRF
    HMRF mrf;
    mrf.importModel(gm);
    cout<<"++++++++++++++++++++ Import to HMRF model ++++++++++++++++++++"<<endl;
    cout<<"Numbers of labels:"<<endl;
    printVector(mrf.getNumberOfLabels());
    cout<<"Factors:"<<endl;
    printVectorVector(mrf.getFactors());
    cout<<"Potentials:"<<endl;
    printVectorVector(mrf.getPotentials());
    cout<<"Node\t \tBelonging to factor"<<endl;



    /*
    /// The dimension the the assignment vector x
    size_t N = 0;
    for(size_t i = 0; i < _numLabels.size(); i++)
        N += _numLabels[i];

    /// The maximum size D of the cliques (i.e. (D-1)-th order MRFs)
    size_t D = 0;
    for(size_t c = 0; c < _factors.size(); c++)
        if(D < _factors[c].size())
            D = _factors[c].size();

    /// assignIndices[i][l] return the corresponding assingment index of assigning label l to node i
    VVS assignIndices(V);
    size_t idx = 0;
    for(size_t i = 0; i < V; i++){
        size_t L = _numLabels[i];
        assignIndices[i].resize(L);
        for(size_t l = 0; l < L; l++){
            assignIndices[i][l] = idx;
            idx++;
        }
    }
    assert(idx == N);


    cout<<"Assignment indices for each node:"<<endl;
    printVectorVector(assignIndices);

    vector<VVS> ind(D); vector<VD> val(D);
    for(size_t d = 0; d < D; d++){
        cout<<"ind["<<d<<"].size() = "<<ind[d].size()<<endl;
    }

    VS numElements(D, 0);

    for(size_t c = 0; c < factors.size(); c++){
        size_t d = factors[c].size() - 1;/// IMPORTANT: -1 because d is indexed from 0
        size_t prod = 1.0;
        for(size_t idx = 0; idx < factors[c].size(); idx++){
            size_t L = numLabels[factors[c][idx]];
            prod *= L;
        }
        numElements[d] += prod;
    }
    for(size_t d = 0; d < D; d++){
        ind[d].resize(numElements[d]);
        val[d].resize(numElements[d]);
    }

    VS startingPosition(D, 0);

    for(size_t c = 0; c < factors.size(); c++){
        size_t d = factors[c].size() - 1;/// IMPORTANT: -1 because d is indexed from 0
        VVS factorTensorIndices;
        VS currentTensorIndex(d + 1);
        getTensorIndicesFromFactor(factorTensorIndices, currentTensorIndex, 0, factors[c], numLabels, assignIndices);
        std::copy(factorTensorIndices.begin(), factorTensorIndices.end(), ind[d].begin() + startingPosition[d]);
        std::copy(potentials[c].begin(), potentials[c].end(), val[d].begin() + startingPosition[d]);
        startingPosition[d] += factorTensorIndices.size();

        // Add the indices of this factor to the list of indices of all factors
        //cout<<"ind["<<d<<"].size() = "<<ind[d].size()<<", factorTensorIndices.size() = "<<factorTensorIndices.size()<<endl;
        //cout<<"val["<<d<<"].size() = "<<val[d].size()<<", potentials[c].size() = "<<potentials[c].size()<<endl;
        //ind[d].reserve(ind[d].size() + factorTensorIndices.size());
        //ind[d].insert(ind[d].end(), factorTensorIndices.begin(), factorTensorIndices.end());
        // Also add the corresponding potentials
        //val[d].reserve(val[d].size() + potentials[c].size());
        //val[d].insert(val[d].end(), potentials[c].begin(), potentials[c].end());
    }

//    cout<<"Remove zeros..."<<endl;
//    for(size_t d = 0; d < val.size(); d++){
//        for(size_t idx = val[d].size() - 1; idx >= 0; idx--){
//            if(val[d][idx] == 0){
//                val[d].erase(val[d].begin() + idx);
//                ind[d].erase(ind[d].begin() + idx);
//            }
//        }
//    }


    /// Displaying

    for(size_t d = 0; d < D; d++){
        cout<<"Size d = "<<d<<": "<<endl;
        assert(ind[d].size() == val[d].size());
        for(size_t i = 0; i < ind[d].size(); i++){
            for(size_t j = 0; j < ind[d][i].size(); j++)
                cout<<ind[d][i][j]<<", ";
            cout<<"\t"<<val[d][i]<<endl;
        }
        cout<<"====================="<<endl;
    }
    */

}




void checkDataConversion()
/// Example
/// (0) ------- (1) ------- (2)
/// (2 labels)  (2 labels)  (3 labels)
{
    VS numLabels;
    numLabels.push_back(2);
    numLabels.push_back(5);
    numLabels.push_back(3);

    //size_t V = numLabels.size();
    VVS factors;
    VS factor0; factor0.push_back(0); factors.push_back(factor0);
    VS factor1; factor1.push_back(1); factors.push_back(factor1);
    VS factor2; factor2.push_back(2); factors.push_back(factor2);
    VS factor01; factor01.push_back(0); factor01.push_back(1); factors.push_back(factor01);
    VS factor12; factor12.push_back(1); factor12.push_back(2); factors.push_back(factor12);

    /// Set the potentials
    VVD potentials(factors.size());
    for(size_t c = 0; c < factors.size(); c++){
        if(factors[c].size() == 1){
            size_t i = factors[c][0]; // The node i
            for(size_t l = 0; l < numLabels[i]; l++){
                potentials[c].push_back(((double)c + 1.0)*l);
            }
        }
        if(factors[c].size() == 2){
            // The edge ij
            size_t i = factors[c][0];
            size_t j = factors[c][1];
            for(size_t k = 0; k < numLabels[i]; k++){
                for(size_t l = 0; l < numLabels[j]; l++){
                    potentials[c].push_back(std::abs(2.0*k - 1.5*l));
                }
            }
        }
    }

    /// Export to OpenGM model
    SpaceType space;
    for(size_t i = 0; i < numLabels.size(); i++){
        space.addVariable(numLabels[i]);
    }
    ModelExplicit gm(space);

    /// Add factors
    for(size_t c = 0; c < factors.size(); c++){
        if(factors[c].size() == 1){
            size_t i = factors[c][0]; // The node i
            size_t L = numLabels[i]; // Number of labels of node i
            const size_t shape[] = {L};
            ExplFunction f(shape, shape + 1);
            for(size_t l = 0; l < L; l++){
                f(l) = potentials[c][l];
            }
            ModelExplicit::FunctionIdentifier fid = gm.addFunction(f);
            size_t variableIndices[] = {i};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }
        if(factors[c].size() == 2){
            // The edge ij
            size_t i = factors[c][0];
            size_t j = factors[c][1];
            size_t K = numLabels[i];
            size_t L = numLabels[j];
            size_t shape[] = {K, L};
            ExplFunction f(shape, shape + 2);
            for(size_t k = 0; k < K; k++){
                for(size_t l = 0; l < L; l++){
                    f(k,l) = potentials[c][k*L+l];
                }
            }
            ModelExplicit::FunctionIdentifier fid = gm.addFunction(f);
            IndexType vars[] = {i,j};
            gm.addFactor(fid, vars, vars + 2);
        }
    }

    /// Import to our model to check
    VS numLabels2;
    VVS factors2;
    VVD potentials2;

//    importFromOpenGM(gm, numLabels2, factors2, potentials2);


//    /// Displaying
//    cout<<"Numbers of labels:"<<endl;
//    printVector(numLabels);
//    cout<<"Factors:"<<endl;
//    printVectorVector(factors);
//    cout<<"Potentials:"<<endl;
//    printVectorVector(potentials);

//    cout<<"Numbers of labels:"<<endl;
//    printVector(numLabels2);
//    cout<<"Factors:"<<endl;
//    printVectorVector(factors2);
//    cout<<"Potentials:"<<endl;
//    printVectorVector(potentials2);
}





void test_eigen()
{
    Eigen::MatrixXd A(3,4);
    A << 1, 2, 6, 9,
            3, 1, 7, 2,
            2, -1, 8, 1;

    /*std::cout << "Column's minimum: " << std::endl
      << mat.colwise().minCoeff() << std::endl;

    vector<int> minIndex(mat.cols());
    VectorXd minVal(mat.cols());
    for(size_t i = 0; i < mat.cols(); i++)
        minVal(i) = mat.col(i).minCoeff( &minIndex[i] );

    cout<<minVal<<endl;
    cout<<"Indices of min:"<<endl;
    for(size_t i = 0; i < minIndex.size(); i++) cout<<minIndex[i]<<endl;;*/

    MatrixXd X(A.rows(), A.cols());
    for(size_t i = 0; i < (size_t)A.cols(); i++){
        VectorXd x = A.col(i).array().exp();
        //VectorXd x = Z.col(i).cwiseProduct(toto);
        x = x/x.sum();
        X.col(i) = x;
    }

    cout<<A<<endl<<"======="<<endl;
    cout<<X<<endl;
}






void test_eigen2()
{
    Matrix3d X, Z;
    X << 0.2, 0.95, 0.9,     // Initialize A. The elements can also be
         0.9, 0.5, 0.3,     // matrices, which are stacked along cols
         0.1, 0.8, 0.91;     // and then the rows are stacked.
    Z << 0.2, 0.95, 0.2,     // Initialize A. The elements can also be
         0.9, 0.5, 0.9,     // matrices, which are stacked along cols
         0.92, 0.8, 0.91;     // and then the rows are stacked.
    cout<<(X.array() >= 0.9)<<endl<<"====="<<endl;
    cout<<(Z.array() >= 0.9)<<endl<<"====="<<endl;
    Matrix3d M = (X.array() >= 0.9).cast<double>() + (Z.array() >= 0.9).cast<double>();
    cout<<M<<endl<<"====="<<endl;
    cout<<(M.array() > 1.0).count()<<endl<<"====="<<endl;
    MatrixXd T = (M.array() > 1.0).cast<double>().cwiseProduct(Z.array() + 1.0);
    cout<<T<<endl;
    VectorXd vv = T.colwise().sum();
    cout<<vv<<endl;
}


void test_read_matrix(){
    vector< vector<size_t> > m;
    //size_t rows;
    //size_t cols;
    vector<size_t> v;
    const char* filename = "/home/khue/Dropbox/Code/MRFs/0_BENCHMARKS/methods/ADML/dev/tsukuba/label_optimal.txt";
    //ReadMatrixFile(filename, m, rows, cols);
    ReadVectorFile(filename, v);
    printVector(v);
    cout<<"elements = "<<v.size()<<endl;
}


void test_write_matrix(){
    double M[] = {0.1, 0.5, -5.6, 5, -7, 10.6};
    vector<double> MM (M, M + sizeof(M) / sizeof(M[0]));

    WriteMatrix2File(MM, 2, "matrix.txt");
}



void test_speed(){

    size_t L = 16;
    size_t V = 200000;
    double rho = 1.5;

    MatrixXd X = MatrixXd::Random(L, V);
    MatrixXd Y = MatrixXd::Random(L, V);

    vector<VectorXd> X2(V);
    vector<VectorXd> Y2(V);
    for(size_t i = 0; i < V; i++){
        X2[i] = X.col(i);
        Y2[i] = Y.col(i);
    }


    myclock::time_point begin, end;
    double timeSec;

    begin = myclock::now();
    for(size_t d = 1; d < 50; d++){
        Y += rho*(X - Y);
    }
    end = myclock::now();
    timeSec = measure_time(begin, end);
    cout << "Matrix version: " << timeSec << "s" << endl;


    begin = myclock::now();
    for(size_t d = 1; d < 50; d++){
        for(size_t i = 0; i < V; i++){
            Y2[i] += rho*(X2[i] - Y2[i]);
        }
    }
    end = myclock::now();
    timeSec = measure_time(begin, end);
    cout << "Vector version: " << timeSec << "s" << endl;

    double r = 0.0;
    for(size_t i = 0; i < V; i++){
        r += (Y2[i] - Y.col(i)).squaredNorm();
    }
    cout << "error = "<<r<<endl;
}



void test_write2hdf(){
    size_t v[] = {16,2,77,29};
    double M[] = {0.1, 0.5, -5.6, 5, -7, -3.6};

    vector<int> vv (v, v + sizeof(v) / sizeof(v[0]));
    vector<double> MM (M, M + sizeof(M) / sizeof(M[0]));

    string filename("testhdf5.h5");

    H5::H5File* file = new H5::H5File( filename, H5F_ACC_TRUNC );

    write2hdf(file, vv, "vector", vv.size(), 1);
    write2hdf(file, MM, "matrix", 3, 2);

    delete file;
}

#endif // UNITTEST_H
