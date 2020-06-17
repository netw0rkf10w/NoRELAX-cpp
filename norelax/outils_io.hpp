#ifndef OUTILS_IO_H
#define OUTILS_IO_H

#include <iostream>
#include <fstream>
#include <iterator>
#include <chrono>
#include <sstream>
#include <assert.h>
#include <vector>
#include <string>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include "H5Cpp.h"

template <typename T>
void printVector(const std::vector<T> &v);

template <typename T>
void printVectorVector(const std::vector< std::vector<T> > &v);

template <typename T>
std::string to_string2(const T a);

template <typename T>
size_t ReadNumbers( const std::string & line, std::vector<T>& v );

template <typename T>
void ReadMatrixFile(const char* filename, std::vector< std::vector<T> > &m, size_t &rows, size_t &cols);

template <typename MatrixType>
void ReadMatrix_EIGEN(const char* filename, MatrixType &M);

template <typename MatrixType>
void ReadProposals_EIGEN(std::vector<MatrixType> &proposals, const std::string& filename);

//int ReadNumbers_CV( const std::string & s, cv::Mat& v );
//void ReadMatrixFile_CV(const char* filename_X, cv::Mat &m, int& rows, int& cols);
//void ReadProposals_CV(std::vector<cv::Mat> &proposals, const std::string& filename);

template <typename T>
void WriteMatrix2File(const std::vector<T>&m, size_t rows, const char* filename);

template <typename MatrixType>
void WriteMatrix2File(const MatrixType &m, const char* filename);

template <typename T>
void ReadVectorFile(const char* filename, std::vector<T> &m);

template <typename T>
void write2hdf(H5::H5File* file, const std::vector<T> &values, const std::string &name, size_t rows, size_t cols);


//using namespace Eigen;
//using namespace std;

template <typename T>
void printVector(const std::vector<T> &v)
{
    for(size_t i = 0; i < v.size(); i++){
        std::cout<<v[i]<<", ";
    }
    std::cout<<std::endl;
}

template <typename T>
void printVectorVector(const std::vector< std::vector<T> > &v)
{
    for(size_t i = 0; i < v.size(); i++){
        printVector(v[i]);
    }
}


template <typename T>
std::string to_string2(const T a){
    std::stringstream ss;
    ss << a;
    return ss.str();
}


/*****************************************
***********	READ MATRIX FILE *************
*****************************************/
template <typename T>
size_t ReadNumbers( const std::string & line, std::vector<T>& v ) {
    std::istringstream is( line );
    T n;
    while( is >> n ) {
        v.push_back( n );
    }
    return v.size();
}


template <typename T>
void ReadMatrixFile(const char* filename, std::vector< std::vector<T> > &m, size_t &rows, size_t &cols)
/// Read the matrix file, set rows and cols, return a CV_32F matrix
{
    std::ifstream myfile;
    std::string myline;

    myfile.open(filename);
    assert (myfile.is_open());

    m.clear();

    size_t i = 0;

    std::vector<T> firstRow;
    std::getline(myfile, myline);
    cols = ReadNumbers( myline, firstRow );
    m.push_back(firstRow);
    //cout << "cols:" << cols << endl;


    for ( i=1; i < SIZE_MAX; i++)
    {
        if ( !std::getline(myfile, myline) ) break;
        std::vector<T> aRow;
        if( ReadNumbers(myline, aRow) == cols )
            m.push_back(aRow);
        else
        {
            std::cerr<<"The number of elements of row "<<i<<" does not match the number of columns!!!"<<std::endl;
            assert(false);
        }
    }
    myfile.close();
    rows = i;
    assert(rows < SIZE_MAX);
    assert(rows == m.size());
}



template <typename MatrixType>
void ReadMatrix_EIGEN(const char* filename, MatrixType &M)
/// Read the matrix file, set rows and cols, return a CV_32F matrix
{
    std::vector< std::vector<double> > m;
    size_t rows, cols;
    ReadMatrixFile(filename, m, rows, cols);
    M.setZero(rows, cols);
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            M(i,j) = m[i][j];
        }
    }
}



template <typename MatrixType>
void ReadProposals_EIGEN(std::vector<MatrixType> &proposals, const std::string& filename)
/// Read the (unary) data terms and the disparities of the proposals
/// Proposal file:
/// row 0: height width 0 0 ....
/// From row 1: each row corresponds to a proposal (either disparity or unary cost, depending on the input file), COLUMN MAJOR
{
    proposals.clear();
    MatrixType M;
    ReadMatrix_EIGEN(filename.c_str(), M);

    //std::cout<<M.leftCols(10)<<std::endl;
    //std::cout<<"Proposal 0"<<M.row(1).leftCols(10)<<std::endl;

    size_t height = M(0,0);
    size_t width = M(0,1);
    size_t numProposals = M.rows() - 1;
    proposals.reserve(numProposals);


//    MatrixType mm = MatrixType::Map(v.data(), height, width);
//    std::cout<<"mm = "<<std::endl;
//    std::cout<<mm.block(0,0,18,10)<<std::endl;

    for(size_t i = 0; i < numProposals; i++){
        // IMPORTANT: Eigen matrix is COLUMN-MAJOR
        Eigen::VectorXd row = M.row(i+1);
        proposals.push_back(MatrixType::Map(row.data(), height, width));
        //cout<<"i = "<<i<<":"<<std::endl;
        std::string fname = "proposal_" + std::to_string(i);
        WriteMatrix2File(proposals[i], fname.c_str());
        //std::cout<<m.block(0,0,18,10)<<std::endl;
        //std::cout<<proposals[i].leftCols(10)<<std::endl;
    }
}



template <typename T>
void WriteMatrix2File(const std::vector<T>&m, size_t rows, const char* filename)
/// Read the matrix file, set rows and cols, return a CV_32F matrix
{
    assert(m.size()%rows == 0);
    size_t cols = m.size()/rows;
    std::ofstream output(filename);
    if (output.is_open())
    {
        for(size_t i = 0; i < m.size(); i++){
            output<<m[i]<<" ";
            if( (i+1)%cols == 0 ){
                output<<std::endl;
            }
        }
        output.close();
    }
    else std::cout << "Unable to open file" <<std::endl;
}


template <typename MatrixType>
void WriteMatrix2File(const MatrixType &m, const char* filename)
/// Read the matrix file, set rows and cols, return a CV_32F matrix
{
    std::ofstream output(filename);
    if (output.is_open())
    {
        output<<m;
        output.close();
    }
    else std::cout << "Unable to open file" <<std::endl;
}


template <typename T>
void ReadVectorFile(const char* filename, std::vector<T> &m)
/// Read the matrix file, set rows and cols, return a CV_32F matrix
{
    std::ifstream myfile;
    std::string myline;
    size_t cols;

    myfile.open(filename);
    assert (myfile.is_open());

    m.clear();

    size_t i = 0;

    std::vector<T> firstRow;
    std::getline(myfile, myline);
    cols = ReadNumbers( myline, firstRow );
    assert(cols == 1);
    m.push_back(firstRow[0]);
    //cout << "cols:" << cols << endl;


    for ( i=1; i < SIZE_MAX; i++)
    {
        if ( !std::getline(myfile, myline) ) break;
        std::vector<T> aRow;
        if( ReadNumbers(myline, aRow) == cols )
            m.push_back(aRow[0]);
        else
        {
            std::cerr<<"The number of elements of row "<<i<<" is greater than 1!!!"<<std::endl;
            assert(false);
        }
    }
    myfile.close();
    assert(i < SIZE_MAX);
    assert(i == m.size());
}



template <typename T>
void write2hdf(H5::H5File* file, const std::vector<T> &values, const std::string &name, size_t rows, size_t cols)
{
    hsize_t fdim[] = {rows, cols}; // dim sizes of ds (on disk)
    H5::DataSpace fspace( 2, fdim );

    // Create property list for a dataset and fill it with 0.
    H5::DSetCreatPropList plist;

    if (typeid(T) == typeid(int) || typeid(T) == typeid(size_t)){
        std::vector<int> newValues(values.begin(), values.end());
        plist.setFillValue(H5::PredType::NATIVE_INT, 0);
        H5::DataSet* dataset = new H5::DataSet(file->createDataSet(name, H5::PredType::NATIVE_INT, fspace, plist));
        dataset->write( newValues.data(), H5::PredType::NATIVE_INT);
        delete dataset;
    }else if(typeid(T) == typeid(double)){
        plist.setFillValue(H5::PredType::NATIVE_DOUBLE, 0);
        H5::DataSet* dataset = new H5::DataSet(file->createDataSet(name, H5::PredType::NATIVE_DOUBLE, fspace, plist));
        dataset->write( values.data(), H5::PredType::NATIVE_DOUBLE);
        delete dataset;
    }else{
        std::cerr<<"Data type not supported!"<<std::endl;
    }

}

#endif // OUTILS_IO_H
