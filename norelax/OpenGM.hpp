#ifndef OPENGM_H
#define OPENGM_H

#include <iostream>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include "opengm/functions/sparsemarray.hxx"
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/fieldofexperts.hxx"


#include <opengm/inference/icm.hxx>
//#include <opengm/inference/messagepassing/messagepassing.hxx>
//#include <opengm/inference/trws/trws_trws.hxx>
//#include <opengm/inference/trws/trws_adsal.hxx>
//#include <opengm/inference/graphcut.hxx>
//#include <opengm/inference/alphaexpansion.hxx>
//#include <opengm/inference/external/fastPD.hxx>
//#include <opengm/inference/external/mplp.hxx>
//#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
//#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>


//#include <opencv2/core/utility.hpp>
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"

//using namespace std; // ’using’ is used only in example code
//using namespace opengm;

//*******************
//** Typedefs
//*******************
typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::Adder OperatorType;
typedef opengm::Minimizer AccumulatorType;
typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;
typedef opengm::ExplicitFunction<ValueType, IndexType, LabelType> ExplFunction;
typedef opengm::SparseFunction<ValueType, IndexType, LabelType> SSparseFunction;
typedef opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType> TruncAbsDifFunction;
typedef opengm::FoEFunction<ValueType, IndexType, LabelType> FoEFunction;
// Set functions for graphical model

typedef opengm::meta::TypeListGenerator< ExplFunction, SSparseFunction,
opengm::PottsFunction<ValueType, IndexType, LabelType>,
opengm::PottsNFunction<ValueType, IndexType, LabelType>,
opengm::PottsGFunction<ValueType, IndexType, LabelType>,
opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
TruncAbsDifFunction >::type FunctionTypeList;


// Explicit model
typedef opengm::GraphicalModel<
     ValueType,
     OperatorType,
     FunctionTypeList,
     SpaceType> GM;

typedef opengm::GraphicalModel<ValueType,OperatorType,ExplFunction,SpaceType>  ModelExplicit;


#endif // OPENGM_H
