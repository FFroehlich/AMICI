#ifndef RETURNDATA_MATLAB_H
#define RETURNDATA_MATLAB_H

#include "amici/rdata.h"

#include <vector>

#include <mex.h>

namespace amici {

/**
 * @brief generates matlab mxArray from a ReturnData object
 * @param rdata ReturnDataObject
 * @return rdatamatlab ReturnDataObject stored as matlab compatible data
 */
mxArray* getReturnDataMatlabFromAmiciCall(ReturnData const* rdata);

/**
 * @brief allocates and initializes solution mxArray with the corresponding
 * fields
 * @param rdata ReturnDataObject
 * @return Solution mxArray
 */
mxArray* initMatlabReturnFields(ReturnData const* rdata);

/**
 * @brief allocates and initializes diagnosis mxArray with the corresponding
 * fields
 * @param rdata ReturnDataObject
 * @return Diagnosis mxArray
 */
mxArray* initMatlabDiagnosisFields(ReturnData const* rdata);

/**
 * @brief initialize vector and attach to the field
 * @param matlabStruct pointer of the field to which the vector will be
 * attached
 * @param fieldName Name of the field to which the vector will be attached
 * @param fieldData Data which will be stored in the field
 */
template <typename T>
void writeMatlabField0(
    mxArray* matlabStruct, char const* fieldName, T fieldData
);

/**
 * @brief initialize vector and attach to the field
 * @param matlabStruct pointer of the field to which the vector will be
 * attached
 * @param fieldName Name of the field to which the vector will be attached
 * @param fieldData Data which will be stored in the field
 * @param dim0 Number of elements in the vector
 */
template <typename T>
void writeMatlabField1(
    mxArray* matlabStruct, char const* fieldName,
    gsl::span<T const> const& fieldData, mwSize const dim0
);

/**
 * @brief initialize matrix, attach to the field and write data
 * @param matlabStruct Pointer to the matlab structure
 * @param fieldName Name of the field to which the tensor will be attached
 * @param fieldData Data which will be stored in the field
 * @param dim0 Number of rows in the tensor
 * @param dim1 Number of columns in the tensor
 * @param perm reordering of dimensions (i.e., transposition)
 */
template <typename T>
void writeMatlabField2(
    mxArray* matlabStruct, char const* fieldName,
    std::vector<T> const& fieldData, mwSize dim0, mwSize dim1,
    std::vector<int> perm
);

/**
 * @brief initialize 3D tensor, attach to the field and write data
 * @param matlabStruct Pointer to the matlab structure
 * @param fieldName Name of the field to which the tensor will be attached
 * @param fieldData Data which will be stored in the field
 * @param dim0 number of rows in the tensor
 * @param dim1 number of columns in the tensor
 * @param dim2 number of elements in the third dimension of the tensor
 * @param perm reordering of dimensions
 */
template <typename T>
void writeMatlabField3(
    mxArray* matlabStruct, char const* fieldName,
    std::vector<T> const& fieldData, mwSize dim0, mwSize dim1, mwSize dim2,
    std::vector<int> perm
);

/**
 * @brief initialize 4D tensor, attach to the field and write data
 * @param matlabStruct Pointer to the matlab structure
 * @param fieldName Name of the field to which the tensor will be attached
 * @param fieldData Data which will be stored in the field
 * @param dim0 number of rows in the tensor
 * @param dim1 number of columns in the tensor
 * @param dim2 number of elements in the third dimension of the tensor
 * @param dim3 number of elements in the fourth dimension of the tensor
 * @param perm reordering of dimensions
 */
template <typename T>
void writeMatlabField4(
    mxArray* matlabStruct, char const* fieldName,
    std::vector<T> const& fieldData, mwSize dim0, mwSize dim1, mwSize dim2,
    mwSize dim3, std::vector<int> perm
);

/**
 * @brief initializes the field fieldName in matlabStruct with dimension dim
 * @param matlabStruct Pointer to the matlab structure
 * @param fieldName Name of the field to which the tensor will be attached
 * @param dim vector of field dimensions
 *
 * @return pointer to field data
 */
double* initAndAttachArray(
    mxArray* matlabStruct, char const* fieldName, std::vector<mwSize> dim
);

/**
 * @brief checks whether fieldNames was properly allocated
 * @param fieldNames array of field names
 * @param fieldCount expected number of fields in fieldNames
 */
void checkFieldNames(char const** fieldNames, int const fieldCount);

/**
 * @brief template function that reorders elements in a std::vector
 *
 * @param input unordered vector
 * @param order dimension permutation
 *
 * @return Reordered vector
 */
template <typename T>
std::vector<T>
reorder(std::vector<T> const& input, std::vector<int> const& order);

} // namespace amici

#endif // RETURNDATA_MATLAB_H
