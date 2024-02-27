#include <amici/sundials_matrix_wrapper.h>
#include <sundials/sundials_matrix.h> // return codes

#include <amici/cblas.h>

#include <new>       // bad_alloc
#include <stdexcept> // invalid_argument and domain_error
#include <utility>

namespace amici {

SUNMatrixWrapper::SUNMatrixWrapper(
    sunindextype M, sunindextype N, sunindextype NNZ, int sparsetype
)
    : matrix_(SUNSparseMatrix(M, N, NNZ, sparsetype))
    , id_(SUNMATRIX_SPARSE)
    , sparsetype_(sparsetype) {

    if (sparsetype != CSC_MAT && sparsetype != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT");

    if (NNZ && M && N && !matrix_)
        throw std::bad_alloc();

    finish_init();
    assert(num_nonzeros() == 0);
    assert(NNZ == capacity() || !matrix_);
    assert(M == rows() || !matrix_);
    assert(N == columns() || !matrix_);
}

SUNMatrixWrapper::SUNMatrixWrapper(sunindextype M, sunindextype N)
    : matrix_(SUNDenseMatrix(M, N))
    , id_(SUNMATRIX_DENSE) {
    if (M && N && !matrix_)
        throw std::bad_alloc();

    finish_init();
    assert(M == rows() || !matrix_);
    assert(N == columns() || !matrix_);
}

SUNMatrixWrapper::SUNMatrixWrapper(
    sunindextype M, sunindextype ubw, sunindextype lbw
)
    : matrix_(SUNBandMatrix(M, ubw, lbw))
    , id_(SUNMATRIX_BAND) {
    if (M && !matrix_)
        throw std::bad_alloc();
    finish_init();
}

SUNMatrixWrapper::SUNMatrixWrapper(
    SUNMatrixWrapper const& A, realtype droptol, int sparsetype
)
    : id_(SUNMATRIX_SPARSE)
    , sparsetype_(sparsetype) {
    if (sparsetype != CSC_MAT && sparsetype != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT");

    switch (A.matrix_id()) {
    case SUNMATRIX_DENSE:
        matrix_ = SUNSparseFromDenseMatrix(A.get(), droptol, sparsetype);
        break;
    case SUNMATRIX_BAND:
        matrix_ = SUNSparseFromBandMatrix(A.get(), droptol, sparsetype);
        break;
    default:
        throw std::invalid_argument("Invalid Matrix. Must be SUNMATRIX_DENSE or"
                                    " SUNMATRIX_BAND");
    }
    if (!matrix_)
        throw std::bad_alloc();
    finish_init();
    num_nonzeros_ = indexptrs_[num_indexptrs()];
}

inline static SUNMatrix_ID get_sparse_id_w_default(SUNMatrix mat) {
    if (mat)
        return SUNMatGetID(mat);
    return SUNMATRIX_CUSTOM;
}

inline static int get_sparse_type_w_default(SUNMatrix mat) {
    if (mat && SUNMatGetID(mat) == SUNMATRIX_SPARSE)
        return SM_SPARSETYPE_S(mat);
    return CSC_MAT;
}

SUNMatrixWrapper::SUNMatrixWrapper(SUNMatrix mat)
    : matrix_(mat)
    , id_(get_sparse_id_w_default(mat))
    , sparsetype_(get_sparse_type_w_default(mat))
    , ownmat(false) {
    finish_init();
}

SUNMatrixWrapper::~SUNMatrixWrapper() {
    if (matrix_ && ownmat)
        SUNMatDestroy(matrix_);
}

SUNMatrixWrapper::SUNMatrixWrapper(SUNMatrixWrapper const& other)
    : id_(get_sparse_id_w_default(other.matrix_))
    , sparsetype_(get_sparse_type_w_default(other.matrix_)) {
    if (!other.matrix_)
        return;

    matrix_ = SUNMatClone(other.matrix_);
    if (!matrix_)
        throw std::bad_alloc();

    SUNMatCopy(other.matrix_, matrix_);
    finish_init();
}

SUNMatrixWrapper::SUNMatrixWrapper(SUNMatrixWrapper&& other)
    : id_(get_sparse_id_w_default(other.matrix_))
    , sparsetype_(get_sparse_type_w_default(other.matrix_)) {
    std::swap(matrix_, other.matrix_);
    finish_init();
}

SUNMatrixWrapper& SUNMatrixWrapper::operator=(SUNMatrixWrapper const& other) {
    if (&other == this)
        return *this;
    return *this = SUNMatrixWrapper(other);
}

SUNMatrixWrapper& SUNMatrixWrapper::operator=(SUNMatrixWrapper&& other) {
    std::swap(matrix_, other.matrix_);
    id_ = other.id_;
    sparsetype_ = other.sparsetype_;
    finish_init();
    return *this;
}

void SUNMatrixWrapper::reallocate(sunindextype NNZ) {
    if (sparsetype() != CSC_MAT && sparsetype() != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT.");

    if (int ret = SUNSparseMatrix_Reallocate(matrix_, NNZ) != SUNMAT_SUCCESS)
        throw std::runtime_error(
            "SUNSparseMatrix_Reallocate failed with "
            "error code "
            + std::to_string(ret) + "."
        );

    update_ptrs();
    capacity_ = NNZ;
    assert((NNZ && columns() && rows()) || !matrix_);
}

void SUNMatrixWrapper::realloc() {
    if (sparsetype() != CSC_MAT && sparsetype() != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT.");
    if (int ret = SUNSparseMatrix_Realloc(matrix_) != SUNMAT_SUCCESS)
        throw std::runtime_error(
            "SUNSparseMatrix_Realloc failed with "
            "error code "
            + std::to_string(ret) + "."
        );

    update_ptrs();
    capacity_ = num_nonzeros_;
    assert(capacity() || !matrix_);
}

sunindextype SUNMatrixWrapper::num_indexptrs() const {
    assert(matrix_id() == SUNMATRIX_SPARSE);
    assert(
        !matrix_
        || (sparsetype() == CSC_MAT ? num_indexptrs_ == num_columns_
                                    : num_indexptrs_ == num_rows_)
    );
    assert(!matrix_ || num_indexptrs_ == SM_NP_S(matrix_));
    return num_indexptrs_;
}

sunindextype SUNMatrixWrapper::capacity() const {
    assert(matrix_id() == SUNMATRIX_SPARSE);
    assert(!matrix_ || capacity_ == SM_NNZ_S(matrix_));
    return capacity_;
}

sunindextype SUNMatrixWrapper::num_nonzeros() const {
    assert(matrix_id() == SUNMATRIX_SPARSE);
    assert(
        !matrix_ || num_nonzeros_ == SM_INDEXPTRS_S(matrix_)[SM_NP_S(matrix_)]
    );
    return num_nonzeros_;
}

realtype const* SUNMatrixWrapper::data() const { return data_; }

realtype* SUNMatrixWrapper::data() { return data_; }

int SUNMatrixWrapper::sparsetype() const {
    assert(matrix_);
    assert(matrix_id() == SUNMATRIX_SPARSE);
    return sparsetype_;
}

void SUNMatrixWrapper::scale(realtype a) {
    if (matrix_) {
        auto cap = capacity();
        for (sunindextype idx = 0; idx < cap; ++idx)
            data_[idx] *= a;
    }
}

void SUNMatrixWrapper::multiply(
    N_Vector c, const_N_Vector b, realtype const alpha
) const {
    multiply(
        gsl::make_span<realtype>(NV_DATA_S(c), NV_LENGTH_S(c)),
        gsl::make_span<realtype const>(NV_DATA_S(b), NV_LENGTH_S(b)), alpha
    );
}

#ifndef NDEBUG
inline static void check_csc(SUNMatrixWrapper const* mat) {
    assert(mat->matrix_id() == SUNMATRIX_SPARSE);
    assert(mat->sparsetype() == CSC_MAT);
}
#else
// avoid "unused parameter" warning
inline static void check_csc(SUNMatrixWrapper const* /*mat*/) {}
#endif

void SUNMatrixWrapper::multiply(
    gsl::span<realtype> c, gsl::span<realtype const> b, realtype const alpha
) const {

    if (!matrix_)
        return;

    assert(rows() == gsl::narrow<sunindextype>(c.size()));
    assert(columns() == gsl::narrow<sunindextype>(b.size()));

    switch (matrix_id()) {
    case SUNMATRIX_DENSE:
        amici_dgemv(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            gsl::narrow<int>(rows()), gsl::narrow<int>(columns()), alpha,
            data(), gsl::narrow<int>(rows()), b.data(), 1, 1.0, c.data(), 1
        );
        break;
    case SUNMATRIX_SPARSE:
        if (!num_nonzeros()) {
            return;
        }
        check_csc(this);
        for (sunindextype icol = 0; icol < columns(); ++icol) {
            scatter(icol, b[icol] * alpha, nullptr, c, icol + 1, nullptr, 0);
        }
        break;
    default:
        throw std::domain_error("Not Implemented.");
    }
}

void SUNMatrixWrapper::multiply(
    N_Vector c, const_N_Vector b, gsl::span<int const> cols, bool transpose
) const {
    multiply(
        gsl::make_span<realtype>(NV_DATA_S(c), NV_LENGTH_S(c)),
        gsl::make_span<realtype const>(NV_DATA_S(b), NV_LENGTH_S(b)), cols,
        transpose
    );
}

void SUNMatrixWrapper::multiply(
    gsl::span<realtype> c, gsl::span<realtype const> b,
    gsl::span<int const> cols, bool transpose
) const {
    if (!matrix_)
        return;

    if (transpose) {
        assert(cols.size() == c.size());
        assert(rows() == gsl::narrow<sunindextype>(b.size()));
    } else {
        assert(rows() == gsl::narrow<sunindextype>(c.size()));
        assert(columns() == gsl::narrow<sunindextype>(b.size()));
    }

    check_csc(this);

    if (!num_nonzeros())
        return;

    /* Carry out actual multiplication */
    auto c_ptr = c.data();
    auto b_ptr = b.data();

    if (transpose) {
        auto cols_size = cols.size();
        for (std::size_t icols = 0; icols < cols_size; ++icols) {
            auto idx_next_col = get_indexptr(cols[icols] + 1);
            for (sunindextype idx = get_indexptr(cols[icols]);
                 idx < idx_next_col; ++idx) {

                auto idx_val = get_indexval(idx);
                assert(icols < c.size());
                assert(gsl::narrow<std::size_t>(idx_val) < b.size());

                c_ptr[icols] += get_data(idx) * b_ptr[idx_val];
            }
        }
    } else {
        auto num_cols = gsl::narrow<std::size_t>(columns());
        for (std::size_t icols = 0; icols < num_cols; ++icols) {
            auto idx_next_col = get_indexptr(cols[icols] + 1);

            for (sunindextype idx = get_indexptr(cols[icols]);
                 idx < idx_next_col; ++idx) {
                auto idx_val = get_indexval(idx);

                assert(icols < b.size());
                assert(gsl::narrow<std::size_t>(idx_val) < c.size());

                c_ptr[idx_val] += get_data(idx) * b_ptr[icols];
            }
        }
    }
}

void SUNMatrixWrapper::sparse_multiply(
    SUNMatrixWrapper& C, SUNMatrixWrapper const& B
) const {
    if (!matrix_ || !B.matrix_ || !C.matrix_)
        return;

    check_csc(this);
    check_csc(&B);
    check_csc(&C);

    assert(rows() == gsl::narrow<sunindextype>(C.rows()));
    assert(C.columns() == B.columns());
    assert(gsl::narrow<sunindextype>(B.rows()) == columns());

    C.zero();

    if (columns() == 0 || rows() == 0 || B.columns() == 0)
        return; // matrix will also have zero size

    if (num_nonzeros() == 0 || B.num_nonzeros() == 0)
        return; // nothing to multiply

    /* see
     * https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_multiply.c
     * modified such that we don't need to use CSparse memory structure and can
     * work with preallocated C. This should minimize number of necessary
     * reallocations as we can assume that C doesn't change size.
     */

    sunindextype nnz = 0; // this keeps track of the nonzero index in C
    sunindextype bcol;
    sunindextype bidx;
    sunindextype cidx;

    auto w = std::vector<sunindextype>(rows()); // sparsity of C(:,j)
    auto x = std::vector<realtype>(rows());     // entries in C(:,j)

    for (bcol = 0; bcol < B.columns();
         bcol++) // k in C(i,j) = sum_k A(i,k)*B(k,j)
    {
        C.set_indexptr(bcol, nnz); /* column j of C starts here */
        if ((B.get_indexptr(bcol + 1) > B.get_indexptr(bcol))
            && (nnz + rows() > C.capacity())) {
            /*
             * if memory usage becomes a concern, remove the factor two here,
             * as it effectively trades memory efficiency against less
             * reallocations
             */
            C.reallocate(2 * C.capacity() + rows());
        }
        for (bidx = B.get_indexptr(bcol); bidx < B.get_indexptr(bcol + 1);
             bidx++) {
            nnz = scatter(
                B.get_indexval(bidx), B.get_data(bidx), w.data(),
                gsl::make_span(x), bcol + 1, &C, nnz
            );
            assert(nnz - C.get_indexptr(bcol) <= rows());
        }
        for (cidx = C.get_indexptr(bcol); cidx < nnz; cidx++)
            C.set_data(cidx, x[C.get_indexval(cidx)]); // copy data to C
    }
    C.set_indexptr(C.num_indexptrs(), nnz);

    /*
     * do not reallocate here since we rather keep a matrix that is a bit
     * bigger than repeatedly resizing this matrix.
     */
}

void SUNMatrixWrapper::sparse_add(
    SUNMatrixWrapper const& A, realtype alpha, SUNMatrixWrapper const& B,
    realtype beta
) {
    // matrix_ == nullptr is allowed on the first call
    if (!A.matrix_ || !B.matrix_)
        return;

    check_csc(this);
    check_csc(&A);
    check_csc(&B);

    assert(rows() == gsl::narrow<sunindextype>(A.rows()));
    assert(rows() == gsl::narrow<sunindextype>(B.rows()));
    assert(columns() == gsl::narrow<sunindextype>(A.columns()));
    assert(columns() == gsl::narrow<sunindextype>(B.columns()));

    zero();

    if (columns() == 0 || rows() == 0
        || (A.num_nonzeros() + B.num_nonzeros() == 0))
        return; // nothing to do

    /* see
     * https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_add.c
     * modified such that we don't need to use CSparse memory structure and can
     * work with preallocated C. This should minimize number of necessary
     * reallocations as we can assume that C doesn't change size.
     */

    sunindextype nnz = 0; // this keeps track of the nonzero index in C

    sunindextype ccol;
    sunindextype cidx;

    // first call, make sure that matrix is initialized with no capacity
    if (!capacity())
        reallocate(A.num_nonzeros() + B.num_nonzeros());

    auto w = std::vector<sunindextype>(rows());
    auto x = std::vector<realtype>(rows());

    for (ccol = 0; ccol < columns(); ccol++) {
        set_indexptr(ccol, nnz); /* column j of C starts here */
        nnz = A.scatter(
            ccol, alpha, w.data(), gsl::make_span(x), ccol + 1, this, nnz
        );
        nnz = B.scatter(
            ccol, beta, w.data(), gsl::make_span(x), ccol + 1, this, nnz
        );
        // no reallocation should happen here
        for (cidx = get_indexptr(ccol); cidx < nnz; cidx++) {
            auto x_idx = get_indexval(cidx);
            assert(x_idx >= 0 && gsl::narrow<std::size_t>(x_idx) < x.size());
            set_data(cidx, x[x_idx]); // copy data to C
        }
    }
    set_indexptr(num_indexptrs(), nnz);
    if (capacity() == A.num_nonzeros() + B.num_nonzeros())
        realloc(
        ); // resize if necessary, will have correct size in future calls
}

void SUNMatrixWrapper::sparse_sum(std::vector<SUNMatrixWrapper> const& mats) {
    // matrix_ == nullptr is allowed on the first call
    auto all_empty
        = std::all_of(mats.begin(), mats.end(), [](SUNMatrixWrapper const& m) {
              return !m.matrix_;
          });
    if (all_empty)
        return;

    check_csc(this);
    int max_total_nonzero = 0;
    for (auto& mat : mats) {
        check_csc(&mat);
        assert(rows() == mat.rows());
        assert(columns() == mat.columns());
        max_total_nonzero += mat.num_nonzeros();
    }

    zero();

    if (columns() == 0 || rows() == 0 || max_total_nonzero == 0)
        return; // nothing to do

    /* see
     * https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_add.c
     * modified such that we don't need to use CSparse memory structure and can
     * work with preallocated C. This should minimize number of necessary
     * reallocations as we can assume that C doesn't change size.
     */

    sunindextype nnz = 0; // this keeps track of the nonzero index in C

    sunindextype acol;
    sunindextype aidx;
    // first call, make sure that matrix is initialized with no capacity
    if (!capacity())
        reallocate(max_total_nonzero);

    auto w = std::vector<sunindextype>(rows());
    auto x = std::vector<realtype>(rows());

    for (acol = 0; acol < columns(); acol++) {
        set_indexptr(acol, nnz); /* column j of A starts here */
        for (auto& mat : mats)
            nnz = mat.scatter(
                acol, 1.0, w.data(), gsl::make_span(x), acol + 1, this, nnz
            );
        // no reallocation should happen here
        for (aidx = get_indexptr(acol); aidx < nnz; aidx++) {
            auto x_idx = get_indexval(aidx);
            assert(x_idx >= 0 && gsl::narrow<std::size_t>(x_idx) < x.size());
            set_data(aidx, x[x_idx]); // copy data to C
        }
    }
    set_indexptr(num_indexptrs(), nnz);
    if (capacity() == max_total_nonzero)
        realloc(); // resize if necessary
}

sunindextype SUNMatrixWrapper::scatter(
    sunindextype const acol, realtype const beta, sunindextype* w,
    gsl::span<realtype> x, sunindextype const mark, SUNMatrixWrapper* C,
    sunindextype nnz
) const {
    if (!matrix_)
        return nnz;

    check_csc(this);
    if (C && C->matrix_)
        check_csc(C);

    if (!num_nonzeros())
        return nnz;

    auto x_data = x.data();
    /* see
     * https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_scatter.c
     */

    sunindextype aidx;
    for (aidx = get_indexptr(acol); aidx < get_indexptr(acol + 1); aidx++) {
        auto arow = get_indexval(aidx); /* A(arow,acol) is nonzero */
        assert(arow >= 0 && gsl::narrow<std::size_t>(arow) <= x.size());
        if (w && w[arow] < mark) {
            w[arow] = mark; /* arow is new entry in C(:,*) */
            if (C)
                C->set_indexval(
                    nnz++, arow
                ); /* add arow to pattern of C(:,*) */
            x_data[arow]
                = beta * get_data(aidx); /* x(arow) = beta*A(arow,acol) */
        } else {
            x_data[arow]
                += beta * get_data(aidx); /* arow exists in C(:,*) already */
        }
    }
    assert(!C || nnz <= C->capacity());
    return nnz;
}

// https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_cumsum.c
/* p [0..n] = cumulative sum of c[0..n-1], and then copy p [0..n-1] into c */
static void cumsum(gsl::span<sunindextype> p, std::vector<sunindextype>& c) {
    sunindextype nz = 0;
    assert(p.size() == c.size() + 1);
    for (sunindextype i = 0; i < gsl::narrow<sunindextype>(c.size()); i++) {
        p[i] = nz;
        nz += c[i];
        c[i] = p[i]; /* also copy p[0..n-1] back into c[0..n-1]*/
    }
    p[c.size()] = nz;
}

void SUNMatrixWrapper::transpose(
    SUNMatrixWrapper& C, realtype const alpha, sunindextype blocksize
) const {
    if (!matrix_ || !C.matrix_)
        return;

    auto C_matrix_id = C.matrix_id();
    if (!((C_matrix_id == SUNMATRIX_SPARSE && C.sparsetype() == CSC_MAT)
          || C_matrix_id == SUNMATRIX_DENSE))
        throw std::domain_error("Not Implemented.");

    check_csc(this);
    assert(rows() == C.rows());
    assert(columns() == C.columns());
    if (C_matrix_id == SUNMATRIX_SPARSE) {
        if (!C.capacity() && num_nonzeros())
            C.reallocate(num_nonzeros());
        assert(C.capacity() >= num_nonzeros());
    }

    assert(columns() % blocksize == 0);
    assert(rows() % blocksize == 0);
    C.zero();

    if (!num_nonzeros() || !columns() || !rows())
        return;

    // see
    // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_transpose.c

    auto nrows = rows();

    if (C_matrix_id == SUNMATRIX_SPARSE) {
        std::vector<sunindextype> w(columns());
        auto w_data = w.data();

        for (sunindextype acol = 0; acol < nrows; acol++) { /* row counts */
            auto next_indexptr = get_indexptr(acol + 1);
            auto widx_offset = (acol / blocksize) * blocksize;
            for (sunindextype aidx = get_indexptr(acol); aidx < next_indexptr;
                 aidx++) {
                sunindextype widx
                    = widx_offset + get_indexval(aidx) % blocksize;
                assert(widx >= 0 && widx < (sunindextype)w.size());
                w_data[widx]++;
                assert(w_data[widx] <= nrows);
            }
        }
        /* row pointers */
        cumsum(gsl::make_span(C.indexptrs_, C.columns() + 1), w);

        for (sunindextype acol = 0; acol < nrows; acol++) {
            auto next_indexptr = get_indexptr(acol + 1);
            auto ccol_offset = (acol / blocksize) * blocksize;
            auto crow_offset = acol % blocksize;
            for (sunindextype aidx = get_indexptr(acol); aidx < next_indexptr;
                 aidx++) {
                auto indexval_aidx = get_indexval(aidx);
                sunindextype ccol = ccol_offset + indexval_aidx % blocksize;
                sunindextype crow
                    = (indexval_aidx / blocksize) * blocksize + crow_offset;
                assert(crow < nrows);
                assert(ccol < columns());
                assert(aidx < capacity());
                assert(ccol >= 0 && ccol < (sunindextype)w.size());
                sunindextype cidx = w_data[ccol]++;
                C.set_indexval(cidx, crow); /* place A(i,j) as entry C(j,i) */
                C.set_data(cidx, alpha * get_data(aidx));
            }
        }
    } else {

        for (sunindextype acol = 0; acol < nrows; acol++) {
            auto next_indexptr = get_indexptr(acol + 1);

            for (sunindextype aidx = get_indexptr(acol); aidx < next_indexptr;
                 aidx++) {
                sunindextype ccol = (acol / blocksize) * blocksize
                                    + get_indexval(aidx) % blocksize;
                sunindextype crow = (get_indexval(aidx) / blocksize) * blocksize
                                    + acol % blocksize;
                assert(crow < nrows);
                assert(ccol < columns());
                C.set_data(crow, ccol, alpha * get_data(aidx));
            }
        }
    }
}

void SUNMatrixWrapper::to_dense(SUNMatrixWrapper& D) const {
    if (!matrix_ || !D.matrix_)
        return;
    check_csc(this);
    assert(rows() == D.rows());
    assert(columns() == D.columns());

    D.zero();
    if (!num_nonzeros())
        return;

    sunindextype icol;
    sunindextype idx;
    for (icol = 0; icol < columns(); ++icol)
        for (idx = get_indexptr(icol); idx < get_indexptr(icol + 1); ++idx) {
            D.set_data(get_indexval(idx), icol, get_data(idx));
        }
}

void SUNMatrixWrapper::to_diag(N_Vector v) const {
    if (!matrix_ || !v)
        return;
    check_csc(this);
    assert(rows() == columns());
    assert(rows() == NV_LENGTH_S(v));

    N_VConst(0.0, v);
    if (!num_nonzeros())
        return;

    sunindextype icol;
    sunindextype idx;
    for (icol = 0; icol < columns(); ++icol)
        for (idx = get_indexptr(icol); idx < get_indexptr(icol + 1); ++idx)
            if (get_indexval(idx) == icol)
                NV_Ith_S(v, icol) = get_data(idx);
}

void SUNMatrixWrapper::zero() {
    if (!matrix_)
        return;
    if (int res = SUNMatZero(matrix_))
        throw std::runtime_error(
            "SUNMatrixWrapper::zero() failed with " + std::to_string(res) + "."
        );
}

void SUNMatrixWrapper::finish_init() {
    update_ptrs();
    update_size();
}

void SUNMatrixWrapper::update_ptrs() {
    if (!matrix_) {
        data_ = nullptr;
        indexptrs_ = nullptr;
        indexvals_ = nullptr;
        return;
    }

    switch (matrix_id()) {
    case SUNMATRIX_DENSE:
        data_ = SM_DATA_D(matrix_);
        break;
    case SUNMATRIX_SPARSE:
        data_ = SM_DATA_S(matrix_);
        indexptrs_ = SM_INDEXPTRS_S(matrix_);
        indexvals_ = SM_INDEXVALS_S(matrix_);
        break;
    default:
        throw std::domain_error("Not Implemented.");
    }
}

void SUNMatrixWrapper::update_size() {
    num_indexptrs_ = 0;
    if (!matrix_) {
        num_rows_ = 0;
        num_columns_ = 0;
        return;
    }

    switch (matrix_id()) {
    case SUNMATRIX_DENSE:
        num_rows_ = SM_ROWS_D(matrix_);
        num_columns_ = SM_COLUMNS_D(matrix_);
        capacity_ = num_rows_ * num_columns_;
        break;
    case SUNMATRIX_SPARSE:
        num_rows_ = SM_ROWS_S(matrix_);
        num_columns_ = SM_COLUMNS_S(matrix_);
        capacity_ = SM_NNZ_S(matrix_);
        num_indexptrs_ = SM_NP_S(matrix_);
        break;
    default:
        throw std::domain_error("Not Implemented.");
    }
}

void SUNMatrixWrapper::refresh() {
    update_ptrs();
    update_size();
    if (matrix_id() == SUNMATRIX_SPARSE)
        num_nonzeros_ = SM_INDEXPTRS_S(matrix_)[SM_NP_S(matrix_)];
}

SUNMatrix SUNMatrixWrapper::get() const { return matrix_; }

std::pair<sunindextype, sunindextype>
unravel_index(sunindextype i, SUNMatrix m) {
    gsl_ExpectsDebug(i >= 0);
    auto mat_id = SUNMatGetID(m);
    if (mat_id == SUNMATRIX_DENSE) {
        gsl_ExpectsDebug(i < SM_COLUMNS_D(m) * SM_ROWS_D(m));

        auto num_rows = SM_ROWS_D(m);
        // col-major
        sunindextype row = i % num_rows;
        sunindextype col = i / num_rows;

        gsl_EnsuresDebug(row >= 0);
        gsl_EnsuresDebug(row < SM_ROWS_D(m));
        gsl_EnsuresDebug(col >= 0);
        gsl_EnsuresDebug(col < SM_COLUMNS_D(m));

        return {row, col};
    }

    if (mat_id == SUNMATRIX_SPARSE) {
        auto nnz = SM_NNZ_S(m);
        auto ncols = SM_COLUMNS_S(m);
        auto index_vals = SM_INDEXVALS_S(m);
        auto index_ptrs = SM_INDEXPTRS_S(m);
        gsl_ExpectsDebug(i < nnz);
        sunindextype row = index_vals[i];
        sunindextype col = 0;
        while (col < ncols && index_ptrs[col + 1] <= i)
            ++col;

        // This can happen if indexvals / indexptrs haven't been set.
        if (col == ncols)
            return {-1, -1};

        gsl_EnsuresDebug(row >= 0);
        gsl_EnsuresDebug(row < SM_ROWS_S(m));
        gsl_EnsuresDebug(col >= 0);
        gsl_EnsuresDebug(col < ncols);
        return {row, col};
    }

    throw amici::AmiException("Unimplemented SUNMatrix type for unravel_index");
}

} // namespace amici
