#include<iostream>
#include<time.h>
#include<vector>
#include<random>
#include<xmmintrin.h>
#include <immintrin.h>
#define NUM1 4
#define TIMES1 100000

#define NUM2 4096
#define TIMES2 100

using namespace std;
std::random_device rd;
std::mt19937 gen;
std::uniform_real_distribution<float> dis;

template <size_t T>
class alignas(64) myVec {
public:
    alignas(64) float vec[T];
    void* operator new(size_t size) {
        return _aligned_malloc(size, 64);
    }
    void operator delete(void* p) {
        _aligned_free(p);
    }
};

// 1. 행 우선 (Single Sum)
template <size_t T>
class alignas(64) RowMatrix {
public:
    alignas(64) float matrix[T][T];
public:
    void* operator new(size_t size) {
        return _aligned_malloc(size, 64);
    }
    void operator delete(void* p) {
        _aligned_free(p);
    }
    RowMatrix() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];

#pragma loop(no_vector)
        for (int i = 0; i < T; i++) {
            float sum = 0;
#pragma loop(no_vector)
            for (int j = 0; j < T; j++) {
                sum += rhs.vec[j] * *(m_ptr + (i * T) + j);
            }
            result.vec[i] = sum;
        }
        return result;
    }
};




// 2. 행 누적 (Accumulator - OoO 최적화)
template <size_t T>
class RowMatrixAccum {
public:
    alignas(64) float matrix[T][T];
public:
    RowMatrixAccum() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];
        for (int i = 0; i < T; i++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            const float* row = m_ptr + (i * T);
            for (int j = 0; j < T; j += 4) {
                s0 += rhs.vec[j] * row[j];
                s1 += rhs.vec[j + 1] * row[j + 1];
                s2 += rhs.vec[j + 2] * row[j + 2];
                s3 += rhs.vec[j + 3] * row[j + 3];
            }
            result.vec[i] = (s0 + s1) + (s2 + s3);
        }
        return result;
    }
};



// 3.행 Simd )
template <size_t T>
class alignas(64) RowMatrixSimd {
public:
    alignas(64) float matrix[T][T];
public:
    void* operator new(size_t size) {
        return _aligned_malloc(size, 64);
    }
    void operator delete(void* p) {
        _aligned_free(p);
    }
    RowMatrixSimd() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];
        vector<__m128> loadedRhs(T / 4);
        for (int j = 0; j < T / 4; ++j) {
            loadedRhs[j] = _mm_load_ps(&rhs.vec[j * 4]);
        }
        for (int i = 0; i < T; i++) {
            float sum = 0;
            const float* row = m_ptr + (i * T);
            __m128 totalResult = _mm_setzero_ps();
            for (int j = 0; j < T / 4; j += 1) {
                __m128 b = _mm_load_ps(&row[j * 4]);
                __m128 result = _mm_mul_ps(loadedRhs[j], b);
                totalResult = _mm_add_ps(result, totalResult);
            }

            totalResult = _mm_hadd_ps(totalResult, totalResult);

            float alignas(64) s[4] = { 0.0f,0.0f ,0.0f ,0.0f };
            _mm_store_ps(s, totalResult);
            sum = (s[0] + s[1]);

            result.vec[i] = sum;
        }
        return result;
    }
};



// 4.행 Simd + OoO최적화)
template <size_t T>
class alignas(64) RowMatrixSimdOoO {
public:
    alignas(64) float matrix[T][T];
public:
    void* operator new(size_t size) {
        return _aligned_malloc(size, 64);
    }
    void operator delete(void* p) {
        _aligned_free(p);
    }
    RowMatrixSimdOoO() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];
        vector<__m128> loadedRhs(T / 4);
        for (int j = 0; j < T / 4; ++j) {
            loadedRhs[j] = _mm_load_ps(&rhs.vec[j * 4]);
        }
        for (int i = 0; i < T; i++) {
            float sum = 0;
            const float* row = m_ptr + (i * T);
            __m128 totalResult = _mm_setzero_ps();
            for (int j = 0; j < T / 4; j += 4) {
                __m128 a = _mm_load_ps(&row[j * 4]);
                __m128 result1 = _mm_mul_ps(loadedRhs[j], a);

                __m128 b = _mm_load_ps(&row[(j + 1) * 4]);
                __m128 result2 = _mm_mul_ps(loadedRhs[j + 1], b);

                __m128 c = _mm_load_ps(&row[(j + 2) * 4]);
                __m128 result3 = _mm_mul_ps(loadedRhs[j + 2], c);

                __m128 d = _mm_load_ps(&row[(j + 3) * 4]);
                __m128 result4 = _mm_mul_ps(loadedRhs[j + 3], d);

                __m128 result5 = _mm_add_ps(result1, result2);
                __m128 result6 = _mm_add_ps(result3, result4);

                __m128 result7 = _mm_add_ps(result5, result6);

                totalResult = _mm_add_ps(result7, totalResult);
            }

            totalResult = _mm_hadd_ps(totalResult, totalResult);

            float alignas(64) s[4] = { 0.0f,0.0f ,0.0f ,0.0f };
            _mm_store_ps(s, totalResult);
            sum = (s[0] + s[1]);

            result.vec[i] = sum;
        }
        return result;
    }
};


// 4.행 Simd + OoO최적화 + fmadd활용)
template <size_t T>
class alignas(64) RowMatrixSimdOoO2 {
public:
    alignas(64) float matrix[T][T];
public:
    void* operator new(size_t size) {
        return _aligned_malloc(size, 64);
    }
    void operator delete(void* p) {
        _aligned_free(p);
    }
    RowMatrixSimdOoO2() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];
        vector<__m128> loadedRhs(T / 4);
        for (int j = 0; j < T / 4; ++j) {
            loadedRhs[j] = _mm_load_ps(&rhs.vec[j * 4]);
        }
        for (int i = 0; i < T; i++) {
            float sum = 0;
            const float* row = m_ptr + (i * T);
            __m128 result1 = _mm_setzero_ps();
            __m128 result2 = _mm_setzero_ps();
            __m128 result3 = _mm_setzero_ps();
            __m128 result4 = _mm_setzero_ps();
            for (int j = 0; j < T / 4; j += 4) {
                __m128 a = _mm_load_ps(&row[j * 4]);
                result1 = _mm_fmadd_ps(loadedRhs[j], a, result1);

                __m128 b = _mm_load_ps(&row[(j + 1) * 4]);
                result2 = _mm_fmadd_ps(loadedRhs[j + 1], b, result2);

                __m128 c = _mm_load_ps(&row[(j + 2) * 4]);
                result3 = _mm_fmadd_ps(loadedRhs[j + 2], c, result3);

                __m128 d = _mm_load_ps(&row[(j + 3) * 4]);
                result4 = _mm_fmadd_ps(loadedRhs[j + 3], d, result4);
            }

            __m128 result5 = _mm_add_ps(result1, result2);
            __m128 result6 = _mm_add_ps(result3, result4);
            __m128 result7 = _mm_add_ps(result5, result6);

            __m128 totalResult = _mm_hadd_ps(result7, result7);

            float alignas(64) s[4] = { 0.0f,0.0f ,0.0f ,0.0f };
            _mm_store_ps(s, totalResult);
            sum = (s[0] + s[1]);

            result.vec[i] = sum;
        }
        return result;
    }
};

// 5. 열 우선 (Single Sum)
template <size_t T>
class ColumnMatrix {
public:
    alignas(64) float matrix[T][T];
public:
    ColumnMatrix() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];
#pragma loop(no_vector)
        for (int i = 0; i < T; i++) {
            float sum = 0;
#pragma loop(no_vector)
            for (int j = 0; j < T; j++) {
                sum += rhs.vec[j] * *(m_ptr + (j * T) + i);
            }
            result.vec[i] = sum;
        }
        return result;
    }
};

// 6. 열 누적 (Accumulator - OoO 최적화)
template <size_t T>
class ColumnMatrixAccum {
public:
    alignas(64) float matrix[T][T];
public:
    ColumnMatrixAccum() {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                matrix[i][j] = (float)dis(gen);
            }
        }
    }
    myVec<T> operator *(myVec<T> rhs) {
        myVec<T> result;
        const float* m_ptr = &matrix[0][0];
#pragma loop(no_vector)
        for (int i = 0; i < T; i++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma loop(no_vector)
            for (int j = 0; j < T; j += 4) {
                s0 += rhs.vec[j] * m_ptr[j * T + i];
                s1 += rhs.vec[j + 1] * m_ptr[(j + 1) * T + i];
                s2 += rhs.vec[j + 2] * m_ptr[(j + 2) * T + i];
                s3 += rhs.vec[j + 3] * m_ptr[(j + 3) * T + i];
            }
            result.vec[i] = (s0 + s1) + (s2 + s3);
        }
        return result;
    }
};

int main() {
    gen = std::mt19937(rd());
    dis = std::uniform_real_distribution<float>(0.0f, 0.1f);
    clock_t start, end;


    /*
    // --- NUM1 테스트 ---
    myVec<NUM1> vec;
    for (int i = 0; i < NUM1; i++) { vec.vec[i] = (float)i; }

    RowMatrix<NUM1>* temp1 = new RowMatrix<NUM1>();
    start = clock();
    for (int i = 0; i < TIMES1; i++) { vec = (*temp1) * vec; }
    end = clock();
    delete temp1;
    std::cout << "행우선  " << NUM1 << " *" << NUM1 << " 행렬 x " << NUM1 << "벡터 " << TIMES1 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec.vec[0] << std::endl;

    for (int i = 0; i < NUM1; i++) { vec.vec[i] = (float)i; }
    ColumnMatrix<NUM1>* temp2 = new ColumnMatrix<NUM1>();
    start = clock();
    for (int i = 0; i < TIMES1; i++) { vec = (*temp2) * vec; }
    end = clock();
    delete temp2;
    std::cout << "열우선  " << NUM1 << " *" << NUM1 << " 행렬 x " << NUM1 << "벡터 " << TIMES1 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec.vec[0] << std::endl;
    */


    // --- NUM2 테스트 ---
    myVec<NUM2> vec2;
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }

    // 1. Row Single
    RowMatrix<NUM2>* temp3 = new RowMatrix<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp3) * vec2; }
    end = clock();
    delete temp3;
    std::cout << "행우선  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;

    // 2. Row Accum
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }
    RowMatrixAccum<NUM2>* temp_ra = new RowMatrixAccum<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp_ra) * vec2; }
    end = clock();
    delete temp_ra;
    std::cout << "행누적  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;


    // 3. Row SImd
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }
    RowMatrixSimd<NUM2>* temp_ra2 = new RowMatrixSimd<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp_ra2) * vec2; }
    end = clock();
    delete temp_ra2;
    std::cout << "행 Simd  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;

    // 4. Row SImd OoO
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }
    RowMatrixSimdOoO<NUM2>* temp_ra3 = new RowMatrixSimdOoO<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp_ra3) * vec2; }
    end = clock();
    delete temp_ra3;
    std::cout << "행 SimdOoO  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;


    // 4. Row SImd OoO
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }
    RowMatrixSimdOoO2<NUM2>* temp_ra4 = new RowMatrixSimdOoO2<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp_ra4) * vec2; }
    end = clock();
    delete temp_ra4;
    std::cout << "행 SimdOoO2  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;

    // 5. Col Single
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }
    ColumnMatrix<NUM2>* temp4 = new ColumnMatrix<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp4) * vec2; }
    end = clock();
    delete temp4;
    std::cout << "열우선  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;

    // 6. Col Accum
    for (int i = 0; i < NUM2; i++) { vec2.vec[i] = (float)i; }
    ColumnMatrixAccum<NUM2>* temp_ca = new ColumnMatrixAccum<NUM2>();
    start = clock();
    for (int i = 0; i < TIMES2; i++) { vec2 = (*temp_ca) * vec2; }
    end = clock();
    delete temp_ca;
    std::cout << "열누적  " << NUM2 << " *" << NUM2 << " 행렬 x " << NUM2 << "벡터 " << TIMES2 << "회  " << end - start << "ms" << std::endl;
    std::cout << "최종 결과 확인: " << vec2.vec[0] << std::endl;

    return 0;
}