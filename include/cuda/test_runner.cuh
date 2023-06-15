/*
    This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
    Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <exception>

#include <iostream>
#include <iomanip>
#include <sstream>

#include <cuda_runtime_api.h>

#define MSG_BUFFER_SIZE 2048

__device__ bool operator==(const float2 &lhs, const float2 &rhs)
{
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

__device__ bool operator!=(const float2 &lhs, const float2 &rhs)
{
    return (lhs.x != rhs.x) || (lhs.y != rhs.y);
}

enum TestResult {TEST_OK=0, TEST_FAILED=1};

__device__ char dig2char(unsigned char digit)
{
    char c = 0;
    switch (digit)
    {
    case 0:
        c = '0'; break;
    case 1:
        c = '1'; break;
    case 2:
        c = '2'; break;
    case 3:
        c = '3'; break;
    case 4:
        c = '4'; break;
    case 5:
        c = '5'; break;
    case 6:
        c = '6'; break;
    case 7:
        c = '7'; break;
    case 8:
        c = '8'; break;
    case 9:
        c = '9'; break;
    default:
        break;
    }
    return c;
}

__device__ void float2str(float num, char *str)
{
    
}

__device__ size_t strlen_cu(const char *str)
{
    size_t len=0;
    while (str[len]!='\0')
        len++;
    return len;
}

__device__ void strcat_cu(char *dest, const char *src, size_t destSize, size_t start=0)
{
    size_t i=0;
    while ((start + i < destSize) && src[i] != '\0')
    {
        dest[start + i] = src[i];
        i++;
    }
    dest[start + i] = '\0';
}

#define ASSERT_EQ_CU(LHS, RHS)                                  \
    if(LHS != RHS)                                              \
    {                                                           \
        strcat_cu(_pMsg, "ASSERTION FAILED: ", MSG_BUFFER_SIZE, 0); \
        strcat_cu(_pMsg, #LHS, MSG_BUFFER_SIZE, strlen_cu(_pMsg));  \
        strcat_cu(_pMsg, " != ", MSG_BUFFER_SIZE, strlen_cu(_pMsg));\
        strcat_cu(_pMsg, #RHS, MSG_BUFFER_SIZE, strlen_cu(_pMsg));  \
        *_pRes = TEST_FAILED; return;                               \
    }

#define CREATE_TEST_KERNEL(KERNEL_NAME) __global__ void KERNEL_NAME(TestResult *_pRes, char *_pMsg)

class CudaKernelTestRunner
{
public:
    CudaKernelTestRunner() : m_fails(0), m_testsCnt(0) {}
    ~CudaKernelTestRunner()
    {
        std::cerr << "=======================" << std::endl;
        std::cerr << "Tests run:" << std::setw(3) << std::right << m_testsCnt << std::endl;
        std::cerr << "Passed:"    << std::setw(6) << std::right << m_testsCnt - m_fails << " ("
                  << std::setw(6) << std::right << std::setprecision(2) << std::fixed
                  << static_cast<float>(m_testsCnt - m_fails)/m_testsCnt*100.f << "%)"
                  << std::endl;
        std::cerr << "Fails:"     << std::setw(7) << std::right << m_fails << " ("
                  << std::setw(6) << std::right << std::setprecision(2) <<
                  static_cast<float>(m_fails)/m_testsCnt*100.f << "%)"
                  << std::endl;
        std::cerr << "=======================" << std::endl;
    }
    template<typename T_func>
    void runTest(T_func test_kernel, const std::string &test_name);
private:
    size_t m_fails;
    size_t m_testsCnt;
};


template<typename T_func>
void CudaKernelTestRunner::runTest(T_func test_kernel, const std::string &test_name)
{
    TestResult *res = nullptr;
    char *msg;
    cudaMallocManaged(&res, sizeof(TestResult));
    cudaMallocManaged(&msg, sizeof(char)*MSG_BUFFER_SIZE);
    *res = TEST_OK;
    test_kernel<<<1,1>>>(res, msg);
    cudaDeviceSynchronize();
    m_testsCnt++;
    if ((*res) == TEST_FAILED)
    {
        m_fails++;
        std::cerr << "- Test " << std::quoted (test_name)  << " failed: " << msg << std::endl;
    }
    else
    {
        std::cerr << "+ Test " << std::quoted (test_name)  << " passed" << std::endl;
    }
    cudaFree(res);
    cudaFree(msg);
}

#define RUN_TEST(RUNNER, TEST) RUNNER.runTest(TEST, #TEST)