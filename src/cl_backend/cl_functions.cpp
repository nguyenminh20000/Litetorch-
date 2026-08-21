#include "cl_functions.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace litetorch {

clGetPlatformIDs_t p_clGetPlatformIDs = nullptr;
clGetDeviceIDs_t p_clGetDeviceIDs = nullptr;
clCreateContext_t p_clCreateContext = nullptr;
clCreateCommandQueue_t p_clCreateCommandQueue = nullptr;
clCreateBuffer_t p_clCreateBuffer = nullptr;
clReleaseMemObject_t p_clReleaseMemObject = nullptr;
clEnqueueReadBuffer_t p_clEnqueueReadBuffer = nullptr;
clEnqueueWriteBuffer_t p_clEnqueueWriteBuffer = nullptr;
clEnqueueCopyBuffer_t p_clEnqueueCopyBuffer = nullptr;
clCreateProgramWithSource_t p_clCreateProgramWithSource = nullptr;
clBuildProgram_t p_clBuildProgram = nullptr;
clCreateKernel_t p_clCreateKernel = nullptr;
clSetKernelArg_t p_clSetKernelArg = nullptr;
clEnqueueNDRangeKernel_t p_clEnqueueNDRangeKernel = nullptr;
clFinish_t p_clFinish = nullptr;
clReleaseKernel_t p_clReleaseKernel = nullptr;
clReleaseProgram_t p_clReleaseProgram = nullptr;
clReleaseCommandQueue_t p_clReleaseCommandQueue = nullptr;
clReleaseContext_t p_clReleaseContext = nullptr;
clGetProgramBuildInfo_t p_clGetProgramBuildInfo = nullptr;
clCreateSubBuffer_t p_clCreateSubBuffer = nullptr;
clGetDeviceInfo_t p_clGetDeviceInfo = nullptr;

}
