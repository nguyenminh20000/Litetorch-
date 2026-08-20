#ifndef LITETORCH_CL_FUNCTIONS_H
#define LITETORCH_CL_FUNCTIONS_H

#include "litetorch/cl_backend.h"

namespace litetorch {

typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*clCreateContext_t)(const void*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (*clCreateCommandQueue_t)(cl_context, cl_device_id, unsigned long long, cl_int*);
typedef cl_mem (*clCreateBuffer_t)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*clReleaseMemObject_t)(cl_mem);
typedef cl_int (*clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*clEnqueueCopyBuffer_t)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
typedef cl_program (*clCreateProgramWithSource_t)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*clBuildProgram_t)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
typedef cl_kernel (*clCreateKernel_t)(cl_program, const char*, cl_int*);
typedef cl_int (*clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*clFinish_t)(cl_command_queue);
typedef cl_int (*clReleaseKernel_t)(cl_kernel);
typedef cl_int (*clReleaseProgram_t)(cl_program);
typedef cl_int (*clReleaseCommandQueue_t)(cl_command_queue);
typedef cl_int (*clReleaseContext_t)(cl_context);
typedef cl_int (*clGetProgramBuildInfo_t)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_mem (*clCreateSubBuffer_t)(cl_mem, cl_mem_flags, cl_uint, const void*, cl_int*);

extern clGetPlatformIDs_t p_clGetPlatformIDs;
extern clGetDeviceIDs_t p_clGetDeviceIDs;
extern clCreateContext_t p_clCreateContext;
extern clCreateCommandQueue_t p_clCreateCommandQueue;
extern clCreateBuffer_t p_clCreateBuffer;
extern clReleaseMemObject_t p_clReleaseMemObject;
extern clEnqueueReadBuffer_t p_clEnqueueReadBuffer;
extern clEnqueueWriteBuffer_t p_clEnqueueWriteBuffer;
extern clEnqueueCopyBuffer_t p_clEnqueueCopyBuffer;
extern clCreateProgramWithSource_t p_clCreateProgramWithSource;
extern clBuildProgram_t p_clBuildProgram;
extern clCreateKernel_t p_clCreateKernel;
extern clSetKernelArg_t p_clSetKernelArg;
extern clEnqueueNDRangeKernel_t p_clEnqueueNDRangeKernel;
extern clFinish_t p_clFinish;
extern clReleaseKernel_t p_clReleaseKernel;
extern clReleaseProgram_t p_clReleaseProgram;
extern clReleaseCommandQueue_t p_clReleaseCommandQueue;
extern clReleaseContext_t p_clReleaseContext;
extern clGetProgramBuildInfo_t p_clGetProgramBuildInfo;
extern clCreateSubBuffer_t p_clCreateSubBuffer;
extern clGetDeviceInfo_t p_clGetDeviceInfo;

}

#endif
