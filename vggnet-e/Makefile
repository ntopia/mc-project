
LINKTAIL  := -Wl,--wrap,clGetPlatformIDs,--wrap,clGetPlatformInfo,--wrap,clGetDeviceIDs,--wrap,clGetDeviceInfo,--wrap,clCreateSubDevices
LINKTAIL  += -Wl,--wrap,clRetainDevice,--wrap,clReleaseDevice,--wrap,clCreateContext,--wrap,clCreateContextFromType,--wrap,clRetainContext
LINKTAIL  += -Wl,--wrap,clReleaseContext,--wrap,clGetContextInfo,--wrap,clCreateCommandQueue,--wrap,clRetainCommandQueue,--wrap,clReleaseCommandQueue
LINKTAIL  += -Wl,--wrap,clGetCommandQueueInfo,--wrap,clCreateBuffer,--wrap,clCreateSubBuffer,--wrap,clCreateImage,--wrap,clRetainMemObject
LINKTAIL  += -Wl,--wrap,clReleaseMemObject,--wrap,clGetSupportedImageFormats,--wrap,clGetMemObjectInfo,--wrap,clGetImageInfo,--wrap,clSetMemObjectDestructorCallback
LINKTAIL  += -Wl,--wrap,clCreateSampler,--wrap,clRetainSampler,--wrap,clReleaseSampler,--wrap,clGetSamplerInfo,--wrap,clCreateProgramWithSource,--wrap,clCreateProgramWithBinary,--wrap,clCreateProgramWithBuiltInKernels,--wrap,clRetainProgram,--wrap,clReleaseProgram
LINKTAIL  += -Wl,--wrap,clBuildProgram,--wrap,clCompileProgram,--wrap,clLinkProgram,--wrap,clUnloadPlatformCompiler,--wrap,clGetProgramInfo
LINKTAIL  += -Wl,--wrap,clGetProgramBuildInfo,--wrap,clCreateKernel,--wrap,clCreateKernelsInProgram,--wrap,clRetainKernel,--wrap,clReleaseKernel
LINKTAIL  += -Wl,--wrap,clSetKernelArg,--wrap,clGetKernelInfo,--wrap,clGetKernelArgInfo,--wrap,clGetKernelWorkGroupInfo,--wrap,clWaitForEvents
LINKTAIL  += -Wl,--wrap,clGetEventInfo,--wrap,clCreateUserEvent,--wrap,clRetainEvent,--wrap,clReleaseEvent,--wrap,clSetUserEventStatus,--wrap,clSetEventCallback,--wrap,clGetEventProfilingInfo,--wrap,clFlush,--wrap,clFinish
LINKTAIL  += -Wl,--wrap,clEnqueueReadBuffer,--wrap,clEnqueueReadBufferRect,--wrap,clEnqueueWriteBuffer,--wrap,clEnqueueWriteBufferRect,--wrap,clEnqueueFillBuffer
LINKTAIL  += -Wl,--wrap,clEnqueueCopyBuffer,--wrap,clEnqueueCopyBufferRect,--wrap,clEnqueueReadImage,--wrap,clEnqueueWriteImage,--wrap,clEnqueueFillImage
LINKTAIL  += -Wl,--wrap,clEnqueueCopyImage,--wrap,clEnqueueCopyImageToBuffer,--wrap,clEnqueueCopyBufferToImage,--wrap,clEnqueueMapBuffer,--wrap,clEnqueueMapImage
LINKTAIL  += -Wl,--wrap,clEnqueueUnmapMemObject,--wrap,clEnqueueMigrateMemObjects,--wrap,clEnqueueNDRangeKernel,--wrap,clEnqueueTask,--wrap,clEnqueueNativeKernel
LINKTAIL  += -Wl,--wrap,clEnqueueMarkerWithWaitList,--wrap,clEnqueueBarrierWithWaitList,--wrap,clSetCommandQueueProperty
LINKTAIL  += -Wl,--wrap,clCreateImage2D,--wrap,clCreateImage3D,--wrap,clEnqueueMarker,--wrap,clEnqueueWaitForEvents,--wrap,clEnqueueBarrier
LINKTAIL  += -Wl,--wrap,clUnloadCompiler
LINKTAIL  += -Wl,--wrap,main


CC=mpicc
CFLAGS=-Wall -g -O2 -std=gnu99 -march=native -msse -mfpmath=sse -funroll-loops -ftree-vectorize -DUNIX -DCLUSTER

LIBS=-lm -lrt -lOpenCL -lpthread -L${SNUCLROOT}/lib -ldl -lsnucl
LDFLAGS = ${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS} ${LINKTAIL}

run: vggnet
	thorq --add --mode snucl --node 4 --device gpu ./vggnet image_list.txt

clean:
	rm -rf vggnet.o main.o vggnet task* snucl_kernels bin obj
