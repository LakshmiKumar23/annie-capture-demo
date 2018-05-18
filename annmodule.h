#ifndef annmodule_h
#define annmodule_h

#include <VX/vx.h>

typedef void * INFHANDLE;

extern "C" {
    VX_API_ENTRY void     VX_API_CALL annGetTensorDimensions(vx_size dimInput[4], vx_size dimOutput[4]);
    VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor input,  vx_tensor output, const char * options);
    VX_API_ENTRY void *   VX_API_CALL annCreateContext(const char * dataFolder_);
    VX_API_ENTRY int     VX_API_CALL annRunInference(void* ctx, float *in_tensor_buff, vx_size in_buffer_size, float *out_tensor_buff, vx_size out_buffer_size);
    VX_API_ENTRY int     VX_API_CALL annReleaseContext(void* ctx);
};

#endif
