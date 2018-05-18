#include "annmodule.h"

#include <vx_ext_amd.h>
#include <VX/vx_khr_nn.h>
#include <vx_amd_nn.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

typedef struct inference_context_t
{
    vx_context context;
    vx_graph graph;
    vx_tensor input, output;
    vx_size dimInput[4], dimOutput[4];
} *inf_context;

#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return nullptr; } }
#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return nullptr; } }
#define ERROR_CHECK_STATUS1(call) { vx_status status = (call); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

static vx_status copyTensor(vx_tensor tensor, std::string fileName, vx_enum usage = VX_WRITE_ONLY)
{
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    vx_size itemsize = sizeof(float);
    if(data_type == VX_TYPE_UINT8 || data_type == VX_TYPE_INT8) {
        itemsize = sizeof(vx_uint8);
    }
    else if(data_type == VX_TYPE_UINT16 || data_type == VX_TYPE_INT16 || data_type == VX_TYPE_FLOAT16) {
        itemsize = sizeof(vx_uint16);
    }
    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_map_id map_id;
    float * ptr;
    vx_status status = vxMapTensorPatch(tensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for " << fileName << std::endl;
        return -1;
    }
    FILE * fp = fopen(fileName.c_str(), usage == VX_WRITE_ONLY ? "rb" : "wb");
    if(!fp) {
        std::cerr << "ERROR: unable to open: " << fileName << std::endl;
        return -1;
    }
    if(usage == VX_WRITE_ONLY) {
        vx_size n = fread(ptr, itemsize, count, fp);
        if(n != count) {
            std::cerr << "ERROR: expected char[" << count*itemsize << "], but got char[" << n*itemsize << "] in " << fileName << std::endl;
            return -1;
        }
    }
    else {
        fwrite(ptr, itemsize, count, fp);
    }
    fclose(fp);
    status = vxUnmapTensorPatch(tensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << fileName << std::endl;
        return -1;
    }
    return 0;
}

VX_API_ENTRY void VX_API_CALL annGetTensorDimensions(vx_size dimInput[4], vx_size dimOutput[4])
{
    dimInput[0] = 299;
    dimInput[1] = 299;
    dimInput[2] = 3;
    dimInput[3] = 1;
    dimOutput[0] = 1;
    dimOutput[1] = 1;
    dimOutput[2] = 1000;
    dimOutput[3] = 1;
}

VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor data, vx_tensor prob, const char * dataFolder_)
{
    // load neural network extension kernels
    ERROR_CHECK_STATUS(vxLoadKernels(context,"vx_nn"));

    // create graph
    vx_graph graph = vxCreateGraph(context); 
    ERROR_CHECK_OBJECT(graph);

    // get dataFolder option
    std::string dataFolder = dataFolder_ ? dataFolder_ : ".", fileName;

    ////
    // initialize the graph
    // conv1_3x3_s2 Layer
    vx_size conv1_3x3_s2_dims[4] = { 149, 149, 32, 1 };
    vx_tensor conv1_3x3_s2;
    conv1_3x3_s2 = vxCreateVirtualTensor(graph,4, conv1_3x3_s2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2);
    vx_size conv1_3x3_s2_W_dims[4] = { 3, 3, 3, 32 };
    vx_tensor conv1_3x3_s2_W;
    conv1_3x3_s2_W = vxCreateTensor(context,4, conv1_3x3_s2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_W, dataFolder + "/weights/conv1_3x3_s2.f32"));
    vx_nn_convolution_params_t conv1_3x3_s2_params;
    conv1_3x3_s2_params.padding_x = 0;
    conv1_3x3_s2_params.padding_y = 0;
    conv1_3x3_s2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv1_3x3_s2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv1_3x3_s2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv1_3x3_s2_params.dilation_x = 0;
    conv1_3x3_s2_params.dilation_y = 0;
    vx_node conv1_3x3_s2_node;
    conv1_3x3_s2_node = vxConvolutionLayer(graph, data, conv1_3x3_s2_W, NULL, &conv1_3x3_s2_params, sizeof(conv1_3x3_s2_params ), conv1_3x3_s2);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_3x3_s2_node));

    // conv1_3x3_s2_bn Layer
    vx_size conv1_3x3_s2_scale_dims[4] = { 149, 149, 32, 1 };
    vx_tensor conv1_3x3_s2_scale;
    conv1_3x3_s2_scale = vxCreateVirtualTensor(graph,4, conv1_3x3_s2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_scale);
    vx_size conv1_3x3_s2_bn_W_dims[1] = { 32 };
    vx_float32 conv1_3x3_s2_bn_eps = 0.001;
    vx_tensor conv1_3x3_s2_bn_W;
    conv1_3x3_s2_bn_W = vxCreateTensor(context,1, conv1_3x3_s2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_bn_W, dataFolder + "/weights/conv1_3x3_s2_bn.f32"));
    vx_size conv1_3x3_s2_bn_B_dims[1] = { 32 };
    vx_tensor conv1_3x3_s2_bn_B;
    conv1_3x3_s2_bn_B = vxCreateTensor(context,1, conv1_3x3_s2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_bn_B, dataFolder + "/bias/conv1_3x3_s2_bn.f32"));
    vx_size conv1_3x3_s2_scale_W_dims[1] = { 32 };
    vx_tensor conv1_3x3_s2_scale_W;
    conv1_3x3_s2_scale_W = vxCreateTensor(context,1, conv1_3x3_s2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_scale_W, dataFolder + "/weights/conv1_3x3_s2_scale.f32"));
    vx_size conv1_3x3_s2_scale_B_dims[1] = { 32 };
    vx_tensor conv1_3x3_s2_scale_B;
    conv1_3x3_s2_scale_B = vxCreateTensor(context,1, conv1_3x3_s2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv1_3x3_s2_scale_B, dataFolder + "/bias/conv1_3x3_s2_scale.f32"));
    vx_node conv1_3x3_s2_bn_node;
    conv1_3x3_s2_bn_node = vxBatchNormalizationLayer(graph, conv1_3x3_s2, conv1_3x3_s2_bn_W, conv1_3x3_s2_bn_B, conv1_3x3_s2_scale_W, conv1_3x3_s2_scale_B, conv1_3x3_s2_bn_eps, conv1_3x3_s2_scale);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_3x3_s2_bn_node));

    // conv1_3x3_s2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv1_3x3_s2_relu Layer
    vx_size conv1_3x3_s2_relu_dims[4] = { 149, 149, 32, 1 };
    vx_tensor conv1_3x3_s2_relu;
    conv1_3x3_s2_relu = vxCreateVirtualTensor(graph,4, conv1_3x3_s2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_relu);
    vx_enum conv1_3x3_s2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv1_3x3_s2_relu_param_a = 0;
    vx_float32 conv1_3x3_s2_relu_param_b = 0;
    vx_node conv1_3x3_s2_relu_node;
    conv1_3x3_s2_relu_node = vxActivationLayer(graph, conv1_3x3_s2_scale, conv1_3x3_s2_relu_mode, conv1_3x3_s2_relu_param_a, conv1_3x3_s2_relu_param_b, conv1_3x3_s2_relu);
    ERROR_CHECK_OBJECT(conv1_3x3_s2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv1_3x3_s2_relu_node));

    // conv2_3x3_s1 Layer
    vx_size conv2_3x3_s1_dims[4] = { 147, 147, 32, 1 };
    vx_tensor conv2_3x3_s1;
    conv2_3x3_s1 = vxCreateVirtualTensor(graph,4, conv2_3x3_s1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1);
    vx_size conv2_3x3_s1_W_dims[4] = { 3, 3, 32, 32 };
    vx_tensor conv2_3x3_s1_W;
    conv2_3x3_s1_W = vxCreateTensor(context,4, conv2_3x3_s1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_W, dataFolder + "/weights/conv2_3x3_s1.f32"));
    vx_nn_convolution_params_t conv2_3x3_s1_params;
    conv2_3x3_s1_params.padding_x = 0;
    conv2_3x3_s1_params.padding_y = 0;
    conv2_3x3_s1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv2_3x3_s1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv2_3x3_s1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv2_3x3_s1_params.dilation_x = 0;
    conv2_3x3_s1_params.dilation_y = 0;
    vx_node conv2_3x3_s1_node;
    conv2_3x3_s1_node = vxConvolutionLayer(graph, conv1_3x3_s2_relu, conv2_3x3_s1_W, NULL, &conv2_3x3_s1_params, sizeof(conv2_3x3_s1_params ), conv2_3x3_s1);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_3x3_s1_node));

    // conv2_3x3_s1_bn Layer
    vx_size conv2_3x3_s1_scale_dims[4] = { 147, 147, 32, 1 };
    vx_tensor conv2_3x3_s1_scale;
    conv2_3x3_s1_scale = vxCreateVirtualTensor(graph,4, conv2_3x3_s1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_scale);
    vx_size conv2_3x3_s1_bn_W_dims[1] = { 32 };
    vx_float32 conv2_3x3_s1_bn_eps = 0.001;
    vx_tensor conv2_3x3_s1_bn_W;
    conv2_3x3_s1_bn_W = vxCreateTensor(context,1, conv2_3x3_s1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_bn_W, dataFolder + "/weights/conv2_3x3_s1_bn.f32"));
    vx_size conv2_3x3_s1_bn_B_dims[1] = { 32 };
    vx_tensor conv2_3x3_s1_bn_B;
    conv2_3x3_s1_bn_B = vxCreateTensor(context,1, conv2_3x3_s1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_bn_B, dataFolder + "/bias/conv2_3x3_s1_bn.f32"));
    vx_size conv2_3x3_s1_scale_W_dims[1] = { 32 };
    vx_tensor conv2_3x3_s1_scale_W;
    conv2_3x3_s1_scale_W = vxCreateTensor(context,1, conv2_3x3_s1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_scale_W, dataFolder + "/weights/conv2_3x3_s1_scale.f32"));
    vx_size conv2_3x3_s1_scale_B_dims[1] = { 32 };
    vx_tensor conv2_3x3_s1_scale_B;
    conv2_3x3_s1_scale_B = vxCreateTensor(context,1, conv2_3x3_s1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv2_3x3_s1_scale_B, dataFolder + "/bias/conv2_3x3_s1_scale.f32"));
    vx_node conv2_3x3_s1_bn_node;
    conv2_3x3_s1_bn_node = vxBatchNormalizationLayer(graph, conv2_3x3_s1, conv2_3x3_s1_bn_W, conv2_3x3_s1_bn_B, conv2_3x3_s1_scale_W, conv2_3x3_s1_scale_B, conv2_3x3_s1_bn_eps, conv2_3x3_s1_scale);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_3x3_s1_bn_node));

    // conv2_3x3_s1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv2_3x3_s1_relu Layer
    vx_size conv2_3x3_s1_relu_dims[4] = { 147, 147, 32, 1 };
    vx_tensor conv2_3x3_s1_relu;
    conv2_3x3_s1_relu = vxCreateVirtualTensor(graph,4, conv2_3x3_s1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_relu);
    vx_enum conv2_3x3_s1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv2_3x3_s1_relu_param_a = 0;
    vx_float32 conv2_3x3_s1_relu_param_b = 0;
    vx_node conv2_3x3_s1_relu_node;
    conv2_3x3_s1_relu_node = vxActivationLayer(graph, conv2_3x3_s1_scale, conv2_3x3_s1_relu_mode, conv2_3x3_s1_relu_param_a, conv2_3x3_s1_relu_param_b, conv2_3x3_s1_relu);
    ERROR_CHECK_OBJECT(conv2_3x3_s1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv2_3x3_s1_relu_node));

    // conv3_3x3_s1 Layer
    vx_size conv3_3x3_s1_dims[4] = { 147, 147, 64, 1 };
    vx_tensor conv3_3x3_s1;
    conv3_3x3_s1 = vxCreateVirtualTensor(graph,4, conv3_3x3_s1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1);
    vx_size conv3_3x3_s1_W_dims[4] = { 3, 3, 32, 64 };
    vx_tensor conv3_3x3_s1_W;
    conv3_3x3_s1_W = vxCreateTensor(context,4, conv3_3x3_s1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_W, dataFolder + "/weights/conv3_3x3_s1.f32"));
    vx_nn_convolution_params_t conv3_3x3_s1_params;
    conv3_3x3_s1_params.padding_x = 1;
    conv3_3x3_s1_params.padding_y = 1;
    conv3_3x3_s1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    conv3_3x3_s1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    conv3_3x3_s1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    conv3_3x3_s1_params.dilation_x = 0;
    conv3_3x3_s1_params.dilation_y = 0;
    vx_node conv3_3x3_s1_node;
    conv3_3x3_s1_node = vxConvolutionLayer(graph, conv2_3x3_s1_relu, conv3_3x3_s1_W, NULL, &conv3_3x3_s1_params, sizeof(conv3_3x3_s1_params ), conv3_3x3_s1);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_3x3_s1_node));

    // conv3_3x3_s1_bn Layer
    vx_size conv3_3x3_s1_scale_dims[4] = { 147, 147, 64, 1 };
    vx_tensor conv3_3x3_s1_scale;
    conv3_3x3_s1_scale = vxCreateVirtualTensor(graph,4, conv3_3x3_s1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_scale);
    vx_size conv3_3x3_s1_bn_W_dims[1] = { 64 };
    vx_float32 conv3_3x3_s1_bn_eps = 0.001;
    vx_tensor conv3_3x3_s1_bn_W;
    conv3_3x3_s1_bn_W = vxCreateTensor(context,1, conv3_3x3_s1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_bn_W, dataFolder + "/weights/conv3_3x3_s1_bn.f32"));
    vx_size conv3_3x3_s1_bn_B_dims[1] = { 64 };
    vx_tensor conv3_3x3_s1_bn_B;
    conv3_3x3_s1_bn_B = vxCreateTensor(context,1, conv3_3x3_s1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_bn_B, dataFolder + "/bias/conv3_3x3_s1_bn.f32"));
    vx_size conv3_3x3_s1_scale_W_dims[1] = { 64 };
    vx_tensor conv3_3x3_s1_scale_W;
    conv3_3x3_s1_scale_W = vxCreateTensor(context,1, conv3_3x3_s1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_scale_W, dataFolder + "/weights/conv3_3x3_s1_scale.f32"));
    vx_size conv3_3x3_s1_scale_B_dims[1] = { 64 };
    vx_tensor conv3_3x3_s1_scale_B;
    conv3_3x3_s1_scale_B = vxCreateTensor(context,1, conv3_3x3_s1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(conv3_3x3_s1_scale_B, dataFolder + "/bias/conv3_3x3_s1_scale.f32"));
    vx_node conv3_3x3_s1_bn_node;
    conv3_3x3_s1_bn_node = vxBatchNormalizationLayer(graph, conv3_3x3_s1, conv3_3x3_s1_bn_W, conv3_3x3_s1_bn_B, conv3_3x3_s1_scale_W, conv3_3x3_s1_scale_B, conv3_3x3_s1_bn_eps, conv3_3x3_s1_scale);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_3x3_s1_bn_node));

    // conv3_3x3_s1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // conv3_3x3_s1_relu Layer
    vx_size conv3_3x3_s1_relu_dims[4] = { 147, 147, 64, 1 };
    vx_tensor conv3_3x3_s1_relu;
    conv3_3x3_s1_relu = vxCreateVirtualTensor(graph,4, conv3_3x3_s1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_relu);
    vx_enum conv3_3x3_s1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 conv3_3x3_s1_relu_param_a = 0;
    vx_float32 conv3_3x3_s1_relu_param_b = 0;
    vx_node conv3_3x3_s1_relu_node;
    conv3_3x3_s1_relu_node = vxActivationLayer(graph, conv3_3x3_s1_scale, conv3_3x3_s1_relu_mode, conv3_3x3_s1_relu_param_a, conv3_3x3_s1_relu_param_b, conv3_3x3_s1_relu);
    ERROR_CHECK_OBJECT(conv3_3x3_s1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&conv3_3x3_s1_relu_node));

    // inception_stem1_3x3_s2 Layer
    vx_size inception_stem1_3x3_s2_dims[4] = { 73, 73, 96, 1 };
    vx_tensor inception_stem1_3x3_s2;
    inception_stem1_3x3_s2 = vxCreateVirtualTensor(graph,4, inception_stem1_3x3_s2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2);
    vx_size inception_stem1_3x3_s2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_stem1_3x3_s2_W;
    inception_stem1_3x3_s2_W = vxCreateTensor(context,4, inception_stem1_3x3_s2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem1_3x3_s2_W, dataFolder + "/weights/inception_stem1_3x3_s2.f32"));
    vx_nn_convolution_params_t inception_stem1_3x3_s2_params;
    inception_stem1_3x3_s2_params.padding_x = 0;
    inception_stem1_3x3_s2_params.padding_y = 0;
    inception_stem1_3x3_s2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem1_3x3_s2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem1_3x3_s2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem1_3x3_s2_params.dilation_x = 0;
    inception_stem1_3x3_s2_params.dilation_y = 0;
    vx_node inception_stem1_3x3_s2_node;
    inception_stem1_3x3_s2_node = vxConvolutionLayer(graph, conv3_3x3_s1_relu, inception_stem1_3x3_s2_W, NULL, &inception_stem1_3x3_s2_params, sizeof(inception_stem1_3x3_s2_params ), inception_stem1_3x3_s2);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem1_3x3_s2_node));

    // inception_stem1_3x3_s2_bn Layer
    vx_size inception_stem1_3x3_s2_scale_dims[4] = { 73, 73, 96, 1 };
    vx_tensor inception_stem1_3x3_s2_scale;
    inception_stem1_3x3_s2_scale = vxCreateVirtualTensor(graph,4, inception_stem1_3x3_s2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_scale);
    vx_size inception_stem1_3x3_s2_bn_W_dims[1] = { 96 };
    vx_float32 inception_stem1_3x3_s2_bn_eps = 0.001;
    vx_tensor inception_stem1_3x3_s2_bn_W;
    inception_stem1_3x3_s2_bn_W = vxCreateTensor(context,1, inception_stem1_3x3_s2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem1_3x3_s2_bn_W, dataFolder + "/weights/inception_stem1_3x3_s2_bn.f32"));
    vx_size inception_stem1_3x3_s2_bn_B_dims[1] = { 96 };
    vx_tensor inception_stem1_3x3_s2_bn_B;
    inception_stem1_3x3_s2_bn_B = vxCreateTensor(context,1, inception_stem1_3x3_s2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem1_3x3_s2_bn_B, dataFolder + "/bias/inception_stem1_3x3_s2_bn.f32"));
    vx_size inception_stem1_3x3_s2_scale_W_dims[1] = { 96 };
    vx_tensor inception_stem1_3x3_s2_scale_W;
    inception_stem1_3x3_s2_scale_W = vxCreateTensor(context,1, inception_stem1_3x3_s2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem1_3x3_s2_scale_W, dataFolder + "/weights/inception_stem1_3x3_s2_scale.f32"));
    vx_size inception_stem1_3x3_s2_scale_B_dims[1] = { 96 };
    vx_tensor inception_stem1_3x3_s2_scale_B;
    inception_stem1_3x3_s2_scale_B = vxCreateTensor(context,1, inception_stem1_3x3_s2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem1_3x3_s2_scale_B, dataFolder + "/bias/inception_stem1_3x3_s2_scale.f32"));
    vx_node inception_stem1_3x3_s2_bn_node;
    inception_stem1_3x3_s2_bn_node = vxBatchNormalizationLayer(graph, inception_stem1_3x3_s2, inception_stem1_3x3_s2_bn_W, inception_stem1_3x3_s2_bn_B, inception_stem1_3x3_s2_scale_W, inception_stem1_3x3_s2_scale_B, inception_stem1_3x3_s2_bn_eps, inception_stem1_3x3_s2_scale);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem1_3x3_s2_bn_node));

    // inception_stem1_3x3_s2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem1_3x3_s2_relu Layer
    vx_size inception_stem1_3x3_s2_relu_dims[4] = { 73, 73, 96, 1 };
    vx_tensor inception_stem1_3x3_s2_relu;
    inception_stem1_3x3_s2_relu = vxCreateVirtualTensor(graph,4, inception_stem1_3x3_s2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_relu);
    vx_enum inception_stem1_3x3_s2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem1_3x3_s2_relu_param_a = 0;
    vx_float32 inception_stem1_3x3_s2_relu_param_b = 0;
    vx_node inception_stem1_3x3_s2_relu_node;
    inception_stem1_3x3_s2_relu_node = vxActivationLayer(graph, inception_stem1_3x3_s2_scale, inception_stem1_3x3_s2_relu_mode, inception_stem1_3x3_s2_relu_param_a, inception_stem1_3x3_s2_relu_param_b, inception_stem1_3x3_s2_relu);
    ERROR_CHECK_OBJECT(inception_stem1_3x3_s2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem1_3x3_s2_relu_node));

    // inception_stem1_pool Layer
    vx_size inception_stem1_pool_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem1_pool;
    inception_stem1_pool = vxCreateVirtualTensor(graph,4, inception_stem1_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem1_pool);
    vx_enum inception_stem1_pool_type = VX_NN_POOLING_MAX;
    vx_size inception_stem1_pool_kernel_w = 3;
    vx_size inception_stem1_pool_kernel_h = 3;
    vx_size inception_stem1_pool_pad_w = 0;
    vx_size inception_stem1_pool_pad_h = 0;
    vx_enum inception_stem1_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_stem1_pool_node;
    inception_stem1_pool_node = vxPoolingLayer(graph, conv3_3x3_s1_relu, inception_stem1_pool_type, inception_stem1_pool_kernel_w, inception_stem1_pool_kernel_h, inception_stem1_pool_pad_w, inception_stem1_pool_pad_h, inception_stem1_pool_roundPolicy, inception_stem1_pool );
    ERROR_CHECK_OBJECT(inception_stem1_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem1_pool_node));

    // inception_stem1 Layer
    vx_size inception_stem1_dims[4] = { 73, 73, 160, 1 };
    vx_tensor inception_stem1;
    inception_stem1 = vxCreateVirtualTensor(graph,4, inception_stem1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem1);
    vx_node inception_stem1_node;
    inception_stem1_node = vxConcatLayer(graph, inception_stem1, inception_stem1_pool, inception_stem1_3x3_s2_relu, NULL, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_stem1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem1_node));

    // inception_stem2_3x3_reduce Layer
    vx_size inception_stem2_3x3_reduce_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_3x3_reduce;
    inception_stem2_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce);
    vx_size inception_stem2_3x3_reduce_W_dims[4] = { 1, 1, 160, 64 };
    vx_tensor inception_stem2_3x3_reduce_W;
    inception_stem2_3x3_reduce_W = vxCreateTensor(context,4, inception_stem2_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_reduce_W, dataFolder + "/weights/inception_stem2_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_stem2_3x3_reduce_params;
    inception_stem2_3x3_reduce_params.padding_x = 0;
    inception_stem2_3x3_reduce_params.padding_y = 0;
    inception_stem2_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem2_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem2_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem2_3x3_reduce_params.dilation_x = 0;
    inception_stem2_3x3_reduce_params.dilation_y = 0;
    vx_node inception_stem2_3x3_reduce_node;
    inception_stem2_3x3_reduce_node = vxConvolutionLayer(graph, inception_stem1, inception_stem2_3x3_reduce_W, NULL, &inception_stem2_3x3_reduce_params, sizeof(inception_stem2_3x3_reduce_params ), inception_stem2_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_reduce_node));

    // inception_stem2_3x3_reduce_bn Layer
    vx_size inception_stem2_3x3_reduce_scale_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_3x3_reduce_scale;
    inception_stem2_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_scale);
    vx_size inception_stem2_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_stem2_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_stem2_3x3_reduce_bn_W;
    inception_stem2_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_stem2_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_reduce_bn_W, dataFolder + "/weights/inception_stem2_3x3_reduce_bn.f32"));
    vx_size inception_stem2_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_stem2_3x3_reduce_bn_B;
    inception_stem2_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_stem2_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_reduce_bn_B, dataFolder + "/bias/inception_stem2_3x3_reduce_bn.f32"));
    vx_size inception_stem2_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_stem2_3x3_reduce_scale_W;
    inception_stem2_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_stem2_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_reduce_scale_W, dataFolder + "/weights/inception_stem2_3x3_reduce_scale.f32"));
    vx_size inception_stem2_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_stem2_3x3_reduce_scale_B;
    inception_stem2_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_stem2_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_reduce_scale_B, dataFolder + "/bias/inception_stem2_3x3_reduce_scale.f32"));
    vx_node inception_stem2_3x3_reduce_bn_node;
    inception_stem2_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_stem2_3x3_reduce, inception_stem2_3x3_reduce_bn_W, inception_stem2_3x3_reduce_bn_B, inception_stem2_3x3_reduce_scale_W, inception_stem2_3x3_reduce_scale_B, inception_stem2_3x3_reduce_bn_eps, inception_stem2_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_reduce_bn_node));

    // inception_stem2_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem2_3x3_reduce_relu Layer
    vx_size inception_stem2_3x3_reduce_relu_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_3x3_reduce_relu;
    inception_stem2_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_relu);
    vx_enum inception_stem2_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem2_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_stem2_3x3_reduce_relu_param_b = 0;
    vx_node inception_stem2_3x3_reduce_relu_node;
    inception_stem2_3x3_reduce_relu_node = vxActivationLayer(graph, inception_stem2_3x3_reduce_scale, inception_stem2_3x3_reduce_relu_mode, inception_stem2_3x3_reduce_relu_param_a, inception_stem2_3x3_reduce_relu_param_b, inception_stem2_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_reduce_relu_node));

    // inception_stem2_3x3 Layer
    vx_size inception_stem2_3x3_dims[4] = { 71, 71, 96, 1 };
    vx_tensor inception_stem2_3x3;
    inception_stem2_3x3 = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3);
    vx_size inception_stem2_3x3_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_stem2_3x3_W;
    inception_stem2_3x3_W = vxCreateTensor(context,4, inception_stem2_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_W, dataFolder + "/weights/inception_stem2_3x3.f32"));
    vx_nn_convolution_params_t inception_stem2_3x3_params;
    inception_stem2_3x3_params.padding_x = 0;
    inception_stem2_3x3_params.padding_y = 0;
    inception_stem2_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem2_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem2_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem2_3x3_params.dilation_x = 0;
    inception_stem2_3x3_params.dilation_y = 0;
    vx_node inception_stem2_3x3_node;
    inception_stem2_3x3_node = vxConvolutionLayer(graph, inception_stem2_3x3_reduce_relu, inception_stem2_3x3_W, NULL, &inception_stem2_3x3_params, sizeof(inception_stem2_3x3_params ), inception_stem2_3x3);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_node));

    // inception_stem2_3x3_bn Layer
    vx_size inception_stem2_3x3_scale_dims[4] = { 71, 71, 96, 1 };
    vx_tensor inception_stem2_3x3_scale;
    inception_stem2_3x3_scale = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_scale);
    vx_size inception_stem2_3x3_bn_W_dims[1] = { 96 };
    vx_float32 inception_stem2_3x3_bn_eps = 0.001;
    vx_tensor inception_stem2_3x3_bn_W;
    inception_stem2_3x3_bn_W = vxCreateTensor(context,1, inception_stem2_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_bn_W, dataFolder + "/weights/inception_stem2_3x3_bn.f32"));
    vx_size inception_stem2_3x3_bn_B_dims[1] = { 96 };
    vx_tensor inception_stem2_3x3_bn_B;
    inception_stem2_3x3_bn_B = vxCreateTensor(context,1, inception_stem2_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_bn_B, dataFolder + "/bias/inception_stem2_3x3_bn.f32"));
    vx_size inception_stem2_3x3_scale_W_dims[1] = { 96 };
    vx_tensor inception_stem2_3x3_scale_W;
    inception_stem2_3x3_scale_W = vxCreateTensor(context,1, inception_stem2_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_scale_W, dataFolder + "/weights/inception_stem2_3x3_scale.f32"));
    vx_size inception_stem2_3x3_scale_B_dims[1] = { 96 };
    vx_tensor inception_stem2_3x3_scale_B;
    inception_stem2_3x3_scale_B = vxCreateTensor(context,1, inception_stem2_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_scale_B, dataFolder + "/bias/inception_stem2_3x3_scale.f32"));
    vx_node inception_stem2_3x3_bn_node;
    inception_stem2_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_stem2_3x3, inception_stem2_3x3_bn_W, inception_stem2_3x3_bn_B, inception_stem2_3x3_scale_W, inception_stem2_3x3_scale_B, inception_stem2_3x3_bn_eps, inception_stem2_3x3_scale);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_bn_node));

    // inception_stem2_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem2_3x3_relu Layer
    vx_size inception_stem2_3x3_relu_dims[4] = { 71, 71, 96, 1 };
    vx_tensor inception_stem2_3x3_relu;
    inception_stem2_3x3_relu = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_relu);
    vx_enum inception_stem2_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem2_3x3_relu_param_a = 0;
    vx_float32 inception_stem2_3x3_relu_param_b = 0;
    vx_node inception_stem2_3x3_relu_node;
    inception_stem2_3x3_relu_node = vxActivationLayer(graph, inception_stem2_3x3_scale, inception_stem2_3x3_relu_mode, inception_stem2_3x3_relu_param_a, inception_stem2_3x3_relu_param_b, inception_stem2_3x3_relu);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_relu_node));

    // inception_stem2_1x7_reduce Layer
    vx_size inception_stem2_1x7_reduce_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_1x7_reduce;
    inception_stem2_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_stem2_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce);
    vx_size inception_stem2_1x7_reduce_W_dims[4] = { 1, 1, 160, 64 };
    vx_tensor inception_stem2_1x7_reduce_W;
    inception_stem2_1x7_reduce_W = vxCreateTensor(context,4, inception_stem2_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_reduce_W, dataFolder + "/weights/inception_stem2_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_stem2_1x7_reduce_params;
    inception_stem2_1x7_reduce_params.padding_x = 0;
    inception_stem2_1x7_reduce_params.padding_y = 0;
    inception_stem2_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem2_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem2_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem2_1x7_reduce_params.dilation_x = 0;
    inception_stem2_1x7_reduce_params.dilation_y = 0;
    vx_node inception_stem2_1x7_reduce_node;
    inception_stem2_1x7_reduce_node = vxConvolutionLayer(graph, inception_stem1, inception_stem2_1x7_reduce_W, NULL, &inception_stem2_1x7_reduce_params, sizeof(inception_stem2_1x7_reduce_params ), inception_stem2_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_1x7_reduce_node));

    // inception_stem2_1x7_reduce_bn Layer
    vx_size inception_stem2_1x7_reduce_scale_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_1x7_reduce_scale;
    inception_stem2_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_stem2_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_scale);
    vx_size inception_stem2_1x7_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_stem2_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_stem2_1x7_reduce_bn_W;
    inception_stem2_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_stem2_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_reduce_bn_W, dataFolder + "/weights/inception_stem2_1x7_reduce_bn.f32"));
    vx_size inception_stem2_1x7_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_stem2_1x7_reduce_bn_B;
    inception_stem2_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_stem2_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_reduce_bn_B, dataFolder + "/bias/inception_stem2_1x7_reduce_bn.f32"));
    vx_size inception_stem2_1x7_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_stem2_1x7_reduce_scale_W;
    inception_stem2_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_stem2_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_reduce_scale_W, dataFolder + "/weights/inception_stem2_1x7_reduce_scale.f32"));
    vx_size inception_stem2_1x7_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_stem2_1x7_reduce_scale_B;
    inception_stem2_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_stem2_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_reduce_scale_B, dataFolder + "/bias/inception_stem2_1x7_reduce_scale.f32"));
    vx_node inception_stem2_1x7_reduce_bn_node;
    inception_stem2_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_stem2_1x7_reduce, inception_stem2_1x7_reduce_bn_W, inception_stem2_1x7_reduce_bn_B, inception_stem2_1x7_reduce_scale_W, inception_stem2_1x7_reduce_scale_B, inception_stem2_1x7_reduce_bn_eps, inception_stem2_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_1x7_reduce_bn_node));

    // inception_stem2_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem2_1x7_reduce_relu Layer
    vx_size inception_stem2_1x7_reduce_relu_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_1x7_reduce_relu;
    inception_stem2_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_stem2_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_relu);
    vx_enum inception_stem2_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem2_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_stem2_1x7_reduce_relu_param_b = 0;
    vx_node inception_stem2_1x7_reduce_relu_node;
    inception_stem2_1x7_reduce_relu_node = vxActivationLayer(graph, inception_stem2_1x7_reduce_scale, inception_stem2_1x7_reduce_relu_mode, inception_stem2_1x7_reduce_relu_param_a, inception_stem2_1x7_reduce_relu_param_b, inception_stem2_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_1x7_reduce_relu_node));

    // inception_stem2_1x7 Layer
    vx_size inception_stem2_1x7_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_1x7;
    inception_stem2_1x7 = vxCreateVirtualTensor(graph,4, inception_stem2_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7);
    vx_size inception_stem2_1x7_W_dims[4] = { 7, 1, 64, 64 };
    vx_tensor inception_stem2_1x7_W;
    inception_stem2_1x7_W = vxCreateTensor(context,4, inception_stem2_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_W, dataFolder + "/weights/inception_stem2_1x7.f32"));
    vx_nn_convolution_params_t inception_stem2_1x7_params;
    inception_stem2_1x7_params.padding_x = 3;
    inception_stem2_1x7_params.padding_y = 0;
    inception_stem2_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem2_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem2_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem2_1x7_params.dilation_x = 0;
    inception_stem2_1x7_params.dilation_y = 0;
    vx_node inception_stem2_1x7_node;
    inception_stem2_1x7_node = vxConvolutionLayer(graph, inception_stem2_1x7_reduce_relu, inception_stem2_1x7_W, NULL, &inception_stem2_1x7_params, sizeof(inception_stem2_1x7_params ), inception_stem2_1x7);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_1x7_node));

    // inception_stem2_1x7_bn Layer
    vx_size inception_stem2_1x7_scale_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_1x7_scale;
    inception_stem2_1x7_scale = vxCreateVirtualTensor(graph,4, inception_stem2_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_scale);
    vx_size inception_stem2_1x7_bn_W_dims[1] = { 64 };
    vx_float32 inception_stem2_1x7_bn_eps = 0.001;
    vx_tensor inception_stem2_1x7_bn_W;
    inception_stem2_1x7_bn_W = vxCreateTensor(context,1, inception_stem2_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_bn_W, dataFolder + "/weights/inception_stem2_1x7_bn.f32"));
    vx_size inception_stem2_1x7_bn_B_dims[1] = { 64 };
    vx_tensor inception_stem2_1x7_bn_B;
    inception_stem2_1x7_bn_B = vxCreateTensor(context,1, inception_stem2_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_bn_B, dataFolder + "/bias/inception_stem2_1x7_bn.f32"));
    vx_size inception_stem2_1x7_scale_W_dims[1] = { 64 };
    vx_tensor inception_stem2_1x7_scale_W;
    inception_stem2_1x7_scale_W = vxCreateTensor(context,1, inception_stem2_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_scale_W, dataFolder + "/weights/inception_stem2_1x7_scale.f32"));
    vx_size inception_stem2_1x7_scale_B_dims[1] = { 64 };
    vx_tensor inception_stem2_1x7_scale_B;
    inception_stem2_1x7_scale_B = vxCreateTensor(context,1, inception_stem2_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_1x7_scale_B, dataFolder + "/bias/inception_stem2_1x7_scale.f32"));
    vx_node inception_stem2_1x7_bn_node;
    inception_stem2_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_stem2_1x7, inception_stem2_1x7_bn_W, inception_stem2_1x7_bn_B, inception_stem2_1x7_scale_W, inception_stem2_1x7_scale_B, inception_stem2_1x7_bn_eps, inception_stem2_1x7_scale);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_1x7_bn_node));

    // inception_stem2_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem2_1x7_relu Layer
    vx_size inception_stem2_1x7_relu_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_1x7_relu;
    inception_stem2_1x7_relu = vxCreateVirtualTensor(graph,4, inception_stem2_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_relu);
    vx_enum inception_stem2_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem2_1x7_relu_param_a = 0;
    vx_float32 inception_stem2_1x7_relu_param_b = 0;
    vx_node inception_stem2_1x7_relu_node;
    inception_stem2_1x7_relu_node = vxActivationLayer(graph, inception_stem2_1x7_scale, inception_stem2_1x7_relu_mode, inception_stem2_1x7_relu_param_a, inception_stem2_1x7_relu_param_b, inception_stem2_1x7_relu);
    ERROR_CHECK_OBJECT(inception_stem2_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_1x7_relu_node));

    // inception_stem2_7x1 Layer
    vx_size inception_stem2_7x1_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_7x1;
    inception_stem2_7x1 = vxCreateVirtualTensor(graph,4, inception_stem2_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1);
    vx_size inception_stem2_7x1_W_dims[4] = { 1, 7, 64, 64 };
    vx_tensor inception_stem2_7x1_W;
    inception_stem2_7x1_W = vxCreateTensor(context,4, inception_stem2_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_7x1_W, dataFolder + "/weights/inception_stem2_7x1.f32"));
    vx_nn_convolution_params_t inception_stem2_7x1_params;
    inception_stem2_7x1_params.padding_x = 0;
    inception_stem2_7x1_params.padding_y = 3;
    inception_stem2_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem2_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem2_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem2_7x1_params.dilation_x = 0;
    inception_stem2_7x1_params.dilation_y = 0;
    vx_node inception_stem2_7x1_node;
    inception_stem2_7x1_node = vxConvolutionLayer(graph, inception_stem2_1x7_relu, inception_stem2_7x1_W, NULL, &inception_stem2_7x1_params, sizeof(inception_stem2_7x1_params ), inception_stem2_7x1);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_7x1_node));

    // inception_stem2_7x1_bn Layer
    vx_size inception_stem2_7x1_scale_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_7x1_scale;
    inception_stem2_7x1_scale = vxCreateVirtualTensor(graph,4, inception_stem2_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_scale);
    vx_size inception_stem2_7x1_bn_W_dims[1] = { 64 };
    vx_float32 inception_stem2_7x1_bn_eps = 0.001;
    vx_tensor inception_stem2_7x1_bn_W;
    inception_stem2_7x1_bn_W = vxCreateTensor(context,1, inception_stem2_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_7x1_bn_W, dataFolder + "/weights/inception_stem2_7x1_bn.f32"));
    vx_size inception_stem2_7x1_bn_B_dims[1] = { 64 };
    vx_tensor inception_stem2_7x1_bn_B;
    inception_stem2_7x1_bn_B = vxCreateTensor(context,1, inception_stem2_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_7x1_bn_B, dataFolder + "/bias/inception_stem2_7x1_bn.f32"));
    vx_size inception_stem2_7x1_scale_W_dims[1] = { 64 };
    vx_tensor inception_stem2_7x1_scale_W;
    inception_stem2_7x1_scale_W = vxCreateTensor(context,1, inception_stem2_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_7x1_scale_W, dataFolder + "/weights/inception_stem2_7x1_scale.f32"));
    vx_size inception_stem2_7x1_scale_B_dims[1] = { 64 };
    vx_tensor inception_stem2_7x1_scale_B;
    inception_stem2_7x1_scale_B = vxCreateTensor(context,1, inception_stem2_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_7x1_scale_B, dataFolder + "/bias/inception_stem2_7x1_scale.f32"));
    vx_node inception_stem2_7x1_bn_node;
    inception_stem2_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_stem2_7x1, inception_stem2_7x1_bn_W, inception_stem2_7x1_bn_B, inception_stem2_7x1_scale_W, inception_stem2_7x1_scale_B, inception_stem2_7x1_bn_eps, inception_stem2_7x1_scale);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_7x1_bn_node));

    // inception_stem2_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem2_7x1_relu Layer
    vx_size inception_stem2_7x1_relu_dims[4] = { 73, 73, 64, 1 };
    vx_tensor inception_stem2_7x1_relu;
    inception_stem2_7x1_relu = vxCreateVirtualTensor(graph,4, inception_stem2_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_relu);
    vx_enum inception_stem2_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem2_7x1_relu_param_a = 0;
    vx_float32 inception_stem2_7x1_relu_param_b = 0;
    vx_node inception_stem2_7x1_relu_node;
    inception_stem2_7x1_relu_node = vxActivationLayer(graph, inception_stem2_7x1_scale, inception_stem2_7x1_relu_mode, inception_stem2_7x1_relu_param_a, inception_stem2_7x1_relu_param_b, inception_stem2_7x1_relu);
    ERROR_CHECK_OBJECT(inception_stem2_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_7x1_relu_node));

    // inception_stem2_3x3_2 Layer
    vx_size inception_stem2_3x3_2_dims[4] = { 71, 71, 96, 1 };
    vx_tensor inception_stem2_3x3_2;
    inception_stem2_3x3_2 = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2);
    vx_size inception_stem2_3x3_2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_stem2_3x3_2_W;
    inception_stem2_3x3_2_W = vxCreateTensor(context,4, inception_stem2_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_2_W, dataFolder + "/weights/inception_stem2_3x3_2.f32"));
    vx_nn_convolution_params_t inception_stem2_3x3_2_params;
    inception_stem2_3x3_2_params.padding_x = 0;
    inception_stem2_3x3_2_params.padding_y = 0;
    inception_stem2_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem2_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem2_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem2_3x3_2_params.dilation_x = 0;
    inception_stem2_3x3_2_params.dilation_y = 0;
    vx_node inception_stem2_3x3_2_node;
    inception_stem2_3x3_2_node = vxConvolutionLayer(graph, inception_stem2_7x1_relu, inception_stem2_3x3_2_W, NULL, &inception_stem2_3x3_2_params, sizeof(inception_stem2_3x3_2_params ), inception_stem2_3x3_2);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_2_node));

    // inception_stem2_3x3_2_bn Layer
    vx_size inception_stem2_3x3_2_scale_dims[4] = { 71, 71, 96, 1 };
    vx_tensor inception_stem2_3x3_2_scale;
    inception_stem2_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_scale);
    vx_size inception_stem2_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_stem2_3x3_2_bn_eps = 0.001;
    vx_tensor inception_stem2_3x3_2_bn_W;
    inception_stem2_3x3_2_bn_W = vxCreateTensor(context,1, inception_stem2_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_2_bn_W, dataFolder + "/weights/inception_stem2_3x3_2_bn.f32"));
    vx_size inception_stem2_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_stem2_3x3_2_bn_B;
    inception_stem2_3x3_2_bn_B = vxCreateTensor(context,1, inception_stem2_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_2_bn_B, dataFolder + "/bias/inception_stem2_3x3_2_bn.f32"));
    vx_size inception_stem2_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_stem2_3x3_2_scale_W;
    inception_stem2_3x3_2_scale_W = vxCreateTensor(context,1, inception_stem2_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_2_scale_W, dataFolder + "/weights/inception_stem2_3x3_2_scale.f32"));
    vx_size inception_stem2_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_stem2_3x3_2_scale_B;
    inception_stem2_3x3_2_scale_B = vxCreateTensor(context,1, inception_stem2_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem2_3x3_2_scale_B, dataFolder + "/bias/inception_stem2_3x3_2_scale.f32"));
    vx_node inception_stem2_3x3_2_bn_node;
    inception_stem2_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_stem2_3x3_2, inception_stem2_3x3_2_bn_W, inception_stem2_3x3_2_bn_B, inception_stem2_3x3_2_scale_W, inception_stem2_3x3_2_scale_B, inception_stem2_3x3_2_bn_eps, inception_stem2_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_2_bn_node));

    // inception_stem2_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem2_3x3_2_relu Layer
    vx_size inception_stem2_3x3_2_relu_dims[4] = { 71, 71, 96, 1 };
    vx_tensor inception_stem2_3x3_2_relu;
    inception_stem2_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_stem2_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_relu);
    vx_enum inception_stem2_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem2_3x3_2_relu_param_a = 0;
    vx_float32 inception_stem2_3x3_2_relu_param_b = 0;
    vx_node inception_stem2_3x3_2_relu_node;
    inception_stem2_3x3_2_relu_node = vxActivationLayer(graph, inception_stem2_3x3_2_scale, inception_stem2_3x3_2_relu_mode, inception_stem2_3x3_2_relu_param_a, inception_stem2_3x3_2_relu_param_b, inception_stem2_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_stem2_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_3x3_2_relu_node));

    // inception_stem2 Layer
    vx_size inception_stem2_dims[4] = { 71, 71, 192, 1 };
    vx_tensor inception_stem2;
    inception_stem2 = vxCreateVirtualTensor(graph,4, inception_stem2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem2);
    vx_node inception_stem2_node;
    inception_stem2_node = vxConcatLayer(graph, inception_stem2, inception_stem2_3x3_relu, inception_stem2_3x3_2_relu, NULL, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_stem2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem2_node));

    // inception_stem3_3x3_s2 Layer
    vx_size inception_stem3_3x3_s2_dims[4] = { 35, 35, 192, 1 };
    vx_tensor inception_stem3_3x3_s2;
    inception_stem3_3x3_s2 = vxCreateVirtualTensor(graph,4, inception_stem3_3x3_s2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2);
    vx_size inception_stem3_3x3_s2_W_dims[4] = { 3, 3, 192, 192 };
    vx_tensor inception_stem3_3x3_s2_W;
    inception_stem3_3x3_s2_W = vxCreateTensor(context,4, inception_stem3_3x3_s2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem3_3x3_s2_W, dataFolder + "/weights/inception_stem3_3x3_s2.f32"));
    vx_nn_convolution_params_t inception_stem3_3x3_s2_params;
    inception_stem3_3x3_s2_params.padding_x = 0;
    inception_stem3_3x3_s2_params.padding_y = 0;
    inception_stem3_3x3_s2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_stem3_3x3_s2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_stem3_3x3_s2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_stem3_3x3_s2_params.dilation_x = 0;
    inception_stem3_3x3_s2_params.dilation_y = 0;
    vx_node inception_stem3_3x3_s2_node;
    inception_stem3_3x3_s2_node = vxConvolutionLayer(graph, inception_stem2, inception_stem3_3x3_s2_W, NULL, &inception_stem3_3x3_s2_params, sizeof(inception_stem3_3x3_s2_params ), inception_stem3_3x3_s2);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem3_3x3_s2_node));

    // inception_stem3_3x3_s2_bn Layer
    vx_size inception_stem3_3x3_s2_scale_dims[4] = { 35, 35, 192, 1 };
    vx_tensor inception_stem3_3x3_s2_scale;
    inception_stem3_3x3_s2_scale = vxCreateVirtualTensor(graph,4, inception_stem3_3x3_s2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_scale);
    vx_size inception_stem3_3x3_s2_bn_W_dims[1] = { 192 };
    vx_float32 inception_stem3_3x3_s2_bn_eps = 0.001;
    vx_tensor inception_stem3_3x3_s2_bn_W;
    inception_stem3_3x3_s2_bn_W = vxCreateTensor(context,1, inception_stem3_3x3_s2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem3_3x3_s2_bn_W, dataFolder + "/weights/inception_stem3_3x3_s2_bn.f32"));
    vx_size inception_stem3_3x3_s2_bn_B_dims[1] = { 192 };
    vx_tensor inception_stem3_3x3_s2_bn_B;
    inception_stem3_3x3_s2_bn_B = vxCreateTensor(context,1, inception_stem3_3x3_s2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem3_3x3_s2_bn_B, dataFolder + "/bias/inception_stem3_3x3_s2_bn.f32"));
    vx_size inception_stem3_3x3_s2_scale_W_dims[1] = { 192 };
    vx_tensor inception_stem3_3x3_s2_scale_W;
    inception_stem3_3x3_s2_scale_W = vxCreateTensor(context,1, inception_stem3_3x3_s2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem3_3x3_s2_scale_W, dataFolder + "/weights/inception_stem3_3x3_s2_scale.f32"));
    vx_size inception_stem3_3x3_s2_scale_B_dims[1] = { 192 };
    vx_tensor inception_stem3_3x3_s2_scale_B;
    inception_stem3_3x3_s2_scale_B = vxCreateTensor(context,1, inception_stem3_3x3_s2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_stem3_3x3_s2_scale_B, dataFolder + "/bias/inception_stem3_3x3_s2_scale.f32"));
    vx_node inception_stem3_3x3_s2_bn_node;
    inception_stem3_3x3_s2_bn_node = vxBatchNormalizationLayer(graph, inception_stem3_3x3_s2, inception_stem3_3x3_s2_bn_W, inception_stem3_3x3_s2_bn_B, inception_stem3_3x3_s2_scale_W, inception_stem3_3x3_s2_scale_B, inception_stem3_3x3_s2_bn_eps, inception_stem3_3x3_s2_scale);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem3_3x3_s2_bn_node));

    // inception_stem3_3x3_s2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_stem3_3x3_s2_relu Layer
    vx_size inception_stem3_3x3_s2_relu_dims[4] = { 35, 35, 192, 1 };
    vx_tensor inception_stem3_3x3_s2_relu;
    inception_stem3_3x3_s2_relu = vxCreateVirtualTensor(graph,4, inception_stem3_3x3_s2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_relu);
    vx_enum inception_stem3_3x3_s2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_stem3_3x3_s2_relu_param_a = 0;
    vx_float32 inception_stem3_3x3_s2_relu_param_b = 0;
    vx_node inception_stem3_3x3_s2_relu_node;
    inception_stem3_3x3_s2_relu_node = vxActivationLayer(graph, inception_stem3_3x3_s2_scale, inception_stem3_3x3_s2_relu_mode, inception_stem3_3x3_s2_relu_param_a, inception_stem3_3x3_s2_relu_param_b, inception_stem3_3x3_s2_relu);
    ERROR_CHECK_OBJECT(inception_stem3_3x3_s2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem3_3x3_s2_relu_node));

    // inception_stem3_pool Layer
    vx_size inception_stem3_pool_dims[4] = { 35, 35, 192, 1 };
    vx_tensor inception_stem3_pool;
    inception_stem3_pool = vxCreateVirtualTensor(graph,4, inception_stem3_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem3_pool);
    vx_enum inception_stem3_pool_type = VX_NN_POOLING_MAX;
    vx_size inception_stem3_pool_kernel_w = 3;
    vx_size inception_stem3_pool_kernel_h = 3;
    vx_size inception_stem3_pool_pad_w = 0;
    vx_size inception_stem3_pool_pad_h = 0;
    vx_enum inception_stem3_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_stem3_pool_node;
    inception_stem3_pool_node = vxPoolingLayer(graph, inception_stem2, inception_stem3_pool_type, inception_stem3_pool_kernel_w, inception_stem3_pool_kernel_h, inception_stem3_pool_pad_w, inception_stem3_pool_pad_h, inception_stem3_pool_roundPolicy, inception_stem3_pool );
    ERROR_CHECK_OBJECT(inception_stem3_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem3_pool_node));

    // inception_stem3 Layer
    vx_size inception_stem3_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_stem3;
    inception_stem3 = vxCreateVirtualTensor(graph,4, inception_stem3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_stem3);
    vx_node inception_stem3_node;
    inception_stem3_node = vxConcatLayer(graph, inception_stem3, inception_stem3_3x3_s2_relu, inception_stem3_pool, NULL, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_stem3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_stem3_node));

    // inception_a1_1x1_2 Layer
    vx_size inception_a1_1x1_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_1x1_2;
    inception_a1_1x1_2 = vxCreateVirtualTensor(graph,4, inception_a1_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2);
    vx_size inception_a1_1x1_2_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a1_1x1_2_W;
    inception_a1_1x1_2_W = vxCreateTensor(context,4, inception_a1_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_2_W, dataFolder + "/weights/inception_a1_1x1_2.f32"));
    vx_nn_convolution_params_t inception_a1_1x1_2_params;
    inception_a1_1x1_2_params.padding_x = 0;
    inception_a1_1x1_2_params.padding_y = 0;
    inception_a1_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_1x1_2_params.dilation_x = 0;
    inception_a1_1x1_2_params.dilation_y = 0;
    vx_node inception_a1_1x1_2_node;
    inception_a1_1x1_2_node = vxConvolutionLayer(graph, inception_stem3, inception_a1_1x1_2_W, NULL, &inception_a1_1x1_2_params, sizeof(inception_a1_1x1_2_params ), inception_a1_1x1_2);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_2_node));

    // inception_a1_1x1_2_bn Layer
    vx_size inception_a1_1x1_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_1x1_2_scale;
    inception_a1_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_a1_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_scale);
    vx_size inception_a1_1x1_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_1x1_2_bn_eps = 0.001;
    vx_tensor inception_a1_1x1_2_bn_W;
    inception_a1_1x1_2_bn_W = vxCreateTensor(context,1, inception_a1_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_2_bn_W, dataFolder + "/weights/inception_a1_1x1_2_bn.f32"));
    vx_size inception_a1_1x1_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_1x1_2_bn_B;
    inception_a1_1x1_2_bn_B = vxCreateTensor(context,1, inception_a1_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_2_bn_B, dataFolder + "/bias/inception_a1_1x1_2_bn.f32"));
    vx_size inception_a1_1x1_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_1x1_2_scale_W;
    inception_a1_1x1_2_scale_W = vxCreateTensor(context,1, inception_a1_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_2_scale_W, dataFolder + "/weights/inception_a1_1x1_2_scale.f32"));
    vx_size inception_a1_1x1_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_1x1_2_scale_B;
    inception_a1_1x1_2_scale_B = vxCreateTensor(context,1, inception_a1_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_2_scale_B, dataFolder + "/bias/inception_a1_1x1_2_scale.f32"));
    vx_node inception_a1_1x1_2_bn_node;
    inception_a1_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_a1_1x1_2, inception_a1_1x1_2_bn_W, inception_a1_1x1_2_bn_B, inception_a1_1x1_2_scale_W, inception_a1_1x1_2_scale_B, inception_a1_1x1_2_bn_eps, inception_a1_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_2_bn_node));

    // inception_a1_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_1x1_2_relu Layer
    vx_size inception_a1_1x1_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_1x1_2_relu;
    inception_a1_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_a1_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_relu);
    vx_enum inception_a1_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_1x1_2_relu_param_a = 0;
    vx_float32 inception_a1_1x1_2_relu_param_b = 0;
    vx_node inception_a1_1x1_2_relu_node;
    inception_a1_1x1_2_relu_node = vxActivationLayer(graph, inception_a1_1x1_2_scale, inception_a1_1x1_2_relu_mode, inception_a1_1x1_2_relu_param_a, inception_a1_1x1_2_relu_param_b, inception_a1_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_a1_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_2_relu_node));

    // inception_a1_3x3_reduce Layer
    vx_size inception_a1_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_reduce;
    inception_a1_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a1_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce);
    vx_size inception_a1_3x3_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a1_3x3_reduce_W;
    inception_a1_3x3_reduce_W = vxCreateTensor(context,4, inception_a1_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_W, dataFolder + "/weights/inception_a1_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_reduce_params;
    inception_a1_3x3_reduce_params.padding_x = 0;
    inception_a1_3x3_reduce_params.padding_y = 0;
    inception_a1_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_reduce_params.dilation_x = 0;
    inception_a1_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a1_3x3_reduce_node;
    inception_a1_3x3_reduce_node = vxConvolutionLayer(graph, inception_stem3, inception_a1_3x3_reduce_W, NULL, &inception_a1_3x3_reduce_params, sizeof(inception_a1_3x3_reduce_params ), inception_a1_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_reduce_node));

    // inception_a1_3x3_reduce_bn Layer
    vx_size inception_a1_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_reduce_scale;
    inception_a1_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_scale);
    vx_size inception_a1_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a1_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_reduce_bn_W;
    inception_a1_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a1_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_bn_W, dataFolder + "/weights/inception_a1_3x3_reduce_bn.f32"));
    vx_size inception_a1_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_reduce_bn_B;
    inception_a1_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a1_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_bn_B, dataFolder + "/bias/inception_a1_3x3_reduce_bn.f32"));
    vx_size inception_a1_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_reduce_scale_W;
    inception_a1_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a1_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_scale_W, dataFolder + "/weights/inception_a1_3x3_reduce_scale.f32"));
    vx_size inception_a1_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_reduce_scale_B;
    inception_a1_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a1_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_reduce_scale_B, dataFolder + "/bias/inception_a1_3x3_reduce_scale.f32"));
    vx_node inception_a1_3x3_reduce_bn_node;
    inception_a1_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_reduce, inception_a1_3x3_reduce_bn_W, inception_a1_3x3_reduce_bn_B, inception_a1_3x3_reduce_scale_W, inception_a1_3x3_reduce_scale_B, inception_a1_3x3_reduce_bn_eps, inception_a1_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_reduce_bn_node));

    // inception_a1_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_reduce_relu Layer
    vx_size inception_a1_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_reduce_relu;
    inception_a1_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_relu);
    vx_enum inception_a1_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a1_3x3_reduce_relu_param_b = 0;
    vx_node inception_a1_3x3_reduce_relu_node;
    inception_a1_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a1_3x3_reduce_scale, inception_a1_3x3_reduce_relu_mode, inception_a1_3x3_reduce_relu_param_a, inception_a1_3x3_reduce_relu_param_b, inception_a1_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_reduce_relu_node));

    // inception_a1_3x3 Layer
    vx_size inception_a1_3x3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3;
    inception_a1_3x3 = vxCreateVirtualTensor(graph,4, inception_a1_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3);
    vx_size inception_a1_3x3_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a1_3x3_W;
    inception_a1_3x3_W = vxCreateTensor(context,4, inception_a1_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_W, dataFolder + "/weights/inception_a1_3x3.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_params;
    inception_a1_3x3_params.padding_x = 1;
    inception_a1_3x3_params.padding_y = 1;
    inception_a1_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_params.dilation_x = 0;
    inception_a1_3x3_params.dilation_y = 0;
    vx_node inception_a1_3x3_node;
    inception_a1_3x3_node = vxConvolutionLayer(graph, inception_a1_3x3_reduce_relu, inception_a1_3x3_W, NULL, &inception_a1_3x3_params, sizeof(inception_a1_3x3_params ), inception_a1_3x3);
    ERROR_CHECK_OBJECT(inception_a1_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_node));

    // inception_a1_3x3_bn Layer
    vx_size inception_a1_3x3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_scale;
    inception_a1_3x3_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_scale);
    vx_size inception_a1_3x3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_3x3_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_bn_W;
    inception_a1_3x3_bn_W = vxCreateTensor(context,1, inception_a1_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_bn_W, dataFolder + "/weights/inception_a1_3x3_bn.f32"));
    vx_size inception_a1_3x3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_bn_B;
    inception_a1_3x3_bn_B = vxCreateTensor(context,1, inception_a1_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_bn_B, dataFolder + "/bias/inception_a1_3x3_bn.f32"));
    vx_size inception_a1_3x3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_scale_W;
    inception_a1_3x3_scale_W = vxCreateTensor(context,1, inception_a1_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_scale_W, dataFolder + "/weights/inception_a1_3x3_scale.f32"));
    vx_size inception_a1_3x3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_scale_B;
    inception_a1_3x3_scale_B = vxCreateTensor(context,1, inception_a1_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_scale_B, dataFolder + "/bias/inception_a1_3x3_scale.f32"));
    vx_node inception_a1_3x3_bn_node;
    inception_a1_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3, inception_a1_3x3_bn_W, inception_a1_3x3_bn_B, inception_a1_3x3_scale_W, inception_a1_3x3_scale_B, inception_a1_3x3_bn_eps, inception_a1_3x3_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_bn_node));

    // inception_a1_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_relu Layer
    vx_size inception_a1_3x3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_relu;
    inception_a1_3x3_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_relu);
    vx_enum inception_a1_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_relu_param_a = 0;
    vx_float32 inception_a1_3x3_relu_param_b = 0;
    vx_node inception_a1_3x3_relu_node;
    inception_a1_3x3_relu_node = vxActivationLayer(graph, inception_a1_3x3_scale, inception_a1_3x3_relu_mode, inception_a1_3x3_relu_param_a, inception_a1_3x3_relu_param_b, inception_a1_3x3_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_relu_node));

    // inception_a1_3x3_2_reduce Layer
    vx_size inception_a1_3x3_2_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_2_reduce;
    inception_a1_3x3_2_reduce = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce);
    vx_size inception_a1_3x3_2_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a1_3x3_2_reduce_W;
    inception_a1_3x3_2_reduce_W = vxCreateTensor(context,4, inception_a1_3x3_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_reduce_W, dataFolder + "/weights/inception_a1_3x3_2_reduce.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_2_reduce_params;
    inception_a1_3x3_2_reduce_params.padding_x = 0;
    inception_a1_3x3_2_reduce_params.padding_y = 0;
    inception_a1_3x3_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_2_reduce_params.dilation_x = 0;
    inception_a1_3x3_2_reduce_params.dilation_y = 0;
    vx_node inception_a1_3x3_2_reduce_node;
    inception_a1_3x3_2_reduce_node = vxConvolutionLayer(graph, inception_stem3, inception_a1_3x3_2_reduce_W, NULL, &inception_a1_3x3_2_reduce_params, sizeof(inception_a1_3x3_2_reduce_params ), inception_a1_3x3_2_reduce);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_reduce_node));

    // inception_a1_3x3_2_reduce_bn Layer
    vx_size inception_a1_3x3_2_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_2_reduce_scale;
    inception_a1_3x3_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_scale);
    vx_size inception_a1_3x3_2_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a1_3x3_2_reduce_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_2_reduce_bn_W;
    inception_a1_3x3_2_reduce_bn_W = vxCreateTensor(context,1, inception_a1_3x3_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_reduce_bn_W, dataFolder + "/weights/inception_a1_3x3_2_reduce_bn.f32"));
    vx_size inception_a1_3x3_2_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_2_reduce_bn_B;
    inception_a1_3x3_2_reduce_bn_B = vxCreateTensor(context,1, inception_a1_3x3_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_reduce_bn_B, dataFolder + "/bias/inception_a1_3x3_2_reduce_bn.f32"));
    vx_size inception_a1_3x3_2_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_2_reduce_scale_W;
    inception_a1_3x3_2_reduce_scale_W = vxCreateTensor(context,1, inception_a1_3x3_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_reduce_scale_W, dataFolder + "/weights/inception_a1_3x3_2_reduce_scale.f32"));
    vx_size inception_a1_3x3_2_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a1_3x3_2_reduce_scale_B;
    inception_a1_3x3_2_reduce_scale_B = vxCreateTensor(context,1, inception_a1_3x3_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_reduce_scale_B, dataFolder + "/bias/inception_a1_3x3_2_reduce_scale.f32"));
    vx_node inception_a1_3x3_2_reduce_bn_node;
    inception_a1_3x3_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_2_reduce, inception_a1_3x3_2_reduce_bn_W, inception_a1_3x3_2_reduce_bn_B, inception_a1_3x3_2_reduce_scale_W, inception_a1_3x3_2_reduce_scale_B, inception_a1_3x3_2_reduce_bn_eps, inception_a1_3x3_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_reduce_bn_node));

    // inception_a1_3x3_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_2_reduce_relu Layer
    vx_size inception_a1_3x3_2_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a1_3x3_2_reduce_relu;
    inception_a1_3x3_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_relu);
    vx_enum inception_a1_3x3_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_2_reduce_relu_param_a = 0;
    vx_float32 inception_a1_3x3_2_reduce_relu_param_b = 0;
    vx_node inception_a1_3x3_2_reduce_relu_node;
    inception_a1_3x3_2_reduce_relu_node = vxActivationLayer(graph, inception_a1_3x3_2_reduce_scale, inception_a1_3x3_2_reduce_relu_mode, inception_a1_3x3_2_reduce_relu_param_a, inception_a1_3x3_2_reduce_relu_param_b, inception_a1_3x3_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_reduce_relu_node));

    // inception_a1_3x3_2 Layer
    vx_size inception_a1_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_2;
    inception_a1_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2);
    vx_size inception_a1_3x3_2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a1_3x3_2_W;
    inception_a1_3x3_2_W = vxCreateTensor(context,4, inception_a1_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_W, dataFolder + "/weights/inception_a1_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_2_params;
    inception_a1_3x3_2_params.padding_x = 1;
    inception_a1_3x3_2_params.padding_y = 1;
    inception_a1_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_2_params.dilation_x = 0;
    inception_a1_3x3_2_params.dilation_y = 0;
    vx_node inception_a1_3x3_2_node;
    inception_a1_3x3_2_node = vxConvolutionLayer(graph, inception_a1_3x3_2_reduce_relu, inception_a1_3x3_2_W, NULL, &inception_a1_3x3_2_params, sizeof(inception_a1_3x3_2_params ), inception_a1_3x3_2);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_node));

    // inception_a1_3x3_2_bn Layer
    vx_size inception_a1_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_2_scale;
    inception_a1_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_scale);
    vx_size inception_a1_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_2_bn_W;
    inception_a1_3x3_2_bn_W = vxCreateTensor(context,1, inception_a1_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_bn_W, dataFolder + "/weights/inception_a1_3x3_2_bn.f32"));
    vx_size inception_a1_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_2_bn_B;
    inception_a1_3x3_2_bn_B = vxCreateTensor(context,1, inception_a1_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_bn_B, dataFolder + "/bias/inception_a1_3x3_2_bn.f32"));
    vx_size inception_a1_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_2_scale_W;
    inception_a1_3x3_2_scale_W = vxCreateTensor(context,1, inception_a1_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_scale_W, dataFolder + "/weights/inception_a1_3x3_2_scale.f32"));
    vx_size inception_a1_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_2_scale_B;
    inception_a1_3x3_2_scale_B = vxCreateTensor(context,1, inception_a1_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_2_scale_B, dataFolder + "/bias/inception_a1_3x3_2_scale.f32"));
    vx_node inception_a1_3x3_2_bn_node;
    inception_a1_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_2, inception_a1_3x3_2_bn_W, inception_a1_3x3_2_bn_B, inception_a1_3x3_2_scale_W, inception_a1_3x3_2_scale_B, inception_a1_3x3_2_bn_eps, inception_a1_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_bn_node));

    // inception_a1_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_2_relu Layer
    vx_size inception_a1_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_2_relu;
    inception_a1_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_relu);
    vx_enum inception_a1_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_2_relu_param_a = 0;
    vx_float32 inception_a1_3x3_2_relu_param_b = 0;
    vx_node inception_a1_3x3_2_relu_node;
    inception_a1_3x3_2_relu_node = vxActivationLayer(graph, inception_a1_3x3_2_scale, inception_a1_3x3_2_relu_mode, inception_a1_3x3_2_relu_param_a, inception_a1_3x3_2_relu_param_b, inception_a1_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_2_relu_node));

    // inception_a1_3x3_3 Layer
    vx_size inception_a1_3x3_3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_3;
    inception_a1_3x3_3 = vxCreateVirtualTensor(graph,4, inception_a1_3x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3);
    vx_size inception_a1_3x3_3_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a1_3x3_3_W;
    inception_a1_3x3_3_W = vxCreateTensor(context,4, inception_a1_3x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_3_W, dataFolder + "/weights/inception_a1_3x3_3.f32"));
    vx_nn_convolution_params_t inception_a1_3x3_3_params;
    inception_a1_3x3_3_params.padding_x = 1;
    inception_a1_3x3_3_params.padding_y = 1;
    inception_a1_3x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_3x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_3x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_3x3_3_params.dilation_x = 0;
    inception_a1_3x3_3_params.dilation_y = 0;
    vx_node inception_a1_3x3_3_node;
    inception_a1_3x3_3_node = vxConvolutionLayer(graph, inception_a1_3x3_2_relu, inception_a1_3x3_3_W, NULL, &inception_a1_3x3_3_params, sizeof(inception_a1_3x3_3_params ), inception_a1_3x3_3);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_3_node));

    // inception_a1_3x3_3_bn Layer
    vx_size inception_a1_3x3_3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_3_scale;
    inception_a1_3x3_3_scale = vxCreateVirtualTensor(graph,4, inception_a1_3x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_scale);
    vx_size inception_a1_3x3_3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_3x3_3_bn_eps = 0.001;
    vx_tensor inception_a1_3x3_3_bn_W;
    inception_a1_3x3_3_bn_W = vxCreateTensor(context,1, inception_a1_3x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_3_bn_W, dataFolder + "/weights/inception_a1_3x3_3_bn.f32"));
    vx_size inception_a1_3x3_3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_3_bn_B;
    inception_a1_3x3_3_bn_B = vxCreateTensor(context,1, inception_a1_3x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_3_bn_B, dataFolder + "/bias/inception_a1_3x3_3_bn.f32"));
    vx_size inception_a1_3x3_3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_3_scale_W;
    inception_a1_3x3_3_scale_W = vxCreateTensor(context,1, inception_a1_3x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_3_scale_W, dataFolder + "/weights/inception_a1_3x3_3_scale.f32"));
    vx_size inception_a1_3x3_3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_3x3_3_scale_B;
    inception_a1_3x3_3_scale_B = vxCreateTensor(context,1, inception_a1_3x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_3x3_3_scale_B, dataFolder + "/bias/inception_a1_3x3_3_scale.f32"));
    vx_node inception_a1_3x3_3_bn_node;
    inception_a1_3x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_a1_3x3_3, inception_a1_3x3_3_bn_W, inception_a1_3x3_3_bn_B, inception_a1_3x3_3_scale_W, inception_a1_3x3_3_scale_B, inception_a1_3x3_3_bn_eps, inception_a1_3x3_3_scale);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_3_bn_node));

    // inception_a1_3x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_3x3_3_relu Layer
    vx_size inception_a1_3x3_3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_3x3_3_relu;
    inception_a1_3x3_3_relu = vxCreateVirtualTensor(graph,4, inception_a1_3x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_relu);
    vx_enum inception_a1_3x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_3x3_3_relu_param_a = 0;
    vx_float32 inception_a1_3x3_3_relu_param_b = 0;
    vx_node inception_a1_3x3_3_relu_node;
    inception_a1_3x3_3_relu_node = vxActivationLayer(graph, inception_a1_3x3_3_scale, inception_a1_3x3_3_relu_mode, inception_a1_3x3_3_relu_param_a, inception_a1_3x3_3_relu_param_b, inception_a1_3x3_3_relu);
    ERROR_CHECK_OBJECT(inception_a1_3x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_3x3_3_relu_node));

    // inception_a1_pool_ave Layer
    vx_size inception_a1_pool_ave_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a1_pool_ave;
    inception_a1_pool_ave = vxCreateVirtualTensor(graph,4, inception_a1_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_pool_ave);
    vx_enum inception_a1_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_a1_pool_ave_kernel_w = 3;
    vx_size inception_a1_pool_ave_kernel_h = 3;
    vx_size inception_a1_pool_ave_pad_w = 1;
    vx_size inception_a1_pool_ave_pad_h = 1;
    vx_enum inception_a1_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a1_pool_ave_node;
    inception_a1_pool_ave_node = vxPoolingLayer(graph, inception_stem3, inception_a1_pool_ave_type, inception_a1_pool_ave_kernel_w, inception_a1_pool_ave_kernel_h, inception_a1_pool_ave_pad_w, inception_a1_pool_ave_pad_h, inception_a1_pool_ave_roundPolicy, inception_a1_pool_ave );
    ERROR_CHECK_OBJECT(inception_a1_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_pool_ave_node));

    // inception_a1_1x1 Layer
    vx_size inception_a1_1x1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_1x1;
    inception_a1_1x1 = vxCreateVirtualTensor(graph,4, inception_a1_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1);
    vx_size inception_a1_1x1_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a1_1x1_W;
    inception_a1_1x1_W = vxCreateTensor(context,4, inception_a1_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_W, dataFolder + "/weights/inception_a1_1x1.f32"));
    vx_nn_convolution_params_t inception_a1_1x1_params;
    inception_a1_1x1_params.padding_x = 0;
    inception_a1_1x1_params.padding_y = 0;
    inception_a1_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a1_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a1_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a1_1x1_params.dilation_x = 0;
    inception_a1_1x1_params.dilation_y = 0;
    vx_node inception_a1_1x1_node;
    inception_a1_1x1_node = vxConvolutionLayer(graph, inception_a1_pool_ave, inception_a1_1x1_W, NULL, &inception_a1_1x1_params, sizeof(inception_a1_1x1_params ), inception_a1_1x1);
    ERROR_CHECK_OBJECT(inception_a1_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_node));

    // inception_a1_1x1_bn Layer
    vx_size inception_a1_1x1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_1x1_scale;
    inception_a1_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a1_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_scale);
    vx_size inception_a1_1x1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a1_1x1_bn_eps = 0.001;
    vx_tensor inception_a1_1x1_bn_W;
    inception_a1_1x1_bn_W = vxCreateTensor(context,1, inception_a1_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_bn_W, dataFolder + "/weights/inception_a1_1x1_bn.f32"));
    vx_size inception_a1_1x1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a1_1x1_bn_B;
    inception_a1_1x1_bn_B = vxCreateTensor(context,1, inception_a1_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_bn_B, dataFolder + "/bias/inception_a1_1x1_bn.f32"));
    vx_size inception_a1_1x1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a1_1x1_scale_W;
    inception_a1_1x1_scale_W = vxCreateTensor(context,1, inception_a1_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_scale_W, dataFolder + "/weights/inception_a1_1x1_scale.f32"));
    vx_size inception_a1_1x1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a1_1x1_scale_B;
    inception_a1_1x1_scale_B = vxCreateTensor(context,1, inception_a1_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a1_1x1_scale_B, dataFolder + "/bias/inception_a1_1x1_scale.f32"));
    vx_node inception_a1_1x1_bn_node;
    inception_a1_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a1_1x1, inception_a1_1x1_bn_W, inception_a1_1x1_bn_B, inception_a1_1x1_scale_W, inception_a1_1x1_scale_B, inception_a1_1x1_bn_eps, inception_a1_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a1_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_bn_node));

    // inception_a1_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a1_1x1_relu Layer
    vx_size inception_a1_1x1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a1_1x1_relu;
    inception_a1_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a1_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_1x1_relu);
    vx_enum inception_a1_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a1_1x1_relu_param_a = 0;
    vx_float32 inception_a1_1x1_relu_param_b = 0;
    vx_node inception_a1_1x1_relu_node;
    inception_a1_1x1_relu_node = vxActivationLayer(graph, inception_a1_1x1_scale, inception_a1_1x1_relu_mode, inception_a1_1x1_relu_param_a, inception_a1_1x1_relu_param_b, inception_a1_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a1_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_1x1_relu_node));

    // inception_a1_concat Layer
    vx_size inception_a1_concat_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a1_concat;
    inception_a1_concat = vxCreateVirtualTensor(graph,4, inception_a1_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a1_concat);
    vx_node inception_a1_concat_node;
    inception_a1_concat_node = vxConcatLayer(graph, inception_a1_concat, inception_a1_1x1_2_relu, inception_a1_3x3_relu, inception_a1_3x3_3_relu, inception_a1_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a1_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a1_concat_node));

    // inception_a2_1x1_2 Layer
    vx_size inception_a2_1x1_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_1x1_2;
    inception_a2_1x1_2 = vxCreateVirtualTensor(graph,4, inception_a2_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2);
    vx_size inception_a2_1x1_2_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a2_1x1_2_W;
    inception_a2_1x1_2_W = vxCreateTensor(context,4, inception_a2_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_2_W, dataFolder + "/weights/inception_a2_1x1_2.f32"));
    vx_nn_convolution_params_t inception_a2_1x1_2_params;
    inception_a2_1x1_2_params.padding_x = 0;
    inception_a2_1x1_2_params.padding_y = 0;
    inception_a2_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_1x1_2_params.dilation_x = 0;
    inception_a2_1x1_2_params.dilation_y = 0;
    vx_node inception_a2_1x1_2_node;
    inception_a2_1x1_2_node = vxConvolutionLayer(graph, inception_a1_concat, inception_a2_1x1_2_W, NULL, &inception_a2_1x1_2_params, sizeof(inception_a2_1x1_2_params ), inception_a2_1x1_2);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_2_node));

    // inception_a2_1x1_2_bn Layer
    vx_size inception_a2_1x1_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_1x1_2_scale;
    inception_a2_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_a2_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_scale);
    vx_size inception_a2_1x1_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_1x1_2_bn_eps = 0.001;
    vx_tensor inception_a2_1x1_2_bn_W;
    inception_a2_1x1_2_bn_W = vxCreateTensor(context,1, inception_a2_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_2_bn_W, dataFolder + "/weights/inception_a2_1x1_2_bn.f32"));
    vx_size inception_a2_1x1_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_1x1_2_bn_B;
    inception_a2_1x1_2_bn_B = vxCreateTensor(context,1, inception_a2_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_2_bn_B, dataFolder + "/bias/inception_a2_1x1_2_bn.f32"));
    vx_size inception_a2_1x1_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_1x1_2_scale_W;
    inception_a2_1x1_2_scale_W = vxCreateTensor(context,1, inception_a2_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_2_scale_W, dataFolder + "/weights/inception_a2_1x1_2_scale.f32"));
    vx_size inception_a2_1x1_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_1x1_2_scale_B;
    inception_a2_1x1_2_scale_B = vxCreateTensor(context,1, inception_a2_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_2_scale_B, dataFolder + "/bias/inception_a2_1x1_2_scale.f32"));
    vx_node inception_a2_1x1_2_bn_node;
    inception_a2_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_a2_1x1_2, inception_a2_1x1_2_bn_W, inception_a2_1x1_2_bn_B, inception_a2_1x1_2_scale_W, inception_a2_1x1_2_scale_B, inception_a2_1x1_2_bn_eps, inception_a2_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_2_bn_node));

    // inception_a2_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_1x1_2_relu Layer
    vx_size inception_a2_1x1_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_1x1_2_relu;
    inception_a2_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_a2_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_relu);
    vx_enum inception_a2_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_1x1_2_relu_param_a = 0;
    vx_float32 inception_a2_1x1_2_relu_param_b = 0;
    vx_node inception_a2_1x1_2_relu_node;
    inception_a2_1x1_2_relu_node = vxActivationLayer(graph, inception_a2_1x1_2_scale, inception_a2_1x1_2_relu_mode, inception_a2_1x1_2_relu_param_a, inception_a2_1x1_2_relu_param_b, inception_a2_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_a2_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_2_relu_node));

    // inception_a2_3x3_reduce Layer
    vx_size inception_a2_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_reduce;
    inception_a2_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a2_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce);
    vx_size inception_a2_3x3_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a2_3x3_reduce_W;
    inception_a2_3x3_reduce_W = vxCreateTensor(context,4, inception_a2_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_W, dataFolder + "/weights/inception_a2_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_reduce_params;
    inception_a2_3x3_reduce_params.padding_x = 0;
    inception_a2_3x3_reduce_params.padding_y = 0;
    inception_a2_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_reduce_params.dilation_x = 0;
    inception_a2_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a2_3x3_reduce_node;
    inception_a2_3x3_reduce_node = vxConvolutionLayer(graph, inception_a1_concat, inception_a2_3x3_reduce_W, NULL, &inception_a2_3x3_reduce_params, sizeof(inception_a2_3x3_reduce_params ), inception_a2_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_reduce_node));

    // inception_a2_3x3_reduce_bn Layer
    vx_size inception_a2_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_reduce_scale;
    inception_a2_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_scale);
    vx_size inception_a2_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a2_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_reduce_bn_W;
    inception_a2_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a2_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_bn_W, dataFolder + "/weights/inception_a2_3x3_reduce_bn.f32"));
    vx_size inception_a2_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_reduce_bn_B;
    inception_a2_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a2_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_bn_B, dataFolder + "/bias/inception_a2_3x3_reduce_bn.f32"));
    vx_size inception_a2_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_reduce_scale_W;
    inception_a2_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a2_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_scale_W, dataFolder + "/weights/inception_a2_3x3_reduce_scale.f32"));
    vx_size inception_a2_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_reduce_scale_B;
    inception_a2_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a2_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_reduce_scale_B, dataFolder + "/bias/inception_a2_3x3_reduce_scale.f32"));
    vx_node inception_a2_3x3_reduce_bn_node;
    inception_a2_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_reduce, inception_a2_3x3_reduce_bn_W, inception_a2_3x3_reduce_bn_B, inception_a2_3x3_reduce_scale_W, inception_a2_3x3_reduce_scale_B, inception_a2_3x3_reduce_bn_eps, inception_a2_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_reduce_bn_node));

    // inception_a2_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_reduce_relu Layer
    vx_size inception_a2_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_reduce_relu;
    inception_a2_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_relu);
    vx_enum inception_a2_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a2_3x3_reduce_relu_param_b = 0;
    vx_node inception_a2_3x3_reduce_relu_node;
    inception_a2_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a2_3x3_reduce_scale, inception_a2_3x3_reduce_relu_mode, inception_a2_3x3_reduce_relu_param_a, inception_a2_3x3_reduce_relu_param_b, inception_a2_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_reduce_relu_node));

    // inception_a2_3x3 Layer
    vx_size inception_a2_3x3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3;
    inception_a2_3x3 = vxCreateVirtualTensor(graph,4, inception_a2_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3);
    vx_size inception_a2_3x3_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a2_3x3_W;
    inception_a2_3x3_W = vxCreateTensor(context,4, inception_a2_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_W, dataFolder + "/weights/inception_a2_3x3.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_params;
    inception_a2_3x3_params.padding_x = 1;
    inception_a2_3x3_params.padding_y = 1;
    inception_a2_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_params.dilation_x = 0;
    inception_a2_3x3_params.dilation_y = 0;
    vx_node inception_a2_3x3_node;
    inception_a2_3x3_node = vxConvolutionLayer(graph, inception_a2_3x3_reduce_relu, inception_a2_3x3_W, NULL, &inception_a2_3x3_params, sizeof(inception_a2_3x3_params ), inception_a2_3x3);
    ERROR_CHECK_OBJECT(inception_a2_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_node));

    // inception_a2_3x3_bn Layer
    vx_size inception_a2_3x3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_scale;
    inception_a2_3x3_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_scale);
    vx_size inception_a2_3x3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_3x3_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_bn_W;
    inception_a2_3x3_bn_W = vxCreateTensor(context,1, inception_a2_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_bn_W, dataFolder + "/weights/inception_a2_3x3_bn.f32"));
    vx_size inception_a2_3x3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_bn_B;
    inception_a2_3x3_bn_B = vxCreateTensor(context,1, inception_a2_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_bn_B, dataFolder + "/bias/inception_a2_3x3_bn.f32"));
    vx_size inception_a2_3x3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_scale_W;
    inception_a2_3x3_scale_W = vxCreateTensor(context,1, inception_a2_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_scale_W, dataFolder + "/weights/inception_a2_3x3_scale.f32"));
    vx_size inception_a2_3x3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_scale_B;
    inception_a2_3x3_scale_B = vxCreateTensor(context,1, inception_a2_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_scale_B, dataFolder + "/bias/inception_a2_3x3_scale.f32"));
    vx_node inception_a2_3x3_bn_node;
    inception_a2_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3, inception_a2_3x3_bn_W, inception_a2_3x3_bn_B, inception_a2_3x3_scale_W, inception_a2_3x3_scale_B, inception_a2_3x3_bn_eps, inception_a2_3x3_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_bn_node));

    // inception_a2_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_relu Layer
    vx_size inception_a2_3x3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_relu;
    inception_a2_3x3_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_relu);
    vx_enum inception_a2_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_relu_param_a = 0;
    vx_float32 inception_a2_3x3_relu_param_b = 0;
    vx_node inception_a2_3x3_relu_node;
    inception_a2_3x3_relu_node = vxActivationLayer(graph, inception_a2_3x3_scale, inception_a2_3x3_relu_mode, inception_a2_3x3_relu_param_a, inception_a2_3x3_relu_param_b, inception_a2_3x3_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_relu_node));

    // inception_a2_3x3_2_reduce Layer
    vx_size inception_a2_3x3_2_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_2_reduce;
    inception_a2_3x3_2_reduce = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce);
    vx_size inception_a2_3x3_2_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a2_3x3_2_reduce_W;
    inception_a2_3x3_2_reduce_W = vxCreateTensor(context,4, inception_a2_3x3_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_reduce_W, dataFolder + "/weights/inception_a2_3x3_2_reduce.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_2_reduce_params;
    inception_a2_3x3_2_reduce_params.padding_x = 0;
    inception_a2_3x3_2_reduce_params.padding_y = 0;
    inception_a2_3x3_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_2_reduce_params.dilation_x = 0;
    inception_a2_3x3_2_reduce_params.dilation_y = 0;
    vx_node inception_a2_3x3_2_reduce_node;
    inception_a2_3x3_2_reduce_node = vxConvolutionLayer(graph, inception_a1_concat, inception_a2_3x3_2_reduce_W, NULL, &inception_a2_3x3_2_reduce_params, sizeof(inception_a2_3x3_2_reduce_params ), inception_a2_3x3_2_reduce);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_reduce_node));

    // inception_a2_3x3_2_reduce_bn Layer
    vx_size inception_a2_3x3_2_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_2_reduce_scale;
    inception_a2_3x3_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_scale);
    vx_size inception_a2_3x3_2_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a2_3x3_2_reduce_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_2_reduce_bn_W;
    inception_a2_3x3_2_reduce_bn_W = vxCreateTensor(context,1, inception_a2_3x3_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_reduce_bn_W, dataFolder + "/weights/inception_a2_3x3_2_reduce_bn.f32"));
    vx_size inception_a2_3x3_2_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_2_reduce_bn_B;
    inception_a2_3x3_2_reduce_bn_B = vxCreateTensor(context,1, inception_a2_3x3_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_reduce_bn_B, dataFolder + "/bias/inception_a2_3x3_2_reduce_bn.f32"));
    vx_size inception_a2_3x3_2_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_2_reduce_scale_W;
    inception_a2_3x3_2_reduce_scale_W = vxCreateTensor(context,1, inception_a2_3x3_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_reduce_scale_W, dataFolder + "/weights/inception_a2_3x3_2_reduce_scale.f32"));
    vx_size inception_a2_3x3_2_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a2_3x3_2_reduce_scale_B;
    inception_a2_3x3_2_reduce_scale_B = vxCreateTensor(context,1, inception_a2_3x3_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_reduce_scale_B, dataFolder + "/bias/inception_a2_3x3_2_reduce_scale.f32"));
    vx_node inception_a2_3x3_2_reduce_bn_node;
    inception_a2_3x3_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_2_reduce, inception_a2_3x3_2_reduce_bn_W, inception_a2_3x3_2_reduce_bn_B, inception_a2_3x3_2_reduce_scale_W, inception_a2_3x3_2_reduce_scale_B, inception_a2_3x3_2_reduce_bn_eps, inception_a2_3x3_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_reduce_bn_node));

    // inception_a2_3x3_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_2_reduce_relu Layer
    vx_size inception_a2_3x3_2_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a2_3x3_2_reduce_relu;
    inception_a2_3x3_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_relu);
    vx_enum inception_a2_3x3_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_2_reduce_relu_param_a = 0;
    vx_float32 inception_a2_3x3_2_reduce_relu_param_b = 0;
    vx_node inception_a2_3x3_2_reduce_relu_node;
    inception_a2_3x3_2_reduce_relu_node = vxActivationLayer(graph, inception_a2_3x3_2_reduce_scale, inception_a2_3x3_2_reduce_relu_mode, inception_a2_3x3_2_reduce_relu_param_a, inception_a2_3x3_2_reduce_relu_param_b, inception_a2_3x3_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_reduce_relu_node));

    // inception_a2_3x3_2 Layer
    vx_size inception_a2_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_2;
    inception_a2_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2);
    vx_size inception_a2_3x3_2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a2_3x3_2_W;
    inception_a2_3x3_2_W = vxCreateTensor(context,4, inception_a2_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_W, dataFolder + "/weights/inception_a2_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_2_params;
    inception_a2_3x3_2_params.padding_x = 1;
    inception_a2_3x3_2_params.padding_y = 1;
    inception_a2_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_2_params.dilation_x = 0;
    inception_a2_3x3_2_params.dilation_y = 0;
    vx_node inception_a2_3x3_2_node;
    inception_a2_3x3_2_node = vxConvolutionLayer(graph, inception_a2_3x3_2_reduce_relu, inception_a2_3x3_2_W, NULL, &inception_a2_3x3_2_params, sizeof(inception_a2_3x3_2_params ), inception_a2_3x3_2);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_node));

    // inception_a2_3x3_2_bn Layer
    vx_size inception_a2_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_2_scale;
    inception_a2_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_scale);
    vx_size inception_a2_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_2_bn_W;
    inception_a2_3x3_2_bn_W = vxCreateTensor(context,1, inception_a2_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_bn_W, dataFolder + "/weights/inception_a2_3x3_2_bn.f32"));
    vx_size inception_a2_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_2_bn_B;
    inception_a2_3x3_2_bn_B = vxCreateTensor(context,1, inception_a2_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_bn_B, dataFolder + "/bias/inception_a2_3x3_2_bn.f32"));
    vx_size inception_a2_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_2_scale_W;
    inception_a2_3x3_2_scale_W = vxCreateTensor(context,1, inception_a2_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_scale_W, dataFolder + "/weights/inception_a2_3x3_2_scale.f32"));
    vx_size inception_a2_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_2_scale_B;
    inception_a2_3x3_2_scale_B = vxCreateTensor(context,1, inception_a2_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_2_scale_B, dataFolder + "/bias/inception_a2_3x3_2_scale.f32"));
    vx_node inception_a2_3x3_2_bn_node;
    inception_a2_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_2, inception_a2_3x3_2_bn_W, inception_a2_3x3_2_bn_B, inception_a2_3x3_2_scale_W, inception_a2_3x3_2_scale_B, inception_a2_3x3_2_bn_eps, inception_a2_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_bn_node));

    // inception_a2_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_2_relu Layer
    vx_size inception_a2_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_2_relu;
    inception_a2_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_relu);
    vx_enum inception_a2_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_2_relu_param_a = 0;
    vx_float32 inception_a2_3x3_2_relu_param_b = 0;
    vx_node inception_a2_3x3_2_relu_node;
    inception_a2_3x3_2_relu_node = vxActivationLayer(graph, inception_a2_3x3_2_scale, inception_a2_3x3_2_relu_mode, inception_a2_3x3_2_relu_param_a, inception_a2_3x3_2_relu_param_b, inception_a2_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_2_relu_node));

    // inception_a2_3x3_3 Layer
    vx_size inception_a2_3x3_3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_3;
    inception_a2_3x3_3 = vxCreateVirtualTensor(graph,4, inception_a2_3x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3);
    vx_size inception_a2_3x3_3_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a2_3x3_3_W;
    inception_a2_3x3_3_W = vxCreateTensor(context,4, inception_a2_3x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_3_W, dataFolder + "/weights/inception_a2_3x3_3.f32"));
    vx_nn_convolution_params_t inception_a2_3x3_3_params;
    inception_a2_3x3_3_params.padding_x = 1;
    inception_a2_3x3_3_params.padding_y = 1;
    inception_a2_3x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_3x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_3x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_3x3_3_params.dilation_x = 0;
    inception_a2_3x3_3_params.dilation_y = 0;
    vx_node inception_a2_3x3_3_node;
    inception_a2_3x3_3_node = vxConvolutionLayer(graph, inception_a2_3x3_2_relu, inception_a2_3x3_3_W, NULL, &inception_a2_3x3_3_params, sizeof(inception_a2_3x3_3_params ), inception_a2_3x3_3);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_3_node));

    // inception_a2_3x3_3_bn Layer
    vx_size inception_a2_3x3_3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_3_scale;
    inception_a2_3x3_3_scale = vxCreateVirtualTensor(graph,4, inception_a2_3x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_scale);
    vx_size inception_a2_3x3_3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_3x3_3_bn_eps = 0.001;
    vx_tensor inception_a2_3x3_3_bn_W;
    inception_a2_3x3_3_bn_W = vxCreateTensor(context,1, inception_a2_3x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_3_bn_W, dataFolder + "/weights/inception_a2_3x3_3_bn.f32"));
    vx_size inception_a2_3x3_3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_3_bn_B;
    inception_a2_3x3_3_bn_B = vxCreateTensor(context,1, inception_a2_3x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_3_bn_B, dataFolder + "/bias/inception_a2_3x3_3_bn.f32"));
    vx_size inception_a2_3x3_3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_3_scale_W;
    inception_a2_3x3_3_scale_W = vxCreateTensor(context,1, inception_a2_3x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_3_scale_W, dataFolder + "/weights/inception_a2_3x3_3_scale.f32"));
    vx_size inception_a2_3x3_3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_3x3_3_scale_B;
    inception_a2_3x3_3_scale_B = vxCreateTensor(context,1, inception_a2_3x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_3x3_3_scale_B, dataFolder + "/bias/inception_a2_3x3_3_scale.f32"));
    vx_node inception_a2_3x3_3_bn_node;
    inception_a2_3x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_a2_3x3_3, inception_a2_3x3_3_bn_W, inception_a2_3x3_3_bn_B, inception_a2_3x3_3_scale_W, inception_a2_3x3_3_scale_B, inception_a2_3x3_3_bn_eps, inception_a2_3x3_3_scale);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_3_bn_node));

    // inception_a2_3x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_3x3_3_relu Layer
    vx_size inception_a2_3x3_3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_3x3_3_relu;
    inception_a2_3x3_3_relu = vxCreateVirtualTensor(graph,4, inception_a2_3x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_relu);
    vx_enum inception_a2_3x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_3x3_3_relu_param_a = 0;
    vx_float32 inception_a2_3x3_3_relu_param_b = 0;
    vx_node inception_a2_3x3_3_relu_node;
    inception_a2_3x3_3_relu_node = vxActivationLayer(graph, inception_a2_3x3_3_scale, inception_a2_3x3_3_relu_mode, inception_a2_3x3_3_relu_param_a, inception_a2_3x3_3_relu_param_b, inception_a2_3x3_3_relu);
    ERROR_CHECK_OBJECT(inception_a2_3x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_3x3_3_relu_node));

    // inception_a2_pool_ave Layer
    vx_size inception_a2_pool_ave_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a2_pool_ave;
    inception_a2_pool_ave = vxCreateVirtualTensor(graph,4, inception_a2_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_pool_ave);
    vx_enum inception_a2_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_a2_pool_ave_kernel_w = 3;
    vx_size inception_a2_pool_ave_kernel_h = 3;
    vx_size inception_a2_pool_ave_pad_w = 1;
    vx_size inception_a2_pool_ave_pad_h = 1;
    vx_enum inception_a2_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a2_pool_ave_node;
    inception_a2_pool_ave_node = vxPoolingLayer(graph, inception_a1_concat, inception_a2_pool_ave_type, inception_a2_pool_ave_kernel_w, inception_a2_pool_ave_kernel_h, inception_a2_pool_ave_pad_w, inception_a2_pool_ave_pad_h, inception_a2_pool_ave_roundPolicy, inception_a2_pool_ave );
    ERROR_CHECK_OBJECT(inception_a2_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_pool_ave_node));

    // inception_a2_1x1 Layer
    vx_size inception_a2_1x1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_1x1;
    inception_a2_1x1 = vxCreateVirtualTensor(graph,4, inception_a2_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1);
    vx_size inception_a2_1x1_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a2_1x1_W;
    inception_a2_1x1_W = vxCreateTensor(context,4, inception_a2_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_W, dataFolder + "/weights/inception_a2_1x1.f32"));
    vx_nn_convolution_params_t inception_a2_1x1_params;
    inception_a2_1x1_params.padding_x = 0;
    inception_a2_1x1_params.padding_y = 0;
    inception_a2_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a2_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a2_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a2_1x1_params.dilation_x = 0;
    inception_a2_1x1_params.dilation_y = 0;
    vx_node inception_a2_1x1_node;
    inception_a2_1x1_node = vxConvolutionLayer(graph, inception_a2_pool_ave, inception_a2_1x1_W, NULL, &inception_a2_1x1_params, sizeof(inception_a2_1x1_params ), inception_a2_1x1);
    ERROR_CHECK_OBJECT(inception_a2_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_node));

    // inception_a2_1x1_bn Layer
    vx_size inception_a2_1x1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_1x1_scale;
    inception_a2_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a2_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_scale);
    vx_size inception_a2_1x1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a2_1x1_bn_eps = 0.001;
    vx_tensor inception_a2_1x1_bn_W;
    inception_a2_1x1_bn_W = vxCreateTensor(context,1, inception_a2_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_bn_W, dataFolder + "/weights/inception_a2_1x1_bn.f32"));
    vx_size inception_a2_1x1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a2_1x1_bn_B;
    inception_a2_1x1_bn_B = vxCreateTensor(context,1, inception_a2_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_bn_B, dataFolder + "/bias/inception_a2_1x1_bn.f32"));
    vx_size inception_a2_1x1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a2_1x1_scale_W;
    inception_a2_1x1_scale_W = vxCreateTensor(context,1, inception_a2_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_scale_W, dataFolder + "/weights/inception_a2_1x1_scale.f32"));
    vx_size inception_a2_1x1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a2_1x1_scale_B;
    inception_a2_1x1_scale_B = vxCreateTensor(context,1, inception_a2_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a2_1x1_scale_B, dataFolder + "/bias/inception_a2_1x1_scale.f32"));
    vx_node inception_a2_1x1_bn_node;
    inception_a2_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a2_1x1, inception_a2_1x1_bn_W, inception_a2_1x1_bn_B, inception_a2_1x1_scale_W, inception_a2_1x1_scale_B, inception_a2_1x1_bn_eps, inception_a2_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a2_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_bn_node));

    // inception_a2_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a2_1x1_relu Layer
    vx_size inception_a2_1x1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a2_1x1_relu;
    inception_a2_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a2_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_1x1_relu);
    vx_enum inception_a2_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a2_1x1_relu_param_a = 0;
    vx_float32 inception_a2_1x1_relu_param_b = 0;
    vx_node inception_a2_1x1_relu_node;
    inception_a2_1x1_relu_node = vxActivationLayer(graph, inception_a2_1x1_scale, inception_a2_1x1_relu_mode, inception_a2_1x1_relu_param_a, inception_a2_1x1_relu_param_b, inception_a2_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a2_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_1x1_relu_node));

    // inception_a2_concat Layer
    vx_size inception_a2_concat_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a2_concat;
    inception_a2_concat = vxCreateVirtualTensor(graph,4, inception_a2_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a2_concat);
    vx_node inception_a2_concat_node;
    inception_a2_concat_node = vxConcatLayer(graph, inception_a2_concat, inception_a2_1x1_2_relu, inception_a2_3x3_relu, inception_a2_3x3_3_relu, inception_a2_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a2_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a2_concat_node));

    // inception_a3_1x1_2 Layer
    vx_size inception_a3_1x1_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_1x1_2;
    inception_a3_1x1_2 = vxCreateVirtualTensor(graph,4, inception_a3_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2);
    vx_size inception_a3_1x1_2_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a3_1x1_2_W;
    inception_a3_1x1_2_W = vxCreateTensor(context,4, inception_a3_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_2_W, dataFolder + "/weights/inception_a3_1x1_2.f32"));
    vx_nn_convolution_params_t inception_a3_1x1_2_params;
    inception_a3_1x1_2_params.padding_x = 0;
    inception_a3_1x1_2_params.padding_y = 0;
    inception_a3_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_1x1_2_params.dilation_x = 0;
    inception_a3_1x1_2_params.dilation_y = 0;
    vx_node inception_a3_1x1_2_node;
    inception_a3_1x1_2_node = vxConvolutionLayer(graph, inception_a2_concat, inception_a3_1x1_2_W, NULL, &inception_a3_1x1_2_params, sizeof(inception_a3_1x1_2_params ), inception_a3_1x1_2);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_2_node));

    // inception_a3_1x1_2_bn Layer
    vx_size inception_a3_1x1_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_1x1_2_scale;
    inception_a3_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_a3_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_scale);
    vx_size inception_a3_1x1_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_1x1_2_bn_eps = 0.001;
    vx_tensor inception_a3_1x1_2_bn_W;
    inception_a3_1x1_2_bn_W = vxCreateTensor(context,1, inception_a3_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_2_bn_W, dataFolder + "/weights/inception_a3_1x1_2_bn.f32"));
    vx_size inception_a3_1x1_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_1x1_2_bn_B;
    inception_a3_1x1_2_bn_B = vxCreateTensor(context,1, inception_a3_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_2_bn_B, dataFolder + "/bias/inception_a3_1x1_2_bn.f32"));
    vx_size inception_a3_1x1_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_1x1_2_scale_W;
    inception_a3_1x1_2_scale_W = vxCreateTensor(context,1, inception_a3_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_2_scale_W, dataFolder + "/weights/inception_a3_1x1_2_scale.f32"));
    vx_size inception_a3_1x1_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_1x1_2_scale_B;
    inception_a3_1x1_2_scale_B = vxCreateTensor(context,1, inception_a3_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_2_scale_B, dataFolder + "/bias/inception_a3_1x1_2_scale.f32"));
    vx_node inception_a3_1x1_2_bn_node;
    inception_a3_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_a3_1x1_2, inception_a3_1x1_2_bn_W, inception_a3_1x1_2_bn_B, inception_a3_1x1_2_scale_W, inception_a3_1x1_2_scale_B, inception_a3_1x1_2_bn_eps, inception_a3_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_2_bn_node));

    // inception_a3_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_1x1_2_relu Layer
    vx_size inception_a3_1x1_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_1x1_2_relu;
    inception_a3_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_a3_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_relu);
    vx_enum inception_a3_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_1x1_2_relu_param_a = 0;
    vx_float32 inception_a3_1x1_2_relu_param_b = 0;
    vx_node inception_a3_1x1_2_relu_node;
    inception_a3_1x1_2_relu_node = vxActivationLayer(graph, inception_a3_1x1_2_scale, inception_a3_1x1_2_relu_mode, inception_a3_1x1_2_relu_param_a, inception_a3_1x1_2_relu_param_b, inception_a3_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_a3_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_2_relu_node));

    // inception_a3_3x3_reduce Layer
    vx_size inception_a3_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_reduce;
    inception_a3_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a3_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce);
    vx_size inception_a3_3x3_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a3_3x3_reduce_W;
    inception_a3_3x3_reduce_W = vxCreateTensor(context,4, inception_a3_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_W, dataFolder + "/weights/inception_a3_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_reduce_params;
    inception_a3_3x3_reduce_params.padding_x = 0;
    inception_a3_3x3_reduce_params.padding_y = 0;
    inception_a3_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_reduce_params.dilation_x = 0;
    inception_a3_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a3_3x3_reduce_node;
    inception_a3_3x3_reduce_node = vxConvolutionLayer(graph, inception_a2_concat, inception_a3_3x3_reduce_W, NULL, &inception_a3_3x3_reduce_params, sizeof(inception_a3_3x3_reduce_params ), inception_a3_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_reduce_node));

    // inception_a3_3x3_reduce_bn Layer
    vx_size inception_a3_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_reduce_scale;
    inception_a3_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_scale);
    vx_size inception_a3_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a3_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_reduce_bn_W;
    inception_a3_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a3_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_bn_W, dataFolder + "/weights/inception_a3_3x3_reduce_bn.f32"));
    vx_size inception_a3_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_reduce_bn_B;
    inception_a3_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a3_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_bn_B, dataFolder + "/bias/inception_a3_3x3_reduce_bn.f32"));
    vx_size inception_a3_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_reduce_scale_W;
    inception_a3_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a3_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_scale_W, dataFolder + "/weights/inception_a3_3x3_reduce_scale.f32"));
    vx_size inception_a3_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_reduce_scale_B;
    inception_a3_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a3_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_reduce_scale_B, dataFolder + "/bias/inception_a3_3x3_reduce_scale.f32"));
    vx_node inception_a3_3x3_reduce_bn_node;
    inception_a3_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_reduce, inception_a3_3x3_reduce_bn_W, inception_a3_3x3_reduce_bn_B, inception_a3_3x3_reduce_scale_W, inception_a3_3x3_reduce_scale_B, inception_a3_3x3_reduce_bn_eps, inception_a3_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_reduce_bn_node));

    // inception_a3_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_reduce_relu Layer
    vx_size inception_a3_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_reduce_relu;
    inception_a3_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_relu);
    vx_enum inception_a3_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a3_3x3_reduce_relu_param_b = 0;
    vx_node inception_a3_3x3_reduce_relu_node;
    inception_a3_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a3_3x3_reduce_scale, inception_a3_3x3_reduce_relu_mode, inception_a3_3x3_reduce_relu_param_a, inception_a3_3x3_reduce_relu_param_b, inception_a3_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_reduce_relu_node));

    // inception_a3_3x3 Layer
    vx_size inception_a3_3x3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3;
    inception_a3_3x3 = vxCreateVirtualTensor(graph,4, inception_a3_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3);
    vx_size inception_a3_3x3_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a3_3x3_W;
    inception_a3_3x3_W = vxCreateTensor(context,4, inception_a3_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_W, dataFolder + "/weights/inception_a3_3x3.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_params;
    inception_a3_3x3_params.padding_x = 1;
    inception_a3_3x3_params.padding_y = 1;
    inception_a3_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_params.dilation_x = 0;
    inception_a3_3x3_params.dilation_y = 0;
    vx_node inception_a3_3x3_node;
    inception_a3_3x3_node = vxConvolutionLayer(graph, inception_a3_3x3_reduce_relu, inception_a3_3x3_W, NULL, &inception_a3_3x3_params, sizeof(inception_a3_3x3_params ), inception_a3_3x3);
    ERROR_CHECK_OBJECT(inception_a3_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_node));

    // inception_a3_3x3_bn Layer
    vx_size inception_a3_3x3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_scale;
    inception_a3_3x3_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_scale);
    vx_size inception_a3_3x3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_3x3_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_bn_W;
    inception_a3_3x3_bn_W = vxCreateTensor(context,1, inception_a3_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_bn_W, dataFolder + "/weights/inception_a3_3x3_bn.f32"));
    vx_size inception_a3_3x3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_bn_B;
    inception_a3_3x3_bn_B = vxCreateTensor(context,1, inception_a3_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_bn_B, dataFolder + "/bias/inception_a3_3x3_bn.f32"));
    vx_size inception_a3_3x3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_scale_W;
    inception_a3_3x3_scale_W = vxCreateTensor(context,1, inception_a3_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_scale_W, dataFolder + "/weights/inception_a3_3x3_scale.f32"));
    vx_size inception_a3_3x3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_scale_B;
    inception_a3_3x3_scale_B = vxCreateTensor(context,1, inception_a3_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_scale_B, dataFolder + "/bias/inception_a3_3x3_scale.f32"));
    vx_node inception_a3_3x3_bn_node;
    inception_a3_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3, inception_a3_3x3_bn_W, inception_a3_3x3_bn_B, inception_a3_3x3_scale_W, inception_a3_3x3_scale_B, inception_a3_3x3_bn_eps, inception_a3_3x3_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_bn_node));

    // inception_a3_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_relu Layer
    vx_size inception_a3_3x3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_relu;
    inception_a3_3x3_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_relu);
    vx_enum inception_a3_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_relu_param_a = 0;
    vx_float32 inception_a3_3x3_relu_param_b = 0;
    vx_node inception_a3_3x3_relu_node;
    inception_a3_3x3_relu_node = vxActivationLayer(graph, inception_a3_3x3_scale, inception_a3_3x3_relu_mode, inception_a3_3x3_relu_param_a, inception_a3_3x3_relu_param_b, inception_a3_3x3_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_relu_node));

    // inception_a3_3x3_2_reduce Layer
    vx_size inception_a3_3x3_2_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_2_reduce;
    inception_a3_3x3_2_reduce = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce);
    vx_size inception_a3_3x3_2_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a3_3x3_2_reduce_W;
    inception_a3_3x3_2_reduce_W = vxCreateTensor(context,4, inception_a3_3x3_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_reduce_W, dataFolder + "/weights/inception_a3_3x3_2_reduce.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_2_reduce_params;
    inception_a3_3x3_2_reduce_params.padding_x = 0;
    inception_a3_3x3_2_reduce_params.padding_y = 0;
    inception_a3_3x3_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_2_reduce_params.dilation_x = 0;
    inception_a3_3x3_2_reduce_params.dilation_y = 0;
    vx_node inception_a3_3x3_2_reduce_node;
    inception_a3_3x3_2_reduce_node = vxConvolutionLayer(graph, inception_a2_concat, inception_a3_3x3_2_reduce_W, NULL, &inception_a3_3x3_2_reduce_params, sizeof(inception_a3_3x3_2_reduce_params ), inception_a3_3x3_2_reduce);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_reduce_node));

    // inception_a3_3x3_2_reduce_bn Layer
    vx_size inception_a3_3x3_2_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_2_reduce_scale;
    inception_a3_3x3_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_scale);
    vx_size inception_a3_3x3_2_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a3_3x3_2_reduce_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_2_reduce_bn_W;
    inception_a3_3x3_2_reduce_bn_W = vxCreateTensor(context,1, inception_a3_3x3_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_reduce_bn_W, dataFolder + "/weights/inception_a3_3x3_2_reduce_bn.f32"));
    vx_size inception_a3_3x3_2_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_2_reduce_bn_B;
    inception_a3_3x3_2_reduce_bn_B = vxCreateTensor(context,1, inception_a3_3x3_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_reduce_bn_B, dataFolder + "/bias/inception_a3_3x3_2_reduce_bn.f32"));
    vx_size inception_a3_3x3_2_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_2_reduce_scale_W;
    inception_a3_3x3_2_reduce_scale_W = vxCreateTensor(context,1, inception_a3_3x3_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_reduce_scale_W, dataFolder + "/weights/inception_a3_3x3_2_reduce_scale.f32"));
    vx_size inception_a3_3x3_2_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a3_3x3_2_reduce_scale_B;
    inception_a3_3x3_2_reduce_scale_B = vxCreateTensor(context,1, inception_a3_3x3_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_reduce_scale_B, dataFolder + "/bias/inception_a3_3x3_2_reduce_scale.f32"));
    vx_node inception_a3_3x3_2_reduce_bn_node;
    inception_a3_3x3_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_2_reduce, inception_a3_3x3_2_reduce_bn_W, inception_a3_3x3_2_reduce_bn_B, inception_a3_3x3_2_reduce_scale_W, inception_a3_3x3_2_reduce_scale_B, inception_a3_3x3_2_reduce_bn_eps, inception_a3_3x3_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_reduce_bn_node));

    // inception_a3_3x3_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_2_reduce_relu Layer
    vx_size inception_a3_3x3_2_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a3_3x3_2_reduce_relu;
    inception_a3_3x3_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_relu);
    vx_enum inception_a3_3x3_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_2_reduce_relu_param_a = 0;
    vx_float32 inception_a3_3x3_2_reduce_relu_param_b = 0;
    vx_node inception_a3_3x3_2_reduce_relu_node;
    inception_a3_3x3_2_reduce_relu_node = vxActivationLayer(graph, inception_a3_3x3_2_reduce_scale, inception_a3_3x3_2_reduce_relu_mode, inception_a3_3x3_2_reduce_relu_param_a, inception_a3_3x3_2_reduce_relu_param_b, inception_a3_3x3_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_reduce_relu_node));

    // inception_a3_3x3_2 Layer
    vx_size inception_a3_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_2;
    inception_a3_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2);
    vx_size inception_a3_3x3_2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a3_3x3_2_W;
    inception_a3_3x3_2_W = vxCreateTensor(context,4, inception_a3_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_W, dataFolder + "/weights/inception_a3_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_2_params;
    inception_a3_3x3_2_params.padding_x = 1;
    inception_a3_3x3_2_params.padding_y = 1;
    inception_a3_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_2_params.dilation_x = 0;
    inception_a3_3x3_2_params.dilation_y = 0;
    vx_node inception_a3_3x3_2_node;
    inception_a3_3x3_2_node = vxConvolutionLayer(graph, inception_a3_3x3_2_reduce_relu, inception_a3_3x3_2_W, NULL, &inception_a3_3x3_2_params, sizeof(inception_a3_3x3_2_params ), inception_a3_3x3_2);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_node));

    // inception_a3_3x3_2_bn Layer
    vx_size inception_a3_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_2_scale;
    inception_a3_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_scale);
    vx_size inception_a3_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_2_bn_W;
    inception_a3_3x3_2_bn_W = vxCreateTensor(context,1, inception_a3_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_bn_W, dataFolder + "/weights/inception_a3_3x3_2_bn.f32"));
    vx_size inception_a3_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_2_bn_B;
    inception_a3_3x3_2_bn_B = vxCreateTensor(context,1, inception_a3_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_bn_B, dataFolder + "/bias/inception_a3_3x3_2_bn.f32"));
    vx_size inception_a3_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_2_scale_W;
    inception_a3_3x3_2_scale_W = vxCreateTensor(context,1, inception_a3_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_scale_W, dataFolder + "/weights/inception_a3_3x3_2_scale.f32"));
    vx_size inception_a3_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_2_scale_B;
    inception_a3_3x3_2_scale_B = vxCreateTensor(context,1, inception_a3_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_2_scale_B, dataFolder + "/bias/inception_a3_3x3_2_scale.f32"));
    vx_node inception_a3_3x3_2_bn_node;
    inception_a3_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_2, inception_a3_3x3_2_bn_W, inception_a3_3x3_2_bn_B, inception_a3_3x3_2_scale_W, inception_a3_3x3_2_scale_B, inception_a3_3x3_2_bn_eps, inception_a3_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_bn_node));

    // inception_a3_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_2_relu Layer
    vx_size inception_a3_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_2_relu;
    inception_a3_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_relu);
    vx_enum inception_a3_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_2_relu_param_a = 0;
    vx_float32 inception_a3_3x3_2_relu_param_b = 0;
    vx_node inception_a3_3x3_2_relu_node;
    inception_a3_3x3_2_relu_node = vxActivationLayer(graph, inception_a3_3x3_2_scale, inception_a3_3x3_2_relu_mode, inception_a3_3x3_2_relu_param_a, inception_a3_3x3_2_relu_param_b, inception_a3_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_2_relu_node));

    // inception_a3_3x3_3 Layer
    vx_size inception_a3_3x3_3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_3;
    inception_a3_3x3_3 = vxCreateVirtualTensor(graph,4, inception_a3_3x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3);
    vx_size inception_a3_3x3_3_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a3_3x3_3_W;
    inception_a3_3x3_3_W = vxCreateTensor(context,4, inception_a3_3x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_3_W, dataFolder + "/weights/inception_a3_3x3_3.f32"));
    vx_nn_convolution_params_t inception_a3_3x3_3_params;
    inception_a3_3x3_3_params.padding_x = 1;
    inception_a3_3x3_3_params.padding_y = 1;
    inception_a3_3x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_3x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_3x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_3x3_3_params.dilation_x = 0;
    inception_a3_3x3_3_params.dilation_y = 0;
    vx_node inception_a3_3x3_3_node;
    inception_a3_3x3_3_node = vxConvolutionLayer(graph, inception_a3_3x3_2_relu, inception_a3_3x3_3_W, NULL, &inception_a3_3x3_3_params, sizeof(inception_a3_3x3_3_params ), inception_a3_3x3_3);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_3_node));

    // inception_a3_3x3_3_bn Layer
    vx_size inception_a3_3x3_3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_3_scale;
    inception_a3_3x3_3_scale = vxCreateVirtualTensor(graph,4, inception_a3_3x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_scale);
    vx_size inception_a3_3x3_3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_3x3_3_bn_eps = 0.001;
    vx_tensor inception_a3_3x3_3_bn_W;
    inception_a3_3x3_3_bn_W = vxCreateTensor(context,1, inception_a3_3x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_3_bn_W, dataFolder + "/weights/inception_a3_3x3_3_bn.f32"));
    vx_size inception_a3_3x3_3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_3_bn_B;
    inception_a3_3x3_3_bn_B = vxCreateTensor(context,1, inception_a3_3x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_3_bn_B, dataFolder + "/bias/inception_a3_3x3_3_bn.f32"));
    vx_size inception_a3_3x3_3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_3_scale_W;
    inception_a3_3x3_3_scale_W = vxCreateTensor(context,1, inception_a3_3x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_3_scale_W, dataFolder + "/weights/inception_a3_3x3_3_scale.f32"));
    vx_size inception_a3_3x3_3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_3x3_3_scale_B;
    inception_a3_3x3_3_scale_B = vxCreateTensor(context,1, inception_a3_3x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_3x3_3_scale_B, dataFolder + "/bias/inception_a3_3x3_3_scale.f32"));
    vx_node inception_a3_3x3_3_bn_node;
    inception_a3_3x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_a3_3x3_3, inception_a3_3x3_3_bn_W, inception_a3_3x3_3_bn_B, inception_a3_3x3_3_scale_W, inception_a3_3x3_3_scale_B, inception_a3_3x3_3_bn_eps, inception_a3_3x3_3_scale);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_3_bn_node));

    // inception_a3_3x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_3x3_3_relu Layer
    vx_size inception_a3_3x3_3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_3x3_3_relu;
    inception_a3_3x3_3_relu = vxCreateVirtualTensor(graph,4, inception_a3_3x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_relu);
    vx_enum inception_a3_3x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_3x3_3_relu_param_a = 0;
    vx_float32 inception_a3_3x3_3_relu_param_b = 0;
    vx_node inception_a3_3x3_3_relu_node;
    inception_a3_3x3_3_relu_node = vxActivationLayer(graph, inception_a3_3x3_3_scale, inception_a3_3x3_3_relu_mode, inception_a3_3x3_3_relu_param_a, inception_a3_3x3_3_relu_param_b, inception_a3_3x3_3_relu);
    ERROR_CHECK_OBJECT(inception_a3_3x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_3x3_3_relu_node));

    // inception_a3_pool_ave Layer
    vx_size inception_a3_pool_ave_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a3_pool_ave;
    inception_a3_pool_ave = vxCreateVirtualTensor(graph,4, inception_a3_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_pool_ave);
    vx_enum inception_a3_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_a3_pool_ave_kernel_w = 3;
    vx_size inception_a3_pool_ave_kernel_h = 3;
    vx_size inception_a3_pool_ave_pad_w = 1;
    vx_size inception_a3_pool_ave_pad_h = 1;
    vx_enum inception_a3_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a3_pool_ave_node;
    inception_a3_pool_ave_node = vxPoolingLayer(graph, inception_a2_concat, inception_a3_pool_ave_type, inception_a3_pool_ave_kernel_w, inception_a3_pool_ave_kernel_h, inception_a3_pool_ave_pad_w, inception_a3_pool_ave_pad_h, inception_a3_pool_ave_roundPolicy, inception_a3_pool_ave );
    ERROR_CHECK_OBJECT(inception_a3_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_pool_ave_node));

    // inception_a3_1x1 Layer
    vx_size inception_a3_1x1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_1x1;
    inception_a3_1x1 = vxCreateVirtualTensor(graph,4, inception_a3_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1);
    vx_size inception_a3_1x1_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a3_1x1_W;
    inception_a3_1x1_W = vxCreateTensor(context,4, inception_a3_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_W, dataFolder + "/weights/inception_a3_1x1.f32"));
    vx_nn_convolution_params_t inception_a3_1x1_params;
    inception_a3_1x1_params.padding_x = 0;
    inception_a3_1x1_params.padding_y = 0;
    inception_a3_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a3_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a3_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a3_1x1_params.dilation_x = 0;
    inception_a3_1x1_params.dilation_y = 0;
    vx_node inception_a3_1x1_node;
    inception_a3_1x1_node = vxConvolutionLayer(graph, inception_a3_pool_ave, inception_a3_1x1_W, NULL, &inception_a3_1x1_params, sizeof(inception_a3_1x1_params ), inception_a3_1x1);
    ERROR_CHECK_OBJECT(inception_a3_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_node));

    // inception_a3_1x1_bn Layer
    vx_size inception_a3_1x1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_1x1_scale;
    inception_a3_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a3_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_scale);
    vx_size inception_a3_1x1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a3_1x1_bn_eps = 0.001;
    vx_tensor inception_a3_1x1_bn_W;
    inception_a3_1x1_bn_W = vxCreateTensor(context,1, inception_a3_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_bn_W, dataFolder + "/weights/inception_a3_1x1_bn.f32"));
    vx_size inception_a3_1x1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a3_1x1_bn_B;
    inception_a3_1x1_bn_B = vxCreateTensor(context,1, inception_a3_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_bn_B, dataFolder + "/bias/inception_a3_1x1_bn.f32"));
    vx_size inception_a3_1x1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a3_1x1_scale_W;
    inception_a3_1x1_scale_W = vxCreateTensor(context,1, inception_a3_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_scale_W, dataFolder + "/weights/inception_a3_1x1_scale.f32"));
    vx_size inception_a3_1x1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a3_1x1_scale_B;
    inception_a3_1x1_scale_B = vxCreateTensor(context,1, inception_a3_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a3_1x1_scale_B, dataFolder + "/bias/inception_a3_1x1_scale.f32"));
    vx_node inception_a3_1x1_bn_node;
    inception_a3_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a3_1x1, inception_a3_1x1_bn_W, inception_a3_1x1_bn_B, inception_a3_1x1_scale_W, inception_a3_1x1_scale_B, inception_a3_1x1_bn_eps, inception_a3_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a3_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_bn_node));

    // inception_a3_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a3_1x1_relu Layer
    vx_size inception_a3_1x1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a3_1x1_relu;
    inception_a3_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a3_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_1x1_relu);
    vx_enum inception_a3_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a3_1x1_relu_param_a = 0;
    vx_float32 inception_a3_1x1_relu_param_b = 0;
    vx_node inception_a3_1x1_relu_node;
    inception_a3_1x1_relu_node = vxActivationLayer(graph, inception_a3_1x1_scale, inception_a3_1x1_relu_mode, inception_a3_1x1_relu_param_a, inception_a3_1x1_relu_param_b, inception_a3_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a3_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_1x1_relu_node));

    // inception_a3_concat Layer
    vx_size inception_a3_concat_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a3_concat;
    inception_a3_concat = vxCreateVirtualTensor(graph,4, inception_a3_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a3_concat);
    vx_node inception_a3_concat_node;
    inception_a3_concat_node = vxConcatLayer(graph, inception_a3_concat, inception_a3_1x1_2_relu, inception_a3_3x3_relu, inception_a3_3x3_3_relu, inception_a3_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a3_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a3_concat_node));

    // inception_a4_1x1_2 Layer
    vx_size inception_a4_1x1_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_1x1_2;
    inception_a4_1x1_2 = vxCreateVirtualTensor(graph,4, inception_a4_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2);
    vx_size inception_a4_1x1_2_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a4_1x1_2_W;
    inception_a4_1x1_2_W = vxCreateTensor(context,4, inception_a4_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_2_W, dataFolder + "/weights/inception_a4_1x1_2.f32"));
    vx_nn_convolution_params_t inception_a4_1x1_2_params;
    inception_a4_1x1_2_params.padding_x = 0;
    inception_a4_1x1_2_params.padding_y = 0;
    inception_a4_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_1x1_2_params.dilation_x = 0;
    inception_a4_1x1_2_params.dilation_y = 0;
    vx_node inception_a4_1x1_2_node;
    inception_a4_1x1_2_node = vxConvolutionLayer(graph, inception_a3_concat, inception_a4_1x1_2_W, NULL, &inception_a4_1x1_2_params, sizeof(inception_a4_1x1_2_params ), inception_a4_1x1_2);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_1x1_2_node));

    // inception_a4_1x1_2_bn Layer
    vx_size inception_a4_1x1_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_1x1_2_scale;
    inception_a4_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_a4_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_scale);
    vx_size inception_a4_1x1_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a4_1x1_2_bn_eps = 0.001;
    vx_tensor inception_a4_1x1_2_bn_W;
    inception_a4_1x1_2_bn_W = vxCreateTensor(context,1, inception_a4_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_2_bn_W, dataFolder + "/weights/inception_a4_1x1_2_bn.f32"));
    vx_size inception_a4_1x1_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a4_1x1_2_bn_B;
    inception_a4_1x1_2_bn_B = vxCreateTensor(context,1, inception_a4_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_2_bn_B, dataFolder + "/bias/inception_a4_1x1_2_bn.f32"));
    vx_size inception_a4_1x1_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a4_1x1_2_scale_W;
    inception_a4_1x1_2_scale_W = vxCreateTensor(context,1, inception_a4_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_2_scale_W, dataFolder + "/weights/inception_a4_1x1_2_scale.f32"));
    vx_size inception_a4_1x1_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a4_1x1_2_scale_B;
    inception_a4_1x1_2_scale_B = vxCreateTensor(context,1, inception_a4_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_2_scale_B, dataFolder + "/bias/inception_a4_1x1_2_scale.f32"));
    vx_node inception_a4_1x1_2_bn_node;
    inception_a4_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_a4_1x1_2, inception_a4_1x1_2_bn_W, inception_a4_1x1_2_bn_B, inception_a4_1x1_2_scale_W, inception_a4_1x1_2_scale_B, inception_a4_1x1_2_bn_eps, inception_a4_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_1x1_2_bn_node));

    // inception_a4_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_1x1_2_relu Layer
    vx_size inception_a4_1x1_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_1x1_2_relu;
    inception_a4_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_a4_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_relu);
    vx_enum inception_a4_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_1x1_2_relu_param_a = 0;
    vx_float32 inception_a4_1x1_2_relu_param_b = 0;
    vx_node inception_a4_1x1_2_relu_node;
    inception_a4_1x1_2_relu_node = vxActivationLayer(graph, inception_a4_1x1_2_scale, inception_a4_1x1_2_relu_mode, inception_a4_1x1_2_relu_param_a, inception_a4_1x1_2_relu_param_b, inception_a4_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_a4_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_1x1_2_relu_node));

    // inception_a4_3x3_reduce Layer
    vx_size inception_a4_3x3_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a4_3x3_reduce;
    inception_a4_3x3_reduce = vxCreateVirtualTensor(graph,4, inception_a4_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce);
    vx_size inception_a4_3x3_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a4_3x3_reduce_W;
    inception_a4_3x3_reduce_W = vxCreateTensor(context,4, inception_a4_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_reduce_W, dataFolder + "/weights/inception_a4_3x3_reduce.f32"));
    vx_nn_convolution_params_t inception_a4_3x3_reduce_params;
    inception_a4_3x3_reduce_params.padding_x = 0;
    inception_a4_3x3_reduce_params.padding_y = 0;
    inception_a4_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_3x3_reduce_params.dilation_x = 0;
    inception_a4_3x3_reduce_params.dilation_y = 0;
    vx_node inception_a4_3x3_reduce_node;
    inception_a4_3x3_reduce_node = vxConvolutionLayer(graph, inception_a3_concat, inception_a4_3x3_reduce_W, NULL, &inception_a4_3x3_reduce_params, sizeof(inception_a4_3x3_reduce_params ), inception_a4_3x3_reduce);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_reduce_node));

    // inception_a4_3x3_reduce_bn Layer
    vx_size inception_a4_3x3_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a4_3x3_reduce_scale;
    inception_a4_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a4_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_scale);
    vx_size inception_a4_3x3_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a4_3x3_reduce_bn_eps = 0.001;
    vx_tensor inception_a4_3x3_reduce_bn_W;
    inception_a4_3x3_reduce_bn_W = vxCreateTensor(context,1, inception_a4_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_reduce_bn_W, dataFolder + "/weights/inception_a4_3x3_reduce_bn.f32"));
    vx_size inception_a4_3x3_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a4_3x3_reduce_bn_B;
    inception_a4_3x3_reduce_bn_B = vxCreateTensor(context,1, inception_a4_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_reduce_bn_B, dataFolder + "/bias/inception_a4_3x3_reduce_bn.f32"));
    vx_size inception_a4_3x3_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a4_3x3_reduce_scale_W;
    inception_a4_3x3_reduce_scale_W = vxCreateTensor(context,1, inception_a4_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_reduce_scale_W, dataFolder + "/weights/inception_a4_3x3_reduce_scale.f32"));
    vx_size inception_a4_3x3_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a4_3x3_reduce_scale_B;
    inception_a4_3x3_reduce_scale_B = vxCreateTensor(context,1, inception_a4_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_reduce_scale_B, dataFolder + "/bias/inception_a4_3x3_reduce_scale.f32"));
    vx_node inception_a4_3x3_reduce_bn_node;
    inception_a4_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a4_3x3_reduce, inception_a4_3x3_reduce_bn_W, inception_a4_3x3_reduce_bn_B, inception_a4_3x3_reduce_scale_W, inception_a4_3x3_reduce_scale_B, inception_a4_3x3_reduce_bn_eps, inception_a4_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_reduce_bn_node));

    // inception_a4_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_3x3_reduce_relu Layer
    vx_size inception_a4_3x3_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a4_3x3_reduce_relu;
    inception_a4_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a4_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_relu);
    vx_enum inception_a4_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_3x3_reduce_relu_param_a = 0;
    vx_float32 inception_a4_3x3_reduce_relu_param_b = 0;
    vx_node inception_a4_3x3_reduce_relu_node;
    inception_a4_3x3_reduce_relu_node = vxActivationLayer(graph, inception_a4_3x3_reduce_scale, inception_a4_3x3_reduce_relu_mode, inception_a4_3x3_reduce_relu_param_a, inception_a4_3x3_reduce_relu_param_b, inception_a4_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a4_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_reduce_relu_node));

    // inception_a4_3x3 Layer
    vx_size inception_a4_3x3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3;
    inception_a4_3x3 = vxCreateVirtualTensor(graph,4, inception_a4_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3);
    vx_size inception_a4_3x3_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a4_3x3_W;
    inception_a4_3x3_W = vxCreateTensor(context,4, inception_a4_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_W, dataFolder + "/weights/inception_a4_3x3.f32"));
    vx_nn_convolution_params_t inception_a4_3x3_params;
    inception_a4_3x3_params.padding_x = 1;
    inception_a4_3x3_params.padding_y = 1;
    inception_a4_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_3x3_params.dilation_x = 0;
    inception_a4_3x3_params.dilation_y = 0;
    vx_node inception_a4_3x3_node;
    inception_a4_3x3_node = vxConvolutionLayer(graph, inception_a4_3x3_reduce_relu, inception_a4_3x3_W, NULL, &inception_a4_3x3_params, sizeof(inception_a4_3x3_params ), inception_a4_3x3);
    ERROR_CHECK_OBJECT(inception_a4_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_node));

    // inception_a4_3x3_bn Layer
    vx_size inception_a4_3x3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_scale;
    inception_a4_3x3_scale = vxCreateVirtualTensor(graph,4, inception_a4_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_scale);
    vx_size inception_a4_3x3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a4_3x3_bn_eps = 0.001;
    vx_tensor inception_a4_3x3_bn_W;
    inception_a4_3x3_bn_W = vxCreateTensor(context,1, inception_a4_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_bn_W, dataFolder + "/weights/inception_a4_3x3_bn.f32"));
    vx_size inception_a4_3x3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_bn_B;
    inception_a4_3x3_bn_B = vxCreateTensor(context,1, inception_a4_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_bn_B, dataFolder + "/bias/inception_a4_3x3_bn.f32"));
    vx_size inception_a4_3x3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_scale_W;
    inception_a4_3x3_scale_W = vxCreateTensor(context,1, inception_a4_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_scale_W, dataFolder + "/weights/inception_a4_3x3_scale.f32"));
    vx_size inception_a4_3x3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_scale_B;
    inception_a4_3x3_scale_B = vxCreateTensor(context,1, inception_a4_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_scale_B, dataFolder + "/bias/inception_a4_3x3_scale.f32"));
    vx_node inception_a4_3x3_bn_node;
    inception_a4_3x3_bn_node = vxBatchNormalizationLayer(graph, inception_a4_3x3, inception_a4_3x3_bn_W, inception_a4_3x3_bn_B, inception_a4_3x3_scale_W, inception_a4_3x3_scale_B, inception_a4_3x3_bn_eps, inception_a4_3x3_scale);
    ERROR_CHECK_OBJECT(inception_a4_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_bn_node));

    // inception_a4_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_3x3_relu Layer
    vx_size inception_a4_3x3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_relu;
    inception_a4_3x3_relu = vxCreateVirtualTensor(graph,4, inception_a4_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_relu);
    vx_enum inception_a4_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_3x3_relu_param_a = 0;
    vx_float32 inception_a4_3x3_relu_param_b = 0;
    vx_node inception_a4_3x3_relu_node;
    inception_a4_3x3_relu_node = vxActivationLayer(graph, inception_a4_3x3_scale, inception_a4_3x3_relu_mode, inception_a4_3x3_relu_param_a, inception_a4_3x3_relu_param_b, inception_a4_3x3_relu);
    ERROR_CHECK_OBJECT(inception_a4_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_relu_node));

    // inception_a4_3x3_2_reduce Layer
    vx_size inception_a4_3x3_2_reduce_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a4_3x3_2_reduce;
    inception_a4_3x3_2_reduce = vxCreateVirtualTensor(graph,4, inception_a4_3x3_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce);
    vx_size inception_a4_3x3_2_reduce_W_dims[4] = { 1, 1, 384, 64 };
    vx_tensor inception_a4_3x3_2_reduce_W;
    inception_a4_3x3_2_reduce_W = vxCreateTensor(context,4, inception_a4_3x3_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_reduce_W, dataFolder + "/weights/inception_a4_3x3_2_reduce.f32"));
    vx_nn_convolution_params_t inception_a4_3x3_2_reduce_params;
    inception_a4_3x3_2_reduce_params.padding_x = 0;
    inception_a4_3x3_2_reduce_params.padding_y = 0;
    inception_a4_3x3_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_3x3_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_3x3_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_3x3_2_reduce_params.dilation_x = 0;
    inception_a4_3x3_2_reduce_params.dilation_y = 0;
    vx_node inception_a4_3x3_2_reduce_node;
    inception_a4_3x3_2_reduce_node = vxConvolutionLayer(graph, inception_a3_concat, inception_a4_3x3_2_reduce_W, NULL, &inception_a4_3x3_2_reduce_params, sizeof(inception_a4_3x3_2_reduce_params ), inception_a4_3x3_2_reduce);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_2_reduce_node));

    // inception_a4_3x3_2_reduce_bn Layer
    vx_size inception_a4_3x3_2_reduce_scale_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a4_3x3_2_reduce_scale;
    inception_a4_3x3_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_a4_3x3_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_scale);
    vx_size inception_a4_3x3_2_reduce_bn_W_dims[1] = { 64 };
    vx_float32 inception_a4_3x3_2_reduce_bn_eps = 0.001;
    vx_tensor inception_a4_3x3_2_reduce_bn_W;
    inception_a4_3x3_2_reduce_bn_W = vxCreateTensor(context,1, inception_a4_3x3_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_reduce_bn_W, dataFolder + "/weights/inception_a4_3x3_2_reduce_bn.f32"));
    vx_size inception_a4_3x3_2_reduce_bn_B_dims[1] = { 64 };
    vx_tensor inception_a4_3x3_2_reduce_bn_B;
    inception_a4_3x3_2_reduce_bn_B = vxCreateTensor(context,1, inception_a4_3x3_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_reduce_bn_B, dataFolder + "/bias/inception_a4_3x3_2_reduce_bn.f32"));
    vx_size inception_a4_3x3_2_reduce_scale_W_dims[1] = { 64 };
    vx_tensor inception_a4_3x3_2_reduce_scale_W;
    inception_a4_3x3_2_reduce_scale_W = vxCreateTensor(context,1, inception_a4_3x3_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_reduce_scale_W, dataFolder + "/weights/inception_a4_3x3_2_reduce_scale.f32"));
    vx_size inception_a4_3x3_2_reduce_scale_B_dims[1] = { 64 };
    vx_tensor inception_a4_3x3_2_reduce_scale_B;
    inception_a4_3x3_2_reduce_scale_B = vxCreateTensor(context,1, inception_a4_3x3_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_reduce_scale_B, dataFolder + "/bias/inception_a4_3x3_2_reduce_scale.f32"));
    vx_node inception_a4_3x3_2_reduce_bn_node;
    inception_a4_3x3_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_a4_3x3_2_reduce, inception_a4_3x3_2_reduce_bn_W, inception_a4_3x3_2_reduce_bn_B, inception_a4_3x3_2_reduce_scale_W, inception_a4_3x3_2_reduce_scale_B, inception_a4_3x3_2_reduce_bn_eps, inception_a4_3x3_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_2_reduce_bn_node));

    // inception_a4_3x3_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_3x3_2_reduce_relu Layer
    vx_size inception_a4_3x3_2_reduce_relu_dims[4] = { 35, 35, 64, 1 };
    vx_tensor inception_a4_3x3_2_reduce_relu;
    inception_a4_3x3_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_a4_3x3_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_relu);
    vx_enum inception_a4_3x3_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_3x3_2_reduce_relu_param_a = 0;
    vx_float32 inception_a4_3x3_2_reduce_relu_param_b = 0;
    vx_node inception_a4_3x3_2_reduce_relu_node;
    inception_a4_3x3_2_reduce_relu_node = vxActivationLayer(graph, inception_a4_3x3_2_reduce_scale, inception_a4_3x3_2_reduce_relu_mode, inception_a4_3x3_2_reduce_relu_param_a, inception_a4_3x3_2_reduce_relu_param_b, inception_a4_3x3_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_2_reduce_relu_node));

    // inception_a4_3x3_2 Layer
    vx_size inception_a4_3x3_2_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_2;
    inception_a4_3x3_2 = vxCreateVirtualTensor(graph,4, inception_a4_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2);
    vx_size inception_a4_3x3_2_W_dims[4] = { 3, 3, 64, 96 };
    vx_tensor inception_a4_3x3_2_W;
    inception_a4_3x3_2_W = vxCreateTensor(context,4, inception_a4_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_W, dataFolder + "/weights/inception_a4_3x3_2.f32"));
    vx_nn_convolution_params_t inception_a4_3x3_2_params;
    inception_a4_3x3_2_params.padding_x = 1;
    inception_a4_3x3_2_params.padding_y = 1;
    inception_a4_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_3x3_2_params.dilation_x = 0;
    inception_a4_3x3_2_params.dilation_y = 0;
    vx_node inception_a4_3x3_2_node;
    inception_a4_3x3_2_node = vxConvolutionLayer(graph, inception_a4_3x3_2_reduce_relu, inception_a4_3x3_2_W, NULL, &inception_a4_3x3_2_params, sizeof(inception_a4_3x3_2_params ), inception_a4_3x3_2);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_2_node));

    // inception_a4_3x3_2_bn Layer
    vx_size inception_a4_3x3_2_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_2_scale;
    inception_a4_3x3_2_scale = vxCreateVirtualTensor(graph,4, inception_a4_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_scale);
    vx_size inception_a4_3x3_2_bn_W_dims[1] = { 96 };
    vx_float32 inception_a4_3x3_2_bn_eps = 0.001;
    vx_tensor inception_a4_3x3_2_bn_W;
    inception_a4_3x3_2_bn_W = vxCreateTensor(context,1, inception_a4_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_bn_W, dataFolder + "/weights/inception_a4_3x3_2_bn.f32"));
    vx_size inception_a4_3x3_2_bn_B_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_2_bn_B;
    inception_a4_3x3_2_bn_B = vxCreateTensor(context,1, inception_a4_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_bn_B, dataFolder + "/bias/inception_a4_3x3_2_bn.f32"));
    vx_size inception_a4_3x3_2_scale_W_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_2_scale_W;
    inception_a4_3x3_2_scale_W = vxCreateTensor(context,1, inception_a4_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_scale_W, dataFolder + "/weights/inception_a4_3x3_2_scale.f32"));
    vx_size inception_a4_3x3_2_scale_B_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_2_scale_B;
    inception_a4_3x3_2_scale_B = vxCreateTensor(context,1, inception_a4_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_2_scale_B, dataFolder + "/bias/inception_a4_3x3_2_scale.f32"));
    vx_node inception_a4_3x3_2_bn_node;
    inception_a4_3x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_a4_3x3_2, inception_a4_3x3_2_bn_W, inception_a4_3x3_2_bn_B, inception_a4_3x3_2_scale_W, inception_a4_3x3_2_scale_B, inception_a4_3x3_2_bn_eps, inception_a4_3x3_2_scale);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_2_bn_node));

    // inception_a4_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_3x3_2_relu Layer
    vx_size inception_a4_3x3_2_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_2_relu;
    inception_a4_3x3_2_relu = vxCreateVirtualTensor(graph,4, inception_a4_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_relu);
    vx_enum inception_a4_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_3x3_2_relu_param_a = 0;
    vx_float32 inception_a4_3x3_2_relu_param_b = 0;
    vx_node inception_a4_3x3_2_relu_node;
    inception_a4_3x3_2_relu_node = vxActivationLayer(graph, inception_a4_3x3_2_scale, inception_a4_3x3_2_relu_mode, inception_a4_3x3_2_relu_param_a, inception_a4_3x3_2_relu_param_b, inception_a4_3x3_2_relu);
    ERROR_CHECK_OBJECT(inception_a4_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_2_relu_node));

    // inception_a4_3x3_3 Layer
    vx_size inception_a4_3x3_3_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_3;
    inception_a4_3x3_3 = vxCreateVirtualTensor(graph,4, inception_a4_3x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3);
    vx_size inception_a4_3x3_3_W_dims[4] = { 3, 3, 96, 96 };
    vx_tensor inception_a4_3x3_3_W;
    inception_a4_3x3_3_W = vxCreateTensor(context,4, inception_a4_3x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_3_W, dataFolder + "/weights/inception_a4_3x3_3.f32"));
    vx_nn_convolution_params_t inception_a4_3x3_3_params;
    inception_a4_3x3_3_params.padding_x = 1;
    inception_a4_3x3_3_params.padding_y = 1;
    inception_a4_3x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_3x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_3x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_3x3_3_params.dilation_x = 0;
    inception_a4_3x3_3_params.dilation_y = 0;
    vx_node inception_a4_3x3_3_node;
    inception_a4_3x3_3_node = vxConvolutionLayer(graph, inception_a4_3x3_2_relu, inception_a4_3x3_3_W, NULL, &inception_a4_3x3_3_params, sizeof(inception_a4_3x3_3_params ), inception_a4_3x3_3);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_3_node));

    // inception_a4_3x3_3_bn Layer
    vx_size inception_a4_3x3_3_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_3_scale;
    inception_a4_3x3_3_scale = vxCreateVirtualTensor(graph,4, inception_a4_3x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_scale);
    vx_size inception_a4_3x3_3_bn_W_dims[1] = { 96 };
    vx_float32 inception_a4_3x3_3_bn_eps = 0.001;
    vx_tensor inception_a4_3x3_3_bn_W;
    inception_a4_3x3_3_bn_W = vxCreateTensor(context,1, inception_a4_3x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_3_bn_W, dataFolder + "/weights/inception_a4_3x3_3_bn.f32"));
    vx_size inception_a4_3x3_3_bn_B_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_3_bn_B;
    inception_a4_3x3_3_bn_B = vxCreateTensor(context,1, inception_a4_3x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_3_bn_B, dataFolder + "/bias/inception_a4_3x3_3_bn.f32"));
    vx_size inception_a4_3x3_3_scale_W_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_3_scale_W;
    inception_a4_3x3_3_scale_W = vxCreateTensor(context,1, inception_a4_3x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_3_scale_W, dataFolder + "/weights/inception_a4_3x3_3_scale.f32"));
    vx_size inception_a4_3x3_3_scale_B_dims[1] = { 96 };
    vx_tensor inception_a4_3x3_3_scale_B;
    inception_a4_3x3_3_scale_B = vxCreateTensor(context,1, inception_a4_3x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_3x3_3_scale_B, dataFolder + "/bias/inception_a4_3x3_3_scale.f32"));
    vx_node inception_a4_3x3_3_bn_node;
    inception_a4_3x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_a4_3x3_3, inception_a4_3x3_3_bn_W, inception_a4_3x3_3_bn_B, inception_a4_3x3_3_scale_W, inception_a4_3x3_3_scale_B, inception_a4_3x3_3_bn_eps, inception_a4_3x3_3_scale);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_3_bn_node));

    // inception_a4_3x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_3x3_3_relu Layer
    vx_size inception_a4_3x3_3_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_3x3_3_relu;
    inception_a4_3x3_3_relu = vxCreateVirtualTensor(graph,4, inception_a4_3x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_relu);
    vx_enum inception_a4_3x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_3x3_3_relu_param_a = 0;
    vx_float32 inception_a4_3x3_3_relu_param_b = 0;
    vx_node inception_a4_3x3_3_relu_node;
    inception_a4_3x3_3_relu_node = vxActivationLayer(graph, inception_a4_3x3_3_scale, inception_a4_3x3_3_relu_mode, inception_a4_3x3_3_relu_param_a, inception_a4_3x3_3_relu_param_b, inception_a4_3x3_3_relu);
    ERROR_CHECK_OBJECT(inception_a4_3x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_3x3_3_relu_node));

    // inception_a4_pool_ave Layer
    vx_size inception_a4_pool_ave_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a4_pool_ave;
    inception_a4_pool_ave = vxCreateVirtualTensor(graph,4, inception_a4_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_pool_ave);
    vx_enum inception_a4_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_a4_pool_ave_kernel_w = 3;
    vx_size inception_a4_pool_ave_kernel_h = 3;
    vx_size inception_a4_pool_ave_pad_w = 1;
    vx_size inception_a4_pool_ave_pad_h = 1;
    vx_enum inception_a4_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_a4_pool_ave_node;
    inception_a4_pool_ave_node = vxPoolingLayer(graph, inception_a3_concat, inception_a4_pool_ave_type, inception_a4_pool_ave_kernel_w, inception_a4_pool_ave_kernel_h, inception_a4_pool_ave_pad_w, inception_a4_pool_ave_pad_h, inception_a4_pool_ave_roundPolicy, inception_a4_pool_ave );
    ERROR_CHECK_OBJECT(inception_a4_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_pool_ave_node));

    // inception_a4_1x1 Layer
    vx_size inception_a4_1x1_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_1x1;
    inception_a4_1x1 = vxCreateVirtualTensor(graph,4, inception_a4_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_1x1);
    vx_size inception_a4_1x1_W_dims[4] = { 1, 1, 384, 96 };
    vx_tensor inception_a4_1x1_W;
    inception_a4_1x1_W = vxCreateTensor(context,4, inception_a4_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_W, dataFolder + "/weights/inception_a4_1x1.f32"));
    vx_nn_convolution_params_t inception_a4_1x1_params;
    inception_a4_1x1_params.padding_x = 0;
    inception_a4_1x1_params.padding_y = 0;
    inception_a4_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_a4_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_a4_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_a4_1x1_params.dilation_x = 0;
    inception_a4_1x1_params.dilation_y = 0;
    vx_node inception_a4_1x1_node;
    inception_a4_1x1_node = vxConvolutionLayer(graph, inception_a4_pool_ave, inception_a4_1x1_W, NULL, &inception_a4_1x1_params, sizeof(inception_a4_1x1_params ), inception_a4_1x1);
    ERROR_CHECK_OBJECT(inception_a4_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_1x1_node));

    // inception_a4_1x1_bn Layer
    vx_size inception_a4_1x1_scale_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_1x1_scale;
    inception_a4_1x1_scale = vxCreateVirtualTensor(graph,4, inception_a4_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_scale);
    vx_size inception_a4_1x1_bn_W_dims[1] = { 96 };
    vx_float32 inception_a4_1x1_bn_eps = 0.001;
    vx_tensor inception_a4_1x1_bn_W;
    inception_a4_1x1_bn_W = vxCreateTensor(context,1, inception_a4_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_bn_W, dataFolder + "/weights/inception_a4_1x1_bn.f32"));
    vx_size inception_a4_1x1_bn_B_dims[1] = { 96 };
    vx_tensor inception_a4_1x1_bn_B;
    inception_a4_1x1_bn_B = vxCreateTensor(context,1, inception_a4_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_bn_B, dataFolder + "/bias/inception_a4_1x1_bn.f32"));
    vx_size inception_a4_1x1_scale_W_dims[1] = { 96 };
    vx_tensor inception_a4_1x1_scale_W;
    inception_a4_1x1_scale_W = vxCreateTensor(context,1, inception_a4_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_scale_W, dataFolder + "/weights/inception_a4_1x1_scale.f32"));
    vx_size inception_a4_1x1_scale_B_dims[1] = { 96 };
    vx_tensor inception_a4_1x1_scale_B;
    inception_a4_1x1_scale_B = vxCreateTensor(context,1, inception_a4_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_a4_1x1_scale_B, dataFolder + "/bias/inception_a4_1x1_scale.f32"));
    vx_node inception_a4_1x1_bn_node;
    inception_a4_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_a4_1x1, inception_a4_1x1_bn_W, inception_a4_1x1_bn_B, inception_a4_1x1_scale_W, inception_a4_1x1_scale_B, inception_a4_1x1_bn_eps, inception_a4_1x1_scale);
    ERROR_CHECK_OBJECT(inception_a4_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_1x1_bn_node));

    // inception_a4_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_a4_1x1_relu Layer
    vx_size inception_a4_1x1_relu_dims[4] = { 35, 35, 96, 1 };
    vx_tensor inception_a4_1x1_relu;
    inception_a4_1x1_relu = vxCreateVirtualTensor(graph,4, inception_a4_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_1x1_relu);
    vx_enum inception_a4_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_a4_1x1_relu_param_a = 0;
    vx_float32 inception_a4_1x1_relu_param_b = 0;
    vx_node inception_a4_1x1_relu_node;
    inception_a4_1x1_relu_node = vxActivationLayer(graph, inception_a4_1x1_scale, inception_a4_1x1_relu_mode, inception_a4_1x1_relu_param_a, inception_a4_1x1_relu_param_b, inception_a4_1x1_relu);
    ERROR_CHECK_OBJECT(inception_a4_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_1x1_relu_node));

    // inception_a4_concat Layer
    vx_size inception_a4_concat_dims[4] = { 35, 35, 384, 1 };
    vx_tensor inception_a4_concat;
    inception_a4_concat = vxCreateVirtualTensor(graph,4, inception_a4_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_a4_concat);
    vx_node inception_a4_concat_node;
    inception_a4_concat_node = vxConcatLayer(graph, inception_a4_concat, inception_a4_1x1_2_relu, inception_a4_3x3_relu, inception_a4_3x3_3_relu, inception_a4_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_a4_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_a4_concat_node));

    // reduction_a_3x3 Layer
    vx_size reduction_a_3x3_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_3x3;
    reduction_a_3x3 = vxCreateVirtualTensor(graph,4, reduction_a_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3);
    vx_size reduction_a_3x3_W_dims[4] = { 3, 3, 384, 384 };
    vx_tensor reduction_a_3x3_W;
    reduction_a_3x3_W = vxCreateTensor(context,4, reduction_a_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_W, dataFolder + "/weights/reduction_a_3x3.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_params;
    reduction_a_3x3_params.padding_x = 0;
    reduction_a_3x3_params.padding_y = 0;
    reduction_a_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_params.dilation_x = 0;
    reduction_a_3x3_params.dilation_y = 0;
    vx_node reduction_a_3x3_node;
    reduction_a_3x3_node = vxConvolutionLayer(graph, inception_a4_concat, reduction_a_3x3_W, NULL, &reduction_a_3x3_params, sizeof(reduction_a_3x3_params ), reduction_a_3x3);
    ERROR_CHECK_OBJECT(reduction_a_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_node));

    // reduction_a_3x3_bn Layer
    vx_size reduction_a_3x3_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_3x3_scale;
    reduction_a_3x3_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_scale);
    vx_size reduction_a_3x3_bn_W_dims[1] = { 384 };
    vx_float32 reduction_a_3x3_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_bn_W;
    reduction_a_3x3_bn_W = vxCreateTensor(context,1, reduction_a_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_bn_W, dataFolder + "/weights/reduction_a_3x3_bn.f32"));
    vx_size reduction_a_3x3_bn_B_dims[1] = { 384 };
    vx_tensor reduction_a_3x3_bn_B;
    reduction_a_3x3_bn_B = vxCreateTensor(context,1, reduction_a_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_bn_B, dataFolder + "/bias/reduction_a_3x3_bn.f32"));
    vx_size reduction_a_3x3_scale_W_dims[1] = { 384 };
    vx_tensor reduction_a_3x3_scale_W;
    reduction_a_3x3_scale_W = vxCreateTensor(context,1, reduction_a_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_scale_W, dataFolder + "/weights/reduction_a_3x3_scale.f32"));
    vx_size reduction_a_3x3_scale_B_dims[1] = { 384 };
    vx_tensor reduction_a_3x3_scale_B;
    reduction_a_3x3_scale_B = vxCreateTensor(context,1, reduction_a_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_scale_B, dataFolder + "/bias/reduction_a_3x3_scale.f32"));
    vx_node reduction_a_3x3_bn_node;
    reduction_a_3x3_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3, reduction_a_3x3_bn_W, reduction_a_3x3_bn_B, reduction_a_3x3_scale_W, reduction_a_3x3_scale_B, reduction_a_3x3_bn_eps, reduction_a_3x3_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_bn_node));

    // reduction_a_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_relu Layer
    vx_size reduction_a_3x3_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_3x3_relu;
    reduction_a_3x3_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_relu);
    vx_enum reduction_a_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_relu_param_a = 0;
    vx_float32 reduction_a_3x3_relu_param_b = 0;
    vx_node reduction_a_3x3_relu_node;
    reduction_a_3x3_relu_node = vxActivationLayer(graph, reduction_a_3x3_scale, reduction_a_3x3_relu_mode, reduction_a_3x3_relu_param_a, reduction_a_3x3_relu_param_b, reduction_a_3x3_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_relu_node));

    // reduction_a_3x3_2_reduce Layer
    vx_size reduction_a_3x3_2_reduce_dims[4] = { 35, 35, 192, 1 };
    vx_tensor reduction_a_3x3_2_reduce;
    reduction_a_3x3_2_reduce = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce);
    vx_size reduction_a_3x3_2_reduce_W_dims[4] = { 1, 1, 384, 192 };
    vx_tensor reduction_a_3x3_2_reduce_W;
    reduction_a_3x3_2_reduce_W = vxCreateTensor(context,4, reduction_a_3x3_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_W, dataFolder + "/weights/reduction_a_3x3_2_reduce.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_2_reduce_params;
    reduction_a_3x3_2_reduce_params.padding_x = 0;
    reduction_a_3x3_2_reduce_params.padding_y = 0;
    reduction_a_3x3_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_2_reduce_params.dilation_x = 0;
    reduction_a_3x3_2_reduce_params.dilation_y = 0;
    vx_node reduction_a_3x3_2_reduce_node;
    reduction_a_3x3_2_reduce_node = vxConvolutionLayer(graph, inception_a4_concat, reduction_a_3x3_2_reduce_W, NULL, &reduction_a_3x3_2_reduce_params, sizeof(reduction_a_3x3_2_reduce_params ), reduction_a_3x3_2_reduce);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_reduce_node));

    // reduction_a_3x3_2_reduce_bn Layer
    vx_size reduction_a_3x3_2_reduce_scale_dims[4] = { 35, 35, 192, 1 };
    vx_tensor reduction_a_3x3_2_reduce_scale;
    reduction_a_3x3_2_reduce_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_scale);
    vx_size reduction_a_3x3_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 reduction_a_3x3_2_reduce_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_2_reduce_bn_W;
    reduction_a_3x3_2_reduce_bn_W = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_bn_W, dataFolder + "/weights/reduction_a_3x3_2_reduce_bn.f32"));
    vx_size reduction_a_3x3_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor reduction_a_3x3_2_reduce_bn_B;
    reduction_a_3x3_2_reduce_bn_B = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_bn_B, dataFolder + "/bias/reduction_a_3x3_2_reduce_bn.f32"));
    vx_size reduction_a_3x3_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor reduction_a_3x3_2_reduce_scale_W;
    reduction_a_3x3_2_reduce_scale_W = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_scale_W, dataFolder + "/weights/reduction_a_3x3_2_reduce_scale.f32"));
    vx_size reduction_a_3x3_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor reduction_a_3x3_2_reduce_scale_B;
    reduction_a_3x3_2_reduce_scale_B = vxCreateTensor(context,1, reduction_a_3x3_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_reduce_scale_B, dataFolder + "/bias/reduction_a_3x3_2_reduce_scale.f32"));
    vx_node reduction_a_3x3_2_reduce_bn_node;
    reduction_a_3x3_2_reduce_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3_2_reduce, reduction_a_3x3_2_reduce_bn_W, reduction_a_3x3_2_reduce_bn_B, reduction_a_3x3_2_reduce_scale_W, reduction_a_3x3_2_reduce_scale_B, reduction_a_3x3_2_reduce_bn_eps, reduction_a_3x3_2_reduce_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_reduce_bn_node));

    // reduction_a_3x3_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_2_reduce_relu Layer
    vx_size reduction_a_3x3_2_reduce_relu_dims[4] = { 35, 35, 192, 1 };
    vx_tensor reduction_a_3x3_2_reduce_relu;
    reduction_a_3x3_2_reduce_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_relu);
    vx_enum reduction_a_3x3_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_2_reduce_relu_param_a = 0;
    vx_float32 reduction_a_3x3_2_reduce_relu_param_b = 0;
    vx_node reduction_a_3x3_2_reduce_relu_node;
    reduction_a_3x3_2_reduce_relu_node = vxActivationLayer(graph, reduction_a_3x3_2_reduce_scale, reduction_a_3x3_2_reduce_relu_mode, reduction_a_3x3_2_reduce_relu_param_a, reduction_a_3x3_2_reduce_relu_param_b, reduction_a_3x3_2_reduce_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_reduce_relu_node));

    // reduction_a_3x3_2 Layer
    vx_size reduction_a_3x3_2_dims[4] = { 35, 35, 224, 1 };
    vx_tensor reduction_a_3x3_2;
    reduction_a_3x3_2 = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2);
    vx_size reduction_a_3x3_2_W_dims[4] = { 3, 3, 192, 224 };
    vx_tensor reduction_a_3x3_2_W;
    reduction_a_3x3_2_W = vxCreateTensor(context,4, reduction_a_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_W, dataFolder + "/weights/reduction_a_3x3_2.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_2_params;
    reduction_a_3x3_2_params.padding_x = 1;
    reduction_a_3x3_2_params.padding_y = 1;
    reduction_a_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_2_params.dilation_x = 0;
    reduction_a_3x3_2_params.dilation_y = 0;
    vx_node reduction_a_3x3_2_node;
    reduction_a_3x3_2_node = vxConvolutionLayer(graph, reduction_a_3x3_2_reduce_relu, reduction_a_3x3_2_W, NULL, &reduction_a_3x3_2_params, sizeof(reduction_a_3x3_2_params ), reduction_a_3x3_2);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_node));

    // reduction_a_3x3_2_bn Layer
    vx_size reduction_a_3x3_2_scale_dims[4] = { 35, 35, 224, 1 };
    vx_tensor reduction_a_3x3_2_scale;
    reduction_a_3x3_2_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_scale);
    vx_size reduction_a_3x3_2_bn_W_dims[1] = { 224 };
    vx_float32 reduction_a_3x3_2_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_2_bn_W;
    reduction_a_3x3_2_bn_W = vxCreateTensor(context,1, reduction_a_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_bn_W, dataFolder + "/weights/reduction_a_3x3_2_bn.f32"));
    vx_size reduction_a_3x3_2_bn_B_dims[1] = { 224 };
    vx_tensor reduction_a_3x3_2_bn_B;
    reduction_a_3x3_2_bn_B = vxCreateTensor(context,1, reduction_a_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_bn_B, dataFolder + "/bias/reduction_a_3x3_2_bn.f32"));
    vx_size reduction_a_3x3_2_scale_W_dims[1] = { 224 };
    vx_tensor reduction_a_3x3_2_scale_W;
    reduction_a_3x3_2_scale_W = vxCreateTensor(context,1, reduction_a_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_scale_W, dataFolder + "/weights/reduction_a_3x3_2_scale.f32"));
    vx_size reduction_a_3x3_2_scale_B_dims[1] = { 224 };
    vx_tensor reduction_a_3x3_2_scale_B;
    reduction_a_3x3_2_scale_B = vxCreateTensor(context,1, reduction_a_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_2_scale_B, dataFolder + "/bias/reduction_a_3x3_2_scale.f32"));
    vx_node reduction_a_3x3_2_bn_node;
    reduction_a_3x3_2_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3_2, reduction_a_3x3_2_bn_W, reduction_a_3x3_2_bn_B, reduction_a_3x3_2_scale_W, reduction_a_3x3_2_scale_B, reduction_a_3x3_2_bn_eps, reduction_a_3x3_2_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_bn_node));

    // reduction_a_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_2_relu Layer
    vx_size reduction_a_3x3_2_relu_dims[4] = { 35, 35, 224, 1 };
    vx_tensor reduction_a_3x3_2_relu;
    reduction_a_3x3_2_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_relu);
    vx_enum reduction_a_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_2_relu_param_a = 0;
    vx_float32 reduction_a_3x3_2_relu_param_b = 0;
    vx_node reduction_a_3x3_2_relu_node;
    reduction_a_3x3_2_relu_node = vxActivationLayer(graph, reduction_a_3x3_2_scale, reduction_a_3x3_2_relu_mode, reduction_a_3x3_2_relu_param_a, reduction_a_3x3_2_relu_param_b, reduction_a_3x3_2_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_2_relu_node));

    // reduction_a_3x3_3 Layer
    vx_size reduction_a_3x3_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_a_3x3_3;
    reduction_a_3x3_3 = vxCreateVirtualTensor(graph,4, reduction_a_3x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3);
    vx_size reduction_a_3x3_3_W_dims[4] = { 3, 3, 224, 256 };
    vx_tensor reduction_a_3x3_3_W;
    reduction_a_3x3_3_W = vxCreateTensor(context,4, reduction_a_3x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_W, dataFolder + "/weights/reduction_a_3x3_3.f32"));
    vx_nn_convolution_params_t reduction_a_3x3_3_params;
    reduction_a_3x3_3_params.padding_x = 0;
    reduction_a_3x3_3_params.padding_y = 0;
    reduction_a_3x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_a_3x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_a_3x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_a_3x3_3_params.dilation_x = 0;
    reduction_a_3x3_3_params.dilation_y = 0;
    vx_node reduction_a_3x3_3_node;
    reduction_a_3x3_3_node = vxConvolutionLayer(graph, reduction_a_3x3_2_relu, reduction_a_3x3_3_W, NULL, &reduction_a_3x3_3_params, sizeof(reduction_a_3x3_3_params ), reduction_a_3x3_3);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_3_node));

    // reduction_a_3x3_3_bn Layer
    vx_size reduction_a_3x3_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_a_3x3_3_scale;
    reduction_a_3x3_3_scale = vxCreateVirtualTensor(graph,4, reduction_a_3x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_scale);
    vx_size reduction_a_3x3_3_bn_W_dims[1] = { 256 };
    vx_float32 reduction_a_3x3_3_bn_eps = 0.001;
    vx_tensor reduction_a_3x3_3_bn_W;
    reduction_a_3x3_3_bn_W = vxCreateTensor(context,1, reduction_a_3x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_bn_W, dataFolder + "/weights/reduction_a_3x3_3_bn.f32"));
    vx_size reduction_a_3x3_3_bn_B_dims[1] = { 256 };
    vx_tensor reduction_a_3x3_3_bn_B;
    reduction_a_3x3_3_bn_B = vxCreateTensor(context,1, reduction_a_3x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_bn_B, dataFolder + "/bias/reduction_a_3x3_3_bn.f32"));
    vx_size reduction_a_3x3_3_scale_W_dims[1] = { 256 };
    vx_tensor reduction_a_3x3_3_scale_W;
    reduction_a_3x3_3_scale_W = vxCreateTensor(context,1, reduction_a_3x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_scale_W, dataFolder + "/weights/reduction_a_3x3_3_scale.f32"));
    vx_size reduction_a_3x3_3_scale_B_dims[1] = { 256 };
    vx_tensor reduction_a_3x3_3_scale_B;
    reduction_a_3x3_3_scale_B = vxCreateTensor(context,1, reduction_a_3x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_a_3x3_3_scale_B, dataFolder + "/bias/reduction_a_3x3_3_scale.f32"));
    vx_node reduction_a_3x3_3_bn_node;
    reduction_a_3x3_3_bn_node = vxBatchNormalizationLayer(graph, reduction_a_3x3_3, reduction_a_3x3_3_bn_W, reduction_a_3x3_3_bn_B, reduction_a_3x3_3_scale_W, reduction_a_3x3_3_scale_B, reduction_a_3x3_3_bn_eps, reduction_a_3x3_3_scale);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_3_bn_node));

    // reduction_a_3x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_a_3x3_3_relu Layer
    vx_size reduction_a_3x3_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_a_3x3_3_relu;
    reduction_a_3x3_3_relu = vxCreateVirtualTensor(graph,4, reduction_a_3x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_relu);
    vx_enum reduction_a_3x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_a_3x3_3_relu_param_a = 0;
    vx_float32 reduction_a_3x3_3_relu_param_b = 0;
    vx_node reduction_a_3x3_3_relu_node;
    reduction_a_3x3_3_relu_node = vxActivationLayer(graph, reduction_a_3x3_3_scale, reduction_a_3x3_3_relu_mode, reduction_a_3x3_3_relu_param_a, reduction_a_3x3_3_relu_param_b, reduction_a_3x3_3_relu);
    ERROR_CHECK_OBJECT(reduction_a_3x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_3x3_3_relu_node));

    // reduction_a_pool Layer
    vx_size reduction_a_pool_dims[4] = { 17, 17, 384, 1 };
    vx_tensor reduction_a_pool;
    reduction_a_pool = vxCreateVirtualTensor(graph,4, reduction_a_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_pool);
    vx_enum reduction_a_pool_type = VX_NN_POOLING_MAX;
    vx_size reduction_a_pool_kernel_w = 3;
    vx_size reduction_a_pool_kernel_h = 3;
    vx_size reduction_a_pool_pad_w = 0;
    vx_size reduction_a_pool_pad_h = 0;
    vx_enum reduction_a_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node reduction_a_pool_node;
    reduction_a_pool_node = vxPoolingLayer(graph, inception_a4_concat, reduction_a_pool_type, reduction_a_pool_kernel_w, reduction_a_pool_kernel_h, reduction_a_pool_pad_w, reduction_a_pool_pad_h, reduction_a_pool_roundPolicy, reduction_a_pool );
    ERROR_CHECK_OBJECT(reduction_a_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_pool_node));

    // reduction_a_concat Layer
    vx_size reduction_a_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor reduction_a_concat;
    reduction_a_concat = vxCreateVirtualTensor(graph,4, reduction_a_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_a_concat);
    vx_node reduction_a_concat_node;
    reduction_a_concat_node = vxConcatLayer(graph, reduction_a_concat, reduction_a_3x3_relu, reduction_a_3x3_3_relu, reduction_a_pool, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(reduction_a_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_a_concat_node));

    // inception_b1_1x1_2 Layer
    vx_size inception_b1_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b1_1x1_2;
    inception_b1_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b1_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2);
    vx_size inception_b1_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b1_1x1_2_W;
    inception_b1_1x1_2_W = vxCreateTensor(context,4, inception_b1_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_W, dataFolder + "/weights/inception_b1_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b1_1x1_2_params;
    inception_b1_1x1_2_params.padding_x = 0;
    inception_b1_1x1_2_params.padding_y = 0;
    inception_b1_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x1_2_params.dilation_x = 0;
    inception_b1_1x1_2_params.dilation_y = 0;
    vx_node inception_b1_1x1_2_node;
    inception_b1_1x1_2_node = vxConvolutionLayer(graph, reduction_a_concat, inception_b1_1x1_2_W, NULL, &inception_b1_1x1_2_params, sizeof(inception_b1_1x1_2_params ), inception_b1_1x1_2);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_2_node));

    // inception_b1_1x1_2_bn Layer
    vx_size inception_b1_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b1_1x1_2_scale;
    inception_b1_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_scale);
    vx_size inception_b1_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b1_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b1_1x1_2_bn_W;
    inception_b1_1x1_2_bn_W = vxCreateTensor(context,1, inception_b1_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_bn_W, dataFolder + "/weights/inception_b1_1x1_2_bn.f32"));
    vx_size inception_b1_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b1_1x1_2_bn_B;
    inception_b1_1x1_2_bn_B = vxCreateTensor(context,1, inception_b1_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_bn_B, dataFolder + "/bias/inception_b1_1x1_2_bn.f32"));
    vx_size inception_b1_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b1_1x1_2_scale_W;
    inception_b1_1x1_2_scale_W = vxCreateTensor(context,1, inception_b1_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_scale_W, dataFolder + "/weights/inception_b1_1x1_2_scale.f32"));
    vx_size inception_b1_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b1_1x1_2_scale_B;
    inception_b1_1x1_2_scale_B = vxCreateTensor(context,1, inception_b1_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_2_scale_B, dataFolder + "/bias/inception_b1_1x1_2_scale.f32"));
    vx_node inception_b1_1x1_2_bn_node;
    inception_b1_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x1_2, inception_b1_1x1_2_bn_W, inception_b1_1x1_2_bn_B, inception_b1_1x1_2_scale_W, inception_b1_1x1_2_scale_B, inception_b1_1x1_2_bn_eps, inception_b1_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_2_bn_node));

    // inception_b1_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x1_2_relu Layer
    vx_size inception_b1_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b1_1x1_2_relu;
    inception_b1_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_relu);
    vx_enum inception_b1_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x1_2_relu_param_a = 0;
    vx_float32 inception_b1_1x1_2_relu_param_b = 0;
    vx_node inception_b1_1x1_2_relu_node;
    inception_b1_1x1_2_relu_node = vxActivationLayer(graph, inception_b1_1x1_2_scale, inception_b1_1x1_2_relu_mode, inception_b1_1x1_2_relu_param_a, inception_b1_1x1_2_relu_param_b, inception_b1_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_2_relu_node));

    // inception_b1_1x7_reduce Layer
    vx_size inception_b1_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x7_reduce;
    inception_b1_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b1_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce);
    vx_size inception_b1_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b1_1x7_reduce_W;
    inception_b1_1x7_reduce_W = vxCreateTensor(context,4, inception_b1_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_W, dataFolder + "/weights/inception_b1_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_reduce_params;
    inception_b1_1x7_reduce_params.padding_x = 0;
    inception_b1_1x7_reduce_params.padding_y = 0;
    inception_b1_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_reduce_params.dilation_x = 0;
    inception_b1_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b1_1x7_reduce_node;
    inception_b1_1x7_reduce_node = vxConvolutionLayer(graph, reduction_a_concat, inception_b1_1x7_reduce_W, NULL, &inception_b1_1x7_reduce_params, sizeof(inception_b1_1x7_reduce_params ), inception_b1_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_reduce_node));

    // inception_b1_1x7_reduce_bn Layer
    vx_size inception_b1_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x7_reduce_scale;
    inception_b1_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_scale);
    vx_size inception_b1_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_reduce_bn_W;
    inception_b1_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b1_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_bn_W, dataFolder + "/weights/inception_b1_1x7_reduce_bn.f32"));
    vx_size inception_b1_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x7_reduce_bn_B;
    inception_b1_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b1_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_bn_B, dataFolder + "/bias/inception_b1_1x7_reduce_bn.f32"));
    vx_size inception_b1_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_1x7_reduce_scale_W;
    inception_b1_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b1_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_scale_W, dataFolder + "/weights/inception_b1_1x7_reduce_scale.f32"));
    vx_size inception_b1_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_1x7_reduce_scale_B;
    inception_b1_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b1_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_reduce_scale_B, dataFolder + "/bias/inception_b1_1x7_reduce_scale.f32"));
    vx_node inception_b1_1x7_reduce_bn_node;
    inception_b1_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7_reduce, inception_b1_1x7_reduce_bn_W, inception_b1_1x7_reduce_bn_B, inception_b1_1x7_reduce_scale_W, inception_b1_1x7_reduce_scale_B, inception_b1_1x7_reduce_bn_eps, inception_b1_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_reduce_bn_node));

    // inception_b1_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_reduce_relu Layer
    vx_size inception_b1_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_1x7_reduce_relu;
    inception_b1_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_relu);
    vx_enum inception_b1_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b1_1x7_reduce_relu_param_b = 0;
    vx_node inception_b1_1x7_reduce_relu_node;
    inception_b1_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b1_1x7_reduce_scale, inception_b1_1x7_reduce_relu_mode, inception_b1_1x7_reduce_relu_param_a, inception_b1_1x7_reduce_relu_param_b, inception_b1_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_reduce_relu_node));

    // inception_b1_1x7 Layer
    vx_size inception_b1_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_1x7;
    inception_b1_1x7 = vxCreateVirtualTensor(graph,4, inception_b1_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7);
    vx_size inception_b1_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b1_1x7_W;
    inception_b1_1x7_W = vxCreateTensor(context,4, inception_b1_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_W, dataFolder + "/weights/inception_b1_1x7.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_params;
    inception_b1_1x7_params.padding_x = 3;
    inception_b1_1x7_params.padding_y = 0;
    inception_b1_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_params.dilation_x = 0;
    inception_b1_1x7_params.dilation_y = 0;
    vx_node inception_b1_1x7_node;
    inception_b1_1x7_node = vxConvolutionLayer(graph, inception_b1_1x7_reduce_relu, inception_b1_1x7_W, NULL, &inception_b1_1x7_params, sizeof(inception_b1_1x7_params ), inception_b1_1x7);
    ERROR_CHECK_OBJECT(inception_b1_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_node));

    // inception_b1_1x7_bn Layer
    vx_size inception_b1_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_1x7_scale;
    inception_b1_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_scale);
    vx_size inception_b1_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b1_1x7_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_bn_W;
    inception_b1_1x7_bn_W = vxCreateTensor(context,1, inception_b1_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_bn_W, dataFolder + "/weights/inception_b1_1x7_bn.f32"));
    vx_size inception_b1_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b1_1x7_bn_B;
    inception_b1_1x7_bn_B = vxCreateTensor(context,1, inception_b1_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_bn_B, dataFolder + "/bias/inception_b1_1x7_bn.f32"));
    vx_size inception_b1_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b1_1x7_scale_W;
    inception_b1_1x7_scale_W = vxCreateTensor(context,1, inception_b1_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_scale_W, dataFolder + "/weights/inception_b1_1x7_scale.f32"));
    vx_size inception_b1_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b1_1x7_scale_B;
    inception_b1_1x7_scale_B = vxCreateTensor(context,1, inception_b1_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_scale_B, dataFolder + "/bias/inception_b1_1x7_scale.f32"));
    vx_node inception_b1_1x7_bn_node;
    inception_b1_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7, inception_b1_1x7_bn_W, inception_b1_1x7_bn_B, inception_b1_1x7_scale_W, inception_b1_1x7_scale_B, inception_b1_1x7_bn_eps, inception_b1_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_bn_node));

    // inception_b1_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_relu Layer
    vx_size inception_b1_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_1x7_relu;
    inception_b1_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_relu);
    vx_enum inception_b1_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_relu_param_a = 0;
    vx_float32 inception_b1_1x7_relu_param_b = 0;
    vx_node inception_b1_1x7_relu_node;
    inception_b1_1x7_relu_node = vxActivationLayer(graph, inception_b1_1x7_scale, inception_b1_1x7_relu_mode, inception_b1_1x7_relu_param_a, inception_b1_1x7_relu_param_b, inception_b1_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_relu_node));

    // inception_b1_7x1 Layer
    vx_size inception_b1_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b1_7x1;
    inception_b1_7x1 = vxCreateVirtualTensor(graph,4, inception_b1_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1);
    vx_size inception_b1_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b1_7x1_W;
    inception_b1_7x1_W = vxCreateTensor(context,4, inception_b1_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_W, dataFolder + "/weights/inception_b1_7x1.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_params;
    inception_b1_7x1_params.padding_x = 0;
    inception_b1_7x1_params.padding_y = 3;
    inception_b1_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_params.dilation_x = 0;
    inception_b1_7x1_params.dilation_y = 0;
    vx_node inception_b1_7x1_node;
    inception_b1_7x1_node = vxConvolutionLayer(graph, inception_b1_1x7_relu, inception_b1_7x1_W, NULL, &inception_b1_7x1_params, sizeof(inception_b1_7x1_params ), inception_b1_7x1);
    ERROR_CHECK_OBJECT(inception_b1_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_node));

    // inception_b1_7x1_bn Layer
    vx_size inception_b1_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b1_7x1_scale;
    inception_b1_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_scale);
    vx_size inception_b1_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b1_7x1_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_bn_W;
    inception_b1_7x1_bn_W = vxCreateTensor(context,1, inception_b1_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_bn_W, dataFolder + "/weights/inception_b1_7x1_bn.f32"));
    vx_size inception_b1_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b1_7x1_bn_B;
    inception_b1_7x1_bn_B = vxCreateTensor(context,1, inception_b1_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_bn_B, dataFolder + "/bias/inception_b1_7x1_bn.f32"));
    vx_size inception_b1_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b1_7x1_scale_W;
    inception_b1_7x1_scale_W = vxCreateTensor(context,1, inception_b1_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_scale_W, dataFolder + "/weights/inception_b1_7x1_scale.f32"));
    vx_size inception_b1_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b1_7x1_scale_B;
    inception_b1_7x1_scale_B = vxCreateTensor(context,1, inception_b1_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_scale_B, dataFolder + "/bias/inception_b1_7x1_scale.f32"));
    vx_node inception_b1_7x1_bn_node;
    inception_b1_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1, inception_b1_7x1_bn_W, inception_b1_7x1_bn_B, inception_b1_7x1_scale_W, inception_b1_7x1_scale_B, inception_b1_7x1_bn_eps, inception_b1_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_bn_node));

    // inception_b1_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_relu Layer
    vx_size inception_b1_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b1_7x1_relu;
    inception_b1_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_relu);
    vx_enum inception_b1_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_relu_param_a = 0;
    vx_float32 inception_b1_7x1_relu_param_b = 0;
    vx_node inception_b1_7x1_relu_node;
    inception_b1_7x1_relu_node = vxActivationLayer(graph, inception_b1_7x1_scale, inception_b1_7x1_relu_mode, inception_b1_7x1_relu_param_a, inception_b1_7x1_relu_param_b, inception_b1_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_relu_node));

    // inception_b1_7x1_2_reduce Layer
    vx_size inception_b1_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_2_reduce;
    inception_b1_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce);
    vx_size inception_b1_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b1_7x1_2_reduce_W;
    inception_b1_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b1_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_reduce_W, dataFolder + "/weights/inception_b1_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_2_reduce_params;
    inception_b1_7x1_2_reduce_params.padding_x = 0;
    inception_b1_7x1_2_reduce_params.padding_y = 0;
    inception_b1_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_2_reduce_params.dilation_x = 0;
    inception_b1_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b1_7x1_2_reduce_node;
    inception_b1_7x1_2_reduce_node = vxConvolutionLayer(graph, reduction_a_concat, inception_b1_7x1_2_reduce_W, NULL, &inception_b1_7x1_2_reduce_params, sizeof(inception_b1_7x1_2_reduce_params ), inception_b1_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_reduce_node));

    // inception_b1_7x1_2_reduce_bn Layer
    vx_size inception_b1_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_2_reduce_scale;
    inception_b1_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_scale);
    vx_size inception_b1_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_2_reduce_bn_W;
    inception_b1_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b1_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b1_7x1_2_reduce_bn.f32"));
    vx_size inception_b1_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_2_reduce_bn_B;
    inception_b1_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b1_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b1_7x1_2_reduce_bn.f32"));
    vx_size inception_b1_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_2_reduce_scale_W;
    inception_b1_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b1_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b1_7x1_2_reduce_scale.f32"));
    vx_size inception_b1_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_2_reduce_scale_B;
    inception_b1_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b1_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b1_7x1_2_reduce_scale.f32"));
    vx_node inception_b1_7x1_2_reduce_bn_node;
    inception_b1_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1_2_reduce, inception_b1_7x1_2_reduce_bn_W, inception_b1_7x1_2_reduce_bn_B, inception_b1_7x1_2_reduce_scale_W, inception_b1_7x1_2_reduce_scale_B, inception_b1_7x1_2_reduce_bn_eps, inception_b1_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_reduce_bn_node));

    // inception_b1_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_2_reduce_relu Layer
    vx_size inception_b1_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_2_reduce_relu;
    inception_b1_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_relu);
    vx_enum inception_b1_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b1_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b1_7x1_2_reduce_relu_node;
    inception_b1_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b1_7x1_2_reduce_scale, inception_b1_7x1_2_reduce_relu_mode, inception_b1_7x1_2_reduce_relu_param_a, inception_b1_7x1_2_reduce_relu_param_b, inception_b1_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_reduce_relu_node));

    // inception_b1_7x1_2 Layer
    vx_size inception_b1_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_2;
    inception_b1_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2);
    vx_size inception_b1_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b1_7x1_2_W;
    inception_b1_7x1_2_W = vxCreateTensor(context,4, inception_b1_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_W, dataFolder + "/weights/inception_b1_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_2_params;
    inception_b1_7x1_2_params.padding_x = 0;
    inception_b1_7x1_2_params.padding_y = 3;
    inception_b1_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_2_params.dilation_x = 0;
    inception_b1_7x1_2_params.dilation_y = 0;
    vx_node inception_b1_7x1_2_node;
    inception_b1_7x1_2_node = vxConvolutionLayer(graph, inception_b1_7x1_2_reduce_relu, inception_b1_7x1_2_W, NULL, &inception_b1_7x1_2_params, sizeof(inception_b1_7x1_2_params ), inception_b1_7x1_2);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_node));

    // inception_b1_7x1_2_bn Layer
    vx_size inception_b1_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_2_scale;
    inception_b1_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_scale);
    vx_size inception_b1_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b1_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_2_bn_W;
    inception_b1_7x1_2_bn_W = vxCreateTensor(context,1, inception_b1_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_bn_W, dataFolder + "/weights/inception_b1_7x1_2_bn.f32"));
    vx_size inception_b1_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_2_bn_B;
    inception_b1_7x1_2_bn_B = vxCreateTensor(context,1, inception_b1_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_bn_B, dataFolder + "/bias/inception_b1_7x1_2_bn.f32"));
    vx_size inception_b1_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_2_scale_W;
    inception_b1_7x1_2_scale_W = vxCreateTensor(context,1, inception_b1_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_scale_W, dataFolder + "/weights/inception_b1_7x1_2_scale.f32"));
    vx_size inception_b1_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b1_7x1_2_scale_B;
    inception_b1_7x1_2_scale_B = vxCreateTensor(context,1, inception_b1_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_2_scale_B, dataFolder + "/bias/inception_b1_7x1_2_scale.f32"));
    vx_node inception_b1_7x1_2_bn_node;
    inception_b1_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1_2, inception_b1_7x1_2_bn_W, inception_b1_7x1_2_bn_B, inception_b1_7x1_2_scale_W, inception_b1_7x1_2_scale_B, inception_b1_7x1_2_bn_eps, inception_b1_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_bn_node));

    // inception_b1_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_2_relu Layer
    vx_size inception_b1_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b1_7x1_2_relu;
    inception_b1_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_relu);
    vx_enum inception_b1_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_2_relu_param_a = 0;
    vx_float32 inception_b1_7x1_2_relu_param_b = 0;
    vx_node inception_b1_7x1_2_relu_node;
    inception_b1_7x1_2_relu_node = vxActivationLayer(graph, inception_b1_7x1_2_scale, inception_b1_7x1_2_relu_mode, inception_b1_7x1_2_relu_param_a, inception_b1_7x1_2_relu_param_b, inception_b1_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_2_relu_node));

    // inception_b1_1x7_2 Layer
    vx_size inception_b1_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_1x7_2;
    inception_b1_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b1_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2);
    vx_size inception_b1_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b1_1x7_2_W;
    inception_b1_1x7_2_W = vxCreateTensor(context,4, inception_b1_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_W, dataFolder + "/weights/inception_b1_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_2_params;
    inception_b1_1x7_2_params.padding_x = 3;
    inception_b1_1x7_2_params.padding_y = 0;
    inception_b1_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_2_params.dilation_x = 0;
    inception_b1_1x7_2_params.dilation_y = 0;
    vx_node inception_b1_1x7_2_node;
    inception_b1_1x7_2_node = vxConvolutionLayer(graph, inception_b1_7x1_2_relu, inception_b1_1x7_2_W, NULL, &inception_b1_1x7_2_params, sizeof(inception_b1_1x7_2_params ), inception_b1_1x7_2);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_2_node));

    // inception_b1_1x7_2_bn Layer
    vx_size inception_b1_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_1x7_2_scale;
    inception_b1_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_scale);
    vx_size inception_b1_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b1_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_2_bn_W;
    inception_b1_1x7_2_bn_W = vxCreateTensor(context,1, inception_b1_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_bn_W, dataFolder + "/weights/inception_b1_1x7_2_bn.f32"));
    vx_size inception_b1_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b1_1x7_2_bn_B;
    inception_b1_1x7_2_bn_B = vxCreateTensor(context,1, inception_b1_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_bn_B, dataFolder + "/bias/inception_b1_1x7_2_bn.f32"));
    vx_size inception_b1_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b1_1x7_2_scale_W;
    inception_b1_1x7_2_scale_W = vxCreateTensor(context,1, inception_b1_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_scale_W, dataFolder + "/weights/inception_b1_1x7_2_scale.f32"));
    vx_size inception_b1_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b1_1x7_2_scale_B;
    inception_b1_1x7_2_scale_B = vxCreateTensor(context,1, inception_b1_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_2_scale_B, dataFolder + "/bias/inception_b1_1x7_2_scale.f32"));
    vx_node inception_b1_1x7_2_bn_node;
    inception_b1_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7_2, inception_b1_1x7_2_bn_W, inception_b1_1x7_2_bn_B, inception_b1_1x7_2_scale_W, inception_b1_1x7_2_scale_B, inception_b1_1x7_2_bn_eps, inception_b1_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_2_bn_node));

    // inception_b1_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_2_relu Layer
    vx_size inception_b1_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_1x7_2_relu;
    inception_b1_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_relu);
    vx_enum inception_b1_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_2_relu_param_a = 0;
    vx_float32 inception_b1_1x7_2_relu_param_b = 0;
    vx_node inception_b1_1x7_2_relu_node;
    inception_b1_1x7_2_relu_node = vxActivationLayer(graph, inception_b1_1x7_2_scale, inception_b1_1x7_2_relu_mode, inception_b1_1x7_2_relu_param_a, inception_b1_1x7_2_relu_param_b, inception_b1_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_2_relu_node));

    // inception_b1_7x1_3 Layer
    vx_size inception_b1_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_7x1_3;
    inception_b1_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b1_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3);
    vx_size inception_b1_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b1_7x1_3_W;
    inception_b1_7x1_3_W = vxCreateTensor(context,4, inception_b1_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_W, dataFolder + "/weights/inception_b1_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b1_7x1_3_params;
    inception_b1_7x1_3_params.padding_x = 0;
    inception_b1_7x1_3_params.padding_y = 3;
    inception_b1_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_7x1_3_params.dilation_x = 0;
    inception_b1_7x1_3_params.dilation_y = 0;
    vx_node inception_b1_7x1_3_node;
    inception_b1_7x1_3_node = vxConvolutionLayer(graph, inception_b1_1x7_2_relu, inception_b1_7x1_3_W, NULL, &inception_b1_7x1_3_params, sizeof(inception_b1_7x1_3_params ), inception_b1_7x1_3);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_3_node));

    // inception_b1_7x1_3_bn Layer
    vx_size inception_b1_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_7x1_3_scale;
    inception_b1_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b1_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_scale);
    vx_size inception_b1_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b1_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b1_7x1_3_bn_W;
    inception_b1_7x1_3_bn_W = vxCreateTensor(context,1, inception_b1_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_bn_W, dataFolder + "/weights/inception_b1_7x1_3_bn.f32"));
    vx_size inception_b1_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b1_7x1_3_bn_B;
    inception_b1_7x1_3_bn_B = vxCreateTensor(context,1, inception_b1_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_bn_B, dataFolder + "/bias/inception_b1_7x1_3_bn.f32"));
    vx_size inception_b1_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b1_7x1_3_scale_W;
    inception_b1_7x1_3_scale_W = vxCreateTensor(context,1, inception_b1_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_scale_W, dataFolder + "/weights/inception_b1_7x1_3_scale.f32"));
    vx_size inception_b1_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b1_7x1_3_scale_B;
    inception_b1_7x1_3_scale_B = vxCreateTensor(context,1, inception_b1_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_7x1_3_scale_B, dataFolder + "/bias/inception_b1_7x1_3_scale.f32"));
    vx_node inception_b1_7x1_3_bn_node;
    inception_b1_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b1_7x1_3, inception_b1_7x1_3_bn_W, inception_b1_7x1_3_bn_B, inception_b1_7x1_3_scale_W, inception_b1_7x1_3_scale_B, inception_b1_7x1_3_bn_eps, inception_b1_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_3_bn_node));

    // inception_b1_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_7x1_3_relu Layer
    vx_size inception_b1_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b1_7x1_3_relu;
    inception_b1_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b1_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_relu);
    vx_enum inception_b1_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_7x1_3_relu_param_a = 0;
    vx_float32 inception_b1_7x1_3_relu_param_b = 0;
    vx_node inception_b1_7x1_3_relu_node;
    inception_b1_7x1_3_relu_node = vxActivationLayer(graph, inception_b1_7x1_3_scale, inception_b1_7x1_3_relu_mode, inception_b1_7x1_3_relu_param_a, inception_b1_7x1_3_relu_param_b, inception_b1_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b1_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_7x1_3_relu_node));

    // inception_b1_1x7_3 Layer
    vx_size inception_b1_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b1_1x7_3;
    inception_b1_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b1_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3);
    vx_size inception_b1_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b1_1x7_3_W;
    inception_b1_1x7_3_W = vxCreateTensor(context,4, inception_b1_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_W, dataFolder + "/weights/inception_b1_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b1_1x7_3_params;
    inception_b1_1x7_3_params.padding_x = 3;
    inception_b1_1x7_3_params.padding_y = 0;
    inception_b1_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x7_3_params.dilation_x = 0;
    inception_b1_1x7_3_params.dilation_y = 0;
    vx_node inception_b1_1x7_3_node;
    inception_b1_1x7_3_node = vxConvolutionLayer(graph, inception_b1_7x1_3_relu, inception_b1_1x7_3_W, NULL, &inception_b1_1x7_3_params, sizeof(inception_b1_1x7_3_params ), inception_b1_1x7_3);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_3_node));

    // inception_b1_1x7_3_bn Layer
    vx_size inception_b1_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b1_1x7_3_scale;
    inception_b1_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_scale);
    vx_size inception_b1_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b1_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b1_1x7_3_bn_W;
    inception_b1_1x7_3_bn_W = vxCreateTensor(context,1, inception_b1_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_bn_W, dataFolder + "/weights/inception_b1_1x7_3_bn.f32"));
    vx_size inception_b1_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b1_1x7_3_bn_B;
    inception_b1_1x7_3_bn_B = vxCreateTensor(context,1, inception_b1_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_bn_B, dataFolder + "/bias/inception_b1_1x7_3_bn.f32"));
    vx_size inception_b1_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b1_1x7_3_scale_W;
    inception_b1_1x7_3_scale_W = vxCreateTensor(context,1, inception_b1_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_scale_W, dataFolder + "/weights/inception_b1_1x7_3_scale.f32"));
    vx_size inception_b1_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b1_1x7_3_scale_B;
    inception_b1_1x7_3_scale_B = vxCreateTensor(context,1, inception_b1_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x7_3_scale_B, dataFolder + "/bias/inception_b1_1x7_3_scale.f32"));
    vx_node inception_b1_1x7_3_bn_node;
    inception_b1_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x7_3, inception_b1_1x7_3_bn_W, inception_b1_1x7_3_bn_B, inception_b1_1x7_3_scale_W, inception_b1_1x7_3_scale_B, inception_b1_1x7_3_bn_eps, inception_b1_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_3_bn_node));

    // inception_b1_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x7_3_relu Layer
    vx_size inception_b1_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b1_1x7_3_relu;
    inception_b1_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_relu);
    vx_enum inception_b1_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x7_3_relu_param_a = 0;
    vx_float32 inception_b1_1x7_3_relu_param_b = 0;
    vx_node inception_b1_1x7_3_relu_node;
    inception_b1_1x7_3_relu_node = vxActivationLayer(graph, inception_b1_1x7_3_scale, inception_b1_1x7_3_relu_mode, inception_b1_1x7_3_relu_param_a, inception_b1_1x7_3_relu_param_b, inception_b1_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x7_3_relu_node));

    // inception_b1_pool_ave Layer
    vx_size inception_b1_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b1_pool_ave;
    inception_b1_pool_ave = vxCreateVirtualTensor(graph,4, inception_b1_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_pool_ave);
    vx_enum inception_b1_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b1_pool_ave_kernel_w = 3;
    vx_size inception_b1_pool_ave_kernel_h = 3;
    vx_size inception_b1_pool_ave_pad_w = 1;
    vx_size inception_b1_pool_ave_pad_h = 1;
    vx_enum inception_b1_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b1_pool_ave_node;
    inception_b1_pool_ave_node = vxPoolingLayer(graph, reduction_a_concat, inception_b1_pool_ave_type, inception_b1_pool_ave_kernel_w, inception_b1_pool_ave_kernel_h, inception_b1_pool_ave_pad_w, inception_b1_pool_ave_pad_h, inception_b1_pool_ave_roundPolicy, inception_b1_pool_ave );
    ERROR_CHECK_OBJECT(inception_b1_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_pool_ave_node));

    // inception_b1_1x1 Layer
    vx_size inception_b1_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x1;
    inception_b1_1x1 = vxCreateVirtualTensor(graph,4, inception_b1_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1);
    vx_size inception_b1_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b1_1x1_W;
    inception_b1_1x1_W = vxCreateTensor(context,4, inception_b1_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_W, dataFolder + "/weights/inception_b1_1x1.f32"));
    vx_nn_convolution_params_t inception_b1_1x1_params;
    inception_b1_1x1_params.padding_x = 0;
    inception_b1_1x1_params.padding_y = 0;
    inception_b1_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b1_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b1_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b1_1x1_params.dilation_x = 0;
    inception_b1_1x1_params.dilation_y = 0;
    vx_node inception_b1_1x1_node;
    inception_b1_1x1_node = vxConvolutionLayer(graph, inception_b1_pool_ave, inception_b1_1x1_W, NULL, &inception_b1_1x1_params, sizeof(inception_b1_1x1_params ), inception_b1_1x1);
    ERROR_CHECK_OBJECT(inception_b1_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_node));

    // inception_b1_1x1_bn Layer
    vx_size inception_b1_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x1_scale;
    inception_b1_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b1_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_scale);
    vx_size inception_b1_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b1_1x1_bn_eps = 0.001;
    vx_tensor inception_b1_1x1_bn_W;
    inception_b1_1x1_bn_W = vxCreateTensor(context,1, inception_b1_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_bn_W, dataFolder + "/weights/inception_b1_1x1_bn.f32"));
    vx_size inception_b1_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x1_bn_B;
    inception_b1_1x1_bn_B = vxCreateTensor(context,1, inception_b1_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_bn_B, dataFolder + "/bias/inception_b1_1x1_bn.f32"));
    vx_size inception_b1_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b1_1x1_scale_W;
    inception_b1_1x1_scale_W = vxCreateTensor(context,1, inception_b1_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_scale_W, dataFolder + "/weights/inception_b1_1x1_scale.f32"));
    vx_size inception_b1_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b1_1x1_scale_B;
    inception_b1_1x1_scale_B = vxCreateTensor(context,1, inception_b1_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b1_1x1_scale_B, dataFolder + "/bias/inception_b1_1x1_scale.f32"));
    vx_node inception_b1_1x1_bn_node;
    inception_b1_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b1_1x1, inception_b1_1x1_bn_W, inception_b1_1x1_bn_B, inception_b1_1x1_scale_W, inception_b1_1x1_scale_B, inception_b1_1x1_bn_eps, inception_b1_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b1_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_bn_node));

    // inception_b1_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b1_1x1_relu Layer
    vx_size inception_b1_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b1_1x1_relu;
    inception_b1_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b1_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_1x1_relu);
    vx_enum inception_b1_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b1_1x1_relu_param_a = 0;
    vx_float32 inception_b1_1x1_relu_param_b = 0;
    vx_node inception_b1_1x1_relu_node;
    inception_b1_1x1_relu_node = vxActivationLayer(graph, inception_b1_1x1_scale, inception_b1_1x1_relu_mode, inception_b1_1x1_relu_param_a, inception_b1_1x1_relu_param_b, inception_b1_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b1_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_1x1_relu_node));

    // inception_b1_concat Layer
    vx_size inception_b1_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b1_concat;
    inception_b1_concat = vxCreateVirtualTensor(graph,4, inception_b1_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b1_concat);
    vx_node inception_b1_concat_node;
    inception_b1_concat_node = vxConcatLayer(graph, inception_b1_concat, inception_b1_1x1_2_relu, inception_b1_7x1_relu, inception_b1_1x7_3_relu, inception_b1_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b1_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b1_concat_node));

    // inception_b2_1x1_2 Layer
    vx_size inception_b2_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b2_1x1_2;
    inception_b2_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b2_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2);
    vx_size inception_b2_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b2_1x1_2_W;
    inception_b2_1x1_2_W = vxCreateTensor(context,4, inception_b2_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_W, dataFolder + "/weights/inception_b2_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b2_1x1_2_params;
    inception_b2_1x1_2_params.padding_x = 0;
    inception_b2_1x1_2_params.padding_y = 0;
    inception_b2_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x1_2_params.dilation_x = 0;
    inception_b2_1x1_2_params.dilation_y = 0;
    vx_node inception_b2_1x1_2_node;
    inception_b2_1x1_2_node = vxConvolutionLayer(graph, inception_b1_concat, inception_b2_1x1_2_W, NULL, &inception_b2_1x1_2_params, sizeof(inception_b2_1x1_2_params ), inception_b2_1x1_2);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_2_node));

    // inception_b2_1x1_2_bn Layer
    vx_size inception_b2_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b2_1x1_2_scale;
    inception_b2_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_scale);
    vx_size inception_b2_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b2_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b2_1x1_2_bn_W;
    inception_b2_1x1_2_bn_W = vxCreateTensor(context,1, inception_b2_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_bn_W, dataFolder + "/weights/inception_b2_1x1_2_bn.f32"));
    vx_size inception_b2_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b2_1x1_2_bn_B;
    inception_b2_1x1_2_bn_B = vxCreateTensor(context,1, inception_b2_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_bn_B, dataFolder + "/bias/inception_b2_1x1_2_bn.f32"));
    vx_size inception_b2_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b2_1x1_2_scale_W;
    inception_b2_1x1_2_scale_W = vxCreateTensor(context,1, inception_b2_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_scale_W, dataFolder + "/weights/inception_b2_1x1_2_scale.f32"));
    vx_size inception_b2_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b2_1x1_2_scale_B;
    inception_b2_1x1_2_scale_B = vxCreateTensor(context,1, inception_b2_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_2_scale_B, dataFolder + "/bias/inception_b2_1x1_2_scale.f32"));
    vx_node inception_b2_1x1_2_bn_node;
    inception_b2_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x1_2, inception_b2_1x1_2_bn_W, inception_b2_1x1_2_bn_B, inception_b2_1x1_2_scale_W, inception_b2_1x1_2_scale_B, inception_b2_1x1_2_bn_eps, inception_b2_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_2_bn_node));

    // inception_b2_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x1_2_relu Layer
    vx_size inception_b2_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b2_1x1_2_relu;
    inception_b2_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_relu);
    vx_enum inception_b2_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x1_2_relu_param_a = 0;
    vx_float32 inception_b2_1x1_2_relu_param_b = 0;
    vx_node inception_b2_1x1_2_relu_node;
    inception_b2_1x1_2_relu_node = vxActivationLayer(graph, inception_b2_1x1_2_scale, inception_b2_1x1_2_relu_mode, inception_b2_1x1_2_relu_param_a, inception_b2_1x1_2_relu_param_b, inception_b2_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_2_relu_node));

    // inception_b2_1x7_reduce Layer
    vx_size inception_b2_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x7_reduce;
    inception_b2_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b2_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce);
    vx_size inception_b2_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b2_1x7_reduce_W;
    inception_b2_1x7_reduce_W = vxCreateTensor(context,4, inception_b2_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_W, dataFolder + "/weights/inception_b2_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_reduce_params;
    inception_b2_1x7_reduce_params.padding_x = 0;
    inception_b2_1x7_reduce_params.padding_y = 0;
    inception_b2_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_reduce_params.dilation_x = 0;
    inception_b2_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b2_1x7_reduce_node;
    inception_b2_1x7_reduce_node = vxConvolutionLayer(graph, inception_b1_concat, inception_b2_1x7_reduce_W, NULL, &inception_b2_1x7_reduce_params, sizeof(inception_b2_1x7_reduce_params ), inception_b2_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_reduce_node));

    // inception_b2_1x7_reduce_bn Layer
    vx_size inception_b2_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x7_reduce_scale;
    inception_b2_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_scale);
    vx_size inception_b2_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_reduce_bn_W;
    inception_b2_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b2_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_bn_W, dataFolder + "/weights/inception_b2_1x7_reduce_bn.f32"));
    vx_size inception_b2_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x7_reduce_bn_B;
    inception_b2_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b2_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_bn_B, dataFolder + "/bias/inception_b2_1x7_reduce_bn.f32"));
    vx_size inception_b2_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_1x7_reduce_scale_W;
    inception_b2_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b2_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_scale_W, dataFolder + "/weights/inception_b2_1x7_reduce_scale.f32"));
    vx_size inception_b2_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_1x7_reduce_scale_B;
    inception_b2_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b2_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_reduce_scale_B, dataFolder + "/bias/inception_b2_1x7_reduce_scale.f32"));
    vx_node inception_b2_1x7_reduce_bn_node;
    inception_b2_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7_reduce, inception_b2_1x7_reduce_bn_W, inception_b2_1x7_reduce_bn_B, inception_b2_1x7_reduce_scale_W, inception_b2_1x7_reduce_scale_B, inception_b2_1x7_reduce_bn_eps, inception_b2_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_reduce_bn_node));

    // inception_b2_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_reduce_relu Layer
    vx_size inception_b2_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_1x7_reduce_relu;
    inception_b2_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_relu);
    vx_enum inception_b2_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b2_1x7_reduce_relu_param_b = 0;
    vx_node inception_b2_1x7_reduce_relu_node;
    inception_b2_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b2_1x7_reduce_scale, inception_b2_1x7_reduce_relu_mode, inception_b2_1x7_reduce_relu_param_a, inception_b2_1x7_reduce_relu_param_b, inception_b2_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_reduce_relu_node));

    // inception_b2_1x7 Layer
    vx_size inception_b2_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_1x7;
    inception_b2_1x7 = vxCreateVirtualTensor(graph,4, inception_b2_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7);
    vx_size inception_b2_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b2_1x7_W;
    inception_b2_1x7_W = vxCreateTensor(context,4, inception_b2_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_W, dataFolder + "/weights/inception_b2_1x7.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_params;
    inception_b2_1x7_params.padding_x = 3;
    inception_b2_1x7_params.padding_y = 0;
    inception_b2_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_params.dilation_x = 0;
    inception_b2_1x7_params.dilation_y = 0;
    vx_node inception_b2_1x7_node;
    inception_b2_1x7_node = vxConvolutionLayer(graph, inception_b2_1x7_reduce_relu, inception_b2_1x7_W, NULL, &inception_b2_1x7_params, sizeof(inception_b2_1x7_params ), inception_b2_1x7);
    ERROR_CHECK_OBJECT(inception_b2_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_node));

    // inception_b2_1x7_bn Layer
    vx_size inception_b2_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_1x7_scale;
    inception_b2_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_scale);
    vx_size inception_b2_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b2_1x7_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_bn_W;
    inception_b2_1x7_bn_W = vxCreateTensor(context,1, inception_b2_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_bn_W, dataFolder + "/weights/inception_b2_1x7_bn.f32"));
    vx_size inception_b2_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b2_1x7_bn_B;
    inception_b2_1x7_bn_B = vxCreateTensor(context,1, inception_b2_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_bn_B, dataFolder + "/bias/inception_b2_1x7_bn.f32"));
    vx_size inception_b2_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b2_1x7_scale_W;
    inception_b2_1x7_scale_W = vxCreateTensor(context,1, inception_b2_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_scale_W, dataFolder + "/weights/inception_b2_1x7_scale.f32"));
    vx_size inception_b2_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b2_1x7_scale_B;
    inception_b2_1x7_scale_B = vxCreateTensor(context,1, inception_b2_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_scale_B, dataFolder + "/bias/inception_b2_1x7_scale.f32"));
    vx_node inception_b2_1x7_bn_node;
    inception_b2_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7, inception_b2_1x7_bn_W, inception_b2_1x7_bn_B, inception_b2_1x7_scale_W, inception_b2_1x7_scale_B, inception_b2_1x7_bn_eps, inception_b2_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_bn_node));

    // inception_b2_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_relu Layer
    vx_size inception_b2_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_1x7_relu;
    inception_b2_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_relu);
    vx_enum inception_b2_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_relu_param_a = 0;
    vx_float32 inception_b2_1x7_relu_param_b = 0;
    vx_node inception_b2_1x7_relu_node;
    inception_b2_1x7_relu_node = vxActivationLayer(graph, inception_b2_1x7_scale, inception_b2_1x7_relu_mode, inception_b2_1x7_relu_param_a, inception_b2_1x7_relu_param_b, inception_b2_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_relu_node));

    // inception_b2_7x1 Layer
    vx_size inception_b2_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b2_7x1;
    inception_b2_7x1 = vxCreateVirtualTensor(graph,4, inception_b2_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1);
    vx_size inception_b2_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b2_7x1_W;
    inception_b2_7x1_W = vxCreateTensor(context,4, inception_b2_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_W, dataFolder + "/weights/inception_b2_7x1.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_params;
    inception_b2_7x1_params.padding_x = 0;
    inception_b2_7x1_params.padding_y = 3;
    inception_b2_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_params.dilation_x = 0;
    inception_b2_7x1_params.dilation_y = 0;
    vx_node inception_b2_7x1_node;
    inception_b2_7x1_node = vxConvolutionLayer(graph, inception_b2_1x7_relu, inception_b2_7x1_W, NULL, &inception_b2_7x1_params, sizeof(inception_b2_7x1_params ), inception_b2_7x1);
    ERROR_CHECK_OBJECT(inception_b2_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_node));

    // inception_b2_7x1_bn Layer
    vx_size inception_b2_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b2_7x1_scale;
    inception_b2_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_scale);
    vx_size inception_b2_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b2_7x1_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_bn_W;
    inception_b2_7x1_bn_W = vxCreateTensor(context,1, inception_b2_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_bn_W, dataFolder + "/weights/inception_b2_7x1_bn.f32"));
    vx_size inception_b2_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b2_7x1_bn_B;
    inception_b2_7x1_bn_B = vxCreateTensor(context,1, inception_b2_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_bn_B, dataFolder + "/bias/inception_b2_7x1_bn.f32"));
    vx_size inception_b2_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b2_7x1_scale_W;
    inception_b2_7x1_scale_W = vxCreateTensor(context,1, inception_b2_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_scale_W, dataFolder + "/weights/inception_b2_7x1_scale.f32"));
    vx_size inception_b2_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b2_7x1_scale_B;
    inception_b2_7x1_scale_B = vxCreateTensor(context,1, inception_b2_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_scale_B, dataFolder + "/bias/inception_b2_7x1_scale.f32"));
    vx_node inception_b2_7x1_bn_node;
    inception_b2_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1, inception_b2_7x1_bn_W, inception_b2_7x1_bn_B, inception_b2_7x1_scale_W, inception_b2_7x1_scale_B, inception_b2_7x1_bn_eps, inception_b2_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_bn_node));

    // inception_b2_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_relu Layer
    vx_size inception_b2_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b2_7x1_relu;
    inception_b2_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_relu);
    vx_enum inception_b2_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_relu_param_a = 0;
    vx_float32 inception_b2_7x1_relu_param_b = 0;
    vx_node inception_b2_7x1_relu_node;
    inception_b2_7x1_relu_node = vxActivationLayer(graph, inception_b2_7x1_scale, inception_b2_7x1_relu_mode, inception_b2_7x1_relu_param_a, inception_b2_7x1_relu_param_b, inception_b2_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_relu_node));

    // inception_b2_7x1_2_reduce Layer
    vx_size inception_b2_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_2_reduce;
    inception_b2_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce);
    vx_size inception_b2_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b2_7x1_2_reduce_W;
    inception_b2_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b2_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_reduce_W, dataFolder + "/weights/inception_b2_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_2_reduce_params;
    inception_b2_7x1_2_reduce_params.padding_x = 0;
    inception_b2_7x1_2_reduce_params.padding_y = 0;
    inception_b2_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_2_reduce_params.dilation_x = 0;
    inception_b2_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b2_7x1_2_reduce_node;
    inception_b2_7x1_2_reduce_node = vxConvolutionLayer(graph, inception_b1_concat, inception_b2_7x1_2_reduce_W, NULL, &inception_b2_7x1_2_reduce_params, sizeof(inception_b2_7x1_2_reduce_params ), inception_b2_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_reduce_node));

    // inception_b2_7x1_2_reduce_bn Layer
    vx_size inception_b2_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_2_reduce_scale;
    inception_b2_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_scale);
    vx_size inception_b2_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_2_reduce_bn_W;
    inception_b2_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b2_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b2_7x1_2_reduce_bn.f32"));
    vx_size inception_b2_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_2_reduce_bn_B;
    inception_b2_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b2_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b2_7x1_2_reduce_bn.f32"));
    vx_size inception_b2_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_2_reduce_scale_W;
    inception_b2_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b2_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b2_7x1_2_reduce_scale.f32"));
    vx_size inception_b2_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_2_reduce_scale_B;
    inception_b2_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b2_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b2_7x1_2_reduce_scale.f32"));
    vx_node inception_b2_7x1_2_reduce_bn_node;
    inception_b2_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1_2_reduce, inception_b2_7x1_2_reduce_bn_W, inception_b2_7x1_2_reduce_bn_B, inception_b2_7x1_2_reduce_scale_W, inception_b2_7x1_2_reduce_scale_B, inception_b2_7x1_2_reduce_bn_eps, inception_b2_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_reduce_bn_node));

    // inception_b2_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_2_reduce_relu Layer
    vx_size inception_b2_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_2_reduce_relu;
    inception_b2_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_relu);
    vx_enum inception_b2_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b2_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b2_7x1_2_reduce_relu_node;
    inception_b2_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b2_7x1_2_reduce_scale, inception_b2_7x1_2_reduce_relu_mode, inception_b2_7x1_2_reduce_relu_param_a, inception_b2_7x1_2_reduce_relu_param_b, inception_b2_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_reduce_relu_node));

    // inception_b2_7x1_2 Layer
    vx_size inception_b2_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_2;
    inception_b2_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2);
    vx_size inception_b2_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b2_7x1_2_W;
    inception_b2_7x1_2_W = vxCreateTensor(context,4, inception_b2_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_W, dataFolder + "/weights/inception_b2_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_2_params;
    inception_b2_7x1_2_params.padding_x = 0;
    inception_b2_7x1_2_params.padding_y = 3;
    inception_b2_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_2_params.dilation_x = 0;
    inception_b2_7x1_2_params.dilation_y = 0;
    vx_node inception_b2_7x1_2_node;
    inception_b2_7x1_2_node = vxConvolutionLayer(graph, inception_b2_7x1_2_reduce_relu, inception_b2_7x1_2_W, NULL, &inception_b2_7x1_2_params, sizeof(inception_b2_7x1_2_params ), inception_b2_7x1_2);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_node));

    // inception_b2_7x1_2_bn Layer
    vx_size inception_b2_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_2_scale;
    inception_b2_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_scale);
    vx_size inception_b2_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b2_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_2_bn_W;
    inception_b2_7x1_2_bn_W = vxCreateTensor(context,1, inception_b2_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_bn_W, dataFolder + "/weights/inception_b2_7x1_2_bn.f32"));
    vx_size inception_b2_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_2_bn_B;
    inception_b2_7x1_2_bn_B = vxCreateTensor(context,1, inception_b2_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_bn_B, dataFolder + "/bias/inception_b2_7x1_2_bn.f32"));
    vx_size inception_b2_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_2_scale_W;
    inception_b2_7x1_2_scale_W = vxCreateTensor(context,1, inception_b2_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_scale_W, dataFolder + "/weights/inception_b2_7x1_2_scale.f32"));
    vx_size inception_b2_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b2_7x1_2_scale_B;
    inception_b2_7x1_2_scale_B = vxCreateTensor(context,1, inception_b2_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_2_scale_B, dataFolder + "/bias/inception_b2_7x1_2_scale.f32"));
    vx_node inception_b2_7x1_2_bn_node;
    inception_b2_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1_2, inception_b2_7x1_2_bn_W, inception_b2_7x1_2_bn_B, inception_b2_7x1_2_scale_W, inception_b2_7x1_2_scale_B, inception_b2_7x1_2_bn_eps, inception_b2_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_bn_node));

    // inception_b2_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_2_relu Layer
    vx_size inception_b2_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b2_7x1_2_relu;
    inception_b2_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_relu);
    vx_enum inception_b2_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_2_relu_param_a = 0;
    vx_float32 inception_b2_7x1_2_relu_param_b = 0;
    vx_node inception_b2_7x1_2_relu_node;
    inception_b2_7x1_2_relu_node = vxActivationLayer(graph, inception_b2_7x1_2_scale, inception_b2_7x1_2_relu_mode, inception_b2_7x1_2_relu_param_a, inception_b2_7x1_2_relu_param_b, inception_b2_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_2_relu_node));

    // inception_b2_1x7_2 Layer
    vx_size inception_b2_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_1x7_2;
    inception_b2_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b2_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2);
    vx_size inception_b2_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b2_1x7_2_W;
    inception_b2_1x7_2_W = vxCreateTensor(context,4, inception_b2_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_W, dataFolder + "/weights/inception_b2_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_2_params;
    inception_b2_1x7_2_params.padding_x = 3;
    inception_b2_1x7_2_params.padding_y = 0;
    inception_b2_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_2_params.dilation_x = 0;
    inception_b2_1x7_2_params.dilation_y = 0;
    vx_node inception_b2_1x7_2_node;
    inception_b2_1x7_2_node = vxConvolutionLayer(graph, inception_b2_7x1_2_relu, inception_b2_1x7_2_W, NULL, &inception_b2_1x7_2_params, sizeof(inception_b2_1x7_2_params ), inception_b2_1x7_2);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_2_node));

    // inception_b2_1x7_2_bn Layer
    vx_size inception_b2_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_1x7_2_scale;
    inception_b2_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_scale);
    vx_size inception_b2_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b2_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_2_bn_W;
    inception_b2_1x7_2_bn_W = vxCreateTensor(context,1, inception_b2_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_bn_W, dataFolder + "/weights/inception_b2_1x7_2_bn.f32"));
    vx_size inception_b2_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b2_1x7_2_bn_B;
    inception_b2_1x7_2_bn_B = vxCreateTensor(context,1, inception_b2_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_bn_B, dataFolder + "/bias/inception_b2_1x7_2_bn.f32"));
    vx_size inception_b2_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b2_1x7_2_scale_W;
    inception_b2_1x7_2_scale_W = vxCreateTensor(context,1, inception_b2_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_scale_W, dataFolder + "/weights/inception_b2_1x7_2_scale.f32"));
    vx_size inception_b2_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b2_1x7_2_scale_B;
    inception_b2_1x7_2_scale_B = vxCreateTensor(context,1, inception_b2_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_2_scale_B, dataFolder + "/bias/inception_b2_1x7_2_scale.f32"));
    vx_node inception_b2_1x7_2_bn_node;
    inception_b2_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7_2, inception_b2_1x7_2_bn_W, inception_b2_1x7_2_bn_B, inception_b2_1x7_2_scale_W, inception_b2_1x7_2_scale_B, inception_b2_1x7_2_bn_eps, inception_b2_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_2_bn_node));

    // inception_b2_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_2_relu Layer
    vx_size inception_b2_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_1x7_2_relu;
    inception_b2_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_relu);
    vx_enum inception_b2_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_2_relu_param_a = 0;
    vx_float32 inception_b2_1x7_2_relu_param_b = 0;
    vx_node inception_b2_1x7_2_relu_node;
    inception_b2_1x7_2_relu_node = vxActivationLayer(graph, inception_b2_1x7_2_scale, inception_b2_1x7_2_relu_mode, inception_b2_1x7_2_relu_param_a, inception_b2_1x7_2_relu_param_b, inception_b2_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_2_relu_node));

    // inception_b2_7x1_3 Layer
    vx_size inception_b2_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_7x1_3;
    inception_b2_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b2_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3);
    vx_size inception_b2_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b2_7x1_3_W;
    inception_b2_7x1_3_W = vxCreateTensor(context,4, inception_b2_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_W, dataFolder + "/weights/inception_b2_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b2_7x1_3_params;
    inception_b2_7x1_3_params.padding_x = 0;
    inception_b2_7x1_3_params.padding_y = 3;
    inception_b2_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_7x1_3_params.dilation_x = 0;
    inception_b2_7x1_3_params.dilation_y = 0;
    vx_node inception_b2_7x1_3_node;
    inception_b2_7x1_3_node = vxConvolutionLayer(graph, inception_b2_1x7_2_relu, inception_b2_7x1_3_W, NULL, &inception_b2_7x1_3_params, sizeof(inception_b2_7x1_3_params ), inception_b2_7x1_3);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_3_node));

    // inception_b2_7x1_3_bn Layer
    vx_size inception_b2_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_7x1_3_scale;
    inception_b2_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b2_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_scale);
    vx_size inception_b2_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b2_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b2_7x1_3_bn_W;
    inception_b2_7x1_3_bn_W = vxCreateTensor(context,1, inception_b2_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_bn_W, dataFolder + "/weights/inception_b2_7x1_3_bn.f32"));
    vx_size inception_b2_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b2_7x1_3_bn_B;
    inception_b2_7x1_3_bn_B = vxCreateTensor(context,1, inception_b2_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_bn_B, dataFolder + "/bias/inception_b2_7x1_3_bn.f32"));
    vx_size inception_b2_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b2_7x1_3_scale_W;
    inception_b2_7x1_3_scale_W = vxCreateTensor(context,1, inception_b2_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_scale_W, dataFolder + "/weights/inception_b2_7x1_3_scale.f32"));
    vx_size inception_b2_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b2_7x1_3_scale_B;
    inception_b2_7x1_3_scale_B = vxCreateTensor(context,1, inception_b2_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_7x1_3_scale_B, dataFolder + "/bias/inception_b2_7x1_3_scale.f32"));
    vx_node inception_b2_7x1_3_bn_node;
    inception_b2_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b2_7x1_3, inception_b2_7x1_3_bn_W, inception_b2_7x1_3_bn_B, inception_b2_7x1_3_scale_W, inception_b2_7x1_3_scale_B, inception_b2_7x1_3_bn_eps, inception_b2_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_3_bn_node));

    // inception_b2_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_7x1_3_relu Layer
    vx_size inception_b2_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b2_7x1_3_relu;
    inception_b2_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b2_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_relu);
    vx_enum inception_b2_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_7x1_3_relu_param_a = 0;
    vx_float32 inception_b2_7x1_3_relu_param_b = 0;
    vx_node inception_b2_7x1_3_relu_node;
    inception_b2_7x1_3_relu_node = vxActivationLayer(graph, inception_b2_7x1_3_scale, inception_b2_7x1_3_relu_mode, inception_b2_7x1_3_relu_param_a, inception_b2_7x1_3_relu_param_b, inception_b2_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b2_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_7x1_3_relu_node));

    // inception_b2_1x7_3 Layer
    vx_size inception_b2_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b2_1x7_3;
    inception_b2_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b2_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3);
    vx_size inception_b2_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b2_1x7_3_W;
    inception_b2_1x7_3_W = vxCreateTensor(context,4, inception_b2_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_W, dataFolder + "/weights/inception_b2_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b2_1x7_3_params;
    inception_b2_1x7_3_params.padding_x = 3;
    inception_b2_1x7_3_params.padding_y = 0;
    inception_b2_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x7_3_params.dilation_x = 0;
    inception_b2_1x7_3_params.dilation_y = 0;
    vx_node inception_b2_1x7_3_node;
    inception_b2_1x7_3_node = vxConvolutionLayer(graph, inception_b2_7x1_3_relu, inception_b2_1x7_3_W, NULL, &inception_b2_1x7_3_params, sizeof(inception_b2_1x7_3_params ), inception_b2_1x7_3);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_3_node));

    // inception_b2_1x7_3_bn Layer
    vx_size inception_b2_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b2_1x7_3_scale;
    inception_b2_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_scale);
    vx_size inception_b2_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b2_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b2_1x7_3_bn_W;
    inception_b2_1x7_3_bn_W = vxCreateTensor(context,1, inception_b2_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_bn_W, dataFolder + "/weights/inception_b2_1x7_3_bn.f32"));
    vx_size inception_b2_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b2_1x7_3_bn_B;
    inception_b2_1x7_3_bn_B = vxCreateTensor(context,1, inception_b2_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_bn_B, dataFolder + "/bias/inception_b2_1x7_3_bn.f32"));
    vx_size inception_b2_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b2_1x7_3_scale_W;
    inception_b2_1x7_3_scale_W = vxCreateTensor(context,1, inception_b2_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_scale_W, dataFolder + "/weights/inception_b2_1x7_3_scale.f32"));
    vx_size inception_b2_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b2_1x7_3_scale_B;
    inception_b2_1x7_3_scale_B = vxCreateTensor(context,1, inception_b2_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x7_3_scale_B, dataFolder + "/bias/inception_b2_1x7_3_scale.f32"));
    vx_node inception_b2_1x7_3_bn_node;
    inception_b2_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x7_3, inception_b2_1x7_3_bn_W, inception_b2_1x7_3_bn_B, inception_b2_1x7_3_scale_W, inception_b2_1x7_3_scale_B, inception_b2_1x7_3_bn_eps, inception_b2_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_3_bn_node));

    // inception_b2_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x7_3_relu Layer
    vx_size inception_b2_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b2_1x7_3_relu;
    inception_b2_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_relu);
    vx_enum inception_b2_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x7_3_relu_param_a = 0;
    vx_float32 inception_b2_1x7_3_relu_param_b = 0;
    vx_node inception_b2_1x7_3_relu_node;
    inception_b2_1x7_3_relu_node = vxActivationLayer(graph, inception_b2_1x7_3_scale, inception_b2_1x7_3_relu_mode, inception_b2_1x7_3_relu_param_a, inception_b2_1x7_3_relu_param_b, inception_b2_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x7_3_relu_node));

    // inception_b2_pool_ave Layer
    vx_size inception_b2_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b2_pool_ave;
    inception_b2_pool_ave = vxCreateVirtualTensor(graph,4, inception_b2_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_pool_ave);
    vx_enum inception_b2_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b2_pool_ave_kernel_w = 3;
    vx_size inception_b2_pool_ave_kernel_h = 3;
    vx_size inception_b2_pool_ave_pad_w = 1;
    vx_size inception_b2_pool_ave_pad_h = 1;
    vx_enum inception_b2_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b2_pool_ave_node;
    inception_b2_pool_ave_node = vxPoolingLayer(graph, inception_b1_concat, inception_b2_pool_ave_type, inception_b2_pool_ave_kernel_w, inception_b2_pool_ave_kernel_h, inception_b2_pool_ave_pad_w, inception_b2_pool_ave_pad_h, inception_b2_pool_ave_roundPolicy, inception_b2_pool_ave );
    ERROR_CHECK_OBJECT(inception_b2_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_pool_ave_node));

    // inception_b2_1x1 Layer
    vx_size inception_b2_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b2_1x1;
    inception_b2_1x1 = vxCreateVirtualTensor(graph,4, inception_b2_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1);
    vx_size inception_b2_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b2_1x1_W;
    inception_b2_1x1_W = vxCreateTensor(context,4, inception_b2_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_W, dataFolder + "/weights/inception_b2_1x1.f32"));
    vx_nn_convolution_params_t inception_b2_1x1_params;
    inception_b2_1x1_params.padding_x = 0;
    inception_b2_1x1_params.padding_y = 0;
    inception_b2_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b2_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b2_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b2_1x1_params.dilation_x = 0;
    inception_b2_1x1_params.dilation_y = 0;
    vx_node inception_b2_1x1_node;
    inception_b2_1x1_node = vxConvolutionLayer(graph, inception_b2_pool_ave, inception_b2_1x1_W, NULL, &inception_b2_1x1_params, sizeof(inception_b2_1x1_params ), inception_b2_1x1);
    ERROR_CHECK_OBJECT(inception_b2_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_node));

    // inception_b2_1x1_bn Layer
    vx_size inception_b2_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b2_1x1_scale;
    inception_b2_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b2_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_scale);
    vx_size inception_b2_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b2_1x1_bn_eps = 0.001;
    vx_tensor inception_b2_1x1_bn_W;
    inception_b2_1x1_bn_W = vxCreateTensor(context,1, inception_b2_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_bn_W, dataFolder + "/weights/inception_b2_1x1_bn.f32"));
    vx_size inception_b2_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b2_1x1_bn_B;
    inception_b2_1x1_bn_B = vxCreateTensor(context,1, inception_b2_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_bn_B, dataFolder + "/bias/inception_b2_1x1_bn.f32"));
    vx_size inception_b2_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b2_1x1_scale_W;
    inception_b2_1x1_scale_W = vxCreateTensor(context,1, inception_b2_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_scale_W, dataFolder + "/weights/inception_b2_1x1_scale.f32"));
    vx_size inception_b2_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b2_1x1_scale_B;
    inception_b2_1x1_scale_B = vxCreateTensor(context,1, inception_b2_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b2_1x1_scale_B, dataFolder + "/bias/inception_b2_1x1_scale.f32"));
    vx_node inception_b2_1x1_bn_node;
    inception_b2_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b2_1x1, inception_b2_1x1_bn_W, inception_b2_1x1_bn_B, inception_b2_1x1_scale_W, inception_b2_1x1_scale_B, inception_b2_1x1_bn_eps, inception_b2_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b2_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_bn_node));

    // inception_b2_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b2_1x1_relu Layer
    vx_size inception_b2_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b2_1x1_relu;
    inception_b2_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b2_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_1x1_relu);
    vx_enum inception_b2_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b2_1x1_relu_param_a = 0;
    vx_float32 inception_b2_1x1_relu_param_b = 0;
    vx_node inception_b2_1x1_relu_node;
    inception_b2_1x1_relu_node = vxActivationLayer(graph, inception_b2_1x1_scale, inception_b2_1x1_relu_mode, inception_b2_1x1_relu_param_a, inception_b2_1x1_relu_param_b, inception_b2_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b2_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_1x1_relu_node));

    // inception_b2_concat Layer
    vx_size inception_b2_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b2_concat;
    inception_b2_concat = vxCreateVirtualTensor(graph,4, inception_b2_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b2_concat);
    vx_node inception_b2_concat_node;
    inception_b2_concat_node = vxConcatLayer(graph, inception_b2_concat, inception_b2_1x1_2_relu, inception_b2_7x1_relu, inception_b2_1x7_3_relu, inception_b2_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b2_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b2_concat_node));

    // inception_b3_1x1_2 Layer
    vx_size inception_b3_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b3_1x1_2;
    inception_b3_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b3_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2);
    vx_size inception_b3_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b3_1x1_2_W;
    inception_b3_1x1_2_W = vxCreateTensor(context,4, inception_b3_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_W, dataFolder + "/weights/inception_b3_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b3_1x1_2_params;
    inception_b3_1x1_2_params.padding_x = 0;
    inception_b3_1x1_2_params.padding_y = 0;
    inception_b3_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x1_2_params.dilation_x = 0;
    inception_b3_1x1_2_params.dilation_y = 0;
    vx_node inception_b3_1x1_2_node;
    inception_b3_1x1_2_node = vxConvolutionLayer(graph, inception_b2_concat, inception_b3_1x1_2_W, NULL, &inception_b3_1x1_2_params, sizeof(inception_b3_1x1_2_params ), inception_b3_1x1_2);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_2_node));

    // inception_b3_1x1_2_bn Layer
    vx_size inception_b3_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b3_1x1_2_scale;
    inception_b3_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_scale);
    vx_size inception_b3_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b3_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b3_1x1_2_bn_W;
    inception_b3_1x1_2_bn_W = vxCreateTensor(context,1, inception_b3_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_bn_W, dataFolder + "/weights/inception_b3_1x1_2_bn.f32"));
    vx_size inception_b3_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b3_1x1_2_bn_B;
    inception_b3_1x1_2_bn_B = vxCreateTensor(context,1, inception_b3_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_bn_B, dataFolder + "/bias/inception_b3_1x1_2_bn.f32"));
    vx_size inception_b3_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b3_1x1_2_scale_W;
    inception_b3_1x1_2_scale_W = vxCreateTensor(context,1, inception_b3_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_scale_W, dataFolder + "/weights/inception_b3_1x1_2_scale.f32"));
    vx_size inception_b3_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b3_1x1_2_scale_B;
    inception_b3_1x1_2_scale_B = vxCreateTensor(context,1, inception_b3_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_2_scale_B, dataFolder + "/bias/inception_b3_1x1_2_scale.f32"));
    vx_node inception_b3_1x1_2_bn_node;
    inception_b3_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x1_2, inception_b3_1x1_2_bn_W, inception_b3_1x1_2_bn_B, inception_b3_1x1_2_scale_W, inception_b3_1x1_2_scale_B, inception_b3_1x1_2_bn_eps, inception_b3_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_2_bn_node));

    // inception_b3_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x1_2_relu Layer
    vx_size inception_b3_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b3_1x1_2_relu;
    inception_b3_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_relu);
    vx_enum inception_b3_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x1_2_relu_param_a = 0;
    vx_float32 inception_b3_1x1_2_relu_param_b = 0;
    vx_node inception_b3_1x1_2_relu_node;
    inception_b3_1x1_2_relu_node = vxActivationLayer(graph, inception_b3_1x1_2_scale, inception_b3_1x1_2_relu_mode, inception_b3_1x1_2_relu_param_a, inception_b3_1x1_2_relu_param_b, inception_b3_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_2_relu_node));

    // inception_b3_1x7_reduce Layer
    vx_size inception_b3_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x7_reduce;
    inception_b3_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b3_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce);
    vx_size inception_b3_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b3_1x7_reduce_W;
    inception_b3_1x7_reduce_W = vxCreateTensor(context,4, inception_b3_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_W, dataFolder + "/weights/inception_b3_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_reduce_params;
    inception_b3_1x7_reduce_params.padding_x = 0;
    inception_b3_1x7_reduce_params.padding_y = 0;
    inception_b3_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_reduce_params.dilation_x = 0;
    inception_b3_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b3_1x7_reduce_node;
    inception_b3_1x7_reduce_node = vxConvolutionLayer(graph, inception_b2_concat, inception_b3_1x7_reduce_W, NULL, &inception_b3_1x7_reduce_params, sizeof(inception_b3_1x7_reduce_params ), inception_b3_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_reduce_node));

    // inception_b3_1x7_reduce_bn Layer
    vx_size inception_b3_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x7_reduce_scale;
    inception_b3_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_scale);
    vx_size inception_b3_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_reduce_bn_W;
    inception_b3_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b3_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_bn_W, dataFolder + "/weights/inception_b3_1x7_reduce_bn.f32"));
    vx_size inception_b3_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x7_reduce_bn_B;
    inception_b3_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b3_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_bn_B, dataFolder + "/bias/inception_b3_1x7_reduce_bn.f32"));
    vx_size inception_b3_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_1x7_reduce_scale_W;
    inception_b3_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b3_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_scale_W, dataFolder + "/weights/inception_b3_1x7_reduce_scale.f32"));
    vx_size inception_b3_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_1x7_reduce_scale_B;
    inception_b3_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b3_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_reduce_scale_B, dataFolder + "/bias/inception_b3_1x7_reduce_scale.f32"));
    vx_node inception_b3_1x7_reduce_bn_node;
    inception_b3_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7_reduce, inception_b3_1x7_reduce_bn_W, inception_b3_1x7_reduce_bn_B, inception_b3_1x7_reduce_scale_W, inception_b3_1x7_reduce_scale_B, inception_b3_1x7_reduce_bn_eps, inception_b3_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_reduce_bn_node));

    // inception_b3_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_reduce_relu Layer
    vx_size inception_b3_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_1x7_reduce_relu;
    inception_b3_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_relu);
    vx_enum inception_b3_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b3_1x7_reduce_relu_param_b = 0;
    vx_node inception_b3_1x7_reduce_relu_node;
    inception_b3_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b3_1x7_reduce_scale, inception_b3_1x7_reduce_relu_mode, inception_b3_1x7_reduce_relu_param_a, inception_b3_1x7_reduce_relu_param_b, inception_b3_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_reduce_relu_node));

    // inception_b3_1x7 Layer
    vx_size inception_b3_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_1x7;
    inception_b3_1x7 = vxCreateVirtualTensor(graph,4, inception_b3_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7);
    vx_size inception_b3_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b3_1x7_W;
    inception_b3_1x7_W = vxCreateTensor(context,4, inception_b3_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_W, dataFolder + "/weights/inception_b3_1x7.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_params;
    inception_b3_1x7_params.padding_x = 3;
    inception_b3_1x7_params.padding_y = 0;
    inception_b3_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_params.dilation_x = 0;
    inception_b3_1x7_params.dilation_y = 0;
    vx_node inception_b3_1x7_node;
    inception_b3_1x7_node = vxConvolutionLayer(graph, inception_b3_1x7_reduce_relu, inception_b3_1x7_W, NULL, &inception_b3_1x7_params, sizeof(inception_b3_1x7_params ), inception_b3_1x7);
    ERROR_CHECK_OBJECT(inception_b3_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_node));

    // inception_b3_1x7_bn Layer
    vx_size inception_b3_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_1x7_scale;
    inception_b3_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_scale);
    vx_size inception_b3_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b3_1x7_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_bn_W;
    inception_b3_1x7_bn_W = vxCreateTensor(context,1, inception_b3_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_bn_W, dataFolder + "/weights/inception_b3_1x7_bn.f32"));
    vx_size inception_b3_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b3_1x7_bn_B;
    inception_b3_1x7_bn_B = vxCreateTensor(context,1, inception_b3_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_bn_B, dataFolder + "/bias/inception_b3_1x7_bn.f32"));
    vx_size inception_b3_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b3_1x7_scale_W;
    inception_b3_1x7_scale_W = vxCreateTensor(context,1, inception_b3_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_scale_W, dataFolder + "/weights/inception_b3_1x7_scale.f32"));
    vx_size inception_b3_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b3_1x7_scale_B;
    inception_b3_1x7_scale_B = vxCreateTensor(context,1, inception_b3_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_scale_B, dataFolder + "/bias/inception_b3_1x7_scale.f32"));
    vx_node inception_b3_1x7_bn_node;
    inception_b3_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7, inception_b3_1x7_bn_W, inception_b3_1x7_bn_B, inception_b3_1x7_scale_W, inception_b3_1x7_scale_B, inception_b3_1x7_bn_eps, inception_b3_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_bn_node));

    // inception_b3_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_relu Layer
    vx_size inception_b3_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_1x7_relu;
    inception_b3_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_relu);
    vx_enum inception_b3_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_relu_param_a = 0;
    vx_float32 inception_b3_1x7_relu_param_b = 0;
    vx_node inception_b3_1x7_relu_node;
    inception_b3_1x7_relu_node = vxActivationLayer(graph, inception_b3_1x7_scale, inception_b3_1x7_relu_mode, inception_b3_1x7_relu_param_a, inception_b3_1x7_relu_param_b, inception_b3_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_relu_node));

    // inception_b3_7x1 Layer
    vx_size inception_b3_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b3_7x1;
    inception_b3_7x1 = vxCreateVirtualTensor(graph,4, inception_b3_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1);
    vx_size inception_b3_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b3_7x1_W;
    inception_b3_7x1_W = vxCreateTensor(context,4, inception_b3_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_W, dataFolder + "/weights/inception_b3_7x1.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_params;
    inception_b3_7x1_params.padding_x = 0;
    inception_b3_7x1_params.padding_y = 3;
    inception_b3_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_params.dilation_x = 0;
    inception_b3_7x1_params.dilation_y = 0;
    vx_node inception_b3_7x1_node;
    inception_b3_7x1_node = vxConvolutionLayer(graph, inception_b3_1x7_relu, inception_b3_7x1_W, NULL, &inception_b3_7x1_params, sizeof(inception_b3_7x1_params ), inception_b3_7x1);
    ERROR_CHECK_OBJECT(inception_b3_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_node));

    // inception_b3_7x1_bn Layer
    vx_size inception_b3_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b3_7x1_scale;
    inception_b3_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_scale);
    vx_size inception_b3_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b3_7x1_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_bn_W;
    inception_b3_7x1_bn_W = vxCreateTensor(context,1, inception_b3_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_bn_W, dataFolder + "/weights/inception_b3_7x1_bn.f32"));
    vx_size inception_b3_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b3_7x1_bn_B;
    inception_b3_7x1_bn_B = vxCreateTensor(context,1, inception_b3_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_bn_B, dataFolder + "/bias/inception_b3_7x1_bn.f32"));
    vx_size inception_b3_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b3_7x1_scale_W;
    inception_b3_7x1_scale_W = vxCreateTensor(context,1, inception_b3_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_scale_W, dataFolder + "/weights/inception_b3_7x1_scale.f32"));
    vx_size inception_b3_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b3_7x1_scale_B;
    inception_b3_7x1_scale_B = vxCreateTensor(context,1, inception_b3_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_scale_B, dataFolder + "/bias/inception_b3_7x1_scale.f32"));
    vx_node inception_b3_7x1_bn_node;
    inception_b3_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1, inception_b3_7x1_bn_W, inception_b3_7x1_bn_B, inception_b3_7x1_scale_W, inception_b3_7x1_scale_B, inception_b3_7x1_bn_eps, inception_b3_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_bn_node));

    // inception_b3_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_relu Layer
    vx_size inception_b3_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b3_7x1_relu;
    inception_b3_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_relu);
    vx_enum inception_b3_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_relu_param_a = 0;
    vx_float32 inception_b3_7x1_relu_param_b = 0;
    vx_node inception_b3_7x1_relu_node;
    inception_b3_7x1_relu_node = vxActivationLayer(graph, inception_b3_7x1_scale, inception_b3_7x1_relu_mode, inception_b3_7x1_relu_param_a, inception_b3_7x1_relu_param_b, inception_b3_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_relu_node));

    // inception_b3_7x1_2_reduce Layer
    vx_size inception_b3_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_2_reduce;
    inception_b3_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce);
    vx_size inception_b3_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b3_7x1_2_reduce_W;
    inception_b3_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b3_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_reduce_W, dataFolder + "/weights/inception_b3_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_2_reduce_params;
    inception_b3_7x1_2_reduce_params.padding_x = 0;
    inception_b3_7x1_2_reduce_params.padding_y = 0;
    inception_b3_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_2_reduce_params.dilation_x = 0;
    inception_b3_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b3_7x1_2_reduce_node;
    inception_b3_7x1_2_reduce_node = vxConvolutionLayer(graph, inception_b2_concat, inception_b3_7x1_2_reduce_W, NULL, &inception_b3_7x1_2_reduce_params, sizeof(inception_b3_7x1_2_reduce_params ), inception_b3_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_reduce_node));

    // inception_b3_7x1_2_reduce_bn Layer
    vx_size inception_b3_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_2_reduce_scale;
    inception_b3_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_scale);
    vx_size inception_b3_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_2_reduce_bn_W;
    inception_b3_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b3_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b3_7x1_2_reduce_bn.f32"));
    vx_size inception_b3_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_2_reduce_bn_B;
    inception_b3_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b3_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b3_7x1_2_reduce_bn.f32"));
    vx_size inception_b3_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_2_reduce_scale_W;
    inception_b3_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b3_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b3_7x1_2_reduce_scale.f32"));
    vx_size inception_b3_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_2_reduce_scale_B;
    inception_b3_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b3_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b3_7x1_2_reduce_scale.f32"));
    vx_node inception_b3_7x1_2_reduce_bn_node;
    inception_b3_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1_2_reduce, inception_b3_7x1_2_reduce_bn_W, inception_b3_7x1_2_reduce_bn_B, inception_b3_7x1_2_reduce_scale_W, inception_b3_7x1_2_reduce_scale_B, inception_b3_7x1_2_reduce_bn_eps, inception_b3_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_reduce_bn_node));

    // inception_b3_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_2_reduce_relu Layer
    vx_size inception_b3_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_2_reduce_relu;
    inception_b3_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_relu);
    vx_enum inception_b3_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b3_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b3_7x1_2_reduce_relu_node;
    inception_b3_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b3_7x1_2_reduce_scale, inception_b3_7x1_2_reduce_relu_mode, inception_b3_7x1_2_reduce_relu_param_a, inception_b3_7x1_2_reduce_relu_param_b, inception_b3_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_reduce_relu_node));

    // inception_b3_7x1_2 Layer
    vx_size inception_b3_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_2;
    inception_b3_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2);
    vx_size inception_b3_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b3_7x1_2_W;
    inception_b3_7x1_2_W = vxCreateTensor(context,4, inception_b3_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_W, dataFolder + "/weights/inception_b3_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_2_params;
    inception_b3_7x1_2_params.padding_x = 0;
    inception_b3_7x1_2_params.padding_y = 3;
    inception_b3_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_2_params.dilation_x = 0;
    inception_b3_7x1_2_params.dilation_y = 0;
    vx_node inception_b3_7x1_2_node;
    inception_b3_7x1_2_node = vxConvolutionLayer(graph, inception_b3_7x1_2_reduce_relu, inception_b3_7x1_2_W, NULL, &inception_b3_7x1_2_params, sizeof(inception_b3_7x1_2_params ), inception_b3_7x1_2);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_node));

    // inception_b3_7x1_2_bn Layer
    vx_size inception_b3_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_2_scale;
    inception_b3_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_scale);
    vx_size inception_b3_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b3_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_2_bn_W;
    inception_b3_7x1_2_bn_W = vxCreateTensor(context,1, inception_b3_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_bn_W, dataFolder + "/weights/inception_b3_7x1_2_bn.f32"));
    vx_size inception_b3_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_2_bn_B;
    inception_b3_7x1_2_bn_B = vxCreateTensor(context,1, inception_b3_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_bn_B, dataFolder + "/bias/inception_b3_7x1_2_bn.f32"));
    vx_size inception_b3_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_2_scale_W;
    inception_b3_7x1_2_scale_W = vxCreateTensor(context,1, inception_b3_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_scale_W, dataFolder + "/weights/inception_b3_7x1_2_scale.f32"));
    vx_size inception_b3_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b3_7x1_2_scale_B;
    inception_b3_7x1_2_scale_B = vxCreateTensor(context,1, inception_b3_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_2_scale_B, dataFolder + "/bias/inception_b3_7x1_2_scale.f32"));
    vx_node inception_b3_7x1_2_bn_node;
    inception_b3_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1_2, inception_b3_7x1_2_bn_W, inception_b3_7x1_2_bn_B, inception_b3_7x1_2_scale_W, inception_b3_7x1_2_scale_B, inception_b3_7x1_2_bn_eps, inception_b3_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_bn_node));

    // inception_b3_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_2_relu Layer
    vx_size inception_b3_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b3_7x1_2_relu;
    inception_b3_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_relu);
    vx_enum inception_b3_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_2_relu_param_a = 0;
    vx_float32 inception_b3_7x1_2_relu_param_b = 0;
    vx_node inception_b3_7x1_2_relu_node;
    inception_b3_7x1_2_relu_node = vxActivationLayer(graph, inception_b3_7x1_2_scale, inception_b3_7x1_2_relu_mode, inception_b3_7x1_2_relu_param_a, inception_b3_7x1_2_relu_param_b, inception_b3_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_2_relu_node));

    // inception_b3_1x7_2 Layer
    vx_size inception_b3_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_1x7_2;
    inception_b3_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b3_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2);
    vx_size inception_b3_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b3_1x7_2_W;
    inception_b3_1x7_2_W = vxCreateTensor(context,4, inception_b3_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_W, dataFolder + "/weights/inception_b3_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_2_params;
    inception_b3_1x7_2_params.padding_x = 3;
    inception_b3_1x7_2_params.padding_y = 0;
    inception_b3_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_2_params.dilation_x = 0;
    inception_b3_1x7_2_params.dilation_y = 0;
    vx_node inception_b3_1x7_2_node;
    inception_b3_1x7_2_node = vxConvolutionLayer(graph, inception_b3_7x1_2_relu, inception_b3_1x7_2_W, NULL, &inception_b3_1x7_2_params, sizeof(inception_b3_1x7_2_params ), inception_b3_1x7_2);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_2_node));

    // inception_b3_1x7_2_bn Layer
    vx_size inception_b3_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_1x7_2_scale;
    inception_b3_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_scale);
    vx_size inception_b3_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b3_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_2_bn_W;
    inception_b3_1x7_2_bn_W = vxCreateTensor(context,1, inception_b3_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_bn_W, dataFolder + "/weights/inception_b3_1x7_2_bn.f32"));
    vx_size inception_b3_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b3_1x7_2_bn_B;
    inception_b3_1x7_2_bn_B = vxCreateTensor(context,1, inception_b3_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_bn_B, dataFolder + "/bias/inception_b3_1x7_2_bn.f32"));
    vx_size inception_b3_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b3_1x7_2_scale_W;
    inception_b3_1x7_2_scale_W = vxCreateTensor(context,1, inception_b3_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_scale_W, dataFolder + "/weights/inception_b3_1x7_2_scale.f32"));
    vx_size inception_b3_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b3_1x7_2_scale_B;
    inception_b3_1x7_2_scale_B = vxCreateTensor(context,1, inception_b3_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_2_scale_B, dataFolder + "/bias/inception_b3_1x7_2_scale.f32"));
    vx_node inception_b3_1x7_2_bn_node;
    inception_b3_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7_2, inception_b3_1x7_2_bn_W, inception_b3_1x7_2_bn_B, inception_b3_1x7_2_scale_W, inception_b3_1x7_2_scale_B, inception_b3_1x7_2_bn_eps, inception_b3_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_2_bn_node));

    // inception_b3_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_2_relu Layer
    vx_size inception_b3_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_1x7_2_relu;
    inception_b3_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_relu);
    vx_enum inception_b3_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_2_relu_param_a = 0;
    vx_float32 inception_b3_1x7_2_relu_param_b = 0;
    vx_node inception_b3_1x7_2_relu_node;
    inception_b3_1x7_2_relu_node = vxActivationLayer(graph, inception_b3_1x7_2_scale, inception_b3_1x7_2_relu_mode, inception_b3_1x7_2_relu_param_a, inception_b3_1x7_2_relu_param_b, inception_b3_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_2_relu_node));

    // inception_b3_7x1_3 Layer
    vx_size inception_b3_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_7x1_3;
    inception_b3_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b3_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3);
    vx_size inception_b3_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b3_7x1_3_W;
    inception_b3_7x1_3_W = vxCreateTensor(context,4, inception_b3_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_W, dataFolder + "/weights/inception_b3_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b3_7x1_3_params;
    inception_b3_7x1_3_params.padding_x = 0;
    inception_b3_7x1_3_params.padding_y = 3;
    inception_b3_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_7x1_3_params.dilation_x = 0;
    inception_b3_7x1_3_params.dilation_y = 0;
    vx_node inception_b3_7x1_3_node;
    inception_b3_7x1_3_node = vxConvolutionLayer(graph, inception_b3_1x7_2_relu, inception_b3_7x1_3_W, NULL, &inception_b3_7x1_3_params, sizeof(inception_b3_7x1_3_params ), inception_b3_7x1_3);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_3_node));

    // inception_b3_7x1_3_bn Layer
    vx_size inception_b3_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_7x1_3_scale;
    inception_b3_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b3_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_scale);
    vx_size inception_b3_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b3_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b3_7x1_3_bn_W;
    inception_b3_7x1_3_bn_W = vxCreateTensor(context,1, inception_b3_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_bn_W, dataFolder + "/weights/inception_b3_7x1_3_bn.f32"));
    vx_size inception_b3_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b3_7x1_3_bn_B;
    inception_b3_7x1_3_bn_B = vxCreateTensor(context,1, inception_b3_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_bn_B, dataFolder + "/bias/inception_b3_7x1_3_bn.f32"));
    vx_size inception_b3_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b3_7x1_3_scale_W;
    inception_b3_7x1_3_scale_W = vxCreateTensor(context,1, inception_b3_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_scale_W, dataFolder + "/weights/inception_b3_7x1_3_scale.f32"));
    vx_size inception_b3_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b3_7x1_3_scale_B;
    inception_b3_7x1_3_scale_B = vxCreateTensor(context,1, inception_b3_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_7x1_3_scale_B, dataFolder + "/bias/inception_b3_7x1_3_scale.f32"));
    vx_node inception_b3_7x1_3_bn_node;
    inception_b3_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b3_7x1_3, inception_b3_7x1_3_bn_W, inception_b3_7x1_3_bn_B, inception_b3_7x1_3_scale_W, inception_b3_7x1_3_scale_B, inception_b3_7x1_3_bn_eps, inception_b3_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_3_bn_node));

    // inception_b3_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_7x1_3_relu Layer
    vx_size inception_b3_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b3_7x1_3_relu;
    inception_b3_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b3_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_relu);
    vx_enum inception_b3_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_7x1_3_relu_param_a = 0;
    vx_float32 inception_b3_7x1_3_relu_param_b = 0;
    vx_node inception_b3_7x1_3_relu_node;
    inception_b3_7x1_3_relu_node = vxActivationLayer(graph, inception_b3_7x1_3_scale, inception_b3_7x1_3_relu_mode, inception_b3_7x1_3_relu_param_a, inception_b3_7x1_3_relu_param_b, inception_b3_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b3_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_7x1_3_relu_node));

    // inception_b3_1x7_3 Layer
    vx_size inception_b3_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b3_1x7_3;
    inception_b3_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b3_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3);
    vx_size inception_b3_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b3_1x7_3_W;
    inception_b3_1x7_3_W = vxCreateTensor(context,4, inception_b3_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_W, dataFolder + "/weights/inception_b3_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b3_1x7_3_params;
    inception_b3_1x7_3_params.padding_x = 3;
    inception_b3_1x7_3_params.padding_y = 0;
    inception_b3_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x7_3_params.dilation_x = 0;
    inception_b3_1x7_3_params.dilation_y = 0;
    vx_node inception_b3_1x7_3_node;
    inception_b3_1x7_3_node = vxConvolutionLayer(graph, inception_b3_7x1_3_relu, inception_b3_1x7_3_W, NULL, &inception_b3_1x7_3_params, sizeof(inception_b3_1x7_3_params ), inception_b3_1x7_3);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_3_node));

    // inception_b3_1x7_3_bn Layer
    vx_size inception_b3_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b3_1x7_3_scale;
    inception_b3_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_scale);
    vx_size inception_b3_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b3_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b3_1x7_3_bn_W;
    inception_b3_1x7_3_bn_W = vxCreateTensor(context,1, inception_b3_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_bn_W, dataFolder + "/weights/inception_b3_1x7_3_bn.f32"));
    vx_size inception_b3_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b3_1x7_3_bn_B;
    inception_b3_1x7_3_bn_B = vxCreateTensor(context,1, inception_b3_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_bn_B, dataFolder + "/bias/inception_b3_1x7_3_bn.f32"));
    vx_size inception_b3_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b3_1x7_3_scale_W;
    inception_b3_1x7_3_scale_W = vxCreateTensor(context,1, inception_b3_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_scale_W, dataFolder + "/weights/inception_b3_1x7_3_scale.f32"));
    vx_size inception_b3_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b3_1x7_3_scale_B;
    inception_b3_1x7_3_scale_B = vxCreateTensor(context,1, inception_b3_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x7_3_scale_B, dataFolder + "/bias/inception_b3_1x7_3_scale.f32"));
    vx_node inception_b3_1x7_3_bn_node;
    inception_b3_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x7_3, inception_b3_1x7_3_bn_W, inception_b3_1x7_3_bn_B, inception_b3_1x7_3_scale_W, inception_b3_1x7_3_scale_B, inception_b3_1x7_3_bn_eps, inception_b3_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_3_bn_node));

    // inception_b3_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x7_3_relu Layer
    vx_size inception_b3_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b3_1x7_3_relu;
    inception_b3_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_relu);
    vx_enum inception_b3_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x7_3_relu_param_a = 0;
    vx_float32 inception_b3_1x7_3_relu_param_b = 0;
    vx_node inception_b3_1x7_3_relu_node;
    inception_b3_1x7_3_relu_node = vxActivationLayer(graph, inception_b3_1x7_3_scale, inception_b3_1x7_3_relu_mode, inception_b3_1x7_3_relu_param_a, inception_b3_1x7_3_relu_param_b, inception_b3_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x7_3_relu_node));

    // inception_b3_pool_ave Layer
    vx_size inception_b3_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b3_pool_ave;
    inception_b3_pool_ave = vxCreateVirtualTensor(graph,4, inception_b3_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_pool_ave);
    vx_enum inception_b3_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b3_pool_ave_kernel_w = 3;
    vx_size inception_b3_pool_ave_kernel_h = 3;
    vx_size inception_b3_pool_ave_pad_w = 1;
    vx_size inception_b3_pool_ave_pad_h = 1;
    vx_enum inception_b3_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b3_pool_ave_node;
    inception_b3_pool_ave_node = vxPoolingLayer(graph, inception_b2_concat, inception_b3_pool_ave_type, inception_b3_pool_ave_kernel_w, inception_b3_pool_ave_kernel_h, inception_b3_pool_ave_pad_w, inception_b3_pool_ave_pad_h, inception_b3_pool_ave_roundPolicy, inception_b3_pool_ave );
    ERROR_CHECK_OBJECT(inception_b3_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_pool_ave_node));

    // inception_b3_1x1 Layer
    vx_size inception_b3_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b3_1x1;
    inception_b3_1x1 = vxCreateVirtualTensor(graph,4, inception_b3_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1);
    vx_size inception_b3_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b3_1x1_W;
    inception_b3_1x1_W = vxCreateTensor(context,4, inception_b3_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_W, dataFolder + "/weights/inception_b3_1x1.f32"));
    vx_nn_convolution_params_t inception_b3_1x1_params;
    inception_b3_1x1_params.padding_x = 0;
    inception_b3_1x1_params.padding_y = 0;
    inception_b3_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b3_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b3_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b3_1x1_params.dilation_x = 0;
    inception_b3_1x1_params.dilation_y = 0;
    vx_node inception_b3_1x1_node;
    inception_b3_1x1_node = vxConvolutionLayer(graph, inception_b3_pool_ave, inception_b3_1x1_W, NULL, &inception_b3_1x1_params, sizeof(inception_b3_1x1_params ), inception_b3_1x1);
    ERROR_CHECK_OBJECT(inception_b3_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_node));

    // inception_b3_1x1_bn Layer
    vx_size inception_b3_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b3_1x1_scale;
    inception_b3_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b3_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_scale);
    vx_size inception_b3_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b3_1x1_bn_eps = 0.001;
    vx_tensor inception_b3_1x1_bn_W;
    inception_b3_1x1_bn_W = vxCreateTensor(context,1, inception_b3_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_bn_W, dataFolder + "/weights/inception_b3_1x1_bn.f32"));
    vx_size inception_b3_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b3_1x1_bn_B;
    inception_b3_1x1_bn_B = vxCreateTensor(context,1, inception_b3_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_bn_B, dataFolder + "/bias/inception_b3_1x1_bn.f32"));
    vx_size inception_b3_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b3_1x1_scale_W;
    inception_b3_1x1_scale_W = vxCreateTensor(context,1, inception_b3_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_scale_W, dataFolder + "/weights/inception_b3_1x1_scale.f32"));
    vx_size inception_b3_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b3_1x1_scale_B;
    inception_b3_1x1_scale_B = vxCreateTensor(context,1, inception_b3_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b3_1x1_scale_B, dataFolder + "/bias/inception_b3_1x1_scale.f32"));
    vx_node inception_b3_1x1_bn_node;
    inception_b3_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b3_1x1, inception_b3_1x1_bn_W, inception_b3_1x1_bn_B, inception_b3_1x1_scale_W, inception_b3_1x1_scale_B, inception_b3_1x1_bn_eps, inception_b3_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b3_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_bn_node));

    // inception_b3_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b3_1x1_relu Layer
    vx_size inception_b3_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b3_1x1_relu;
    inception_b3_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b3_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_1x1_relu);
    vx_enum inception_b3_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b3_1x1_relu_param_a = 0;
    vx_float32 inception_b3_1x1_relu_param_b = 0;
    vx_node inception_b3_1x1_relu_node;
    inception_b3_1x1_relu_node = vxActivationLayer(graph, inception_b3_1x1_scale, inception_b3_1x1_relu_mode, inception_b3_1x1_relu_param_a, inception_b3_1x1_relu_param_b, inception_b3_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b3_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_1x1_relu_node));

    // inception_b3_concat Layer
    vx_size inception_b3_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b3_concat;
    inception_b3_concat = vxCreateVirtualTensor(graph,4, inception_b3_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b3_concat);
    vx_node inception_b3_concat_node;
    inception_b3_concat_node = vxConcatLayer(graph, inception_b3_concat, inception_b3_1x1_2_relu, inception_b3_7x1_relu, inception_b3_1x7_3_relu, inception_b3_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b3_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b3_concat_node));

    // inception_b4_1x1_2 Layer
    vx_size inception_b4_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b4_1x1_2;
    inception_b4_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b4_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2);
    vx_size inception_b4_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b4_1x1_2_W;
    inception_b4_1x1_2_W = vxCreateTensor(context,4, inception_b4_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_W, dataFolder + "/weights/inception_b4_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b4_1x1_2_params;
    inception_b4_1x1_2_params.padding_x = 0;
    inception_b4_1x1_2_params.padding_y = 0;
    inception_b4_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x1_2_params.dilation_x = 0;
    inception_b4_1x1_2_params.dilation_y = 0;
    vx_node inception_b4_1x1_2_node;
    inception_b4_1x1_2_node = vxConvolutionLayer(graph, inception_b3_concat, inception_b4_1x1_2_W, NULL, &inception_b4_1x1_2_params, sizeof(inception_b4_1x1_2_params ), inception_b4_1x1_2);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_2_node));

    // inception_b4_1x1_2_bn Layer
    vx_size inception_b4_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b4_1x1_2_scale;
    inception_b4_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_scale);
    vx_size inception_b4_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b4_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b4_1x1_2_bn_W;
    inception_b4_1x1_2_bn_W = vxCreateTensor(context,1, inception_b4_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_bn_W, dataFolder + "/weights/inception_b4_1x1_2_bn.f32"));
    vx_size inception_b4_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b4_1x1_2_bn_B;
    inception_b4_1x1_2_bn_B = vxCreateTensor(context,1, inception_b4_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_bn_B, dataFolder + "/bias/inception_b4_1x1_2_bn.f32"));
    vx_size inception_b4_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b4_1x1_2_scale_W;
    inception_b4_1x1_2_scale_W = vxCreateTensor(context,1, inception_b4_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_scale_W, dataFolder + "/weights/inception_b4_1x1_2_scale.f32"));
    vx_size inception_b4_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b4_1x1_2_scale_B;
    inception_b4_1x1_2_scale_B = vxCreateTensor(context,1, inception_b4_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_2_scale_B, dataFolder + "/bias/inception_b4_1x1_2_scale.f32"));
    vx_node inception_b4_1x1_2_bn_node;
    inception_b4_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x1_2, inception_b4_1x1_2_bn_W, inception_b4_1x1_2_bn_B, inception_b4_1x1_2_scale_W, inception_b4_1x1_2_scale_B, inception_b4_1x1_2_bn_eps, inception_b4_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_2_bn_node));

    // inception_b4_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x1_2_relu Layer
    vx_size inception_b4_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b4_1x1_2_relu;
    inception_b4_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_relu);
    vx_enum inception_b4_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x1_2_relu_param_a = 0;
    vx_float32 inception_b4_1x1_2_relu_param_b = 0;
    vx_node inception_b4_1x1_2_relu_node;
    inception_b4_1x1_2_relu_node = vxActivationLayer(graph, inception_b4_1x1_2_scale, inception_b4_1x1_2_relu_mode, inception_b4_1x1_2_relu_param_a, inception_b4_1x1_2_relu_param_b, inception_b4_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_2_relu_node));

    // inception_b4_1x7_reduce Layer
    vx_size inception_b4_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_reduce;
    inception_b4_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b4_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce);
    vx_size inception_b4_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b4_1x7_reduce_W;
    inception_b4_1x7_reduce_W = vxCreateTensor(context,4, inception_b4_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_W, dataFolder + "/weights/inception_b4_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_reduce_params;
    inception_b4_1x7_reduce_params.padding_x = 0;
    inception_b4_1x7_reduce_params.padding_y = 0;
    inception_b4_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_reduce_params.dilation_x = 0;
    inception_b4_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b4_1x7_reduce_node;
    inception_b4_1x7_reduce_node = vxConvolutionLayer(graph, inception_b3_concat, inception_b4_1x7_reduce_W, NULL, &inception_b4_1x7_reduce_params, sizeof(inception_b4_1x7_reduce_params ), inception_b4_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_reduce_node));

    // inception_b4_1x7_reduce_bn Layer
    vx_size inception_b4_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_reduce_scale;
    inception_b4_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_scale);
    vx_size inception_b4_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_reduce_bn_W;
    inception_b4_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b4_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_bn_W, dataFolder + "/weights/inception_b4_1x7_reduce_bn.f32"));
    vx_size inception_b4_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_reduce_bn_B;
    inception_b4_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b4_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_bn_B, dataFolder + "/bias/inception_b4_1x7_reduce_bn.f32"));
    vx_size inception_b4_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_reduce_scale_W;
    inception_b4_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b4_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_scale_W, dataFolder + "/weights/inception_b4_1x7_reduce_scale.f32"));
    vx_size inception_b4_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_1x7_reduce_scale_B;
    inception_b4_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b4_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_reduce_scale_B, dataFolder + "/bias/inception_b4_1x7_reduce_scale.f32"));
    vx_node inception_b4_1x7_reduce_bn_node;
    inception_b4_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7_reduce, inception_b4_1x7_reduce_bn_W, inception_b4_1x7_reduce_bn_B, inception_b4_1x7_reduce_scale_W, inception_b4_1x7_reduce_scale_B, inception_b4_1x7_reduce_bn_eps, inception_b4_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_reduce_bn_node));

    // inception_b4_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_reduce_relu Layer
    vx_size inception_b4_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_1x7_reduce_relu;
    inception_b4_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_relu);
    vx_enum inception_b4_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b4_1x7_reduce_relu_param_b = 0;
    vx_node inception_b4_1x7_reduce_relu_node;
    inception_b4_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b4_1x7_reduce_scale, inception_b4_1x7_reduce_relu_mode, inception_b4_1x7_reduce_relu_param_a, inception_b4_1x7_reduce_relu_param_b, inception_b4_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_reduce_relu_node));

    // inception_b4_1x7 Layer
    vx_size inception_b4_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_1x7;
    inception_b4_1x7 = vxCreateVirtualTensor(graph,4, inception_b4_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7);
    vx_size inception_b4_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b4_1x7_W;
    inception_b4_1x7_W = vxCreateTensor(context,4, inception_b4_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_W, dataFolder + "/weights/inception_b4_1x7.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_params;
    inception_b4_1x7_params.padding_x = 3;
    inception_b4_1x7_params.padding_y = 0;
    inception_b4_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_params.dilation_x = 0;
    inception_b4_1x7_params.dilation_y = 0;
    vx_node inception_b4_1x7_node;
    inception_b4_1x7_node = vxConvolutionLayer(graph, inception_b4_1x7_reduce_relu, inception_b4_1x7_W, NULL, &inception_b4_1x7_params, sizeof(inception_b4_1x7_params ), inception_b4_1x7);
    ERROR_CHECK_OBJECT(inception_b4_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_node));

    // inception_b4_1x7_bn Layer
    vx_size inception_b4_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_1x7_scale;
    inception_b4_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_scale);
    vx_size inception_b4_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b4_1x7_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_bn_W;
    inception_b4_1x7_bn_W = vxCreateTensor(context,1, inception_b4_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_bn_W, dataFolder + "/weights/inception_b4_1x7_bn.f32"));
    vx_size inception_b4_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b4_1x7_bn_B;
    inception_b4_1x7_bn_B = vxCreateTensor(context,1, inception_b4_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_bn_B, dataFolder + "/bias/inception_b4_1x7_bn.f32"));
    vx_size inception_b4_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b4_1x7_scale_W;
    inception_b4_1x7_scale_W = vxCreateTensor(context,1, inception_b4_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_scale_W, dataFolder + "/weights/inception_b4_1x7_scale.f32"));
    vx_size inception_b4_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b4_1x7_scale_B;
    inception_b4_1x7_scale_B = vxCreateTensor(context,1, inception_b4_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_scale_B, dataFolder + "/bias/inception_b4_1x7_scale.f32"));
    vx_node inception_b4_1x7_bn_node;
    inception_b4_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7, inception_b4_1x7_bn_W, inception_b4_1x7_bn_B, inception_b4_1x7_scale_W, inception_b4_1x7_scale_B, inception_b4_1x7_bn_eps, inception_b4_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_bn_node));

    // inception_b4_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_relu Layer
    vx_size inception_b4_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_1x7_relu;
    inception_b4_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_relu);
    vx_enum inception_b4_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_relu_param_a = 0;
    vx_float32 inception_b4_1x7_relu_param_b = 0;
    vx_node inception_b4_1x7_relu_node;
    inception_b4_1x7_relu_node = vxActivationLayer(graph, inception_b4_1x7_scale, inception_b4_1x7_relu_mode, inception_b4_1x7_relu_param_a, inception_b4_1x7_relu_param_b, inception_b4_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_relu_node));

    // inception_b4_7x1 Layer
    vx_size inception_b4_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b4_7x1;
    inception_b4_7x1 = vxCreateVirtualTensor(graph,4, inception_b4_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1);
    vx_size inception_b4_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b4_7x1_W;
    inception_b4_7x1_W = vxCreateTensor(context,4, inception_b4_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_W, dataFolder + "/weights/inception_b4_7x1.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_params;
    inception_b4_7x1_params.padding_x = 0;
    inception_b4_7x1_params.padding_y = 3;
    inception_b4_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_params.dilation_x = 0;
    inception_b4_7x1_params.dilation_y = 0;
    vx_node inception_b4_7x1_node;
    inception_b4_7x1_node = vxConvolutionLayer(graph, inception_b4_1x7_relu, inception_b4_7x1_W, NULL, &inception_b4_7x1_params, sizeof(inception_b4_7x1_params ), inception_b4_7x1);
    ERROR_CHECK_OBJECT(inception_b4_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_node));

    // inception_b4_7x1_bn Layer
    vx_size inception_b4_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b4_7x1_scale;
    inception_b4_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_scale);
    vx_size inception_b4_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b4_7x1_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_bn_W;
    inception_b4_7x1_bn_W = vxCreateTensor(context,1, inception_b4_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_bn_W, dataFolder + "/weights/inception_b4_7x1_bn.f32"));
    vx_size inception_b4_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b4_7x1_bn_B;
    inception_b4_7x1_bn_B = vxCreateTensor(context,1, inception_b4_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_bn_B, dataFolder + "/bias/inception_b4_7x1_bn.f32"));
    vx_size inception_b4_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b4_7x1_scale_W;
    inception_b4_7x1_scale_W = vxCreateTensor(context,1, inception_b4_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_scale_W, dataFolder + "/weights/inception_b4_7x1_scale.f32"));
    vx_size inception_b4_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b4_7x1_scale_B;
    inception_b4_7x1_scale_B = vxCreateTensor(context,1, inception_b4_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_scale_B, dataFolder + "/bias/inception_b4_7x1_scale.f32"));
    vx_node inception_b4_7x1_bn_node;
    inception_b4_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1, inception_b4_7x1_bn_W, inception_b4_7x1_bn_B, inception_b4_7x1_scale_W, inception_b4_7x1_scale_B, inception_b4_7x1_bn_eps, inception_b4_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_bn_node));

    // inception_b4_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_relu Layer
    vx_size inception_b4_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b4_7x1_relu;
    inception_b4_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_relu);
    vx_enum inception_b4_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_relu_param_a = 0;
    vx_float32 inception_b4_7x1_relu_param_b = 0;
    vx_node inception_b4_7x1_relu_node;
    inception_b4_7x1_relu_node = vxActivationLayer(graph, inception_b4_7x1_scale, inception_b4_7x1_relu_mode, inception_b4_7x1_relu_param_a, inception_b4_7x1_relu_param_b, inception_b4_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_relu_node));

    // inception_b4_7x1_2_reduce Layer
    vx_size inception_b4_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_reduce;
    inception_b4_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce);
    vx_size inception_b4_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b4_7x1_2_reduce_W;
    inception_b4_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b4_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_reduce_W, dataFolder + "/weights/inception_b4_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_2_reduce_params;
    inception_b4_7x1_2_reduce_params.padding_x = 0;
    inception_b4_7x1_2_reduce_params.padding_y = 0;
    inception_b4_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_2_reduce_params.dilation_x = 0;
    inception_b4_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b4_7x1_2_reduce_node;
    inception_b4_7x1_2_reduce_node = vxConvolutionLayer(graph, inception_b3_concat, inception_b4_7x1_2_reduce_W, NULL, &inception_b4_7x1_2_reduce_params, sizeof(inception_b4_7x1_2_reduce_params ), inception_b4_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_reduce_node));

    // inception_b4_7x1_2_reduce_bn Layer
    vx_size inception_b4_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_reduce_scale;
    inception_b4_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_scale);
    vx_size inception_b4_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_2_reduce_bn_W;
    inception_b4_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b4_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b4_7x1_2_reduce_bn.f32"));
    vx_size inception_b4_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_reduce_bn_B;
    inception_b4_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b4_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b4_7x1_2_reduce_bn.f32"));
    vx_size inception_b4_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_reduce_scale_W;
    inception_b4_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b4_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b4_7x1_2_reduce_scale.f32"));
    vx_size inception_b4_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_reduce_scale_B;
    inception_b4_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b4_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b4_7x1_2_reduce_scale.f32"));
    vx_node inception_b4_7x1_2_reduce_bn_node;
    inception_b4_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1_2_reduce, inception_b4_7x1_2_reduce_bn_W, inception_b4_7x1_2_reduce_bn_B, inception_b4_7x1_2_reduce_scale_W, inception_b4_7x1_2_reduce_scale_B, inception_b4_7x1_2_reduce_bn_eps, inception_b4_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_reduce_bn_node));

    // inception_b4_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_2_reduce_relu Layer
    vx_size inception_b4_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_reduce_relu;
    inception_b4_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_relu);
    vx_enum inception_b4_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b4_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b4_7x1_2_reduce_relu_node;
    inception_b4_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b4_7x1_2_reduce_scale, inception_b4_7x1_2_reduce_relu_mode, inception_b4_7x1_2_reduce_relu_param_a, inception_b4_7x1_2_reduce_relu_param_b, inception_b4_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_reduce_relu_node));

    // inception_b4_7x1_2 Layer
    vx_size inception_b4_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2;
    inception_b4_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2);
    vx_size inception_b4_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b4_7x1_2_W;
    inception_b4_7x1_2_W = vxCreateTensor(context,4, inception_b4_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_W, dataFolder + "/weights/inception_b4_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_2_params;
    inception_b4_7x1_2_params.padding_x = 0;
    inception_b4_7x1_2_params.padding_y = 3;
    inception_b4_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_2_params.dilation_x = 0;
    inception_b4_7x1_2_params.dilation_y = 0;
    vx_node inception_b4_7x1_2_node;
    inception_b4_7x1_2_node = vxConvolutionLayer(graph, inception_b4_7x1_2_reduce_relu, inception_b4_7x1_2_W, NULL, &inception_b4_7x1_2_params, sizeof(inception_b4_7x1_2_params ), inception_b4_7x1_2);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_node));

    // inception_b4_7x1_2_bn Layer
    vx_size inception_b4_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_scale;
    inception_b4_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_scale);
    vx_size inception_b4_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b4_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_2_bn_W;
    inception_b4_7x1_2_bn_W = vxCreateTensor(context,1, inception_b4_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_bn_W, dataFolder + "/weights/inception_b4_7x1_2_bn.f32"));
    vx_size inception_b4_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_bn_B;
    inception_b4_7x1_2_bn_B = vxCreateTensor(context,1, inception_b4_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_bn_B, dataFolder + "/bias/inception_b4_7x1_2_bn.f32"));
    vx_size inception_b4_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_scale_W;
    inception_b4_7x1_2_scale_W = vxCreateTensor(context,1, inception_b4_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_scale_W, dataFolder + "/weights/inception_b4_7x1_2_scale.f32"));
    vx_size inception_b4_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b4_7x1_2_scale_B;
    inception_b4_7x1_2_scale_B = vxCreateTensor(context,1, inception_b4_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_2_scale_B, dataFolder + "/bias/inception_b4_7x1_2_scale.f32"));
    vx_node inception_b4_7x1_2_bn_node;
    inception_b4_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1_2, inception_b4_7x1_2_bn_W, inception_b4_7x1_2_bn_B, inception_b4_7x1_2_scale_W, inception_b4_7x1_2_scale_B, inception_b4_7x1_2_bn_eps, inception_b4_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_bn_node));

    // inception_b4_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_2_relu Layer
    vx_size inception_b4_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b4_7x1_2_relu;
    inception_b4_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_relu);
    vx_enum inception_b4_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_2_relu_param_a = 0;
    vx_float32 inception_b4_7x1_2_relu_param_b = 0;
    vx_node inception_b4_7x1_2_relu_node;
    inception_b4_7x1_2_relu_node = vxActivationLayer(graph, inception_b4_7x1_2_scale, inception_b4_7x1_2_relu_mode, inception_b4_7x1_2_relu_param_a, inception_b4_7x1_2_relu_param_b, inception_b4_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_2_relu_node));

    // inception_b4_1x7_2 Layer
    vx_size inception_b4_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_1x7_2;
    inception_b4_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b4_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2);
    vx_size inception_b4_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b4_1x7_2_W;
    inception_b4_1x7_2_W = vxCreateTensor(context,4, inception_b4_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_W, dataFolder + "/weights/inception_b4_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_2_params;
    inception_b4_1x7_2_params.padding_x = 3;
    inception_b4_1x7_2_params.padding_y = 0;
    inception_b4_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_2_params.dilation_x = 0;
    inception_b4_1x7_2_params.dilation_y = 0;
    vx_node inception_b4_1x7_2_node;
    inception_b4_1x7_2_node = vxConvolutionLayer(graph, inception_b4_7x1_2_relu, inception_b4_1x7_2_W, NULL, &inception_b4_1x7_2_params, sizeof(inception_b4_1x7_2_params ), inception_b4_1x7_2);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_2_node));

    // inception_b4_1x7_2_bn Layer
    vx_size inception_b4_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_1x7_2_scale;
    inception_b4_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_scale);
    vx_size inception_b4_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b4_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_2_bn_W;
    inception_b4_1x7_2_bn_W = vxCreateTensor(context,1, inception_b4_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_bn_W, dataFolder + "/weights/inception_b4_1x7_2_bn.f32"));
    vx_size inception_b4_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b4_1x7_2_bn_B;
    inception_b4_1x7_2_bn_B = vxCreateTensor(context,1, inception_b4_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_bn_B, dataFolder + "/bias/inception_b4_1x7_2_bn.f32"));
    vx_size inception_b4_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b4_1x7_2_scale_W;
    inception_b4_1x7_2_scale_W = vxCreateTensor(context,1, inception_b4_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_scale_W, dataFolder + "/weights/inception_b4_1x7_2_scale.f32"));
    vx_size inception_b4_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b4_1x7_2_scale_B;
    inception_b4_1x7_2_scale_B = vxCreateTensor(context,1, inception_b4_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_2_scale_B, dataFolder + "/bias/inception_b4_1x7_2_scale.f32"));
    vx_node inception_b4_1x7_2_bn_node;
    inception_b4_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7_2, inception_b4_1x7_2_bn_W, inception_b4_1x7_2_bn_B, inception_b4_1x7_2_scale_W, inception_b4_1x7_2_scale_B, inception_b4_1x7_2_bn_eps, inception_b4_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_2_bn_node));

    // inception_b4_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_2_relu Layer
    vx_size inception_b4_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_1x7_2_relu;
    inception_b4_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_relu);
    vx_enum inception_b4_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_2_relu_param_a = 0;
    vx_float32 inception_b4_1x7_2_relu_param_b = 0;
    vx_node inception_b4_1x7_2_relu_node;
    inception_b4_1x7_2_relu_node = vxActivationLayer(graph, inception_b4_1x7_2_scale, inception_b4_1x7_2_relu_mode, inception_b4_1x7_2_relu_param_a, inception_b4_1x7_2_relu_param_b, inception_b4_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_2_relu_node));

    // inception_b4_7x1_3 Layer
    vx_size inception_b4_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_7x1_3;
    inception_b4_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b4_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3);
    vx_size inception_b4_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b4_7x1_3_W;
    inception_b4_7x1_3_W = vxCreateTensor(context,4, inception_b4_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_W, dataFolder + "/weights/inception_b4_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b4_7x1_3_params;
    inception_b4_7x1_3_params.padding_x = 0;
    inception_b4_7x1_3_params.padding_y = 3;
    inception_b4_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_7x1_3_params.dilation_x = 0;
    inception_b4_7x1_3_params.dilation_y = 0;
    vx_node inception_b4_7x1_3_node;
    inception_b4_7x1_3_node = vxConvolutionLayer(graph, inception_b4_1x7_2_relu, inception_b4_7x1_3_W, NULL, &inception_b4_7x1_3_params, sizeof(inception_b4_7x1_3_params ), inception_b4_7x1_3);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_3_node));

    // inception_b4_7x1_3_bn Layer
    vx_size inception_b4_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_7x1_3_scale;
    inception_b4_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b4_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_scale);
    vx_size inception_b4_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b4_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b4_7x1_3_bn_W;
    inception_b4_7x1_3_bn_W = vxCreateTensor(context,1, inception_b4_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_bn_W, dataFolder + "/weights/inception_b4_7x1_3_bn.f32"));
    vx_size inception_b4_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b4_7x1_3_bn_B;
    inception_b4_7x1_3_bn_B = vxCreateTensor(context,1, inception_b4_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_bn_B, dataFolder + "/bias/inception_b4_7x1_3_bn.f32"));
    vx_size inception_b4_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b4_7x1_3_scale_W;
    inception_b4_7x1_3_scale_W = vxCreateTensor(context,1, inception_b4_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_scale_W, dataFolder + "/weights/inception_b4_7x1_3_scale.f32"));
    vx_size inception_b4_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b4_7x1_3_scale_B;
    inception_b4_7x1_3_scale_B = vxCreateTensor(context,1, inception_b4_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_7x1_3_scale_B, dataFolder + "/bias/inception_b4_7x1_3_scale.f32"));
    vx_node inception_b4_7x1_3_bn_node;
    inception_b4_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b4_7x1_3, inception_b4_7x1_3_bn_W, inception_b4_7x1_3_bn_B, inception_b4_7x1_3_scale_W, inception_b4_7x1_3_scale_B, inception_b4_7x1_3_bn_eps, inception_b4_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_3_bn_node));

    // inception_b4_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_7x1_3_relu Layer
    vx_size inception_b4_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b4_7x1_3_relu;
    inception_b4_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b4_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_relu);
    vx_enum inception_b4_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_7x1_3_relu_param_a = 0;
    vx_float32 inception_b4_7x1_3_relu_param_b = 0;
    vx_node inception_b4_7x1_3_relu_node;
    inception_b4_7x1_3_relu_node = vxActivationLayer(graph, inception_b4_7x1_3_scale, inception_b4_7x1_3_relu_mode, inception_b4_7x1_3_relu_param_a, inception_b4_7x1_3_relu_param_b, inception_b4_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b4_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_7x1_3_relu_node));

    // inception_b4_1x7_3 Layer
    vx_size inception_b4_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b4_1x7_3;
    inception_b4_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b4_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3);
    vx_size inception_b4_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b4_1x7_3_W;
    inception_b4_1x7_3_W = vxCreateTensor(context,4, inception_b4_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_W, dataFolder + "/weights/inception_b4_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b4_1x7_3_params;
    inception_b4_1x7_3_params.padding_x = 3;
    inception_b4_1x7_3_params.padding_y = 0;
    inception_b4_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x7_3_params.dilation_x = 0;
    inception_b4_1x7_3_params.dilation_y = 0;
    vx_node inception_b4_1x7_3_node;
    inception_b4_1x7_3_node = vxConvolutionLayer(graph, inception_b4_7x1_3_relu, inception_b4_1x7_3_W, NULL, &inception_b4_1x7_3_params, sizeof(inception_b4_1x7_3_params ), inception_b4_1x7_3);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_3_node));

    // inception_b4_1x7_3_bn Layer
    vx_size inception_b4_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b4_1x7_3_scale;
    inception_b4_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_scale);
    vx_size inception_b4_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b4_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b4_1x7_3_bn_W;
    inception_b4_1x7_3_bn_W = vxCreateTensor(context,1, inception_b4_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_bn_W, dataFolder + "/weights/inception_b4_1x7_3_bn.f32"));
    vx_size inception_b4_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b4_1x7_3_bn_B;
    inception_b4_1x7_3_bn_B = vxCreateTensor(context,1, inception_b4_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_bn_B, dataFolder + "/bias/inception_b4_1x7_3_bn.f32"));
    vx_size inception_b4_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b4_1x7_3_scale_W;
    inception_b4_1x7_3_scale_W = vxCreateTensor(context,1, inception_b4_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_scale_W, dataFolder + "/weights/inception_b4_1x7_3_scale.f32"));
    vx_size inception_b4_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b4_1x7_3_scale_B;
    inception_b4_1x7_3_scale_B = vxCreateTensor(context,1, inception_b4_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x7_3_scale_B, dataFolder + "/bias/inception_b4_1x7_3_scale.f32"));
    vx_node inception_b4_1x7_3_bn_node;
    inception_b4_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x7_3, inception_b4_1x7_3_bn_W, inception_b4_1x7_3_bn_B, inception_b4_1x7_3_scale_W, inception_b4_1x7_3_scale_B, inception_b4_1x7_3_bn_eps, inception_b4_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_3_bn_node));

    // inception_b4_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x7_3_relu Layer
    vx_size inception_b4_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b4_1x7_3_relu;
    inception_b4_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_relu);
    vx_enum inception_b4_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x7_3_relu_param_a = 0;
    vx_float32 inception_b4_1x7_3_relu_param_b = 0;
    vx_node inception_b4_1x7_3_relu_node;
    inception_b4_1x7_3_relu_node = vxActivationLayer(graph, inception_b4_1x7_3_scale, inception_b4_1x7_3_relu_mode, inception_b4_1x7_3_relu_param_a, inception_b4_1x7_3_relu_param_b, inception_b4_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x7_3_relu_node));

    // inception_b4_pool_ave Layer
    vx_size inception_b4_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b4_pool_ave;
    inception_b4_pool_ave = vxCreateVirtualTensor(graph,4, inception_b4_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_pool_ave);
    vx_enum inception_b4_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b4_pool_ave_kernel_w = 3;
    vx_size inception_b4_pool_ave_kernel_h = 3;
    vx_size inception_b4_pool_ave_pad_w = 1;
    vx_size inception_b4_pool_ave_pad_h = 1;
    vx_enum inception_b4_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b4_pool_ave_node;
    inception_b4_pool_ave_node = vxPoolingLayer(graph, inception_b3_concat, inception_b4_pool_ave_type, inception_b4_pool_ave_kernel_w, inception_b4_pool_ave_kernel_h, inception_b4_pool_ave_pad_w, inception_b4_pool_ave_pad_h, inception_b4_pool_ave_roundPolicy, inception_b4_pool_ave );
    ERROR_CHECK_OBJECT(inception_b4_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_pool_ave_node));

    // inception_b4_1x1 Layer
    vx_size inception_b4_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b4_1x1;
    inception_b4_1x1 = vxCreateVirtualTensor(graph,4, inception_b4_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1);
    vx_size inception_b4_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b4_1x1_W;
    inception_b4_1x1_W = vxCreateTensor(context,4, inception_b4_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_W, dataFolder + "/weights/inception_b4_1x1.f32"));
    vx_nn_convolution_params_t inception_b4_1x1_params;
    inception_b4_1x1_params.padding_x = 0;
    inception_b4_1x1_params.padding_y = 0;
    inception_b4_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b4_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b4_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b4_1x1_params.dilation_x = 0;
    inception_b4_1x1_params.dilation_y = 0;
    vx_node inception_b4_1x1_node;
    inception_b4_1x1_node = vxConvolutionLayer(graph, inception_b4_pool_ave, inception_b4_1x1_W, NULL, &inception_b4_1x1_params, sizeof(inception_b4_1x1_params ), inception_b4_1x1);
    ERROR_CHECK_OBJECT(inception_b4_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_node));

    // inception_b4_1x1_bn Layer
    vx_size inception_b4_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b4_1x1_scale;
    inception_b4_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b4_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_scale);
    vx_size inception_b4_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b4_1x1_bn_eps = 0.001;
    vx_tensor inception_b4_1x1_bn_W;
    inception_b4_1x1_bn_W = vxCreateTensor(context,1, inception_b4_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_bn_W, dataFolder + "/weights/inception_b4_1x1_bn.f32"));
    vx_size inception_b4_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b4_1x1_bn_B;
    inception_b4_1x1_bn_B = vxCreateTensor(context,1, inception_b4_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_bn_B, dataFolder + "/bias/inception_b4_1x1_bn.f32"));
    vx_size inception_b4_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b4_1x1_scale_W;
    inception_b4_1x1_scale_W = vxCreateTensor(context,1, inception_b4_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_scale_W, dataFolder + "/weights/inception_b4_1x1_scale.f32"));
    vx_size inception_b4_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b4_1x1_scale_B;
    inception_b4_1x1_scale_B = vxCreateTensor(context,1, inception_b4_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b4_1x1_scale_B, dataFolder + "/bias/inception_b4_1x1_scale.f32"));
    vx_node inception_b4_1x1_bn_node;
    inception_b4_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b4_1x1, inception_b4_1x1_bn_W, inception_b4_1x1_bn_B, inception_b4_1x1_scale_W, inception_b4_1x1_scale_B, inception_b4_1x1_bn_eps, inception_b4_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b4_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_bn_node));

    // inception_b4_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b4_1x1_relu Layer
    vx_size inception_b4_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b4_1x1_relu;
    inception_b4_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b4_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_1x1_relu);
    vx_enum inception_b4_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b4_1x1_relu_param_a = 0;
    vx_float32 inception_b4_1x1_relu_param_b = 0;
    vx_node inception_b4_1x1_relu_node;
    inception_b4_1x1_relu_node = vxActivationLayer(graph, inception_b4_1x1_scale, inception_b4_1x1_relu_mode, inception_b4_1x1_relu_param_a, inception_b4_1x1_relu_param_b, inception_b4_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b4_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_1x1_relu_node));

    // inception_b4_concat Layer
    vx_size inception_b4_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b4_concat;
    inception_b4_concat = vxCreateVirtualTensor(graph,4, inception_b4_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b4_concat);
    vx_node inception_b4_concat_node;
    inception_b4_concat_node = vxConcatLayer(graph, inception_b4_concat, inception_b4_1x1_2_relu, inception_b4_7x1_relu, inception_b4_1x7_3_relu, inception_b4_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b4_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b4_concat_node));

    // inception_b5_1x1_2 Layer
    vx_size inception_b5_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b5_1x1_2;
    inception_b5_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b5_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2);
    vx_size inception_b5_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b5_1x1_2_W;
    inception_b5_1x1_2_W = vxCreateTensor(context,4, inception_b5_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_2_W, dataFolder + "/weights/inception_b5_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b5_1x1_2_params;
    inception_b5_1x1_2_params.padding_x = 0;
    inception_b5_1x1_2_params.padding_y = 0;
    inception_b5_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_1x1_2_params.dilation_x = 0;
    inception_b5_1x1_2_params.dilation_y = 0;
    vx_node inception_b5_1x1_2_node;
    inception_b5_1x1_2_node = vxConvolutionLayer(graph, inception_b4_concat, inception_b5_1x1_2_W, NULL, &inception_b5_1x1_2_params, sizeof(inception_b5_1x1_2_params ), inception_b5_1x1_2);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x1_2_node));

    // inception_b5_1x1_2_bn Layer
    vx_size inception_b5_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b5_1x1_2_scale;
    inception_b5_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b5_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_scale);
    vx_size inception_b5_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b5_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b5_1x1_2_bn_W;
    inception_b5_1x1_2_bn_W = vxCreateTensor(context,1, inception_b5_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_2_bn_W, dataFolder + "/weights/inception_b5_1x1_2_bn.f32"));
    vx_size inception_b5_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b5_1x1_2_bn_B;
    inception_b5_1x1_2_bn_B = vxCreateTensor(context,1, inception_b5_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_2_bn_B, dataFolder + "/bias/inception_b5_1x1_2_bn.f32"));
    vx_size inception_b5_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b5_1x1_2_scale_W;
    inception_b5_1x1_2_scale_W = vxCreateTensor(context,1, inception_b5_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_2_scale_W, dataFolder + "/weights/inception_b5_1x1_2_scale.f32"));
    vx_size inception_b5_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b5_1x1_2_scale_B;
    inception_b5_1x1_2_scale_B = vxCreateTensor(context,1, inception_b5_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_2_scale_B, dataFolder + "/bias/inception_b5_1x1_2_scale.f32"));
    vx_node inception_b5_1x1_2_bn_node;
    inception_b5_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b5_1x1_2, inception_b5_1x1_2_bn_W, inception_b5_1x1_2_bn_B, inception_b5_1x1_2_scale_W, inception_b5_1x1_2_scale_B, inception_b5_1x1_2_bn_eps, inception_b5_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x1_2_bn_node));

    // inception_b5_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_1x1_2_relu Layer
    vx_size inception_b5_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b5_1x1_2_relu;
    inception_b5_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b5_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_relu);
    vx_enum inception_b5_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_1x1_2_relu_param_a = 0;
    vx_float32 inception_b5_1x1_2_relu_param_b = 0;
    vx_node inception_b5_1x1_2_relu_node;
    inception_b5_1x1_2_relu_node = vxActivationLayer(graph, inception_b5_1x1_2_scale, inception_b5_1x1_2_relu_mode, inception_b5_1x1_2_relu_param_a, inception_b5_1x1_2_relu_param_b, inception_b5_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b5_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x1_2_relu_node));

    // inception_b5_1x7_reduce Layer
    vx_size inception_b5_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_1x7_reduce;
    inception_b5_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b5_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce);
    vx_size inception_b5_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b5_1x7_reduce_W;
    inception_b5_1x7_reduce_W = vxCreateTensor(context,4, inception_b5_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_reduce_W, dataFolder + "/weights/inception_b5_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b5_1x7_reduce_params;
    inception_b5_1x7_reduce_params.padding_x = 0;
    inception_b5_1x7_reduce_params.padding_y = 0;
    inception_b5_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_1x7_reduce_params.dilation_x = 0;
    inception_b5_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b5_1x7_reduce_node;
    inception_b5_1x7_reduce_node = vxConvolutionLayer(graph, inception_b4_concat, inception_b5_1x7_reduce_W, NULL, &inception_b5_1x7_reduce_params, sizeof(inception_b5_1x7_reduce_params ), inception_b5_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_reduce_node));

    // inception_b5_1x7_reduce_bn Layer
    vx_size inception_b5_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_1x7_reduce_scale;
    inception_b5_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b5_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_scale);
    vx_size inception_b5_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b5_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b5_1x7_reduce_bn_W;
    inception_b5_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b5_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_reduce_bn_W, dataFolder + "/weights/inception_b5_1x7_reduce_bn.f32"));
    vx_size inception_b5_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b5_1x7_reduce_bn_B;
    inception_b5_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b5_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_reduce_bn_B, dataFolder + "/bias/inception_b5_1x7_reduce_bn.f32"));
    vx_size inception_b5_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b5_1x7_reduce_scale_W;
    inception_b5_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b5_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_reduce_scale_W, dataFolder + "/weights/inception_b5_1x7_reduce_scale.f32"));
    vx_size inception_b5_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b5_1x7_reduce_scale_B;
    inception_b5_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b5_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_reduce_scale_B, dataFolder + "/bias/inception_b5_1x7_reduce_scale.f32"));
    vx_node inception_b5_1x7_reduce_bn_node;
    inception_b5_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b5_1x7_reduce, inception_b5_1x7_reduce_bn_W, inception_b5_1x7_reduce_bn_B, inception_b5_1x7_reduce_scale_W, inception_b5_1x7_reduce_scale_B, inception_b5_1x7_reduce_bn_eps, inception_b5_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_reduce_bn_node));

    // inception_b5_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_1x7_reduce_relu Layer
    vx_size inception_b5_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_1x7_reduce_relu;
    inception_b5_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b5_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_relu);
    vx_enum inception_b5_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b5_1x7_reduce_relu_param_b = 0;
    vx_node inception_b5_1x7_reduce_relu_node;
    inception_b5_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b5_1x7_reduce_scale, inception_b5_1x7_reduce_relu_mode, inception_b5_1x7_reduce_relu_param_a, inception_b5_1x7_reduce_relu_param_b, inception_b5_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b5_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_reduce_relu_node));

    // inception_b5_1x7 Layer
    vx_size inception_b5_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_1x7;
    inception_b5_1x7 = vxCreateVirtualTensor(graph,4, inception_b5_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7);
    vx_size inception_b5_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b5_1x7_W;
    inception_b5_1x7_W = vxCreateTensor(context,4, inception_b5_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_W, dataFolder + "/weights/inception_b5_1x7.f32"));
    vx_nn_convolution_params_t inception_b5_1x7_params;
    inception_b5_1x7_params.padding_x = 3;
    inception_b5_1x7_params.padding_y = 0;
    inception_b5_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_1x7_params.dilation_x = 0;
    inception_b5_1x7_params.dilation_y = 0;
    vx_node inception_b5_1x7_node;
    inception_b5_1x7_node = vxConvolutionLayer(graph, inception_b5_1x7_reduce_relu, inception_b5_1x7_W, NULL, &inception_b5_1x7_params, sizeof(inception_b5_1x7_params ), inception_b5_1x7);
    ERROR_CHECK_OBJECT(inception_b5_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_node));

    // inception_b5_1x7_bn Layer
    vx_size inception_b5_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_1x7_scale;
    inception_b5_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b5_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_scale);
    vx_size inception_b5_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b5_1x7_bn_eps = 0.001;
    vx_tensor inception_b5_1x7_bn_W;
    inception_b5_1x7_bn_W = vxCreateTensor(context,1, inception_b5_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_bn_W, dataFolder + "/weights/inception_b5_1x7_bn.f32"));
    vx_size inception_b5_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b5_1x7_bn_B;
    inception_b5_1x7_bn_B = vxCreateTensor(context,1, inception_b5_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_bn_B, dataFolder + "/bias/inception_b5_1x7_bn.f32"));
    vx_size inception_b5_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b5_1x7_scale_W;
    inception_b5_1x7_scale_W = vxCreateTensor(context,1, inception_b5_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_scale_W, dataFolder + "/weights/inception_b5_1x7_scale.f32"));
    vx_size inception_b5_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b5_1x7_scale_B;
    inception_b5_1x7_scale_B = vxCreateTensor(context,1, inception_b5_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_scale_B, dataFolder + "/bias/inception_b5_1x7_scale.f32"));
    vx_node inception_b5_1x7_bn_node;
    inception_b5_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b5_1x7, inception_b5_1x7_bn_W, inception_b5_1x7_bn_B, inception_b5_1x7_scale_W, inception_b5_1x7_scale_B, inception_b5_1x7_bn_eps, inception_b5_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b5_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_bn_node));

    // inception_b5_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_1x7_relu Layer
    vx_size inception_b5_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_1x7_relu;
    inception_b5_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b5_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_relu);
    vx_enum inception_b5_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_1x7_relu_param_a = 0;
    vx_float32 inception_b5_1x7_relu_param_b = 0;
    vx_node inception_b5_1x7_relu_node;
    inception_b5_1x7_relu_node = vxActivationLayer(graph, inception_b5_1x7_scale, inception_b5_1x7_relu_mode, inception_b5_1x7_relu_param_a, inception_b5_1x7_relu_param_b, inception_b5_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b5_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_relu_node));

    // inception_b5_7x1 Layer
    vx_size inception_b5_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b5_7x1;
    inception_b5_7x1 = vxCreateVirtualTensor(graph,4, inception_b5_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1);
    vx_size inception_b5_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b5_7x1_W;
    inception_b5_7x1_W = vxCreateTensor(context,4, inception_b5_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_W, dataFolder + "/weights/inception_b5_7x1.f32"));
    vx_nn_convolution_params_t inception_b5_7x1_params;
    inception_b5_7x1_params.padding_x = 0;
    inception_b5_7x1_params.padding_y = 3;
    inception_b5_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_7x1_params.dilation_x = 0;
    inception_b5_7x1_params.dilation_y = 0;
    vx_node inception_b5_7x1_node;
    inception_b5_7x1_node = vxConvolutionLayer(graph, inception_b5_1x7_relu, inception_b5_7x1_W, NULL, &inception_b5_7x1_params, sizeof(inception_b5_7x1_params ), inception_b5_7x1);
    ERROR_CHECK_OBJECT(inception_b5_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_node));

    // inception_b5_7x1_bn Layer
    vx_size inception_b5_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b5_7x1_scale;
    inception_b5_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b5_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_scale);
    vx_size inception_b5_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b5_7x1_bn_eps = 0.001;
    vx_tensor inception_b5_7x1_bn_W;
    inception_b5_7x1_bn_W = vxCreateTensor(context,1, inception_b5_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_bn_W, dataFolder + "/weights/inception_b5_7x1_bn.f32"));
    vx_size inception_b5_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b5_7x1_bn_B;
    inception_b5_7x1_bn_B = vxCreateTensor(context,1, inception_b5_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_bn_B, dataFolder + "/bias/inception_b5_7x1_bn.f32"));
    vx_size inception_b5_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b5_7x1_scale_W;
    inception_b5_7x1_scale_W = vxCreateTensor(context,1, inception_b5_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_scale_W, dataFolder + "/weights/inception_b5_7x1_scale.f32"));
    vx_size inception_b5_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b5_7x1_scale_B;
    inception_b5_7x1_scale_B = vxCreateTensor(context,1, inception_b5_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_scale_B, dataFolder + "/bias/inception_b5_7x1_scale.f32"));
    vx_node inception_b5_7x1_bn_node;
    inception_b5_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b5_7x1, inception_b5_7x1_bn_W, inception_b5_7x1_bn_B, inception_b5_7x1_scale_W, inception_b5_7x1_scale_B, inception_b5_7x1_bn_eps, inception_b5_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b5_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_bn_node));

    // inception_b5_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_7x1_relu Layer
    vx_size inception_b5_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b5_7x1_relu;
    inception_b5_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b5_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_relu);
    vx_enum inception_b5_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_7x1_relu_param_a = 0;
    vx_float32 inception_b5_7x1_relu_param_b = 0;
    vx_node inception_b5_7x1_relu_node;
    inception_b5_7x1_relu_node = vxActivationLayer(graph, inception_b5_7x1_scale, inception_b5_7x1_relu_mode, inception_b5_7x1_relu_param_a, inception_b5_7x1_relu_param_b, inception_b5_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b5_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_relu_node));

    // inception_b5_7x1_2_reduce Layer
    vx_size inception_b5_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_7x1_2_reduce;
    inception_b5_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b5_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce);
    vx_size inception_b5_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b5_7x1_2_reduce_W;
    inception_b5_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b5_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_reduce_W, dataFolder + "/weights/inception_b5_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b5_7x1_2_reduce_params;
    inception_b5_7x1_2_reduce_params.padding_x = 0;
    inception_b5_7x1_2_reduce_params.padding_y = 0;
    inception_b5_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_7x1_2_reduce_params.dilation_x = 0;
    inception_b5_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b5_7x1_2_reduce_node;
    inception_b5_7x1_2_reduce_node = vxConvolutionLayer(graph, inception_b4_concat, inception_b5_7x1_2_reduce_W, NULL, &inception_b5_7x1_2_reduce_params, sizeof(inception_b5_7x1_2_reduce_params ), inception_b5_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_2_reduce_node));

    // inception_b5_7x1_2_reduce_bn Layer
    vx_size inception_b5_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_7x1_2_reduce_scale;
    inception_b5_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b5_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_scale);
    vx_size inception_b5_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b5_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b5_7x1_2_reduce_bn_W;
    inception_b5_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b5_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b5_7x1_2_reduce_bn.f32"));
    vx_size inception_b5_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b5_7x1_2_reduce_bn_B;
    inception_b5_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b5_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b5_7x1_2_reduce_bn.f32"));
    vx_size inception_b5_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b5_7x1_2_reduce_scale_W;
    inception_b5_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b5_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b5_7x1_2_reduce_scale.f32"));
    vx_size inception_b5_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b5_7x1_2_reduce_scale_B;
    inception_b5_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b5_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b5_7x1_2_reduce_scale.f32"));
    vx_node inception_b5_7x1_2_reduce_bn_node;
    inception_b5_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b5_7x1_2_reduce, inception_b5_7x1_2_reduce_bn_W, inception_b5_7x1_2_reduce_bn_B, inception_b5_7x1_2_reduce_scale_W, inception_b5_7x1_2_reduce_scale_B, inception_b5_7x1_2_reduce_bn_eps, inception_b5_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_2_reduce_bn_node));

    // inception_b5_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_7x1_2_reduce_relu Layer
    vx_size inception_b5_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_7x1_2_reduce_relu;
    inception_b5_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b5_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_relu);
    vx_enum inception_b5_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b5_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b5_7x1_2_reduce_relu_node;
    inception_b5_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b5_7x1_2_reduce_scale, inception_b5_7x1_2_reduce_relu_mode, inception_b5_7x1_2_reduce_relu_param_a, inception_b5_7x1_2_reduce_relu_param_b, inception_b5_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_2_reduce_relu_node));

    // inception_b5_7x1_2 Layer
    vx_size inception_b5_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_7x1_2;
    inception_b5_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b5_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2);
    vx_size inception_b5_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b5_7x1_2_W;
    inception_b5_7x1_2_W = vxCreateTensor(context,4, inception_b5_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_W, dataFolder + "/weights/inception_b5_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b5_7x1_2_params;
    inception_b5_7x1_2_params.padding_x = 0;
    inception_b5_7x1_2_params.padding_y = 3;
    inception_b5_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_7x1_2_params.dilation_x = 0;
    inception_b5_7x1_2_params.dilation_y = 0;
    vx_node inception_b5_7x1_2_node;
    inception_b5_7x1_2_node = vxConvolutionLayer(graph, inception_b5_7x1_2_reduce_relu, inception_b5_7x1_2_W, NULL, &inception_b5_7x1_2_params, sizeof(inception_b5_7x1_2_params ), inception_b5_7x1_2);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_2_node));

    // inception_b5_7x1_2_bn Layer
    vx_size inception_b5_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_7x1_2_scale;
    inception_b5_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b5_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_scale);
    vx_size inception_b5_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b5_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b5_7x1_2_bn_W;
    inception_b5_7x1_2_bn_W = vxCreateTensor(context,1, inception_b5_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_bn_W, dataFolder + "/weights/inception_b5_7x1_2_bn.f32"));
    vx_size inception_b5_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b5_7x1_2_bn_B;
    inception_b5_7x1_2_bn_B = vxCreateTensor(context,1, inception_b5_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_bn_B, dataFolder + "/bias/inception_b5_7x1_2_bn.f32"));
    vx_size inception_b5_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b5_7x1_2_scale_W;
    inception_b5_7x1_2_scale_W = vxCreateTensor(context,1, inception_b5_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_scale_W, dataFolder + "/weights/inception_b5_7x1_2_scale.f32"));
    vx_size inception_b5_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b5_7x1_2_scale_B;
    inception_b5_7x1_2_scale_B = vxCreateTensor(context,1, inception_b5_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_2_scale_B, dataFolder + "/bias/inception_b5_7x1_2_scale.f32"));
    vx_node inception_b5_7x1_2_bn_node;
    inception_b5_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b5_7x1_2, inception_b5_7x1_2_bn_W, inception_b5_7x1_2_bn_B, inception_b5_7x1_2_scale_W, inception_b5_7x1_2_scale_B, inception_b5_7x1_2_bn_eps, inception_b5_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_2_bn_node));

    // inception_b5_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_7x1_2_relu Layer
    vx_size inception_b5_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b5_7x1_2_relu;
    inception_b5_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b5_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_relu);
    vx_enum inception_b5_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_7x1_2_relu_param_a = 0;
    vx_float32 inception_b5_7x1_2_relu_param_b = 0;
    vx_node inception_b5_7x1_2_relu_node;
    inception_b5_7x1_2_relu_node = vxActivationLayer(graph, inception_b5_7x1_2_scale, inception_b5_7x1_2_relu_mode, inception_b5_7x1_2_relu_param_a, inception_b5_7x1_2_relu_param_b, inception_b5_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b5_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_2_relu_node));

    // inception_b5_1x7_2 Layer
    vx_size inception_b5_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_1x7_2;
    inception_b5_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b5_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2);
    vx_size inception_b5_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b5_1x7_2_W;
    inception_b5_1x7_2_W = vxCreateTensor(context,4, inception_b5_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_2_W, dataFolder + "/weights/inception_b5_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b5_1x7_2_params;
    inception_b5_1x7_2_params.padding_x = 3;
    inception_b5_1x7_2_params.padding_y = 0;
    inception_b5_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_1x7_2_params.dilation_x = 0;
    inception_b5_1x7_2_params.dilation_y = 0;
    vx_node inception_b5_1x7_2_node;
    inception_b5_1x7_2_node = vxConvolutionLayer(graph, inception_b5_7x1_2_relu, inception_b5_1x7_2_W, NULL, &inception_b5_1x7_2_params, sizeof(inception_b5_1x7_2_params ), inception_b5_1x7_2);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_2_node));

    // inception_b5_1x7_2_bn Layer
    vx_size inception_b5_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_1x7_2_scale;
    inception_b5_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b5_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_scale);
    vx_size inception_b5_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b5_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b5_1x7_2_bn_W;
    inception_b5_1x7_2_bn_W = vxCreateTensor(context,1, inception_b5_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_2_bn_W, dataFolder + "/weights/inception_b5_1x7_2_bn.f32"));
    vx_size inception_b5_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b5_1x7_2_bn_B;
    inception_b5_1x7_2_bn_B = vxCreateTensor(context,1, inception_b5_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_2_bn_B, dataFolder + "/bias/inception_b5_1x7_2_bn.f32"));
    vx_size inception_b5_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b5_1x7_2_scale_W;
    inception_b5_1x7_2_scale_W = vxCreateTensor(context,1, inception_b5_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_2_scale_W, dataFolder + "/weights/inception_b5_1x7_2_scale.f32"));
    vx_size inception_b5_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b5_1x7_2_scale_B;
    inception_b5_1x7_2_scale_B = vxCreateTensor(context,1, inception_b5_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_2_scale_B, dataFolder + "/bias/inception_b5_1x7_2_scale.f32"));
    vx_node inception_b5_1x7_2_bn_node;
    inception_b5_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b5_1x7_2, inception_b5_1x7_2_bn_W, inception_b5_1x7_2_bn_B, inception_b5_1x7_2_scale_W, inception_b5_1x7_2_scale_B, inception_b5_1x7_2_bn_eps, inception_b5_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_2_bn_node));

    // inception_b5_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_1x7_2_relu Layer
    vx_size inception_b5_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_1x7_2_relu;
    inception_b5_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b5_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_relu);
    vx_enum inception_b5_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_1x7_2_relu_param_a = 0;
    vx_float32 inception_b5_1x7_2_relu_param_b = 0;
    vx_node inception_b5_1x7_2_relu_node;
    inception_b5_1x7_2_relu_node = vxActivationLayer(graph, inception_b5_1x7_2_scale, inception_b5_1x7_2_relu_mode, inception_b5_1x7_2_relu_param_a, inception_b5_1x7_2_relu_param_b, inception_b5_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b5_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_2_relu_node));

    // inception_b5_7x1_3 Layer
    vx_size inception_b5_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_7x1_3;
    inception_b5_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b5_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3);
    vx_size inception_b5_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b5_7x1_3_W;
    inception_b5_7x1_3_W = vxCreateTensor(context,4, inception_b5_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_3_W, dataFolder + "/weights/inception_b5_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b5_7x1_3_params;
    inception_b5_7x1_3_params.padding_x = 0;
    inception_b5_7x1_3_params.padding_y = 3;
    inception_b5_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_7x1_3_params.dilation_x = 0;
    inception_b5_7x1_3_params.dilation_y = 0;
    vx_node inception_b5_7x1_3_node;
    inception_b5_7x1_3_node = vxConvolutionLayer(graph, inception_b5_1x7_2_relu, inception_b5_7x1_3_W, NULL, &inception_b5_7x1_3_params, sizeof(inception_b5_7x1_3_params ), inception_b5_7x1_3);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_3_node));

    // inception_b5_7x1_3_bn Layer
    vx_size inception_b5_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_7x1_3_scale;
    inception_b5_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b5_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_scale);
    vx_size inception_b5_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b5_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b5_7x1_3_bn_W;
    inception_b5_7x1_3_bn_W = vxCreateTensor(context,1, inception_b5_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_3_bn_W, dataFolder + "/weights/inception_b5_7x1_3_bn.f32"));
    vx_size inception_b5_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b5_7x1_3_bn_B;
    inception_b5_7x1_3_bn_B = vxCreateTensor(context,1, inception_b5_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_3_bn_B, dataFolder + "/bias/inception_b5_7x1_3_bn.f32"));
    vx_size inception_b5_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b5_7x1_3_scale_W;
    inception_b5_7x1_3_scale_W = vxCreateTensor(context,1, inception_b5_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_3_scale_W, dataFolder + "/weights/inception_b5_7x1_3_scale.f32"));
    vx_size inception_b5_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b5_7x1_3_scale_B;
    inception_b5_7x1_3_scale_B = vxCreateTensor(context,1, inception_b5_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_7x1_3_scale_B, dataFolder + "/bias/inception_b5_7x1_3_scale.f32"));
    vx_node inception_b5_7x1_3_bn_node;
    inception_b5_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b5_7x1_3, inception_b5_7x1_3_bn_W, inception_b5_7x1_3_bn_B, inception_b5_7x1_3_scale_W, inception_b5_7x1_3_scale_B, inception_b5_7x1_3_bn_eps, inception_b5_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_3_bn_node));

    // inception_b5_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_7x1_3_relu Layer
    vx_size inception_b5_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b5_7x1_3_relu;
    inception_b5_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b5_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_relu);
    vx_enum inception_b5_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_7x1_3_relu_param_a = 0;
    vx_float32 inception_b5_7x1_3_relu_param_b = 0;
    vx_node inception_b5_7x1_3_relu_node;
    inception_b5_7x1_3_relu_node = vxActivationLayer(graph, inception_b5_7x1_3_scale, inception_b5_7x1_3_relu_mode, inception_b5_7x1_3_relu_param_a, inception_b5_7x1_3_relu_param_b, inception_b5_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b5_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_7x1_3_relu_node));

    // inception_b5_1x7_3 Layer
    vx_size inception_b5_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b5_1x7_3;
    inception_b5_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b5_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3);
    vx_size inception_b5_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b5_1x7_3_W;
    inception_b5_1x7_3_W = vxCreateTensor(context,4, inception_b5_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_3_W, dataFolder + "/weights/inception_b5_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b5_1x7_3_params;
    inception_b5_1x7_3_params.padding_x = 3;
    inception_b5_1x7_3_params.padding_y = 0;
    inception_b5_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_1x7_3_params.dilation_x = 0;
    inception_b5_1x7_3_params.dilation_y = 0;
    vx_node inception_b5_1x7_3_node;
    inception_b5_1x7_3_node = vxConvolutionLayer(graph, inception_b5_7x1_3_relu, inception_b5_1x7_3_W, NULL, &inception_b5_1x7_3_params, sizeof(inception_b5_1x7_3_params ), inception_b5_1x7_3);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_3_node));

    // inception_b5_1x7_3_bn Layer
    vx_size inception_b5_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b5_1x7_3_scale;
    inception_b5_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b5_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_scale);
    vx_size inception_b5_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b5_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b5_1x7_3_bn_W;
    inception_b5_1x7_3_bn_W = vxCreateTensor(context,1, inception_b5_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_3_bn_W, dataFolder + "/weights/inception_b5_1x7_3_bn.f32"));
    vx_size inception_b5_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b5_1x7_3_bn_B;
    inception_b5_1x7_3_bn_B = vxCreateTensor(context,1, inception_b5_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_3_bn_B, dataFolder + "/bias/inception_b5_1x7_3_bn.f32"));
    vx_size inception_b5_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b5_1x7_3_scale_W;
    inception_b5_1x7_3_scale_W = vxCreateTensor(context,1, inception_b5_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_3_scale_W, dataFolder + "/weights/inception_b5_1x7_3_scale.f32"));
    vx_size inception_b5_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b5_1x7_3_scale_B;
    inception_b5_1x7_3_scale_B = vxCreateTensor(context,1, inception_b5_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x7_3_scale_B, dataFolder + "/bias/inception_b5_1x7_3_scale.f32"));
    vx_node inception_b5_1x7_3_bn_node;
    inception_b5_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b5_1x7_3, inception_b5_1x7_3_bn_W, inception_b5_1x7_3_bn_B, inception_b5_1x7_3_scale_W, inception_b5_1x7_3_scale_B, inception_b5_1x7_3_bn_eps, inception_b5_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_3_bn_node));

    // inception_b5_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_1x7_3_relu Layer
    vx_size inception_b5_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b5_1x7_3_relu;
    inception_b5_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b5_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_relu);
    vx_enum inception_b5_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_1x7_3_relu_param_a = 0;
    vx_float32 inception_b5_1x7_3_relu_param_b = 0;
    vx_node inception_b5_1x7_3_relu_node;
    inception_b5_1x7_3_relu_node = vxActivationLayer(graph, inception_b5_1x7_3_scale, inception_b5_1x7_3_relu_mode, inception_b5_1x7_3_relu_param_a, inception_b5_1x7_3_relu_param_b, inception_b5_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b5_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x7_3_relu_node));

    // inception_b5_pool_ave Layer
    vx_size inception_b5_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b5_pool_ave;
    inception_b5_pool_ave = vxCreateVirtualTensor(graph,4, inception_b5_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_pool_ave);
    vx_enum inception_b5_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b5_pool_ave_kernel_w = 3;
    vx_size inception_b5_pool_ave_kernel_h = 3;
    vx_size inception_b5_pool_ave_pad_w = 1;
    vx_size inception_b5_pool_ave_pad_h = 1;
    vx_enum inception_b5_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b5_pool_ave_node;
    inception_b5_pool_ave_node = vxPoolingLayer(graph, inception_b4_concat, inception_b5_pool_ave_type, inception_b5_pool_ave_kernel_w, inception_b5_pool_ave_kernel_h, inception_b5_pool_ave_pad_w, inception_b5_pool_ave_pad_h, inception_b5_pool_ave_roundPolicy, inception_b5_pool_ave );
    ERROR_CHECK_OBJECT(inception_b5_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_pool_ave_node));

    // inception_b5_1x1 Layer
    vx_size inception_b5_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b5_1x1;
    inception_b5_1x1 = vxCreateVirtualTensor(graph,4, inception_b5_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x1);
    vx_size inception_b5_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b5_1x1_W;
    inception_b5_1x1_W = vxCreateTensor(context,4, inception_b5_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_W, dataFolder + "/weights/inception_b5_1x1.f32"));
    vx_nn_convolution_params_t inception_b5_1x1_params;
    inception_b5_1x1_params.padding_x = 0;
    inception_b5_1x1_params.padding_y = 0;
    inception_b5_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b5_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b5_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b5_1x1_params.dilation_x = 0;
    inception_b5_1x1_params.dilation_y = 0;
    vx_node inception_b5_1x1_node;
    inception_b5_1x1_node = vxConvolutionLayer(graph, inception_b5_pool_ave, inception_b5_1x1_W, NULL, &inception_b5_1x1_params, sizeof(inception_b5_1x1_params ), inception_b5_1x1);
    ERROR_CHECK_OBJECT(inception_b5_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x1_node));

    // inception_b5_1x1_bn Layer
    vx_size inception_b5_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b5_1x1_scale;
    inception_b5_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b5_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_scale);
    vx_size inception_b5_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b5_1x1_bn_eps = 0.001;
    vx_tensor inception_b5_1x1_bn_W;
    inception_b5_1x1_bn_W = vxCreateTensor(context,1, inception_b5_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_bn_W, dataFolder + "/weights/inception_b5_1x1_bn.f32"));
    vx_size inception_b5_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b5_1x1_bn_B;
    inception_b5_1x1_bn_B = vxCreateTensor(context,1, inception_b5_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_bn_B, dataFolder + "/bias/inception_b5_1x1_bn.f32"));
    vx_size inception_b5_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b5_1x1_scale_W;
    inception_b5_1x1_scale_W = vxCreateTensor(context,1, inception_b5_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_scale_W, dataFolder + "/weights/inception_b5_1x1_scale.f32"));
    vx_size inception_b5_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b5_1x1_scale_B;
    inception_b5_1x1_scale_B = vxCreateTensor(context,1, inception_b5_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b5_1x1_scale_B, dataFolder + "/bias/inception_b5_1x1_scale.f32"));
    vx_node inception_b5_1x1_bn_node;
    inception_b5_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b5_1x1, inception_b5_1x1_bn_W, inception_b5_1x1_bn_B, inception_b5_1x1_scale_W, inception_b5_1x1_scale_B, inception_b5_1x1_bn_eps, inception_b5_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b5_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x1_bn_node));

    // inception_b5_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b5_1x1_relu Layer
    vx_size inception_b5_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b5_1x1_relu;
    inception_b5_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b5_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_1x1_relu);
    vx_enum inception_b5_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b5_1x1_relu_param_a = 0;
    vx_float32 inception_b5_1x1_relu_param_b = 0;
    vx_node inception_b5_1x1_relu_node;
    inception_b5_1x1_relu_node = vxActivationLayer(graph, inception_b5_1x1_scale, inception_b5_1x1_relu_mode, inception_b5_1x1_relu_param_a, inception_b5_1x1_relu_param_b, inception_b5_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b5_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_1x1_relu_node));

    // inception_b5_concat Layer
    vx_size inception_b5_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b5_concat;
    inception_b5_concat = vxCreateVirtualTensor(graph,4, inception_b5_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b5_concat);
    vx_node inception_b5_concat_node;
    inception_b5_concat_node = vxConcatLayer(graph, inception_b5_concat, inception_b5_1x1_2_relu, inception_b5_7x1_relu, inception_b5_1x7_3_relu, inception_b5_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b5_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b5_concat_node));

    // inception_b6_1x1_2 Layer
    vx_size inception_b6_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b6_1x1_2;
    inception_b6_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b6_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2);
    vx_size inception_b6_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b6_1x1_2_W;
    inception_b6_1x1_2_W = vxCreateTensor(context,4, inception_b6_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_2_W, dataFolder + "/weights/inception_b6_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b6_1x1_2_params;
    inception_b6_1x1_2_params.padding_x = 0;
    inception_b6_1x1_2_params.padding_y = 0;
    inception_b6_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_1x1_2_params.dilation_x = 0;
    inception_b6_1x1_2_params.dilation_y = 0;
    vx_node inception_b6_1x1_2_node;
    inception_b6_1x1_2_node = vxConvolutionLayer(graph, inception_b5_concat, inception_b6_1x1_2_W, NULL, &inception_b6_1x1_2_params, sizeof(inception_b6_1x1_2_params ), inception_b6_1x1_2);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x1_2_node));

    // inception_b6_1x1_2_bn Layer
    vx_size inception_b6_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b6_1x1_2_scale;
    inception_b6_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b6_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_scale);
    vx_size inception_b6_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b6_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b6_1x1_2_bn_W;
    inception_b6_1x1_2_bn_W = vxCreateTensor(context,1, inception_b6_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_2_bn_W, dataFolder + "/weights/inception_b6_1x1_2_bn.f32"));
    vx_size inception_b6_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b6_1x1_2_bn_B;
    inception_b6_1x1_2_bn_B = vxCreateTensor(context,1, inception_b6_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_2_bn_B, dataFolder + "/bias/inception_b6_1x1_2_bn.f32"));
    vx_size inception_b6_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b6_1x1_2_scale_W;
    inception_b6_1x1_2_scale_W = vxCreateTensor(context,1, inception_b6_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_2_scale_W, dataFolder + "/weights/inception_b6_1x1_2_scale.f32"));
    vx_size inception_b6_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b6_1x1_2_scale_B;
    inception_b6_1x1_2_scale_B = vxCreateTensor(context,1, inception_b6_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_2_scale_B, dataFolder + "/bias/inception_b6_1x1_2_scale.f32"));
    vx_node inception_b6_1x1_2_bn_node;
    inception_b6_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b6_1x1_2, inception_b6_1x1_2_bn_W, inception_b6_1x1_2_bn_B, inception_b6_1x1_2_scale_W, inception_b6_1x1_2_scale_B, inception_b6_1x1_2_bn_eps, inception_b6_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x1_2_bn_node));

    // inception_b6_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_1x1_2_relu Layer
    vx_size inception_b6_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b6_1x1_2_relu;
    inception_b6_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b6_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_relu);
    vx_enum inception_b6_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_1x1_2_relu_param_a = 0;
    vx_float32 inception_b6_1x1_2_relu_param_b = 0;
    vx_node inception_b6_1x1_2_relu_node;
    inception_b6_1x1_2_relu_node = vxActivationLayer(graph, inception_b6_1x1_2_scale, inception_b6_1x1_2_relu_mode, inception_b6_1x1_2_relu_param_a, inception_b6_1x1_2_relu_param_b, inception_b6_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b6_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x1_2_relu_node));

    // inception_b6_1x7_reduce Layer
    vx_size inception_b6_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_1x7_reduce;
    inception_b6_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b6_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce);
    vx_size inception_b6_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b6_1x7_reduce_W;
    inception_b6_1x7_reduce_W = vxCreateTensor(context,4, inception_b6_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_reduce_W, dataFolder + "/weights/inception_b6_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b6_1x7_reduce_params;
    inception_b6_1x7_reduce_params.padding_x = 0;
    inception_b6_1x7_reduce_params.padding_y = 0;
    inception_b6_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_1x7_reduce_params.dilation_x = 0;
    inception_b6_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b6_1x7_reduce_node;
    inception_b6_1x7_reduce_node = vxConvolutionLayer(graph, inception_b5_concat, inception_b6_1x7_reduce_W, NULL, &inception_b6_1x7_reduce_params, sizeof(inception_b6_1x7_reduce_params ), inception_b6_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_reduce_node));

    // inception_b6_1x7_reduce_bn Layer
    vx_size inception_b6_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_1x7_reduce_scale;
    inception_b6_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b6_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_scale);
    vx_size inception_b6_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b6_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b6_1x7_reduce_bn_W;
    inception_b6_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b6_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_reduce_bn_W, dataFolder + "/weights/inception_b6_1x7_reduce_bn.f32"));
    vx_size inception_b6_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b6_1x7_reduce_bn_B;
    inception_b6_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b6_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_reduce_bn_B, dataFolder + "/bias/inception_b6_1x7_reduce_bn.f32"));
    vx_size inception_b6_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b6_1x7_reduce_scale_W;
    inception_b6_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b6_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_reduce_scale_W, dataFolder + "/weights/inception_b6_1x7_reduce_scale.f32"));
    vx_size inception_b6_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b6_1x7_reduce_scale_B;
    inception_b6_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b6_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_reduce_scale_B, dataFolder + "/bias/inception_b6_1x7_reduce_scale.f32"));
    vx_node inception_b6_1x7_reduce_bn_node;
    inception_b6_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b6_1x7_reduce, inception_b6_1x7_reduce_bn_W, inception_b6_1x7_reduce_bn_B, inception_b6_1x7_reduce_scale_W, inception_b6_1x7_reduce_scale_B, inception_b6_1x7_reduce_bn_eps, inception_b6_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_reduce_bn_node));

    // inception_b6_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_1x7_reduce_relu Layer
    vx_size inception_b6_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_1x7_reduce_relu;
    inception_b6_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b6_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_relu);
    vx_enum inception_b6_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b6_1x7_reduce_relu_param_b = 0;
    vx_node inception_b6_1x7_reduce_relu_node;
    inception_b6_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b6_1x7_reduce_scale, inception_b6_1x7_reduce_relu_mode, inception_b6_1x7_reduce_relu_param_a, inception_b6_1x7_reduce_relu_param_b, inception_b6_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b6_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_reduce_relu_node));

    // inception_b6_1x7 Layer
    vx_size inception_b6_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_1x7;
    inception_b6_1x7 = vxCreateVirtualTensor(graph,4, inception_b6_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7);
    vx_size inception_b6_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b6_1x7_W;
    inception_b6_1x7_W = vxCreateTensor(context,4, inception_b6_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_W, dataFolder + "/weights/inception_b6_1x7.f32"));
    vx_nn_convolution_params_t inception_b6_1x7_params;
    inception_b6_1x7_params.padding_x = 3;
    inception_b6_1x7_params.padding_y = 0;
    inception_b6_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_1x7_params.dilation_x = 0;
    inception_b6_1x7_params.dilation_y = 0;
    vx_node inception_b6_1x7_node;
    inception_b6_1x7_node = vxConvolutionLayer(graph, inception_b6_1x7_reduce_relu, inception_b6_1x7_W, NULL, &inception_b6_1x7_params, sizeof(inception_b6_1x7_params ), inception_b6_1x7);
    ERROR_CHECK_OBJECT(inception_b6_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_node));

    // inception_b6_1x7_bn Layer
    vx_size inception_b6_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_1x7_scale;
    inception_b6_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b6_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_scale);
    vx_size inception_b6_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b6_1x7_bn_eps = 0.001;
    vx_tensor inception_b6_1x7_bn_W;
    inception_b6_1x7_bn_W = vxCreateTensor(context,1, inception_b6_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_bn_W, dataFolder + "/weights/inception_b6_1x7_bn.f32"));
    vx_size inception_b6_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b6_1x7_bn_B;
    inception_b6_1x7_bn_B = vxCreateTensor(context,1, inception_b6_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_bn_B, dataFolder + "/bias/inception_b6_1x7_bn.f32"));
    vx_size inception_b6_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b6_1x7_scale_W;
    inception_b6_1x7_scale_W = vxCreateTensor(context,1, inception_b6_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_scale_W, dataFolder + "/weights/inception_b6_1x7_scale.f32"));
    vx_size inception_b6_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b6_1x7_scale_B;
    inception_b6_1x7_scale_B = vxCreateTensor(context,1, inception_b6_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_scale_B, dataFolder + "/bias/inception_b6_1x7_scale.f32"));
    vx_node inception_b6_1x7_bn_node;
    inception_b6_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b6_1x7, inception_b6_1x7_bn_W, inception_b6_1x7_bn_B, inception_b6_1x7_scale_W, inception_b6_1x7_scale_B, inception_b6_1x7_bn_eps, inception_b6_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b6_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_bn_node));

    // inception_b6_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_1x7_relu Layer
    vx_size inception_b6_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_1x7_relu;
    inception_b6_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b6_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_relu);
    vx_enum inception_b6_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_1x7_relu_param_a = 0;
    vx_float32 inception_b6_1x7_relu_param_b = 0;
    vx_node inception_b6_1x7_relu_node;
    inception_b6_1x7_relu_node = vxActivationLayer(graph, inception_b6_1x7_scale, inception_b6_1x7_relu_mode, inception_b6_1x7_relu_param_a, inception_b6_1x7_relu_param_b, inception_b6_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b6_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_relu_node));

    // inception_b6_7x1 Layer
    vx_size inception_b6_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b6_7x1;
    inception_b6_7x1 = vxCreateVirtualTensor(graph,4, inception_b6_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1);
    vx_size inception_b6_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b6_7x1_W;
    inception_b6_7x1_W = vxCreateTensor(context,4, inception_b6_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_W, dataFolder + "/weights/inception_b6_7x1.f32"));
    vx_nn_convolution_params_t inception_b6_7x1_params;
    inception_b6_7x1_params.padding_x = 0;
    inception_b6_7x1_params.padding_y = 3;
    inception_b6_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_7x1_params.dilation_x = 0;
    inception_b6_7x1_params.dilation_y = 0;
    vx_node inception_b6_7x1_node;
    inception_b6_7x1_node = vxConvolutionLayer(graph, inception_b6_1x7_relu, inception_b6_7x1_W, NULL, &inception_b6_7x1_params, sizeof(inception_b6_7x1_params ), inception_b6_7x1);
    ERROR_CHECK_OBJECT(inception_b6_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_node));

    // inception_b6_7x1_bn Layer
    vx_size inception_b6_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b6_7x1_scale;
    inception_b6_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b6_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_scale);
    vx_size inception_b6_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b6_7x1_bn_eps = 0.001;
    vx_tensor inception_b6_7x1_bn_W;
    inception_b6_7x1_bn_W = vxCreateTensor(context,1, inception_b6_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_bn_W, dataFolder + "/weights/inception_b6_7x1_bn.f32"));
    vx_size inception_b6_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b6_7x1_bn_B;
    inception_b6_7x1_bn_B = vxCreateTensor(context,1, inception_b6_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_bn_B, dataFolder + "/bias/inception_b6_7x1_bn.f32"));
    vx_size inception_b6_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b6_7x1_scale_W;
    inception_b6_7x1_scale_W = vxCreateTensor(context,1, inception_b6_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_scale_W, dataFolder + "/weights/inception_b6_7x1_scale.f32"));
    vx_size inception_b6_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b6_7x1_scale_B;
    inception_b6_7x1_scale_B = vxCreateTensor(context,1, inception_b6_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_scale_B, dataFolder + "/bias/inception_b6_7x1_scale.f32"));
    vx_node inception_b6_7x1_bn_node;
    inception_b6_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b6_7x1, inception_b6_7x1_bn_W, inception_b6_7x1_bn_B, inception_b6_7x1_scale_W, inception_b6_7x1_scale_B, inception_b6_7x1_bn_eps, inception_b6_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b6_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_bn_node));

    // inception_b6_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_7x1_relu Layer
    vx_size inception_b6_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b6_7x1_relu;
    inception_b6_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b6_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_relu);
    vx_enum inception_b6_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_7x1_relu_param_a = 0;
    vx_float32 inception_b6_7x1_relu_param_b = 0;
    vx_node inception_b6_7x1_relu_node;
    inception_b6_7x1_relu_node = vxActivationLayer(graph, inception_b6_7x1_scale, inception_b6_7x1_relu_mode, inception_b6_7x1_relu_param_a, inception_b6_7x1_relu_param_b, inception_b6_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b6_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_relu_node));

    // inception_b6_7x1_2_reduce Layer
    vx_size inception_b6_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_7x1_2_reduce;
    inception_b6_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b6_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce);
    vx_size inception_b6_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b6_7x1_2_reduce_W;
    inception_b6_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b6_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_reduce_W, dataFolder + "/weights/inception_b6_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b6_7x1_2_reduce_params;
    inception_b6_7x1_2_reduce_params.padding_x = 0;
    inception_b6_7x1_2_reduce_params.padding_y = 0;
    inception_b6_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_7x1_2_reduce_params.dilation_x = 0;
    inception_b6_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b6_7x1_2_reduce_node;
    inception_b6_7x1_2_reduce_node = vxConvolutionLayer(graph, inception_b5_concat, inception_b6_7x1_2_reduce_W, NULL, &inception_b6_7x1_2_reduce_params, sizeof(inception_b6_7x1_2_reduce_params ), inception_b6_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_2_reduce_node));

    // inception_b6_7x1_2_reduce_bn Layer
    vx_size inception_b6_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_7x1_2_reduce_scale;
    inception_b6_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b6_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_scale);
    vx_size inception_b6_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b6_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b6_7x1_2_reduce_bn_W;
    inception_b6_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b6_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b6_7x1_2_reduce_bn.f32"));
    vx_size inception_b6_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b6_7x1_2_reduce_bn_B;
    inception_b6_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b6_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b6_7x1_2_reduce_bn.f32"));
    vx_size inception_b6_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b6_7x1_2_reduce_scale_W;
    inception_b6_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b6_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b6_7x1_2_reduce_scale.f32"));
    vx_size inception_b6_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b6_7x1_2_reduce_scale_B;
    inception_b6_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b6_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b6_7x1_2_reduce_scale.f32"));
    vx_node inception_b6_7x1_2_reduce_bn_node;
    inception_b6_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b6_7x1_2_reduce, inception_b6_7x1_2_reduce_bn_W, inception_b6_7x1_2_reduce_bn_B, inception_b6_7x1_2_reduce_scale_W, inception_b6_7x1_2_reduce_scale_B, inception_b6_7x1_2_reduce_bn_eps, inception_b6_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_2_reduce_bn_node));

    // inception_b6_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_7x1_2_reduce_relu Layer
    vx_size inception_b6_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_7x1_2_reduce_relu;
    inception_b6_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b6_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_relu);
    vx_enum inception_b6_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b6_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b6_7x1_2_reduce_relu_node;
    inception_b6_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b6_7x1_2_reduce_scale, inception_b6_7x1_2_reduce_relu_mode, inception_b6_7x1_2_reduce_relu_param_a, inception_b6_7x1_2_reduce_relu_param_b, inception_b6_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_2_reduce_relu_node));

    // inception_b6_7x1_2 Layer
    vx_size inception_b6_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_7x1_2;
    inception_b6_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b6_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2);
    vx_size inception_b6_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b6_7x1_2_W;
    inception_b6_7x1_2_W = vxCreateTensor(context,4, inception_b6_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_W, dataFolder + "/weights/inception_b6_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b6_7x1_2_params;
    inception_b6_7x1_2_params.padding_x = 0;
    inception_b6_7x1_2_params.padding_y = 3;
    inception_b6_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_7x1_2_params.dilation_x = 0;
    inception_b6_7x1_2_params.dilation_y = 0;
    vx_node inception_b6_7x1_2_node;
    inception_b6_7x1_2_node = vxConvolutionLayer(graph, inception_b6_7x1_2_reduce_relu, inception_b6_7x1_2_W, NULL, &inception_b6_7x1_2_params, sizeof(inception_b6_7x1_2_params ), inception_b6_7x1_2);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_2_node));

    // inception_b6_7x1_2_bn Layer
    vx_size inception_b6_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_7x1_2_scale;
    inception_b6_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b6_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_scale);
    vx_size inception_b6_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b6_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b6_7x1_2_bn_W;
    inception_b6_7x1_2_bn_W = vxCreateTensor(context,1, inception_b6_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_bn_W, dataFolder + "/weights/inception_b6_7x1_2_bn.f32"));
    vx_size inception_b6_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b6_7x1_2_bn_B;
    inception_b6_7x1_2_bn_B = vxCreateTensor(context,1, inception_b6_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_bn_B, dataFolder + "/bias/inception_b6_7x1_2_bn.f32"));
    vx_size inception_b6_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b6_7x1_2_scale_W;
    inception_b6_7x1_2_scale_W = vxCreateTensor(context,1, inception_b6_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_scale_W, dataFolder + "/weights/inception_b6_7x1_2_scale.f32"));
    vx_size inception_b6_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b6_7x1_2_scale_B;
    inception_b6_7x1_2_scale_B = vxCreateTensor(context,1, inception_b6_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_2_scale_B, dataFolder + "/bias/inception_b6_7x1_2_scale.f32"));
    vx_node inception_b6_7x1_2_bn_node;
    inception_b6_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b6_7x1_2, inception_b6_7x1_2_bn_W, inception_b6_7x1_2_bn_B, inception_b6_7x1_2_scale_W, inception_b6_7x1_2_scale_B, inception_b6_7x1_2_bn_eps, inception_b6_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_2_bn_node));

    // inception_b6_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_7x1_2_relu Layer
    vx_size inception_b6_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b6_7x1_2_relu;
    inception_b6_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b6_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_relu);
    vx_enum inception_b6_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_7x1_2_relu_param_a = 0;
    vx_float32 inception_b6_7x1_2_relu_param_b = 0;
    vx_node inception_b6_7x1_2_relu_node;
    inception_b6_7x1_2_relu_node = vxActivationLayer(graph, inception_b6_7x1_2_scale, inception_b6_7x1_2_relu_mode, inception_b6_7x1_2_relu_param_a, inception_b6_7x1_2_relu_param_b, inception_b6_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b6_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_2_relu_node));

    // inception_b6_1x7_2 Layer
    vx_size inception_b6_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_1x7_2;
    inception_b6_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b6_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2);
    vx_size inception_b6_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b6_1x7_2_W;
    inception_b6_1x7_2_W = vxCreateTensor(context,4, inception_b6_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_2_W, dataFolder + "/weights/inception_b6_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b6_1x7_2_params;
    inception_b6_1x7_2_params.padding_x = 3;
    inception_b6_1x7_2_params.padding_y = 0;
    inception_b6_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_1x7_2_params.dilation_x = 0;
    inception_b6_1x7_2_params.dilation_y = 0;
    vx_node inception_b6_1x7_2_node;
    inception_b6_1x7_2_node = vxConvolutionLayer(graph, inception_b6_7x1_2_relu, inception_b6_1x7_2_W, NULL, &inception_b6_1x7_2_params, sizeof(inception_b6_1x7_2_params ), inception_b6_1x7_2);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_2_node));

    // inception_b6_1x7_2_bn Layer
    vx_size inception_b6_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_1x7_2_scale;
    inception_b6_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b6_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_scale);
    vx_size inception_b6_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b6_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b6_1x7_2_bn_W;
    inception_b6_1x7_2_bn_W = vxCreateTensor(context,1, inception_b6_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_2_bn_W, dataFolder + "/weights/inception_b6_1x7_2_bn.f32"));
    vx_size inception_b6_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b6_1x7_2_bn_B;
    inception_b6_1x7_2_bn_B = vxCreateTensor(context,1, inception_b6_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_2_bn_B, dataFolder + "/bias/inception_b6_1x7_2_bn.f32"));
    vx_size inception_b6_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b6_1x7_2_scale_W;
    inception_b6_1x7_2_scale_W = vxCreateTensor(context,1, inception_b6_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_2_scale_W, dataFolder + "/weights/inception_b6_1x7_2_scale.f32"));
    vx_size inception_b6_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b6_1x7_2_scale_B;
    inception_b6_1x7_2_scale_B = vxCreateTensor(context,1, inception_b6_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_2_scale_B, dataFolder + "/bias/inception_b6_1x7_2_scale.f32"));
    vx_node inception_b6_1x7_2_bn_node;
    inception_b6_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b6_1x7_2, inception_b6_1x7_2_bn_W, inception_b6_1x7_2_bn_B, inception_b6_1x7_2_scale_W, inception_b6_1x7_2_scale_B, inception_b6_1x7_2_bn_eps, inception_b6_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_2_bn_node));

    // inception_b6_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_1x7_2_relu Layer
    vx_size inception_b6_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_1x7_2_relu;
    inception_b6_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b6_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_relu);
    vx_enum inception_b6_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_1x7_2_relu_param_a = 0;
    vx_float32 inception_b6_1x7_2_relu_param_b = 0;
    vx_node inception_b6_1x7_2_relu_node;
    inception_b6_1x7_2_relu_node = vxActivationLayer(graph, inception_b6_1x7_2_scale, inception_b6_1x7_2_relu_mode, inception_b6_1x7_2_relu_param_a, inception_b6_1x7_2_relu_param_b, inception_b6_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b6_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_2_relu_node));

    // inception_b6_7x1_3 Layer
    vx_size inception_b6_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_7x1_3;
    inception_b6_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b6_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3);
    vx_size inception_b6_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b6_7x1_3_W;
    inception_b6_7x1_3_W = vxCreateTensor(context,4, inception_b6_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_3_W, dataFolder + "/weights/inception_b6_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b6_7x1_3_params;
    inception_b6_7x1_3_params.padding_x = 0;
    inception_b6_7x1_3_params.padding_y = 3;
    inception_b6_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_7x1_3_params.dilation_x = 0;
    inception_b6_7x1_3_params.dilation_y = 0;
    vx_node inception_b6_7x1_3_node;
    inception_b6_7x1_3_node = vxConvolutionLayer(graph, inception_b6_1x7_2_relu, inception_b6_7x1_3_W, NULL, &inception_b6_7x1_3_params, sizeof(inception_b6_7x1_3_params ), inception_b6_7x1_3);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_3_node));

    // inception_b6_7x1_3_bn Layer
    vx_size inception_b6_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_7x1_3_scale;
    inception_b6_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b6_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_scale);
    vx_size inception_b6_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b6_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b6_7x1_3_bn_W;
    inception_b6_7x1_3_bn_W = vxCreateTensor(context,1, inception_b6_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_3_bn_W, dataFolder + "/weights/inception_b6_7x1_3_bn.f32"));
    vx_size inception_b6_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b6_7x1_3_bn_B;
    inception_b6_7x1_3_bn_B = vxCreateTensor(context,1, inception_b6_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_3_bn_B, dataFolder + "/bias/inception_b6_7x1_3_bn.f32"));
    vx_size inception_b6_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b6_7x1_3_scale_W;
    inception_b6_7x1_3_scale_W = vxCreateTensor(context,1, inception_b6_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_3_scale_W, dataFolder + "/weights/inception_b6_7x1_3_scale.f32"));
    vx_size inception_b6_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b6_7x1_3_scale_B;
    inception_b6_7x1_3_scale_B = vxCreateTensor(context,1, inception_b6_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_7x1_3_scale_B, dataFolder + "/bias/inception_b6_7x1_3_scale.f32"));
    vx_node inception_b6_7x1_3_bn_node;
    inception_b6_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b6_7x1_3, inception_b6_7x1_3_bn_W, inception_b6_7x1_3_bn_B, inception_b6_7x1_3_scale_W, inception_b6_7x1_3_scale_B, inception_b6_7x1_3_bn_eps, inception_b6_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_3_bn_node));

    // inception_b6_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_7x1_3_relu Layer
    vx_size inception_b6_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b6_7x1_3_relu;
    inception_b6_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b6_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_relu);
    vx_enum inception_b6_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_7x1_3_relu_param_a = 0;
    vx_float32 inception_b6_7x1_3_relu_param_b = 0;
    vx_node inception_b6_7x1_3_relu_node;
    inception_b6_7x1_3_relu_node = vxActivationLayer(graph, inception_b6_7x1_3_scale, inception_b6_7x1_3_relu_mode, inception_b6_7x1_3_relu_param_a, inception_b6_7x1_3_relu_param_b, inception_b6_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b6_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_7x1_3_relu_node));

    // inception_b6_1x7_3 Layer
    vx_size inception_b6_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b6_1x7_3;
    inception_b6_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b6_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3);
    vx_size inception_b6_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b6_1x7_3_W;
    inception_b6_1x7_3_W = vxCreateTensor(context,4, inception_b6_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_3_W, dataFolder + "/weights/inception_b6_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b6_1x7_3_params;
    inception_b6_1x7_3_params.padding_x = 3;
    inception_b6_1x7_3_params.padding_y = 0;
    inception_b6_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_1x7_3_params.dilation_x = 0;
    inception_b6_1x7_3_params.dilation_y = 0;
    vx_node inception_b6_1x7_3_node;
    inception_b6_1x7_3_node = vxConvolutionLayer(graph, inception_b6_7x1_3_relu, inception_b6_1x7_3_W, NULL, &inception_b6_1x7_3_params, sizeof(inception_b6_1x7_3_params ), inception_b6_1x7_3);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_3_node));

    // inception_b6_1x7_3_bn Layer
    vx_size inception_b6_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b6_1x7_3_scale;
    inception_b6_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b6_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_scale);
    vx_size inception_b6_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b6_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b6_1x7_3_bn_W;
    inception_b6_1x7_3_bn_W = vxCreateTensor(context,1, inception_b6_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_3_bn_W, dataFolder + "/weights/inception_b6_1x7_3_bn.f32"));
    vx_size inception_b6_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b6_1x7_3_bn_B;
    inception_b6_1x7_3_bn_B = vxCreateTensor(context,1, inception_b6_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_3_bn_B, dataFolder + "/bias/inception_b6_1x7_3_bn.f32"));
    vx_size inception_b6_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b6_1x7_3_scale_W;
    inception_b6_1x7_3_scale_W = vxCreateTensor(context,1, inception_b6_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_3_scale_W, dataFolder + "/weights/inception_b6_1x7_3_scale.f32"));
    vx_size inception_b6_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b6_1x7_3_scale_B;
    inception_b6_1x7_3_scale_B = vxCreateTensor(context,1, inception_b6_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x7_3_scale_B, dataFolder + "/bias/inception_b6_1x7_3_scale.f32"));
    vx_node inception_b6_1x7_3_bn_node;
    inception_b6_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b6_1x7_3, inception_b6_1x7_3_bn_W, inception_b6_1x7_3_bn_B, inception_b6_1x7_3_scale_W, inception_b6_1x7_3_scale_B, inception_b6_1x7_3_bn_eps, inception_b6_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_3_bn_node));

    // inception_b6_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_1x7_3_relu Layer
    vx_size inception_b6_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b6_1x7_3_relu;
    inception_b6_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b6_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_relu);
    vx_enum inception_b6_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_1x7_3_relu_param_a = 0;
    vx_float32 inception_b6_1x7_3_relu_param_b = 0;
    vx_node inception_b6_1x7_3_relu_node;
    inception_b6_1x7_3_relu_node = vxActivationLayer(graph, inception_b6_1x7_3_scale, inception_b6_1x7_3_relu_mode, inception_b6_1x7_3_relu_param_a, inception_b6_1x7_3_relu_param_b, inception_b6_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b6_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x7_3_relu_node));

    // inception_b6_pool_ave Layer
    vx_size inception_b6_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b6_pool_ave;
    inception_b6_pool_ave = vxCreateVirtualTensor(graph,4, inception_b6_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_pool_ave);
    vx_enum inception_b6_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b6_pool_ave_kernel_w = 3;
    vx_size inception_b6_pool_ave_kernel_h = 3;
    vx_size inception_b6_pool_ave_pad_w = 1;
    vx_size inception_b6_pool_ave_pad_h = 1;
    vx_enum inception_b6_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b6_pool_ave_node;
    inception_b6_pool_ave_node = vxPoolingLayer(graph, inception_b5_concat, inception_b6_pool_ave_type, inception_b6_pool_ave_kernel_w, inception_b6_pool_ave_kernel_h, inception_b6_pool_ave_pad_w, inception_b6_pool_ave_pad_h, inception_b6_pool_ave_roundPolicy, inception_b6_pool_ave );
    ERROR_CHECK_OBJECT(inception_b6_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_pool_ave_node));

    // inception_b6_1x1 Layer
    vx_size inception_b6_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b6_1x1;
    inception_b6_1x1 = vxCreateVirtualTensor(graph,4, inception_b6_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x1);
    vx_size inception_b6_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b6_1x1_W;
    inception_b6_1x1_W = vxCreateTensor(context,4, inception_b6_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_W, dataFolder + "/weights/inception_b6_1x1.f32"));
    vx_nn_convolution_params_t inception_b6_1x1_params;
    inception_b6_1x1_params.padding_x = 0;
    inception_b6_1x1_params.padding_y = 0;
    inception_b6_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b6_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b6_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b6_1x1_params.dilation_x = 0;
    inception_b6_1x1_params.dilation_y = 0;
    vx_node inception_b6_1x1_node;
    inception_b6_1x1_node = vxConvolutionLayer(graph, inception_b6_pool_ave, inception_b6_1x1_W, NULL, &inception_b6_1x1_params, sizeof(inception_b6_1x1_params ), inception_b6_1x1);
    ERROR_CHECK_OBJECT(inception_b6_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x1_node));

    // inception_b6_1x1_bn Layer
    vx_size inception_b6_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b6_1x1_scale;
    inception_b6_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b6_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_scale);
    vx_size inception_b6_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b6_1x1_bn_eps = 0.001;
    vx_tensor inception_b6_1x1_bn_W;
    inception_b6_1x1_bn_W = vxCreateTensor(context,1, inception_b6_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_bn_W, dataFolder + "/weights/inception_b6_1x1_bn.f32"));
    vx_size inception_b6_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b6_1x1_bn_B;
    inception_b6_1x1_bn_B = vxCreateTensor(context,1, inception_b6_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_bn_B, dataFolder + "/bias/inception_b6_1x1_bn.f32"));
    vx_size inception_b6_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b6_1x1_scale_W;
    inception_b6_1x1_scale_W = vxCreateTensor(context,1, inception_b6_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_scale_W, dataFolder + "/weights/inception_b6_1x1_scale.f32"));
    vx_size inception_b6_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b6_1x1_scale_B;
    inception_b6_1x1_scale_B = vxCreateTensor(context,1, inception_b6_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b6_1x1_scale_B, dataFolder + "/bias/inception_b6_1x1_scale.f32"));
    vx_node inception_b6_1x1_bn_node;
    inception_b6_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b6_1x1, inception_b6_1x1_bn_W, inception_b6_1x1_bn_B, inception_b6_1x1_scale_W, inception_b6_1x1_scale_B, inception_b6_1x1_bn_eps, inception_b6_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b6_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x1_bn_node));

    // inception_b6_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b6_1x1_relu Layer
    vx_size inception_b6_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b6_1x1_relu;
    inception_b6_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b6_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_1x1_relu);
    vx_enum inception_b6_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b6_1x1_relu_param_a = 0;
    vx_float32 inception_b6_1x1_relu_param_b = 0;
    vx_node inception_b6_1x1_relu_node;
    inception_b6_1x1_relu_node = vxActivationLayer(graph, inception_b6_1x1_scale, inception_b6_1x1_relu_mode, inception_b6_1x1_relu_param_a, inception_b6_1x1_relu_param_b, inception_b6_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b6_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_1x1_relu_node));

    // inception_b6_concat Layer
    vx_size inception_b6_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b6_concat;
    inception_b6_concat = vxCreateVirtualTensor(graph,4, inception_b6_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b6_concat);
    vx_node inception_b6_concat_node;
    inception_b6_concat_node = vxConcatLayer(graph, inception_b6_concat, inception_b6_1x1_2_relu, inception_b6_7x1_relu, inception_b6_1x7_3_relu, inception_b6_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b6_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b6_concat_node));

    // inception_b7_1x1_2 Layer
    vx_size inception_b7_1x1_2_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b7_1x1_2;
    inception_b7_1x1_2 = vxCreateVirtualTensor(graph,4, inception_b7_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2);
    vx_size inception_b7_1x1_2_W_dims[4] = { 1, 1, 1024, 384 };
    vx_tensor inception_b7_1x1_2_W;
    inception_b7_1x1_2_W = vxCreateTensor(context,4, inception_b7_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_2_W, dataFolder + "/weights/inception_b7_1x1_2.f32"));
    vx_nn_convolution_params_t inception_b7_1x1_2_params;
    inception_b7_1x1_2_params.padding_x = 0;
    inception_b7_1x1_2_params.padding_y = 0;
    inception_b7_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_1x1_2_params.dilation_x = 0;
    inception_b7_1x1_2_params.dilation_y = 0;
    vx_node inception_b7_1x1_2_node;
    inception_b7_1x1_2_node = vxConvolutionLayer(graph, inception_b6_concat, inception_b7_1x1_2_W, NULL, &inception_b7_1x1_2_params, sizeof(inception_b7_1x1_2_params ), inception_b7_1x1_2);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x1_2_node));

    // inception_b7_1x1_2_bn Layer
    vx_size inception_b7_1x1_2_scale_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b7_1x1_2_scale;
    inception_b7_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b7_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_scale);
    vx_size inception_b7_1x1_2_bn_W_dims[1] = { 384 };
    vx_float32 inception_b7_1x1_2_bn_eps = 0.001;
    vx_tensor inception_b7_1x1_2_bn_W;
    inception_b7_1x1_2_bn_W = vxCreateTensor(context,1, inception_b7_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_2_bn_W, dataFolder + "/weights/inception_b7_1x1_2_bn.f32"));
    vx_size inception_b7_1x1_2_bn_B_dims[1] = { 384 };
    vx_tensor inception_b7_1x1_2_bn_B;
    inception_b7_1x1_2_bn_B = vxCreateTensor(context,1, inception_b7_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_2_bn_B, dataFolder + "/bias/inception_b7_1x1_2_bn.f32"));
    vx_size inception_b7_1x1_2_scale_W_dims[1] = { 384 };
    vx_tensor inception_b7_1x1_2_scale_W;
    inception_b7_1x1_2_scale_W = vxCreateTensor(context,1, inception_b7_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_2_scale_W, dataFolder + "/weights/inception_b7_1x1_2_scale.f32"));
    vx_size inception_b7_1x1_2_scale_B_dims[1] = { 384 };
    vx_tensor inception_b7_1x1_2_scale_B;
    inception_b7_1x1_2_scale_B = vxCreateTensor(context,1, inception_b7_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_2_scale_B, dataFolder + "/bias/inception_b7_1x1_2_scale.f32"));
    vx_node inception_b7_1x1_2_bn_node;
    inception_b7_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b7_1x1_2, inception_b7_1x1_2_bn_W, inception_b7_1x1_2_bn_B, inception_b7_1x1_2_scale_W, inception_b7_1x1_2_scale_B, inception_b7_1x1_2_bn_eps, inception_b7_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x1_2_bn_node));

    // inception_b7_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_1x1_2_relu Layer
    vx_size inception_b7_1x1_2_relu_dims[4] = { 17, 17, 384, 1 };
    vx_tensor inception_b7_1x1_2_relu;
    inception_b7_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b7_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_relu);
    vx_enum inception_b7_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_1x1_2_relu_param_a = 0;
    vx_float32 inception_b7_1x1_2_relu_param_b = 0;
    vx_node inception_b7_1x1_2_relu_node;
    inception_b7_1x1_2_relu_node = vxActivationLayer(graph, inception_b7_1x1_2_scale, inception_b7_1x1_2_relu_mode, inception_b7_1x1_2_relu_param_a, inception_b7_1x1_2_relu_param_b, inception_b7_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b7_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x1_2_relu_node));

    // inception_b7_1x7_reduce Layer
    vx_size inception_b7_1x7_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_1x7_reduce;
    inception_b7_1x7_reduce = vxCreateVirtualTensor(graph,4, inception_b7_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce);
    vx_size inception_b7_1x7_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b7_1x7_reduce_W;
    inception_b7_1x7_reduce_W = vxCreateTensor(context,4, inception_b7_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_reduce_W, dataFolder + "/weights/inception_b7_1x7_reduce.f32"));
    vx_nn_convolution_params_t inception_b7_1x7_reduce_params;
    inception_b7_1x7_reduce_params.padding_x = 0;
    inception_b7_1x7_reduce_params.padding_y = 0;
    inception_b7_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_1x7_reduce_params.dilation_x = 0;
    inception_b7_1x7_reduce_params.dilation_y = 0;
    vx_node inception_b7_1x7_reduce_node;
    inception_b7_1x7_reduce_node = vxConvolutionLayer(graph, inception_b6_concat, inception_b7_1x7_reduce_W, NULL, &inception_b7_1x7_reduce_params, sizeof(inception_b7_1x7_reduce_params ), inception_b7_1x7_reduce);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_reduce_node));

    // inception_b7_1x7_reduce_bn Layer
    vx_size inception_b7_1x7_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_1x7_reduce_scale;
    inception_b7_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b7_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_scale);
    vx_size inception_b7_1x7_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b7_1x7_reduce_bn_eps = 0.001;
    vx_tensor inception_b7_1x7_reduce_bn_W;
    inception_b7_1x7_reduce_bn_W = vxCreateTensor(context,1, inception_b7_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_reduce_bn_W, dataFolder + "/weights/inception_b7_1x7_reduce_bn.f32"));
    vx_size inception_b7_1x7_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b7_1x7_reduce_bn_B;
    inception_b7_1x7_reduce_bn_B = vxCreateTensor(context,1, inception_b7_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_reduce_bn_B, dataFolder + "/bias/inception_b7_1x7_reduce_bn.f32"));
    vx_size inception_b7_1x7_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b7_1x7_reduce_scale_W;
    inception_b7_1x7_reduce_scale_W = vxCreateTensor(context,1, inception_b7_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_reduce_scale_W, dataFolder + "/weights/inception_b7_1x7_reduce_scale.f32"));
    vx_size inception_b7_1x7_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b7_1x7_reduce_scale_B;
    inception_b7_1x7_reduce_scale_B = vxCreateTensor(context,1, inception_b7_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_reduce_scale_B, dataFolder + "/bias/inception_b7_1x7_reduce_scale.f32"));
    vx_node inception_b7_1x7_reduce_bn_node;
    inception_b7_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b7_1x7_reduce, inception_b7_1x7_reduce_bn_W, inception_b7_1x7_reduce_bn_B, inception_b7_1x7_reduce_scale_W, inception_b7_1x7_reduce_scale_B, inception_b7_1x7_reduce_bn_eps, inception_b7_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_reduce_bn_node));

    // inception_b7_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_1x7_reduce_relu Layer
    vx_size inception_b7_1x7_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_1x7_reduce_relu;
    inception_b7_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b7_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_relu);
    vx_enum inception_b7_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_1x7_reduce_relu_param_a = 0;
    vx_float32 inception_b7_1x7_reduce_relu_param_b = 0;
    vx_node inception_b7_1x7_reduce_relu_node;
    inception_b7_1x7_reduce_relu_node = vxActivationLayer(graph, inception_b7_1x7_reduce_scale, inception_b7_1x7_reduce_relu_mode, inception_b7_1x7_reduce_relu_param_a, inception_b7_1x7_reduce_relu_param_b, inception_b7_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b7_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_reduce_relu_node));

    // inception_b7_1x7 Layer
    vx_size inception_b7_1x7_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_1x7;
    inception_b7_1x7 = vxCreateVirtualTensor(graph,4, inception_b7_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7);
    vx_size inception_b7_1x7_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b7_1x7_W;
    inception_b7_1x7_W = vxCreateTensor(context,4, inception_b7_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_W, dataFolder + "/weights/inception_b7_1x7.f32"));
    vx_nn_convolution_params_t inception_b7_1x7_params;
    inception_b7_1x7_params.padding_x = 3;
    inception_b7_1x7_params.padding_y = 0;
    inception_b7_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_1x7_params.dilation_x = 0;
    inception_b7_1x7_params.dilation_y = 0;
    vx_node inception_b7_1x7_node;
    inception_b7_1x7_node = vxConvolutionLayer(graph, inception_b7_1x7_reduce_relu, inception_b7_1x7_W, NULL, &inception_b7_1x7_params, sizeof(inception_b7_1x7_params ), inception_b7_1x7);
    ERROR_CHECK_OBJECT(inception_b7_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_node));

    // inception_b7_1x7_bn Layer
    vx_size inception_b7_1x7_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_1x7_scale;
    inception_b7_1x7_scale = vxCreateVirtualTensor(graph,4, inception_b7_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_scale);
    vx_size inception_b7_1x7_bn_W_dims[1] = { 224 };
    vx_float32 inception_b7_1x7_bn_eps = 0.001;
    vx_tensor inception_b7_1x7_bn_W;
    inception_b7_1x7_bn_W = vxCreateTensor(context,1, inception_b7_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_bn_W, dataFolder + "/weights/inception_b7_1x7_bn.f32"));
    vx_size inception_b7_1x7_bn_B_dims[1] = { 224 };
    vx_tensor inception_b7_1x7_bn_B;
    inception_b7_1x7_bn_B = vxCreateTensor(context,1, inception_b7_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_bn_B, dataFolder + "/bias/inception_b7_1x7_bn.f32"));
    vx_size inception_b7_1x7_scale_W_dims[1] = { 224 };
    vx_tensor inception_b7_1x7_scale_W;
    inception_b7_1x7_scale_W = vxCreateTensor(context,1, inception_b7_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_scale_W, dataFolder + "/weights/inception_b7_1x7_scale.f32"));
    vx_size inception_b7_1x7_scale_B_dims[1] = { 224 };
    vx_tensor inception_b7_1x7_scale_B;
    inception_b7_1x7_scale_B = vxCreateTensor(context,1, inception_b7_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_scale_B, dataFolder + "/bias/inception_b7_1x7_scale.f32"));
    vx_node inception_b7_1x7_bn_node;
    inception_b7_1x7_bn_node = vxBatchNormalizationLayer(graph, inception_b7_1x7, inception_b7_1x7_bn_W, inception_b7_1x7_bn_B, inception_b7_1x7_scale_W, inception_b7_1x7_scale_B, inception_b7_1x7_bn_eps, inception_b7_1x7_scale);
    ERROR_CHECK_OBJECT(inception_b7_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_bn_node));

    // inception_b7_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_1x7_relu Layer
    vx_size inception_b7_1x7_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_1x7_relu;
    inception_b7_1x7_relu = vxCreateVirtualTensor(graph,4, inception_b7_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_relu);
    vx_enum inception_b7_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_1x7_relu_param_a = 0;
    vx_float32 inception_b7_1x7_relu_param_b = 0;
    vx_node inception_b7_1x7_relu_node;
    inception_b7_1x7_relu_node = vxActivationLayer(graph, inception_b7_1x7_scale, inception_b7_1x7_relu_mode, inception_b7_1x7_relu_param_a, inception_b7_1x7_relu_param_b, inception_b7_1x7_relu);
    ERROR_CHECK_OBJECT(inception_b7_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_relu_node));

    // inception_b7_7x1 Layer
    vx_size inception_b7_7x1_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b7_7x1;
    inception_b7_7x1 = vxCreateVirtualTensor(graph,4, inception_b7_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1);
    vx_size inception_b7_7x1_W_dims[4] = { 1, 7, 224, 256 };
    vx_tensor inception_b7_7x1_W;
    inception_b7_7x1_W = vxCreateTensor(context,4, inception_b7_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_W, dataFolder + "/weights/inception_b7_7x1.f32"));
    vx_nn_convolution_params_t inception_b7_7x1_params;
    inception_b7_7x1_params.padding_x = 0;
    inception_b7_7x1_params.padding_y = 3;
    inception_b7_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_7x1_params.dilation_x = 0;
    inception_b7_7x1_params.dilation_y = 0;
    vx_node inception_b7_7x1_node;
    inception_b7_7x1_node = vxConvolutionLayer(graph, inception_b7_1x7_relu, inception_b7_7x1_W, NULL, &inception_b7_7x1_params, sizeof(inception_b7_7x1_params ), inception_b7_7x1);
    ERROR_CHECK_OBJECT(inception_b7_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_node));

    // inception_b7_7x1_bn Layer
    vx_size inception_b7_7x1_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b7_7x1_scale;
    inception_b7_7x1_scale = vxCreateVirtualTensor(graph,4, inception_b7_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_scale);
    vx_size inception_b7_7x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_b7_7x1_bn_eps = 0.001;
    vx_tensor inception_b7_7x1_bn_W;
    inception_b7_7x1_bn_W = vxCreateTensor(context,1, inception_b7_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_bn_W, dataFolder + "/weights/inception_b7_7x1_bn.f32"));
    vx_size inception_b7_7x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_b7_7x1_bn_B;
    inception_b7_7x1_bn_B = vxCreateTensor(context,1, inception_b7_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_bn_B, dataFolder + "/bias/inception_b7_7x1_bn.f32"));
    vx_size inception_b7_7x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_b7_7x1_scale_W;
    inception_b7_7x1_scale_W = vxCreateTensor(context,1, inception_b7_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_scale_W, dataFolder + "/weights/inception_b7_7x1_scale.f32"));
    vx_size inception_b7_7x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_b7_7x1_scale_B;
    inception_b7_7x1_scale_B = vxCreateTensor(context,1, inception_b7_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_scale_B, dataFolder + "/bias/inception_b7_7x1_scale.f32"));
    vx_node inception_b7_7x1_bn_node;
    inception_b7_7x1_bn_node = vxBatchNormalizationLayer(graph, inception_b7_7x1, inception_b7_7x1_bn_W, inception_b7_7x1_bn_B, inception_b7_7x1_scale_W, inception_b7_7x1_scale_B, inception_b7_7x1_bn_eps, inception_b7_7x1_scale);
    ERROR_CHECK_OBJECT(inception_b7_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_bn_node));

    // inception_b7_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_7x1_relu Layer
    vx_size inception_b7_7x1_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b7_7x1_relu;
    inception_b7_7x1_relu = vxCreateVirtualTensor(graph,4, inception_b7_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_relu);
    vx_enum inception_b7_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_7x1_relu_param_a = 0;
    vx_float32 inception_b7_7x1_relu_param_b = 0;
    vx_node inception_b7_7x1_relu_node;
    inception_b7_7x1_relu_node = vxActivationLayer(graph, inception_b7_7x1_scale, inception_b7_7x1_relu_mode, inception_b7_7x1_relu_param_a, inception_b7_7x1_relu_param_b, inception_b7_7x1_relu);
    ERROR_CHECK_OBJECT(inception_b7_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_relu_node));

    // inception_b7_7x1_2_reduce Layer
    vx_size inception_b7_7x1_2_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_7x1_2_reduce;
    inception_b7_7x1_2_reduce = vxCreateVirtualTensor(graph,4, inception_b7_7x1_2_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce);
    vx_size inception_b7_7x1_2_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor inception_b7_7x1_2_reduce_W;
    inception_b7_7x1_2_reduce_W = vxCreateTensor(context,4, inception_b7_7x1_2_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_reduce_W, dataFolder + "/weights/inception_b7_7x1_2_reduce.f32"));
    vx_nn_convolution_params_t inception_b7_7x1_2_reduce_params;
    inception_b7_7x1_2_reduce_params.padding_x = 0;
    inception_b7_7x1_2_reduce_params.padding_y = 0;
    inception_b7_7x1_2_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_7x1_2_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_7x1_2_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_7x1_2_reduce_params.dilation_x = 0;
    inception_b7_7x1_2_reduce_params.dilation_y = 0;
    vx_node inception_b7_7x1_2_reduce_node;
    inception_b7_7x1_2_reduce_node = vxConvolutionLayer(graph, inception_b6_concat, inception_b7_7x1_2_reduce_W, NULL, &inception_b7_7x1_2_reduce_params, sizeof(inception_b7_7x1_2_reduce_params ), inception_b7_7x1_2_reduce);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_2_reduce_node));

    // inception_b7_7x1_2_reduce_bn Layer
    vx_size inception_b7_7x1_2_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_7x1_2_reduce_scale;
    inception_b7_7x1_2_reduce_scale = vxCreateVirtualTensor(graph,4, inception_b7_7x1_2_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_scale);
    vx_size inception_b7_7x1_2_reduce_bn_W_dims[1] = { 192 };
    vx_float32 inception_b7_7x1_2_reduce_bn_eps = 0.001;
    vx_tensor inception_b7_7x1_2_reduce_bn_W;
    inception_b7_7x1_2_reduce_bn_W = vxCreateTensor(context,1, inception_b7_7x1_2_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_reduce_bn_W, dataFolder + "/weights/inception_b7_7x1_2_reduce_bn.f32"));
    vx_size inception_b7_7x1_2_reduce_bn_B_dims[1] = { 192 };
    vx_tensor inception_b7_7x1_2_reduce_bn_B;
    inception_b7_7x1_2_reduce_bn_B = vxCreateTensor(context,1, inception_b7_7x1_2_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_reduce_bn_B, dataFolder + "/bias/inception_b7_7x1_2_reduce_bn.f32"));
    vx_size inception_b7_7x1_2_reduce_scale_W_dims[1] = { 192 };
    vx_tensor inception_b7_7x1_2_reduce_scale_W;
    inception_b7_7x1_2_reduce_scale_W = vxCreateTensor(context,1, inception_b7_7x1_2_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_reduce_scale_W, dataFolder + "/weights/inception_b7_7x1_2_reduce_scale.f32"));
    vx_size inception_b7_7x1_2_reduce_scale_B_dims[1] = { 192 };
    vx_tensor inception_b7_7x1_2_reduce_scale_B;
    inception_b7_7x1_2_reduce_scale_B = vxCreateTensor(context,1, inception_b7_7x1_2_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_reduce_scale_B, dataFolder + "/bias/inception_b7_7x1_2_reduce_scale.f32"));
    vx_node inception_b7_7x1_2_reduce_bn_node;
    inception_b7_7x1_2_reduce_bn_node = vxBatchNormalizationLayer(graph, inception_b7_7x1_2_reduce, inception_b7_7x1_2_reduce_bn_W, inception_b7_7x1_2_reduce_bn_B, inception_b7_7x1_2_reduce_scale_W, inception_b7_7x1_2_reduce_scale_B, inception_b7_7x1_2_reduce_bn_eps, inception_b7_7x1_2_reduce_scale);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_2_reduce_bn_node));

    // inception_b7_7x1_2_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_7x1_2_reduce_relu Layer
    vx_size inception_b7_7x1_2_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_7x1_2_reduce_relu;
    inception_b7_7x1_2_reduce_relu = vxCreateVirtualTensor(graph,4, inception_b7_7x1_2_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_relu);
    vx_enum inception_b7_7x1_2_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_7x1_2_reduce_relu_param_a = 0;
    vx_float32 inception_b7_7x1_2_reduce_relu_param_b = 0;
    vx_node inception_b7_7x1_2_reduce_relu_node;
    inception_b7_7x1_2_reduce_relu_node = vxActivationLayer(graph, inception_b7_7x1_2_reduce_scale, inception_b7_7x1_2_reduce_relu_mode, inception_b7_7x1_2_reduce_relu_param_a, inception_b7_7x1_2_reduce_relu_param_b, inception_b7_7x1_2_reduce_relu);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_2_reduce_relu_node));

    // inception_b7_7x1_2 Layer
    vx_size inception_b7_7x1_2_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_7x1_2;
    inception_b7_7x1_2 = vxCreateVirtualTensor(graph,4, inception_b7_7x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2);
    vx_size inception_b7_7x1_2_W_dims[4] = { 1, 7, 192, 192 };
    vx_tensor inception_b7_7x1_2_W;
    inception_b7_7x1_2_W = vxCreateTensor(context,4, inception_b7_7x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_W, dataFolder + "/weights/inception_b7_7x1_2.f32"));
    vx_nn_convolution_params_t inception_b7_7x1_2_params;
    inception_b7_7x1_2_params.padding_x = 0;
    inception_b7_7x1_2_params.padding_y = 3;
    inception_b7_7x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_7x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_7x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_7x1_2_params.dilation_x = 0;
    inception_b7_7x1_2_params.dilation_y = 0;
    vx_node inception_b7_7x1_2_node;
    inception_b7_7x1_2_node = vxConvolutionLayer(graph, inception_b7_7x1_2_reduce_relu, inception_b7_7x1_2_W, NULL, &inception_b7_7x1_2_params, sizeof(inception_b7_7x1_2_params ), inception_b7_7x1_2);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_2_node));

    // inception_b7_7x1_2_bn Layer
    vx_size inception_b7_7x1_2_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_7x1_2_scale;
    inception_b7_7x1_2_scale = vxCreateVirtualTensor(graph,4, inception_b7_7x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_scale);
    vx_size inception_b7_7x1_2_bn_W_dims[1] = { 192 };
    vx_float32 inception_b7_7x1_2_bn_eps = 0.001;
    vx_tensor inception_b7_7x1_2_bn_W;
    inception_b7_7x1_2_bn_W = vxCreateTensor(context,1, inception_b7_7x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_bn_W, dataFolder + "/weights/inception_b7_7x1_2_bn.f32"));
    vx_size inception_b7_7x1_2_bn_B_dims[1] = { 192 };
    vx_tensor inception_b7_7x1_2_bn_B;
    inception_b7_7x1_2_bn_B = vxCreateTensor(context,1, inception_b7_7x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_bn_B, dataFolder + "/bias/inception_b7_7x1_2_bn.f32"));
    vx_size inception_b7_7x1_2_scale_W_dims[1] = { 192 };
    vx_tensor inception_b7_7x1_2_scale_W;
    inception_b7_7x1_2_scale_W = vxCreateTensor(context,1, inception_b7_7x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_scale_W, dataFolder + "/weights/inception_b7_7x1_2_scale.f32"));
    vx_size inception_b7_7x1_2_scale_B_dims[1] = { 192 };
    vx_tensor inception_b7_7x1_2_scale_B;
    inception_b7_7x1_2_scale_B = vxCreateTensor(context,1, inception_b7_7x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_2_scale_B, dataFolder + "/bias/inception_b7_7x1_2_scale.f32"));
    vx_node inception_b7_7x1_2_bn_node;
    inception_b7_7x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_b7_7x1_2, inception_b7_7x1_2_bn_W, inception_b7_7x1_2_bn_B, inception_b7_7x1_2_scale_W, inception_b7_7x1_2_scale_B, inception_b7_7x1_2_bn_eps, inception_b7_7x1_2_scale);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_2_bn_node));

    // inception_b7_7x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_7x1_2_relu Layer
    vx_size inception_b7_7x1_2_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor inception_b7_7x1_2_relu;
    inception_b7_7x1_2_relu = vxCreateVirtualTensor(graph,4, inception_b7_7x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_relu);
    vx_enum inception_b7_7x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_7x1_2_relu_param_a = 0;
    vx_float32 inception_b7_7x1_2_relu_param_b = 0;
    vx_node inception_b7_7x1_2_relu_node;
    inception_b7_7x1_2_relu_node = vxActivationLayer(graph, inception_b7_7x1_2_scale, inception_b7_7x1_2_relu_mode, inception_b7_7x1_2_relu_param_a, inception_b7_7x1_2_relu_param_b, inception_b7_7x1_2_relu);
    ERROR_CHECK_OBJECT(inception_b7_7x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_2_relu_node));

    // inception_b7_1x7_2 Layer
    vx_size inception_b7_1x7_2_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_1x7_2;
    inception_b7_1x7_2 = vxCreateVirtualTensor(graph,4, inception_b7_1x7_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2);
    vx_size inception_b7_1x7_2_W_dims[4] = { 7, 1, 192, 224 };
    vx_tensor inception_b7_1x7_2_W;
    inception_b7_1x7_2_W = vxCreateTensor(context,4, inception_b7_1x7_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_2_W, dataFolder + "/weights/inception_b7_1x7_2.f32"));
    vx_nn_convolution_params_t inception_b7_1x7_2_params;
    inception_b7_1x7_2_params.padding_x = 3;
    inception_b7_1x7_2_params.padding_y = 0;
    inception_b7_1x7_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_1x7_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_1x7_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_1x7_2_params.dilation_x = 0;
    inception_b7_1x7_2_params.dilation_y = 0;
    vx_node inception_b7_1x7_2_node;
    inception_b7_1x7_2_node = vxConvolutionLayer(graph, inception_b7_7x1_2_relu, inception_b7_1x7_2_W, NULL, &inception_b7_1x7_2_params, sizeof(inception_b7_1x7_2_params ), inception_b7_1x7_2);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_2_node));

    // inception_b7_1x7_2_bn Layer
    vx_size inception_b7_1x7_2_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_1x7_2_scale;
    inception_b7_1x7_2_scale = vxCreateVirtualTensor(graph,4, inception_b7_1x7_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_scale);
    vx_size inception_b7_1x7_2_bn_W_dims[1] = { 224 };
    vx_float32 inception_b7_1x7_2_bn_eps = 0.001;
    vx_tensor inception_b7_1x7_2_bn_W;
    inception_b7_1x7_2_bn_W = vxCreateTensor(context,1, inception_b7_1x7_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_2_bn_W, dataFolder + "/weights/inception_b7_1x7_2_bn.f32"));
    vx_size inception_b7_1x7_2_bn_B_dims[1] = { 224 };
    vx_tensor inception_b7_1x7_2_bn_B;
    inception_b7_1x7_2_bn_B = vxCreateTensor(context,1, inception_b7_1x7_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_2_bn_B, dataFolder + "/bias/inception_b7_1x7_2_bn.f32"));
    vx_size inception_b7_1x7_2_scale_W_dims[1] = { 224 };
    vx_tensor inception_b7_1x7_2_scale_W;
    inception_b7_1x7_2_scale_W = vxCreateTensor(context,1, inception_b7_1x7_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_2_scale_W, dataFolder + "/weights/inception_b7_1x7_2_scale.f32"));
    vx_size inception_b7_1x7_2_scale_B_dims[1] = { 224 };
    vx_tensor inception_b7_1x7_2_scale_B;
    inception_b7_1x7_2_scale_B = vxCreateTensor(context,1, inception_b7_1x7_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_2_scale_B, dataFolder + "/bias/inception_b7_1x7_2_scale.f32"));
    vx_node inception_b7_1x7_2_bn_node;
    inception_b7_1x7_2_bn_node = vxBatchNormalizationLayer(graph, inception_b7_1x7_2, inception_b7_1x7_2_bn_W, inception_b7_1x7_2_bn_B, inception_b7_1x7_2_scale_W, inception_b7_1x7_2_scale_B, inception_b7_1x7_2_bn_eps, inception_b7_1x7_2_scale);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_2_bn_node));

    // inception_b7_1x7_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_1x7_2_relu Layer
    vx_size inception_b7_1x7_2_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_1x7_2_relu;
    inception_b7_1x7_2_relu = vxCreateVirtualTensor(graph,4, inception_b7_1x7_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_relu);
    vx_enum inception_b7_1x7_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_1x7_2_relu_param_a = 0;
    vx_float32 inception_b7_1x7_2_relu_param_b = 0;
    vx_node inception_b7_1x7_2_relu_node;
    inception_b7_1x7_2_relu_node = vxActivationLayer(graph, inception_b7_1x7_2_scale, inception_b7_1x7_2_relu_mode, inception_b7_1x7_2_relu_param_a, inception_b7_1x7_2_relu_param_b, inception_b7_1x7_2_relu);
    ERROR_CHECK_OBJECT(inception_b7_1x7_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_2_relu_node));

    // inception_b7_7x1_3 Layer
    vx_size inception_b7_7x1_3_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_7x1_3;
    inception_b7_7x1_3 = vxCreateVirtualTensor(graph,4, inception_b7_7x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3);
    vx_size inception_b7_7x1_3_W_dims[4] = { 1, 7, 224, 224 };
    vx_tensor inception_b7_7x1_3_W;
    inception_b7_7x1_3_W = vxCreateTensor(context,4, inception_b7_7x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_3_W, dataFolder + "/weights/inception_b7_7x1_3.f32"));
    vx_nn_convolution_params_t inception_b7_7x1_3_params;
    inception_b7_7x1_3_params.padding_x = 0;
    inception_b7_7x1_3_params.padding_y = 3;
    inception_b7_7x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_7x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_7x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_7x1_3_params.dilation_x = 0;
    inception_b7_7x1_3_params.dilation_y = 0;
    vx_node inception_b7_7x1_3_node;
    inception_b7_7x1_3_node = vxConvolutionLayer(graph, inception_b7_1x7_2_relu, inception_b7_7x1_3_W, NULL, &inception_b7_7x1_3_params, sizeof(inception_b7_7x1_3_params ), inception_b7_7x1_3);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_3_node));

    // inception_b7_7x1_3_bn Layer
    vx_size inception_b7_7x1_3_scale_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_7x1_3_scale;
    inception_b7_7x1_3_scale = vxCreateVirtualTensor(graph,4, inception_b7_7x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_scale);
    vx_size inception_b7_7x1_3_bn_W_dims[1] = { 224 };
    vx_float32 inception_b7_7x1_3_bn_eps = 0.001;
    vx_tensor inception_b7_7x1_3_bn_W;
    inception_b7_7x1_3_bn_W = vxCreateTensor(context,1, inception_b7_7x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_3_bn_W, dataFolder + "/weights/inception_b7_7x1_3_bn.f32"));
    vx_size inception_b7_7x1_3_bn_B_dims[1] = { 224 };
    vx_tensor inception_b7_7x1_3_bn_B;
    inception_b7_7x1_3_bn_B = vxCreateTensor(context,1, inception_b7_7x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_3_bn_B, dataFolder + "/bias/inception_b7_7x1_3_bn.f32"));
    vx_size inception_b7_7x1_3_scale_W_dims[1] = { 224 };
    vx_tensor inception_b7_7x1_3_scale_W;
    inception_b7_7x1_3_scale_W = vxCreateTensor(context,1, inception_b7_7x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_3_scale_W, dataFolder + "/weights/inception_b7_7x1_3_scale.f32"));
    vx_size inception_b7_7x1_3_scale_B_dims[1] = { 224 };
    vx_tensor inception_b7_7x1_3_scale_B;
    inception_b7_7x1_3_scale_B = vxCreateTensor(context,1, inception_b7_7x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_7x1_3_scale_B, dataFolder + "/bias/inception_b7_7x1_3_scale.f32"));
    vx_node inception_b7_7x1_3_bn_node;
    inception_b7_7x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_b7_7x1_3, inception_b7_7x1_3_bn_W, inception_b7_7x1_3_bn_B, inception_b7_7x1_3_scale_W, inception_b7_7x1_3_scale_B, inception_b7_7x1_3_bn_eps, inception_b7_7x1_3_scale);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_3_bn_node));

    // inception_b7_7x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_7x1_3_relu Layer
    vx_size inception_b7_7x1_3_relu_dims[4] = { 17, 17, 224, 1 };
    vx_tensor inception_b7_7x1_3_relu;
    inception_b7_7x1_3_relu = vxCreateVirtualTensor(graph,4, inception_b7_7x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_relu);
    vx_enum inception_b7_7x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_7x1_3_relu_param_a = 0;
    vx_float32 inception_b7_7x1_3_relu_param_b = 0;
    vx_node inception_b7_7x1_3_relu_node;
    inception_b7_7x1_3_relu_node = vxActivationLayer(graph, inception_b7_7x1_3_scale, inception_b7_7x1_3_relu_mode, inception_b7_7x1_3_relu_param_a, inception_b7_7x1_3_relu_param_b, inception_b7_7x1_3_relu);
    ERROR_CHECK_OBJECT(inception_b7_7x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_7x1_3_relu_node));

    // inception_b7_1x7_3 Layer
    vx_size inception_b7_1x7_3_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b7_1x7_3;
    inception_b7_1x7_3 = vxCreateVirtualTensor(graph,4, inception_b7_1x7_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3);
    vx_size inception_b7_1x7_3_W_dims[4] = { 7, 1, 224, 256 };
    vx_tensor inception_b7_1x7_3_W;
    inception_b7_1x7_3_W = vxCreateTensor(context,4, inception_b7_1x7_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_3_W, dataFolder + "/weights/inception_b7_1x7_3.f32"));
    vx_nn_convolution_params_t inception_b7_1x7_3_params;
    inception_b7_1x7_3_params.padding_x = 3;
    inception_b7_1x7_3_params.padding_y = 0;
    inception_b7_1x7_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_1x7_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_1x7_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_1x7_3_params.dilation_x = 0;
    inception_b7_1x7_3_params.dilation_y = 0;
    vx_node inception_b7_1x7_3_node;
    inception_b7_1x7_3_node = vxConvolutionLayer(graph, inception_b7_7x1_3_relu, inception_b7_1x7_3_W, NULL, &inception_b7_1x7_3_params, sizeof(inception_b7_1x7_3_params ), inception_b7_1x7_3);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_3_node));

    // inception_b7_1x7_3_bn Layer
    vx_size inception_b7_1x7_3_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b7_1x7_3_scale;
    inception_b7_1x7_3_scale = vxCreateVirtualTensor(graph,4, inception_b7_1x7_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_scale);
    vx_size inception_b7_1x7_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_b7_1x7_3_bn_eps = 0.001;
    vx_tensor inception_b7_1x7_3_bn_W;
    inception_b7_1x7_3_bn_W = vxCreateTensor(context,1, inception_b7_1x7_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_3_bn_W, dataFolder + "/weights/inception_b7_1x7_3_bn.f32"));
    vx_size inception_b7_1x7_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_b7_1x7_3_bn_B;
    inception_b7_1x7_3_bn_B = vxCreateTensor(context,1, inception_b7_1x7_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_3_bn_B, dataFolder + "/bias/inception_b7_1x7_3_bn.f32"));
    vx_size inception_b7_1x7_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_b7_1x7_3_scale_W;
    inception_b7_1x7_3_scale_W = vxCreateTensor(context,1, inception_b7_1x7_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_3_scale_W, dataFolder + "/weights/inception_b7_1x7_3_scale.f32"));
    vx_size inception_b7_1x7_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_b7_1x7_3_scale_B;
    inception_b7_1x7_3_scale_B = vxCreateTensor(context,1, inception_b7_1x7_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x7_3_scale_B, dataFolder + "/bias/inception_b7_1x7_3_scale.f32"));
    vx_node inception_b7_1x7_3_bn_node;
    inception_b7_1x7_3_bn_node = vxBatchNormalizationLayer(graph, inception_b7_1x7_3, inception_b7_1x7_3_bn_W, inception_b7_1x7_3_bn_B, inception_b7_1x7_3_scale_W, inception_b7_1x7_3_scale_B, inception_b7_1x7_3_bn_eps, inception_b7_1x7_3_scale);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_3_bn_node));

    // inception_b7_1x7_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_1x7_3_relu Layer
    vx_size inception_b7_1x7_3_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor inception_b7_1x7_3_relu;
    inception_b7_1x7_3_relu = vxCreateVirtualTensor(graph,4, inception_b7_1x7_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_relu);
    vx_enum inception_b7_1x7_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_1x7_3_relu_param_a = 0;
    vx_float32 inception_b7_1x7_3_relu_param_b = 0;
    vx_node inception_b7_1x7_3_relu_node;
    inception_b7_1x7_3_relu_node = vxActivationLayer(graph, inception_b7_1x7_3_scale, inception_b7_1x7_3_relu_mode, inception_b7_1x7_3_relu_param_a, inception_b7_1x7_3_relu_param_b, inception_b7_1x7_3_relu);
    ERROR_CHECK_OBJECT(inception_b7_1x7_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x7_3_relu_node));

    // inception_b7_pool_ave Layer
    vx_size inception_b7_pool_ave_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b7_pool_ave;
    inception_b7_pool_ave = vxCreateVirtualTensor(graph,4, inception_b7_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_pool_ave);
    vx_enum inception_b7_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_b7_pool_ave_kernel_w = 3;
    vx_size inception_b7_pool_ave_kernel_h = 3;
    vx_size inception_b7_pool_ave_pad_w = 1;
    vx_size inception_b7_pool_ave_pad_h = 1;
    vx_enum inception_b7_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_b7_pool_ave_node;
    inception_b7_pool_ave_node = vxPoolingLayer(graph, inception_b6_concat, inception_b7_pool_ave_type, inception_b7_pool_ave_kernel_w, inception_b7_pool_ave_kernel_h, inception_b7_pool_ave_pad_w, inception_b7_pool_ave_pad_h, inception_b7_pool_ave_roundPolicy, inception_b7_pool_ave );
    ERROR_CHECK_OBJECT(inception_b7_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_pool_ave_node));

    // inception_b7_1x1 Layer
    vx_size inception_b7_1x1_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b7_1x1;
    inception_b7_1x1 = vxCreateVirtualTensor(graph,4, inception_b7_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x1);
    vx_size inception_b7_1x1_W_dims[4] = { 1, 1, 1024, 128 };
    vx_tensor inception_b7_1x1_W;
    inception_b7_1x1_W = vxCreateTensor(context,4, inception_b7_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_W, dataFolder + "/weights/inception_b7_1x1.f32"));
    vx_nn_convolution_params_t inception_b7_1x1_params;
    inception_b7_1x1_params.padding_x = 0;
    inception_b7_1x1_params.padding_y = 0;
    inception_b7_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_b7_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_b7_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_b7_1x1_params.dilation_x = 0;
    inception_b7_1x1_params.dilation_y = 0;
    vx_node inception_b7_1x1_node;
    inception_b7_1x1_node = vxConvolutionLayer(graph, inception_b7_pool_ave, inception_b7_1x1_W, NULL, &inception_b7_1x1_params, sizeof(inception_b7_1x1_params ), inception_b7_1x1);
    ERROR_CHECK_OBJECT(inception_b7_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x1_node));

    // inception_b7_1x1_bn Layer
    vx_size inception_b7_1x1_scale_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b7_1x1_scale;
    inception_b7_1x1_scale = vxCreateVirtualTensor(graph,4, inception_b7_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_scale);
    vx_size inception_b7_1x1_bn_W_dims[1] = { 128 };
    vx_float32 inception_b7_1x1_bn_eps = 0.001;
    vx_tensor inception_b7_1x1_bn_W;
    inception_b7_1x1_bn_W = vxCreateTensor(context,1, inception_b7_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_bn_W, dataFolder + "/weights/inception_b7_1x1_bn.f32"));
    vx_size inception_b7_1x1_bn_B_dims[1] = { 128 };
    vx_tensor inception_b7_1x1_bn_B;
    inception_b7_1x1_bn_B = vxCreateTensor(context,1, inception_b7_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_bn_B, dataFolder + "/bias/inception_b7_1x1_bn.f32"));
    vx_size inception_b7_1x1_scale_W_dims[1] = { 128 };
    vx_tensor inception_b7_1x1_scale_W;
    inception_b7_1x1_scale_W = vxCreateTensor(context,1, inception_b7_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_scale_W, dataFolder + "/weights/inception_b7_1x1_scale.f32"));
    vx_size inception_b7_1x1_scale_B_dims[1] = { 128 };
    vx_tensor inception_b7_1x1_scale_B;
    inception_b7_1x1_scale_B = vxCreateTensor(context,1, inception_b7_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_b7_1x1_scale_B, dataFolder + "/bias/inception_b7_1x1_scale.f32"));
    vx_node inception_b7_1x1_bn_node;
    inception_b7_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_b7_1x1, inception_b7_1x1_bn_W, inception_b7_1x1_bn_B, inception_b7_1x1_scale_W, inception_b7_1x1_scale_B, inception_b7_1x1_bn_eps, inception_b7_1x1_scale);
    ERROR_CHECK_OBJECT(inception_b7_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x1_bn_node));

    // inception_b7_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_b7_1x1_relu Layer
    vx_size inception_b7_1x1_relu_dims[4] = { 17, 17, 128, 1 };
    vx_tensor inception_b7_1x1_relu;
    inception_b7_1x1_relu = vxCreateVirtualTensor(graph,4, inception_b7_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_1x1_relu);
    vx_enum inception_b7_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_b7_1x1_relu_param_a = 0;
    vx_float32 inception_b7_1x1_relu_param_b = 0;
    vx_node inception_b7_1x1_relu_node;
    inception_b7_1x1_relu_node = vxActivationLayer(graph, inception_b7_1x1_scale, inception_b7_1x1_relu_mode, inception_b7_1x1_relu_param_a, inception_b7_1x1_relu_param_b, inception_b7_1x1_relu);
    ERROR_CHECK_OBJECT(inception_b7_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_1x1_relu_node));

    // inception_b7_concat Layer
    vx_size inception_b7_concat_dims[4] = { 17, 17, 1024, 1 };
    vx_tensor inception_b7_concat;
    inception_b7_concat = vxCreateVirtualTensor(graph,4, inception_b7_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_b7_concat);
    vx_node inception_b7_concat_node;
    inception_b7_concat_node = vxConcatLayer(graph, inception_b7_concat, inception_b7_1x1_2_relu, inception_b7_7x1_relu, inception_b7_1x7_3_relu, inception_b7_1x1_relu, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_b7_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_b7_concat_node));

    // reduction_b_3x3_reduce Layer
    vx_size reduction_b_3x3_reduce_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_3x3_reduce;
    reduction_b_3x3_reduce = vxCreateVirtualTensor(graph,4, reduction_b_3x3_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce);
    vx_size reduction_b_3x3_reduce_W_dims[4] = { 1, 1, 1024, 192 };
    vx_tensor reduction_b_3x3_reduce_W;
    reduction_b_3x3_reduce_W = vxCreateTensor(context,4, reduction_b_3x3_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_W, dataFolder + "/weights/reduction_b_3x3_reduce.f32"));
    vx_nn_convolution_params_t reduction_b_3x3_reduce_params;
    reduction_b_3x3_reduce_params.padding_x = 0;
    reduction_b_3x3_reduce_params.padding_y = 0;
    reduction_b_3x3_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_3x3_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_3x3_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_3x3_reduce_params.dilation_x = 0;
    reduction_b_3x3_reduce_params.dilation_y = 0;
    vx_node reduction_b_3x3_reduce_node;
    reduction_b_3x3_reduce_node = vxConvolutionLayer(graph, inception_b7_concat, reduction_b_3x3_reduce_W, NULL, &reduction_b_3x3_reduce_params, sizeof(reduction_b_3x3_reduce_params ), reduction_b_3x3_reduce);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_reduce_node));

    // reduction_b_3x3_reduce_bn Layer
    vx_size reduction_b_3x3_reduce_scale_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_3x3_reduce_scale;
    reduction_b_3x3_reduce_scale = vxCreateVirtualTensor(graph,4, reduction_b_3x3_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_scale);
    vx_size reduction_b_3x3_reduce_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_3x3_reduce_bn_eps = 0.001;
    vx_tensor reduction_b_3x3_reduce_bn_W;
    reduction_b_3x3_reduce_bn_W = vxCreateTensor(context,1, reduction_b_3x3_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_bn_W, dataFolder + "/weights/reduction_b_3x3_reduce_bn.f32"));
    vx_size reduction_b_3x3_reduce_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_reduce_bn_B;
    reduction_b_3x3_reduce_bn_B = vxCreateTensor(context,1, reduction_b_3x3_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_bn_B, dataFolder + "/bias/reduction_b_3x3_reduce_bn.f32"));
    vx_size reduction_b_3x3_reduce_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_reduce_scale_W;
    reduction_b_3x3_reduce_scale_W = vxCreateTensor(context,1, reduction_b_3x3_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_scale_W, dataFolder + "/weights/reduction_b_3x3_reduce_scale.f32"));
    vx_size reduction_b_3x3_reduce_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_reduce_scale_B;
    reduction_b_3x3_reduce_scale_B = vxCreateTensor(context,1, reduction_b_3x3_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_reduce_scale_B, dataFolder + "/bias/reduction_b_3x3_reduce_scale.f32"));
    vx_node reduction_b_3x3_reduce_bn_node;
    reduction_b_3x3_reduce_bn_node = vxBatchNormalizationLayer(graph, reduction_b_3x3_reduce, reduction_b_3x3_reduce_bn_W, reduction_b_3x3_reduce_bn_B, reduction_b_3x3_reduce_scale_W, reduction_b_3x3_reduce_scale_B, reduction_b_3x3_reduce_bn_eps, reduction_b_3x3_reduce_scale);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_reduce_bn_node));

    // reduction_b_3x3_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_3x3_reduce_relu Layer
    vx_size reduction_b_3x3_reduce_relu_dims[4] = { 17, 17, 192, 1 };
    vx_tensor reduction_b_3x3_reduce_relu;
    reduction_b_3x3_reduce_relu = vxCreateVirtualTensor(graph,4, reduction_b_3x3_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_relu);
    vx_enum reduction_b_3x3_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_3x3_reduce_relu_param_a = 0;
    vx_float32 reduction_b_3x3_reduce_relu_param_b = 0;
    vx_node reduction_b_3x3_reduce_relu_node;
    reduction_b_3x3_reduce_relu_node = vxActivationLayer(graph, reduction_b_3x3_reduce_scale, reduction_b_3x3_reduce_relu_mode, reduction_b_3x3_reduce_relu_param_a, reduction_b_3x3_reduce_relu_param_b, reduction_b_3x3_reduce_relu);
    ERROR_CHECK_OBJECT(reduction_b_3x3_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_reduce_relu_node));

    // reduction_b_3x3 Layer
    vx_size reduction_b_3x3_dims[4] = { 8, 8, 192, 1 };
    vx_tensor reduction_b_3x3;
    reduction_b_3x3 = vxCreateVirtualTensor(graph,4, reduction_b_3x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3);
    vx_size reduction_b_3x3_W_dims[4] = { 3, 3, 192, 192 };
    vx_tensor reduction_b_3x3_W;
    reduction_b_3x3_W = vxCreateTensor(context,4, reduction_b_3x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_W, dataFolder + "/weights/reduction_b_3x3.f32"));
    vx_nn_convolution_params_t reduction_b_3x3_params;
    reduction_b_3x3_params.padding_x = 0;
    reduction_b_3x3_params.padding_y = 0;
    reduction_b_3x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_3x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_3x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_3x3_params.dilation_x = 0;
    reduction_b_3x3_params.dilation_y = 0;
    vx_node reduction_b_3x3_node;
    reduction_b_3x3_node = vxConvolutionLayer(graph, reduction_b_3x3_reduce_relu, reduction_b_3x3_W, NULL, &reduction_b_3x3_params, sizeof(reduction_b_3x3_params ), reduction_b_3x3);
    ERROR_CHECK_OBJECT(reduction_b_3x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_node));

    // reduction_b_3x3_bn Layer
    vx_size reduction_b_3x3_scale_dims[4] = { 8, 8, 192, 1 };
    vx_tensor reduction_b_3x3_scale;
    reduction_b_3x3_scale = vxCreateVirtualTensor(graph,4, reduction_b_3x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_scale);
    vx_size reduction_b_3x3_bn_W_dims[1] = { 192 };
    vx_float32 reduction_b_3x3_bn_eps = 0.001;
    vx_tensor reduction_b_3x3_bn_W;
    reduction_b_3x3_bn_W = vxCreateTensor(context,1, reduction_b_3x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_bn_W, dataFolder + "/weights/reduction_b_3x3_bn.f32"));
    vx_size reduction_b_3x3_bn_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_bn_B;
    reduction_b_3x3_bn_B = vxCreateTensor(context,1, reduction_b_3x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_bn_B, dataFolder + "/bias/reduction_b_3x3_bn.f32"));
    vx_size reduction_b_3x3_scale_W_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_scale_W;
    reduction_b_3x3_scale_W = vxCreateTensor(context,1, reduction_b_3x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_scale_W, dataFolder + "/weights/reduction_b_3x3_scale.f32"));
    vx_size reduction_b_3x3_scale_B_dims[1] = { 192 };
    vx_tensor reduction_b_3x3_scale_B;
    reduction_b_3x3_scale_B = vxCreateTensor(context,1, reduction_b_3x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_scale_B, dataFolder + "/bias/reduction_b_3x3_scale.f32"));
    vx_node reduction_b_3x3_bn_node;
    reduction_b_3x3_bn_node = vxBatchNormalizationLayer(graph, reduction_b_3x3, reduction_b_3x3_bn_W, reduction_b_3x3_bn_B, reduction_b_3x3_scale_W, reduction_b_3x3_scale_B, reduction_b_3x3_bn_eps, reduction_b_3x3_scale);
    ERROR_CHECK_OBJECT(reduction_b_3x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_bn_node));

    // reduction_b_3x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_3x3_relu Layer
    vx_size reduction_b_3x3_relu_dims[4] = { 8, 8, 192, 1 };
    vx_tensor reduction_b_3x3_relu;
    reduction_b_3x3_relu = vxCreateVirtualTensor(graph,4, reduction_b_3x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_relu);
    vx_enum reduction_b_3x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_3x3_relu_param_a = 0;
    vx_float32 reduction_b_3x3_relu_param_b = 0;
    vx_node reduction_b_3x3_relu_node;
    reduction_b_3x3_relu_node = vxActivationLayer(graph, reduction_b_3x3_scale, reduction_b_3x3_relu_mode, reduction_b_3x3_relu_param_a, reduction_b_3x3_relu_param_b, reduction_b_3x3_relu);
    ERROR_CHECK_OBJECT(reduction_b_3x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_relu_node));

    // reduction_b_1x7_reduce Layer
    vx_size reduction_b_1x7_reduce_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_b_1x7_reduce;
    reduction_b_1x7_reduce = vxCreateVirtualTensor(graph,4, reduction_b_1x7_reduce_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce);
    vx_size reduction_b_1x7_reduce_W_dims[4] = { 1, 1, 1024, 256 };
    vx_tensor reduction_b_1x7_reduce_W;
    reduction_b_1x7_reduce_W = vxCreateTensor(context,4, reduction_b_1x7_reduce_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_W, dataFolder + "/weights/reduction_b_1x7_reduce.f32"));
    vx_nn_convolution_params_t reduction_b_1x7_reduce_params;
    reduction_b_1x7_reduce_params.padding_x = 0;
    reduction_b_1x7_reduce_params.padding_y = 0;
    reduction_b_1x7_reduce_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_1x7_reduce_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_1x7_reduce_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_1x7_reduce_params.dilation_x = 0;
    reduction_b_1x7_reduce_params.dilation_y = 0;
    vx_node reduction_b_1x7_reduce_node;
    reduction_b_1x7_reduce_node = vxConvolutionLayer(graph, inception_b7_concat, reduction_b_1x7_reduce_W, NULL, &reduction_b_1x7_reduce_params, sizeof(reduction_b_1x7_reduce_params ), reduction_b_1x7_reduce);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_reduce_node));

    // reduction_b_1x7_reduce_bn Layer
    vx_size reduction_b_1x7_reduce_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_b_1x7_reduce_scale;
    reduction_b_1x7_reduce_scale = vxCreateVirtualTensor(graph,4, reduction_b_1x7_reduce_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_scale);
    vx_size reduction_b_1x7_reduce_bn_W_dims[1] = { 256 };
    vx_float32 reduction_b_1x7_reduce_bn_eps = 0.001;
    vx_tensor reduction_b_1x7_reduce_bn_W;
    reduction_b_1x7_reduce_bn_W = vxCreateTensor(context,1, reduction_b_1x7_reduce_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_bn_W, dataFolder + "/weights/reduction_b_1x7_reduce_bn.f32"));
    vx_size reduction_b_1x7_reduce_bn_B_dims[1] = { 256 };
    vx_tensor reduction_b_1x7_reduce_bn_B;
    reduction_b_1x7_reduce_bn_B = vxCreateTensor(context,1, reduction_b_1x7_reduce_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_bn_B, dataFolder + "/bias/reduction_b_1x7_reduce_bn.f32"));
    vx_size reduction_b_1x7_reduce_scale_W_dims[1] = { 256 };
    vx_tensor reduction_b_1x7_reduce_scale_W;
    reduction_b_1x7_reduce_scale_W = vxCreateTensor(context,1, reduction_b_1x7_reduce_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_scale_W, dataFolder + "/weights/reduction_b_1x7_reduce_scale.f32"));
    vx_size reduction_b_1x7_reduce_scale_B_dims[1] = { 256 };
    vx_tensor reduction_b_1x7_reduce_scale_B;
    reduction_b_1x7_reduce_scale_B = vxCreateTensor(context,1, reduction_b_1x7_reduce_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_reduce_scale_B, dataFolder + "/bias/reduction_b_1x7_reduce_scale.f32"));
    vx_node reduction_b_1x7_reduce_bn_node;
    reduction_b_1x7_reduce_bn_node = vxBatchNormalizationLayer(graph, reduction_b_1x7_reduce, reduction_b_1x7_reduce_bn_W, reduction_b_1x7_reduce_bn_B, reduction_b_1x7_reduce_scale_W, reduction_b_1x7_reduce_scale_B, reduction_b_1x7_reduce_bn_eps, reduction_b_1x7_reduce_scale);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_reduce_bn_node));

    // reduction_b_1x7_reduce_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_1x7_reduce_relu Layer
    vx_size reduction_b_1x7_reduce_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_b_1x7_reduce_relu;
    reduction_b_1x7_reduce_relu = vxCreateVirtualTensor(graph,4, reduction_b_1x7_reduce_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_relu);
    vx_enum reduction_b_1x7_reduce_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_1x7_reduce_relu_param_a = 0;
    vx_float32 reduction_b_1x7_reduce_relu_param_b = 0;
    vx_node reduction_b_1x7_reduce_relu_node;
    reduction_b_1x7_reduce_relu_node = vxActivationLayer(graph, reduction_b_1x7_reduce_scale, reduction_b_1x7_reduce_relu_mode, reduction_b_1x7_reduce_relu_param_a, reduction_b_1x7_reduce_relu_param_b, reduction_b_1x7_reduce_relu);
    ERROR_CHECK_OBJECT(reduction_b_1x7_reduce_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_reduce_relu_node));

    // reduction_b_1x7 Layer
    vx_size reduction_b_1x7_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_b_1x7;
    reduction_b_1x7 = vxCreateVirtualTensor(graph,4, reduction_b_1x7_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7);
    vx_size reduction_b_1x7_W_dims[4] = { 7, 1, 256, 256 };
    vx_tensor reduction_b_1x7_W;
    reduction_b_1x7_W = vxCreateTensor(context,4, reduction_b_1x7_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_W, dataFolder + "/weights/reduction_b_1x7.f32"));
    vx_nn_convolution_params_t reduction_b_1x7_params;
    reduction_b_1x7_params.padding_x = 3;
    reduction_b_1x7_params.padding_y = 0;
    reduction_b_1x7_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_1x7_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_1x7_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_1x7_params.dilation_x = 0;
    reduction_b_1x7_params.dilation_y = 0;
    vx_node reduction_b_1x7_node;
    reduction_b_1x7_node = vxConvolutionLayer(graph, reduction_b_1x7_reduce_relu, reduction_b_1x7_W, NULL, &reduction_b_1x7_params, sizeof(reduction_b_1x7_params ), reduction_b_1x7);
    ERROR_CHECK_OBJECT(reduction_b_1x7_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_node));

    // reduction_b_1x7_bn Layer
    vx_size reduction_b_1x7_scale_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_b_1x7_scale;
    reduction_b_1x7_scale = vxCreateVirtualTensor(graph,4, reduction_b_1x7_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_scale);
    vx_size reduction_b_1x7_bn_W_dims[1] = { 256 };
    vx_float32 reduction_b_1x7_bn_eps = 0.001;
    vx_tensor reduction_b_1x7_bn_W;
    reduction_b_1x7_bn_W = vxCreateTensor(context,1, reduction_b_1x7_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_bn_W, dataFolder + "/weights/reduction_b_1x7_bn.f32"));
    vx_size reduction_b_1x7_bn_B_dims[1] = { 256 };
    vx_tensor reduction_b_1x7_bn_B;
    reduction_b_1x7_bn_B = vxCreateTensor(context,1, reduction_b_1x7_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_bn_B, dataFolder + "/bias/reduction_b_1x7_bn.f32"));
    vx_size reduction_b_1x7_scale_W_dims[1] = { 256 };
    vx_tensor reduction_b_1x7_scale_W;
    reduction_b_1x7_scale_W = vxCreateTensor(context,1, reduction_b_1x7_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_scale_W, dataFolder + "/weights/reduction_b_1x7_scale.f32"));
    vx_size reduction_b_1x7_scale_B_dims[1] = { 256 };
    vx_tensor reduction_b_1x7_scale_B;
    reduction_b_1x7_scale_B = vxCreateTensor(context,1, reduction_b_1x7_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_1x7_scale_B, dataFolder + "/bias/reduction_b_1x7_scale.f32"));
    vx_node reduction_b_1x7_bn_node;
    reduction_b_1x7_bn_node = vxBatchNormalizationLayer(graph, reduction_b_1x7, reduction_b_1x7_bn_W, reduction_b_1x7_bn_B, reduction_b_1x7_scale_W, reduction_b_1x7_scale_B, reduction_b_1x7_bn_eps, reduction_b_1x7_scale);
    ERROR_CHECK_OBJECT(reduction_b_1x7_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_bn_node));

    // reduction_b_1x7_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_1x7_relu Layer
    vx_size reduction_b_1x7_relu_dims[4] = { 17, 17, 256, 1 };
    vx_tensor reduction_b_1x7_relu;
    reduction_b_1x7_relu = vxCreateVirtualTensor(graph,4, reduction_b_1x7_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_1x7_relu);
    vx_enum reduction_b_1x7_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_1x7_relu_param_a = 0;
    vx_float32 reduction_b_1x7_relu_param_b = 0;
    vx_node reduction_b_1x7_relu_node;
    reduction_b_1x7_relu_node = vxActivationLayer(graph, reduction_b_1x7_scale, reduction_b_1x7_relu_mode, reduction_b_1x7_relu_param_a, reduction_b_1x7_relu_param_b, reduction_b_1x7_relu);
    ERROR_CHECK_OBJECT(reduction_b_1x7_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_1x7_relu_node));

    // reduction_b_7x1 Layer
    vx_size reduction_b_7x1_dims[4] = { 17, 17, 320, 1 };
    vx_tensor reduction_b_7x1;
    reduction_b_7x1 = vxCreateVirtualTensor(graph,4, reduction_b_7x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_7x1);
    vx_size reduction_b_7x1_W_dims[4] = { 1, 7, 256, 320 };
    vx_tensor reduction_b_7x1_W;
    reduction_b_7x1_W = vxCreateTensor(context,4, reduction_b_7x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_W, dataFolder + "/weights/reduction_b_7x1.f32"));
    vx_nn_convolution_params_t reduction_b_7x1_params;
    reduction_b_7x1_params.padding_x = 0;
    reduction_b_7x1_params.padding_y = 3;
    reduction_b_7x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_7x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_7x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_7x1_params.dilation_x = 0;
    reduction_b_7x1_params.dilation_y = 0;
    vx_node reduction_b_7x1_node;
    reduction_b_7x1_node = vxConvolutionLayer(graph, reduction_b_1x7_relu, reduction_b_7x1_W, NULL, &reduction_b_7x1_params, sizeof(reduction_b_7x1_params ), reduction_b_7x1);
    ERROR_CHECK_OBJECT(reduction_b_7x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_7x1_node));

    // reduction_b_7x1_bn Layer
    vx_size reduction_b_7x1_scale_dims[4] = { 17, 17, 320, 1 };
    vx_tensor reduction_b_7x1_scale;
    reduction_b_7x1_scale = vxCreateVirtualTensor(graph,4, reduction_b_7x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_scale);
    vx_size reduction_b_7x1_bn_W_dims[1] = { 320 };
    vx_float32 reduction_b_7x1_bn_eps = 0.001;
    vx_tensor reduction_b_7x1_bn_W;
    reduction_b_7x1_bn_W = vxCreateTensor(context,1, reduction_b_7x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_bn_W, dataFolder + "/weights/reduction_b_7x1_bn.f32"));
    vx_size reduction_b_7x1_bn_B_dims[1] = { 320 };
    vx_tensor reduction_b_7x1_bn_B;
    reduction_b_7x1_bn_B = vxCreateTensor(context,1, reduction_b_7x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_bn_B, dataFolder + "/bias/reduction_b_7x1_bn.f32"));
    vx_size reduction_b_7x1_scale_W_dims[1] = { 320 };
    vx_tensor reduction_b_7x1_scale_W;
    reduction_b_7x1_scale_W = vxCreateTensor(context,1, reduction_b_7x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_scale_W, dataFolder + "/weights/reduction_b_7x1_scale.f32"));
    vx_size reduction_b_7x1_scale_B_dims[1] = { 320 };
    vx_tensor reduction_b_7x1_scale_B;
    reduction_b_7x1_scale_B = vxCreateTensor(context,1, reduction_b_7x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_7x1_scale_B, dataFolder + "/bias/reduction_b_7x1_scale.f32"));
    vx_node reduction_b_7x1_bn_node;
    reduction_b_7x1_bn_node = vxBatchNormalizationLayer(graph, reduction_b_7x1, reduction_b_7x1_bn_W, reduction_b_7x1_bn_B, reduction_b_7x1_scale_W, reduction_b_7x1_scale_B, reduction_b_7x1_bn_eps, reduction_b_7x1_scale);
    ERROR_CHECK_OBJECT(reduction_b_7x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_7x1_bn_node));

    // reduction_b_7x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_7x1_relu Layer
    vx_size reduction_b_7x1_relu_dims[4] = { 17, 17, 320, 1 };
    vx_tensor reduction_b_7x1_relu;
    reduction_b_7x1_relu = vxCreateVirtualTensor(graph,4, reduction_b_7x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_7x1_relu);
    vx_enum reduction_b_7x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_7x1_relu_param_a = 0;
    vx_float32 reduction_b_7x1_relu_param_b = 0;
    vx_node reduction_b_7x1_relu_node;
    reduction_b_7x1_relu_node = vxActivationLayer(graph, reduction_b_7x1_scale, reduction_b_7x1_relu_mode, reduction_b_7x1_relu_param_a, reduction_b_7x1_relu_param_b, reduction_b_7x1_relu);
    ERROR_CHECK_OBJECT(reduction_b_7x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_7x1_relu_node));

    // reduction_b_3x3_2 Layer
    vx_size reduction_b_3x3_2_dims[4] = { 8, 8, 320, 1 };
    vx_tensor reduction_b_3x3_2;
    reduction_b_3x3_2 = vxCreateVirtualTensor(graph,4, reduction_b_3x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2);
    vx_size reduction_b_3x3_2_W_dims[4] = { 3, 3, 320, 320 };
    vx_tensor reduction_b_3x3_2_W;
    reduction_b_3x3_2_W = vxCreateTensor(context,4, reduction_b_3x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_W, dataFolder + "/weights/reduction_b_3x3_2.f32"));
    vx_nn_convolution_params_t reduction_b_3x3_2_params;
    reduction_b_3x3_2_params.padding_x = 0;
    reduction_b_3x3_2_params.padding_y = 0;
    reduction_b_3x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    reduction_b_3x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    reduction_b_3x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    reduction_b_3x3_2_params.dilation_x = 0;
    reduction_b_3x3_2_params.dilation_y = 0;
    vx_node reduction_b_3x3_2_node;
    reduction_b_3x3_2_node = vxConvolutionLayer(graph, reduction_b_7x1_relu, reduction_b_3x3_2_W, NULL, &reduction_b_3x3_2_params, sizeof(reduction_b_3x3_2_params ), reduction_b_3x3_2);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_2_node));

    // reduction_b_3x3_2_bn Layer
    vx_size reduction_b_3x3_2_scale_dims[4] = { 8, 8, 320, 1 };
    vx_tensor reduction_b_3x3_2_scale;
    reduction_b_3x3_2_scale = vxCreateVirtualTensor(graph,4, reduction_b_3x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_scale);
    vx_size reduction_b_3x3_2_bn_W_dims[1] = { 320 };
    vx_float32 reduction_b_3x3_2_bn_eps = 0.001;
    vx_tensor reduction_b_3x3_2_bn_W;
    reduction_b_3x3_2_bn_W = vxCreateTensor(context,1, reduction_b_3x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_bn_W, dataFolder + "/weights/reduction_b_3x3_2_bn.f32"));
    vx_size reduction_b_3x3_2_bn_B_dims[1] = { 320 };
    vx_tensor reduction_b_3x3_2_bn_B;
    reduction_b_3x3_2_bn_B = vxCreateTensor(context,1, reduction_b_3x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_bn_B, dataFolder + "/bias/reduction_b_3x3_2_bn.f32"));
    vx_size reduction_b_3x3_2_scale_W_dims[1] = { 320 };
    vx_tensor reduction_b_3x3_2_scale_W;
    reduction_b_3x3_2_scale_W = vxCreateTensor(context,1, reduction_b_3x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_scale_W, dataFolder + "/weights/reduction_b_3x3_2_scale.f32"));
    vx_size reduction_b_3x3_2_scale_B_dims[1] = { 320 };
    vx_tensor reduction_b_3x3_2_scale_B;
    reduction_b_3x3_2_scale_B = vxCreateTensor(context,1, reduction_b_3x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(reduction_b_3x3_2_scale_B, dataFolder + "/bias/reduction_b_3x3_2_scale.f32"));
    vx_node reduction_b_3x3_2_bn_node;
    reduction_b_3x3_2_bn_node = vxBatchNormalizationLayer(graph, reduction_b_3x3_2, reduction_b_3x3_2_bn_W, reduction_b_3x3_2_bn_B, reduction_b_3x3_2_scale_W, reduction_b_3x3_2_scale_B, reduction_b_3x3_2_bn_eps, reduction_b_3x3_2_scale);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_2_bn_node));

    // reduction_b_3x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // reduction_b_3x3_2_relu Layer
    vx_size reduction_b_3x3_2_relu_dims[4] = { 8, 8, 320, 1 };
    vx_tensor reduction_b_3x3_2_relu;
    reduction_b_3x3_2_relu = vxCreateVirtualTensor(graph,4, reduction_b_3x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_relu);
    vx_enum reduction_b_3x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 reduction_b_3x3_2_relu_param_a = 0;
    vx_float32 reduction_b_3x3_2_relu_param_b = 0;
    vx_node reduction_b_3x3_2_relu_node;
    reduction_b_3x3_2_relu_node = vxActivationLayer(graph, reduction_b_3x3_2_scale, reduction_b_3x3_2_relu_mode, reduction_b_3x3_2_relu_param_a, reduction_b_3x3_2_relu_param_b, reduction_b_3x3_2_relu);
    ERROR_CHECK_OBJECT(reduction_b_3x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_3x3_2_relu_node));

    // reduction_b_pool Layer
    vx_size reduction_b_pool_dims[4] = { 8, 8, 1024, 1 };
    vx_tensor reduction_b_pool;
    reduction_b_pool = vxCreateVirtualTensor(graph,4, reduction_b_pool_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_pool);
    vx_enum reduction_b_pool_type = VX_NN_POOLING_MAX;
    vx_size reduction_b_pool_kernel_w = 3;
    vx_size reduction_b_pool_kernel_h = 3;
    vx_size reduction_b_pool_pad_w = 0;
    vx_size reduction_b_pool_pad_h = 0;
    vx_enum reduction_b_pool_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node reduction_b_pool_node;
    reduction_b_pool_node = vxPoolingLayer(graph, inception_b7_concat, reduction_b_pool_type, reduction_b_pool_kernel_w, reduction_b_pool_kernel_h, reduction_b_pool_pad_w, reduction_b_pool_pad_h, reduction_b_pool_roundPolicy, reduction_b_pool );
    ERROR_CHECK_OBJECT(reduction_b_pool_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_pool_node));

    // reduction_b_concat Layer
    vx_size reduction_b_concat_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor reduction_b_concat;
    reduction_b_concat = vxCreateVirtualTensor(graph,4, reduction_b_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(reduction_b_concat);
    vx_node reduction_b_concat_node;
    reduction_b_concat_node = vxConcatLayer(graph, reduction_b_concat, reduction_b_3x3_relu, reduction_b_3x3_2_relu, reduction_b_pool, NULL, NULL, NULL, NULL, NULL );
    ERROR_CHECK_OBJECT(reduction_b_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&reduction_b_concat_node));

    // inception_c1_1x1_2 Layer
    vx_size inception_c1_1x1_2_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x1_2;
    inception_c1_1x1_2 = vxCreateVirtualTensor(graph,4, inception_c1_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2);
    vx_size inception_c1_1x1_2_W_dims[4] = { 1, 1, 1536, 256 };
    vx_tensor inception_c1_1x1_2_W;
    inception_c1_1x1_2_W = vxCreateTensor(context,4, inception_c1_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_W, dataFolder + "/weights/inception_c1_1x1_2.f32"));
    vx_nn_convolution_params_t inception_c1_1x1_2_params;
    inception_c1_1x1_2_params.padding_x = 0;
    inception_c1_1x1_2_params.padding_y = 0;
    inception_c1_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x1_2_params.dilation_x = 0;
    inception_c1_1x1_2_params.dilation_y = 0;
    vx_node inception_c1_1x1_2_node;
    inception_c1_1x1_2_node = vxConvolutionLayer(graph, reduction_b_concat, inception_c1_1x1_2_W, NULL, &inception_c1_1x1_2_params, sizeof(inception_c1_1x1_2_params ), inception_c1_1x1_2);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_2_node));

    // inception_c1_1x1_2_bn Layer
    vx_size inception_c1_1x1_2_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x1_2_scale;
    inception_c1_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_scale);
    vx_size inception_c1_1x1_2_bn_W_dims[1] = { 256 };
    vx_float32 inception_c1_1x1_2_bn_eps = 0.001;
    vx_tensor inception_c1_1x1_2_bn_W;
    inception_c1_1x1_2_bn_W = vxCreateTensor(context,1, inception_c1_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_bn_W, dataFolder + "/weights/inception_c1_1x1_2_bn.f32"));
    vx_size inception_c1_1x1_2_bn_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x1_2_bn_B;
    inception_c1_1x1_2_bn_B = vxCreateTensor(context,1, inception_c1_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_bn_B, dataFolder + "/bias/inception_c1_1x1_2_bn.f32"));
    vx_size inception_c1_1x1_2_scale_W_dims[1] = { 256 };
    vx_tensor inception_c1_1x1_2_scale_W;
    inception_c1_1x1_2_scale_W = vxCreateTensor(context,1, inception_c1_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_scale_W, dataFolder + "/weights/inception_c1_1x1_2_scale.f32"));
    vx_size inception_c1_1x1_2_scale_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x1_2_scale_B;
    inception_c1_1x1_2_scale_B = vxCreateTensor(context,1, inception_c1_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_2_scale_B, dataFolder + "/bias/inception_c1_1x1_2_scale.f32"));
    vx_node inception_c1_1x1_2_bn_node;
    inception_c1_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x1_2, inception_c1_1x1_2_bn_W, inception_c1_1x1_2_bn_B, inception_c1_1x1_2_scale_W, inception_c1_1x1_2_scale_B, inception_c1_1x1_2_bn_eps, inception_c1_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_2_bn_node));

    // inception_c1_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x1_2_relu Layer
    vx_size inception_c1_1x1_2_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x1_2_relu;
    inception_c1_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_relu);
    vx_enum inception_c1_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x1_2_relu_param_a = 0;
    vx_float32 inception_c1_1x1_2_relu_param_b = 0;
    vx_node inception_c1_1x1_2_relu_node;
    inception_c1_1x1_2_relu_node = vxActivationLayer(graph, inception_c1_1x1_2_scale, inception_c1_1x1_2_relu_mode, inception_c1_1x1_2_relu_param_a, inception_c1_1x1_2_relu_param_b, inception_c1_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_2_relu_node));

    // inception_c1_1x1_3 Layer
    vx_size inception_c1_1x1_3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x1_3;
    inception_c1_1x1_3 = vxCreateVirtualTensor(graph,4, inception_c1_1x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3);
    vx_size inception_c1_1x1_3_W_dims[4] = { 1, 1, 1536, 384 };
    vx_tensor inception_c1_1x1_3_W;
    inception_c1_1x1_3_W = vxCreateTensor(context,4, inception_c1_1x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_3_W, dataFolder + "/weights/inception_c1_1x1_3.f32"));
    vx_nn_convolution_params_t inception_c1_1x1_3_params;
    inception_c1_1x1_3_params.padding_x = 0;
    inception_c1_1x1_3_params.padding_y = 0;
    inception_c1_1x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x1_3_params.dilation_x = 0;
    inception_c1_1x1_3_params.dilation_y = 0;
    vx_node inception_c1_1x1_3_node;
    inception_c1_1x1_3_node = vxConvolutionLayer(graph, reduction_b_concat, inception_c1_1x1_3_W, NULL, &inception_c1_1x1_3_params, sizeof(inception_c1_1x1_3_params ), inception_c1_1x1_3);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_3_node));

    // inception_c1_1x1_3_bn Layer
    vx_size inception_c1_1x1_3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x1_3_scale;
    inception_c1_1x1_3_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_scale);
    vx_size inception_c1_1x1_3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_1x1_3_bn_eps = 0.001;
    vx_tensor inception_c1_1x1_3_bn_W;
    inception_c1_1x1_3_bn_W = vxCreateTensor(context,1, inception_c1_1x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_3_bn_W, dataFolder + "/weights/inception_c1_1x1_3_bn.f32"));
    vx_size inception_c1_1x1_3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x1_3_bn_B;
    inception_c1_1x1_3_bn_B = vxCreateTensor(context,1, inception_c1_1x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_3_bn_B, dataFolder + "/bias/inception_c1_1x1_3_bn.f32"));
    vx_size inception_c1_1x1_3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_1x1_3_scale_W;
    inception_c1_1x1_3_scale_W = vxCreateTensor(context,1, inception_c1_1x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_3_scale_W, dataFolder + "/weights/inception_c1_1x1_3_scale.f32"));
    vx_size inception_c1_1x1_3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x1_3_scale_B;
    inception_c1_1x1_3_scale_B = vxCreateTensor(context,1, inception_c1_1x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_3_scale_B, dataFolder + "/bias/inception_c1_1x1_3_scale.f32"));
    vx_node inception_c1_1x1_3_bn_node;
    inception_c1_1x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x1_3, inception_c1_1x1_3_bn_W, inception_c1_1x1_3_bn_B, inception_c1_1x1_3_scale_W, inception_c1_1x1_3_scale_B, inception_c1_1x1_3_bn_eps, inception_c1_1x1_3_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_3_bn_node));

    // inception_c1_1x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x1_3_relu Layer
    vx_size inception_c1_1x1_3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x1_3_relu;
    inception_c1_1x1_3_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_relu);
    vx_enum inception_c1_1x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x1_3_relu_param_a = 0;
    vx_float32 inception_c1_1x1_3_relu_param_b = 0;
    vx_node inception_c1_1x1_3_relu_node;
    inception_c1_1x1_3_relu_node = vxActivationLayer(graph, inception_c1_1x1_3_scale, inception_c1_1x1_3_relu_mode, inception_c1_1x1_3_relu_param_a, inception_c1_1x1_3_relu_param_b, inception_c1_1x1_3_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_3_relu_node));

    // inception_c1_1x3 Layer
    vx_size inception_c1_1x3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x3;
    inception_c1_1x3 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3);
    vx_size inception_c1_1x3_W_dims[4] = { 3, 1, 384, 256 };
    vx_tensor inception_c1_1x3_W;
    inception_c1_1x3_W = vxCreateTensor(context,4, inception_c1_1x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_W, dataFolder + "/weights/inception_c1_1x3.f32"));
    vx_nn_convolution_params_t inception_c1_1x3_params;
    inception_c1_1x3_params.padding_x = 1;
    inception_c1_1x3_params.padding_y = 0;
    inception_c1_1x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x3_params.dilation_x = 0;
    inception_c1_1x3_params.dilation_y = 0;
    vx_node inception_c1_1x3_node;
    inception_c1_1x3_node = vxConvolutionLayer(graph, inception_c1_1x1_3_relu, inception_c1_1x3_W, NULL, &inception_c1_1x3_params, sizeof(inception_c1_1x3_params ), inception_c1_1x3);
    ERROR_CHECK_OBJECT(inception_c1_1x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_node));

    // inception_c1_1x3_bn Layer
    vx_size inception_c1_1x3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x3_scale;
    inception_c1_1x3_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_scale);
    vx_size inception_c1_1x3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c1_1x3_bn_eps = 0.001;
    vx_tensor inception_c1_1x3_bn_W;
    inception_c1_1x3_bn_W = vxCreateTensor(context,1, inception_c1_1x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_bn_W, dataFolder + "/weights/inception_c1_1x3_bn.f32"));
    vx_size inception_c1_1x3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x3_bn_B;
    inception_c1_1x3_bn_B = vxCreateTensor(context,1, inception_c1_1x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_bn_B, dataFolder + "/bias/inception_c1_1x3_bn.f32"));
    vx_size inception_c1_1x3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c1_1x3_scale_W;
    inception_c1_1x3_scale_W = vxCreateTensor(context,1, inception_c1_1x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_scale_W, dataFolder + "/weights/inception_c1_1x3_scale.f32"));
    vx_size inception_c1_1x3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x3_scale_B;
    inception_c1_1x3_scale_B = vxCreateTensor(context,1, inception_c1_1x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_scale_B, dataFolder + "/bias/inception_c1_1x3_scale.f32"));
    vx_node inception_c1_1x3_bn_node;
    inception_c1_1x3_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x3, inception_c1_1x3_bn_W, inception_c1_1x3_bn_B, inception_c1_1x3_scale_W, inception_c1_1x3_scale_B, inception_c1_1x3_bn_eps, inception_c1_1x3_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_bn_node));

    // inception_c1_1x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x3_relu Layer
    vx_size inception_c1_1x3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x3_relu;
    inception_c1_1x3_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_relu);
    vx_enum inception_c1_1x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x3_relu_param_a = 0;
    vx_float32 inception_c1_1x3_relu_param_b = 0;
    vx_node inception_c1_1x3_relu_node;
    inception_c1_1x3_relu_node = vxActivationLayer(graph, inception_c1_1x3_scale, inception_c1_1x3_relu_mode, inception_c1_1x3_relu_param_a, inception_c1_1x3_relu_param_b, inception_c1_1x3_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_relu_node));

    // inception_c1_3x1 Layer
    vx_size inception_c1_3x1_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_3x1;
    inception_c1_3x1 = vxCreateVirtualTensor(graph,4, inception_c1_3x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1);
    vx_size inception_c1_3x1_W_dims[4] = { 1, 3, 384, 256 };
    vx_tensor inception_c1_3x1_W;
    inception_c1_3x1_W = vxCreateTensor(context,4, inception_c1_3x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_W, dataFolder + "/weights/inception_c1_3x1.f32"));
    vx_nn_convolution_params_t inception_c1_3x1_params;
    inception_c1_3x1_params.padding_x = 0;
    inception_c1_3x1_params.padding_y = 1;
    inception_c1_3x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x1_params.dilation_x = 0;
    inception_c1_3x1_params.dilation_y = 0;
    vx_node inception_c1_3x1_node;
    inception_c1_3x1_node = vxConvolutionLayer(graph, inception_c1_1x1_3_relu, inception_c1_3x1_W, NULL, &inception_c1_3x1_params, sizeof(inception_c1_3x1_params ), inception_c1_3x1);
    ERROR_CHECK_OBJECT(inception_c1_3x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_node));

    // inception_c1_3x1_bn Layer
    vx_size inception_c1_3x1_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_3x1_scale;
    inception_c1_3x1_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_scale);
    vx_size inception_c1_3x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_c1_3x1_bn_eps = 0.001;
    vx_tensor inception_c1_3x1_bn_W;
    inception_c1_3x1_bn_W = vxCreateTensor(context,1, inception_c1_3x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_bn_W, dataFolder + "/weights/inception_c1_3x1_bn.f32"));
    vx_size inception_c1_3x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_c1_3x1_bn_B;
    inception_c1_3x1_bn_B = vxCreateTensor(context,1, inception_c1_3x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_bn_B, dataFolder + "/bias/inception_c1_3x1_bn.f32"));
    vx_size inception_c1_3x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_c1_3x1_scale_W;
    inception_c1_3x1_scale_W = vxCreateTensor(context,1, inception_c1_3x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_scale_W, dataFolder + "/weights/inception_c1_3x1_scale.f32"));
    vx_size inception_c1_3x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_c1_3x1_scale_B;
    inception_c1_3x1_scale_B = vxCreateTensor(context,1, inception_c1_3x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_scale_B, dataFolder + "/bias/inception_c1_3x1_scale.f32"));
    vx_node inception_c1_3x1_bn_node;
    inception_c1_3x1_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x1, inception_c1_3x1_bn_W, inception_c1_3x1_bn_B, inception_c1_3x1_scale_W, inception_c1_3x1_scale_B, inception_c1_3x1_bn_eps, inception_c1_3x1_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_bn_node));

    // inception_c1_3x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x1_relu Layer
    vx_size inception_c1_3x1_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_3x1_relu;
    inception_c1_3x1_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_relu);
    vx_enum inception_c1_3x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x1_relu_param_a = 0;
    vx_float32 inception_c1_3x1_relu_param_b = 0;
    vx_node inception_c1_3x1_relu_node;
    inception_c1_3x1_relu_node = vxActivationLayer(graph, inception_c1_3x1_scale, inception_c1_3x1_relu_mode, inception_c1_3x1_relu_param_a, inception_c1_3x1_relu_param_b, inception_c1_3x1_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_relu_node));

    // inception_c1_1x1_4 Layer
    vx_size inception_c1_1x1_4_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x1_4;
    inception_c1_1x1_4 = vxCreateVirtualTensor(graph,4, inception_c1_1x1_4_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4);
    vx_size inception_c1_1x1_4_W_dims[4] = { 1, 1, 1536, 384 };
    vx_tensor inception_c1_1x1_4_W;
    inception_c1_1x1_4_W = vxCreateTensor(context,4, inception_c1_1x1_4_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_4_W, dataFolder + "/weights/inception_c1_1x1_4.f32"));
    vx_nn_convolution_params_t inception_c1_1x1_4_params;
    inception_c1_1x1_4_params.padding_x = 0;
    inception_c1_1x1_4_params.padding_y = 0;
    inception_c1_1x1_4_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x1_4_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x1_4_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x1_4_params.dilation_x = 0;
    inception_c1_1x1_4_params.dilation_y = 0;
    vx_node inception_c1_1x1_4_node;
    inception_c1_1x1_4_node = vxConvolutionLayer(graph, reduction_b_concat, inception_c1_1x1_4_W, NULL, &inception_c1_1x1_4_params, sizeof(inception_c1_1x1_4_params ), inception_c1_1x1_4);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_4_node));

    // inception_c1_1x1_4_bn Layer
    vx_size inception_c1_1x1_4_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x1_4_scale;
    inception_c1_1x1_4_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x1_4_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_scale);
    vx_size inception_c1_1x1_4_bn_W_dims[1] = { 384 };
    vx_float32 inception_c1_1x1_4_bn_eps = 0.001;
    vx_tensor inception_c1_1x1_4_bn_W;
    inception_c1_1x1_4_bn_W = vxCreateTensor(context,1, inception_c1_1x1_4_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_4_bn_W, dataFolder + "/weights/inception_c1_1x1_4_bn.f32"));
    vx_size inception_c1_1x1_4_bn_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x1_4_bn_B;
    inception_c1_1x1_4_bn_B = vxCreateTensor(context,1, inception_c1_1x1_4_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_4_bn_B, dataFolder + "/bias/inception_c1_1x1_4_bn.f32"));
    vx_size inception_c1_1x1_4_scale_W_dims[1] = { 384 };
    vx_tensor inception_c1_1x1_4_scale_W;
    inception_c1_1x1_4_scale_W = vxCreateTensor(context,1, inception_c1_1x1_4_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_4_scale_W, dataFolder + "/weights/inception_c1_1x1_4_scale.f32"));
    vx_size inception_c1_1x1_4_scale_B_dims[1] = { 384 };
    vx_tensor inception_c1_1x1_4_scale_B;
    inception_c1_1x1_4_scale_B = vxCreateTensor(context,1, inception_c1_1x1_4_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_4_scale_B, dataFolder + "/bias/inception_c1_1x1_4_scale.f32"));
    vx_node inception_c1_1x1_4_bn_node;
    inception_c1_1x1_4_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x1_4, inception_c1_1x1_4_bn_W, inception_c1_1x1_4_bn_B, inception_c1_1x1_4_scale_W, inception_c1_1x1_4_scale_B, inception_c1_1x1_4_bn_eps, inception_c1_1x1_4_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_4_bn_node));

    // inception_c1_1x1_4_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x1_4_relu Layer
    vx_size inception_c1_1x1_4_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c1_1x1_4_relu;
    inception_c1_1x1_4_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x1_4_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_relu);
    vx_enum inception_c1_1x1_4_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x1_4_relu_param_a = 0;
    vx_float32 inception_c1_1x1_4_relu_param_b = 0;
    vx_node inception_c1_1x1_4_relu_node;
    inception_c1_1x1_4_relu_node = vxActivationLayer(graph, inception_c1_1x1_4_scale, inception_c1_1x1_4_relu_mode, inception_c1_1x1_4_relu_param_a, inception_c1_1x1_4_relu_param_b, inception_c1_1x1_4_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x1_4_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_4_relu_node));

    // inception_c1_3x1_2 Layer
    vx_size inception_c1_3x1_2_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c1_3x1_2;
    inception_c1_3x1_2 = vxCreateVirtualTensor(graph,4, inception_c1_3x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2);
    vx_size inception_c1_3x1_2_W_dims[4] = { 1, 3, 384, 448 };
    vx_tensor inception_c1_3x1_2_W;
    inception_c1_3x1_2_W = vxCreateTensor(context,4, inception_c1_3x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_W, dataFolder + "/weights/inception_c1_3x1_2.f32"));
    vx_nn_convolution_params_t inception_c1_3x1_2_params;
    inception_c1_3x1_2_params.padding_x = 0;
    inception_c1_3x1_2_params.padding_y = 1;
    inception_c1_3x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x1_2_params.dilation_x = 0;
    inception_c1_3x1_2_params.dilation_y = 0;
    vx_node inception_c1_3x1_2_node;
    inception_c1_3x1_2_node = vxConvolutionLayer(graph, inception_c1_1x1_4_relu, inception_c1_3x1_2_W, NULL, &inception_c1_3x1_2_params, sizeof(inception_c1_3x1_2_params ), inception_c1_3x1_2);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_2_node));

    // inception_c1_3x1_2_bn Layer
    vx_size inception_c1_3x1_2_scale_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c1_3x1_2_scale;
    inception_c1_3x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_scale);
    vx_size inception_c1_3x1_2_bn_W_dims[1] = { 448 };
    vx_float32 inception_c1_3x1_2_bn_eps = 0.001;
    vx_tensor inception_c1_3x1_2_bn_W;
    inception_c1_3x1_2_bn_W = vxCreateTensor(context,1, inception_c1_3x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_bn_W, dataFolder + "/weights/inception_c1_3x1_2_bn.f32"));
    vx_size inception_c1_3x1_2_bn_B_dims[1] = { 448 };
    vx_tensor inception_c1_3x1_2_bn_B;
    inception_c1_3x1_2_bn_B = vxCreateTensor(context,1, inception_c1_3x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_bn_B, dataFolder + "/bias/inception_c1_3x1_2_bn.f32"));
    vx_size inception_c1_3x1_2_scale_W_dims[1] = { 448 };
    vx_tensor inception_c1_3x1_2_scale_W;
    inception_c1_3x1_2_scale_W = vxCreateTensor(context,1, inception_c1_3x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_scale_W, dataFolder + "/weights/inception_c1_3x1_2_scale.f32"));
    vx_size inception_c1_3x1_2_scale_B_dims[1] = { 448 };
    vx_tensor inception_c1_3x1_2_scale_B;
    inception_c1_3x1_2_scale_B = vxCreateTensor(context,1, inception_c1_3x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_2_scale_B, dataFolder + "/bias/inception_c1_3x1_2_scale.f32"));
    vx_node inception_c1_3x1_2_bn_node;
    inception_c1_3x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x1_2, inception_c1_3x1_2_bn_W, inception_c1_3x1_2_bn_B, inception_c1_3x1_2_scale_W, inception_c1_3x1_2_scale_B, inception_c1_3x1_2_bn_eps, inception_c1_3x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_2_bn_node));

    // inception_c1_3x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x1_2_relu Layer
    vx_size inception_c1_3x1_2_relu_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c1_3x1_2_relu;
    inception_c1_3x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_relu);
    vx_enum inception_c1_3x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x1_2_relu_param_a = 0;
    vx_float32 inception_c1_3x1_2_relu_param_b = 0;
    vx_node inception_c1_3x1_2_relu_node;
    inception_c1_3x1_2_relu_node = vxActivationLayer(graph, inception_c1_3x1_2_scale, inception_c1_3x1_2_relu_mode, inception_c1_3x1_2_relu_param_a, inception_c1_3x1_2_relu_param_b, inception_c1_3x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_2_relu_node));

    // inception_c1_1x3_2 Layer
    vx_size inception_c1_1x3_2_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c1_1x3_2;
    inception_c1_1x3_2 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2);
    vx_size inception_c1_1x3_2_W_dims[4] = { 3, 1, 448, 512 };
    vx_tensor inception_c1_1x3_2_W;
    inception_c1_1x3_2_W = vxCreateTensor(context,4, inception_c1_1x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_W, dataFolder + "/weights/inception_c1_1x3_2.f32"));
    vx_nn_convolution_params_t inception_c1_1x3_2_params;
    inception_c1_1x3_2_params.padding_x = 1;
    inception_c1_1x3_2_params.padding_y = 0;
    inception_c1_1x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x3_2_params.dilation_x = 0;
    inception_c1_1x3_2_params.dilation_y = 0;
    vx_node inception_c1_1x3_2_node;
    inception_c1_1x3_2_node = vxConvolutionLayer(graph, inception_c1_3x1_2_relu, inception_c1_1x3_2_W, NULL, &inception_c1_1x3_2_params, sizeof(inception_c1_1x3_2_params ), inception_c1_1x3_2);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_2_node));

    // inception_c1_1x3_2_bn Layer
    vx_size inception_c1_1x3_2_scale_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c1_1x3_2_scale;
    inception_c1_1x3_2_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_scale);
    vx_size inception_c1_1x3_2_bn_W_dims[1] = { 512 };
    vx_float32 inception_c1_1x3_2_bn_eps = 0.001;
    vx_tensor inception_c1_1x3_2_bn_W;
    inception_c1_1x3_2_bn_W = vxCreateTensor(context,1, inception_c1_1x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_bn_W, dataFolder + "/weights/inception_c1_1x3_2_bn.f32"));
    vx_size inception_c1_1x3_2_bn_B_dims[1] = { 512 };
    vx_tensor inception_c1_1x3_2_bn_B;
    inception_c1_1x3_2_bn_B = vxCreateTensor(context,1, inception_c1_1x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_bn_B, dataFolder + "/bias/inception_c1_1x3_2_bn.f32"));
    vx_size inception_c1_1x3_2_scale_W_dims[1] = { 512 };
    vx_tensor inception_c1_1x3_2_scale_W;
    inception_c1_1x3_2_scale_W = vxCreateTensor(context,1, inception_c1_1x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_scale_W, dataFolder + "/weights/inception_c1_1x3_2_scale.f32"));
    vx_size inception_c1_1x3_2_scale_B_dims[1] = { 512 };
    vx_tensor inception_c1_1x3_2_scale_B;
    inception_c1_1x3_2_scale_B = vxCreateTensor(context,1, inception_c1_1x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_2_scale_B, dataFolder + "/bias/inception_c1_1x3_2_scale.f32"));
    vx_node inception_c1_1x3_2_bn_node;
    inception_c1_1x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x3_2, inception_c1_1x3_2_bn_W, inception_c1_1x3_2_bn_B, inception_c1_1x3_2_scale_W, inception_c1_1x3_2_scale_B, inception_c1_1x3_2_bn_eps, inception_c1_1x3_2_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_2_bn_node));

    // inception_c1_1x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x3_2_relu Layer
    vx_size inception_c1_1x3_2_relu_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c1_1x3_2_relu;
    inception_c1_1x3_2_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_relu);
    vx_enum inception_c1_1x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x3_2_relu_param_a = 0;
    vx_float32 inception_c1_1x3_2_relu_param_b = 0;
    vx_node inception_c1_1x3_2_relu_node;
    inception_c1_1x3_2_relu_node = vxActivationLayer(graph, inception_c1_1x3_2_scale, inception_c1_1x3_2_relu_mode, inception_c1_1x3_2_relu_param_a, inception_c1_1x3_2_relu_param_b, inception_c1_1x3_2_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_2_relu_node));

    // inception_c1_1x3_3 Layer
    vx_size inception_c1_1x3_3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x3_3;
    inception_c1_1x3_3 = vxCreateVirtualTensor(graph,4, inception_c1_1x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3);
    vx_size inception_c1_1x3_3_W_dims[4] = { 3, 1, 512, 256 };
    vx_tensor inception_c1_1x3_3_W;
    inception_c1_1x3_3_W = vxCreateTensor(context,4, inception_c1_1x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_3_W, dataFolder + "/weights/inception_c1_1x3_3.f32"));
    vx_nn_convolution_params_t inception_c1_1x3_3_params;
    inception_c1_1x3_3_params.padding_x = 1;
    inception_c1_1x3_3_params.padding_y = 0;
    inception_c1_1x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x3_3_params.dilation_x = 0;
    inception_c1_1x3_3_params.dilation_y = 0;
    vx_node inception_c1_1x3_3_node;
    inception_c1_1x3_3_node = vxConvolutionLayer(graph, inception_c1_1x3_2_relu, inception_c1_1x3_3_W, NULL, &inception_c1_1x3_3_params, sizeof(inception_c1_1x3_3_params ), inception_c1_1x3_3);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_3_node));

    // inception_c1_1x3_3_bn Layer
    vx_size inception_c1_1x3_3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x3_3_scale;
    inception_c1_1x3_3_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_scale);
    vx_size inception_c1_1x3_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c1_1x3_3_bn_eps = 0.001;
    vx_tensor inception_c1_1x3_3_bn_W;
    inception_c1_1x3_3_bn_W = vxCreateTensor(context,1, inception_c1_1x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_3_bn_W, dataFolder + "/weights/inception_c1_1x3_3_bn.f32"));
    vx_size inception_c1_1x3_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x3_3_bn_B;
    inception_c1_1x3_3_bn_B = vxCreateTensor(context,1, inception_c1_1x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_3_bn_B, dataFolder + "/bias/inception_c1_1x3_3_bn.f32"));
    vx_size inception_c1_1x3_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c1_1x3_3_scale_W;
    inception_c1_1x3_3_scale_W = vxCreateTensor(context,1, inception_c1_1x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_3_scale_W, dataFolder + "/weights/inception_c1_1x3_3_scale.f32"));
    vx_size inception_c1_1x3_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x3_3_scale_B;
    inception_c1_1x3_3_scale_B = vxCreateTensor(context,1, inception_c1_1x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x3_3_scale_B, dataFolder + "/bias/inception_c1_1x3_3_scale.f32"));
    vx_node inception_c1_1x3_3_bn_node;
    inception_c1_1x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x3_3, inception_c1_1x3_3_bn_W, inception_c1_1x3_3_bn_B, inception_c1_1x3_3_scale_W, inception_c1_1x3_3_scale_B, inception_c1_1x3_3_bn_eps, inception_c1_1x3_3_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_3_bn_node));

    // inception_c1_1x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x3_3_relu Layer
    vx_size inception_c1_1x3_3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x3_3_relu;
    inception_c1_1x3_3_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_relu);
    vx_enum inception_c1_1x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x3_3_relu_param_a = 0;
    vx_float32 inception_c1_1x3_3_relu_param_b = 0;
    vx_node inception_c1_1x3_3_relu_node;
    inception_c1_1x3_3_relu_node = vxActivationLayer(graph, inception_c1_1x3_3_scale, inception_c1_1x3_3_relu_mode, inception_c1_1x3_3_relu_param_a, inception_c1_1x3_3_relu_param_b, inception_c1_1x3_3_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x3_3_relu_node));

    // inception_c1_3x1_3 Layer
    vx_size inception_c1_3x1_3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_3x1_3;
    inception_c1_3x1_3 = vxCreateVirtualTensor(graph,4, inception_c1_3x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3);
    vx_size inception_c1_3x1_3_W_dims[4] = { 1, 3, 512, 256 };
    vx_tensor inception_c1_3x1_3_W;
    inception_c1_3x1_3_W = vxCreateTensor(context,4, inception_c1_3x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_3_W, dataFolder + "/weights/inception_c1_3x1_3.f32"));
    vx_nn_convolution_params_t inception_c1_3x1_3_params;
    inception_c1_3x1_3_params.padding_x = 0;
    inception_c1_3x1_3_params.padding_y = 1;
    inception_c1_3x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_3x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_3x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_3x1_3_params.dilation_x = 0;
    inception_c1_3x1_3_params.dilation_y = 0;
    vx_node inception_c1_3x1_3_node;
    inception_c1_3x1_3_node = vxConvolutionLayer(graph, inception_c1_1x3_2_relu, inception_c1_3x1_3_W, NULL, &inception_c1_3x1_3_params, sizeof(inception_c1_3x1_3_params ), inception_c1_3x1_3);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_3_node));

    // inception_c1_3x1_3_bn Layer
    vx_size inception_c1_3x1_3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_3x1_3_scale;
    inception_c1_3x1_3_scale = vxCreateVirtualTensor(graph,4, inception_c1_3x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_scale);
    vx_size inception_c1_3x1_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c1_3x1_3_bn_eps = 0.001;
    vx_tensor inception_c1_3x1_3_bn_W;
    inception_c1_3x1_3_bn_W = vxCreateTensor(context,1, inception_c1_3x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_3_bn_W, dataFolder + "/weights/inception_c1_3x1_3_bn.f32"));
    vx_size inception_c1_3x1_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c1_3x1_3_bn_B;
    inception_c1_3x1_3_bn_B = vxCreateTensor(context,1, inception_c1_3x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_3_bn_B, dataFolder + "/bias/inception_c1_3x1_3_bn.f32"));
    vx_size inception_c1_3x1_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c1_3x1_3_scale_W;
    inception_c1_3x1_3_scale_W = vxCreateTensor(context,1, inception_c1_3x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_3_scale_W, dataFolder + "/weights/inception_c1_3x1_3_scale.f32"));
    vx_size inception_c1_3x1_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c1_3x1_3_scale_B;
    inception_c1_3x1_3_scale_B = vxCreateTensor(context,1, inception_c1_3x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_3x1_3_scale_B, dataFolder + "/bias/inception_c1_3x1_3_scale.f32"));
    vx_node inception_c1_3x1_3_bn_node;
    inception_c1_3x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_c1_3x1_3, inception_c1_3x1_3_bn_W, inception_c1_3x1_3_bn_B, inception_c1_3x1_3_scale_W, inception_c1_3x1_3_scale_B, inception_c1_3x1_3_bn_eps, inception_c1_3x1_3_scale);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_3_bn_node));

    // inception_c1_3x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_3x1_3_relu Layer
    vx_size inception_c1_3x1_3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_3x1_3_relu;
    inception_c1_3x1_3_relu = vxCreateVirtualTensor(graph,4, inception_c1_3x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_relu);
    vx_enum inception_c1_3x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_3x1_3_relu_param_a = 0;
    vx_float32 inception_c1_3x1_3_relu_param_b = 0;
    vx_node inception_c1_3x1_3_relu_node;
    inception_c1_3x1_3_relu_node = vxActivationLayer(graph, inception_c1_3x1_3_scale, inception_c1_3x1_3_relu_mode, inception_c1_3x1_3_relu_param_a, inception_c1_3x1_3_relu_param_b, inception_c1_3x1_3_relu);
    ERROR_CHECK_OBJECT(inception_c1_3x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_3x1_3_relu_node));

    // inception_c1_pool_ave Layer
    vx_size inception_c1_pool_ave_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor inception_c1_pool_ave;
    inception_c1_pool_ave = vxCreateVirtualTensor(graph,4, inception_c1_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_pool_ave);
    vx_enum inception_c1_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_c1_pool_ave_kernel_w = 3;
    vx_size inception_c1_pool_ave_kernel_h = 3;
    vx_size inception_c1_pool_ave_pad_w = 1;
    vx_size inception_c1_pool_ave_pad_h = 1;
    vx_enum inception_c1_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_c1_pool_ave_node;
    inception_c1_pool_ave_node = vxPoolingLayer(graph, reduction_b_concat, inception_c1_pool_ave_type, inception_c1_pool_ave_kernel_w, inception_c1_pool_ave_kernel_h, inception_c1_pool_ave_pad_w, inception_c1_pool_ave_pad_h, inception_c1_pool_ave_roundPolicy, inception_c1_pool_ave );
    ERROR_CHECK_OBJECT(inception_c1_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_pool_ave_node));

    // inception_c1_1x1 Layer
    vx_size inception_c1_1x1_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x1;
    inception_c1_1x1 = vxCreateVirtualTensor(graph,4, inception_c1_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1);
    vx_size inception_c1_1x1_W_dims[4] = { 1, 1, 1536, 256 };
    vx_tensor inception_c1_1x1_W;
    inception_c1_1x1_W = vxCreateTensor(context,4, inception_c1_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_W, dataFolder + "/weights/inception_c1_1x1.f32"));
    vx_nn_convolution_params_t inception_c1_1x1_params;
    inception_c1_1x1_params.padding_x = 0;
    inception_c1_1x1_params.padding_y = 0;
    inception_c1_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c1_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c1_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c1_1x1_params.dilation_x = 0;
    inception_c1_1x1_params.dilation_y = 0;
    vx_node inception_c1_1x1_node;
    inception_c1_1x1_node = vxConvolutionLayer(graph, inception_c1_pool_ave, inception_c1_1x1_W, NULL, &inception_c1_1x1_params, sizeof(inception_c1_1x1_params ), inception_c1_1x1);
    ERROR_CHECK_OBJECT(inception_c1_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_node));

    // inception_c1_1x1_bn Layer
    vx_size inception_c1_1x1_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x1_scale;
    inception_c1_1x1_scale = vxCreateVirtualTensor(graph,4, inception_c1_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_scale);
    vx_size inception_c1_1x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_c1_1x1_bn_eps = 0.001;
    vx_tensor inception_c1_1x1_bn_W;
    inception_c1_1x1_bn_W = vxCreateTensor(context,1, inception_c1_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_bn_W, dataFolder + "/weights/inception_c1_1x1_bn.f32"));
    vx_size inception_c1_1x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x1_bn_B;
    inception_c1_1x1_bn_B = vxCreateTensor(context,1, inception_c1_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_bn_B, dataFolder + "/bias/inception_c1_1x1_bn.f32"));
    vx_size inception_c1_1x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_c1_1x1_scale_W;
    inception_c1_1x1_scale_W = vxCreateTensor(context,1, inception_c1_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_scale_W, dataFolder + "/weights/inception_c1_1x1_scale.f32"));
    vx_size inception_c1_1x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_c1_1x1_scale_B;
    inception_c1_1x1_scale_B = vxCreateTensor(context,1, inception_c1_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c1_1x1_scale_B, dataFolder + "/bias/inception_c1_1x1_scale.f32"));
    vx_node inception_c1_1x1_bn_node;
    inception_c1_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_c1_1x1, inception_c1_1x1_bn_W, inception_c1_1x1_bn_B, inception_c1_1x1_scale_W, inception_c1_1x1_scale_B, inception_c1_1x1_bn_eps, inception_c1_1x1_scale);
    ERROR_CHECK_OBJECT(inception_c1_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_bn_node));

    // inception_c1_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c1_1x1_relu Layer
    vx_size inception_c1_1x1_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c1_1x1_relu;
    inception_c1_1x1_relu = vxCreateVirtualTensor(graph,4, inception_c1_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_1x1_relu);
    vx_enum inception_c1_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c1_1x1_relu_param_a = 0;
    vx_float32 inception_c1_1x1_relu_param_b = 0;
    vx_node inception_c1_1x1_relu_node;
    inception_c1_1x1_relu_node = vxActivationLayer(graph, inception_c1_1x1_scale, inception_c1_1x1_relu_mode, inception_c1_1x1_relu_param_a, inception_c1_1x1_relu_param_b, inception_c1_1x1_relu);
    ERROR_CHECK_OBJECT(inception_c1_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_1x1_relu_node));

    // inception_c1_concat Layer
    vx_size inception_c1_concat_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor inception_c1_concat;
    inception_c1_concat = vxCreateVirtualTensor(graph,4, inception_c1_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c1_concat);
    vx_node inception_c1_concat_node;
    inception_c1_concat_node = vxConcatLayer(graph, inception_c1_concat, inception_c1_1x1_2_relu, inception_c1_1x3_relu, inception_c1_3x1_relu, inception_c1_1x3_3_relu, inception_c1_3x1_3_relu, inception_c1_1x1_relu, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_c1_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c1_concat_node));

    // inception_c2_1x1_2 Layer
    vx_size inception_c2_1x1_2_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x1_2;
    inception_c2_1x1_2 = vxCreateVirtualTensor(graph,4, inception_c2_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2);
    vx_size inception_c2_1x1_2_W_dims[4] = { 1, 1, 1536, 256 };
    vx_tensor inception_c2_1x1_2_W;
    inception_c2_1x1_2_W = vxCreateTensor(context,4, inception_c2_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_W, dataFolder + "/weights/inception_c2_1x1_2.f32"));
    vx_nn_convolution_params_t inception_c2_1x1_2_params;
    inception_c2_1x1_2_params.padding_x = 0;
    inception_c2_1x1_2_params.padding_y = 0;
    inception_c2_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x1_2_params.dilation_x = 0;
    inception_c2_1x1_2_params.dilation_y = 0;
    vx_node inception_c2_1x1_2_node;
    inception_c2_1x1_2_node = vxConvolutionLayer(graph, inception_c1_concat, inception_c2_1x1_2_W, NULL, &inception_c2_1x1_2_params, sizeof(inception_c2_1x1_2_params ), inception_c2_1x1_2);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_2_node));

    // inception_c2_1x1_2_bn Layer
    vx_size inception_c2_1x1_2_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x1_2_scale;
    inception_c2_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_scale);
    vx_size inception_c2_1x1_2_bn_W_dims[1] = { 256 };
    vx_float32 inception_c2_1x1_2_bn_eps = 0.001;
    vx_tensor inception_c2_1x1_2_bn_W;
    inception_c2_1x1_2_bn_W = vxCreateTensor(context,1, inception_c2_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_bn_W, dataFolder + "/weights/inception_c2_1x1_2_bn.f32"));
    vx_size inception_c2_1x1_2_bn_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x1_2_bn_B;
    inception_c2_1x1_2_bn_B = vxCreateTensor(context,1, inception_c2_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_bn_B, dataFolder + "/bias/inception_c2_1x1_2_bn.f32"));
    vx_size inception_c2_1x1_2_scale_W_dims[1] = { 256 };
    vx_tensor inception_c2_1x1_2_scale_W;
    inception_c2_1x1_2_scale_W = vxCreateTensor(context,1, inception_c2_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_scale_W, dataFolder + "/weights/inception_c2_1x1_2_scale.f32"));
    vx_size inception_c2_1x1_2_scale_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x1_2_scale_B;
    inception_c2_1x1_2_scale_B = vxCreateTensor(context,1, inception_c2_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_2_scale_B, dataFolder + "/bias/inception_c2_1x1_2_scale.f32"));
    vx_node inception_c2_1x1_2_bn_node;
    inception_c2_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x1_2, inception_c2_1x1_2_bn_W, inception_c2_1x1_2_bn_B, inception_c2_1x1_2_scale_W, inception_c2_1x1_2_scale_B, inception_c2_1x1_2_bn_eps, inception_c2_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_2_bn_node));

    // inception_c2_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x1_2_relu Layer
    vx_size inception_c2_1x1_2_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x1_2_relu;
    inception_c2_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_relu);
    vx_enum inception_c2_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x1_2_relu_param_a = 0;
    vx_float32 inception_c2_1x1_2_relu_param_b = 0;
    vx_node inception_c2_1x1_2_relu_node;
    inception_c2_1x1_2_relu_node = vxActivationLayer(graph, inception_c2_1x1_2_scale, inception_c2_1x1_2_relu_mode, inception_c2_1x1_2_relu_param_a, inception_c2_1x1_2_relu_param_b, inception_c2_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_2_relu_node));

    // inception_c2_1x1_3 Layer
    vx_size inception_c2_1x1_3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x1_3;
    inception_c2_1x1_3 = vxCreateVirtualTensor(graph,4, inception_c2_1x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3);
    vx_size inception_c2_1x1_3_W_dims[4] = { 1, 1, 1536, 384 };
    vx_tensor inception_c2_1x1_3_W;
    inception_c2_1x1_3_W = vxCreateTensor(context,4, inception_c2_1x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_3_W, dataFolder + "/weights/inception_c2_1x1_3.f32"));
    vx_nn_convolution_params_t inception_c2_1x1_3_params;
    inception_c2_1x1_3_params.padding_x = 0;
    inception_c2_1x1_3_params.padding_y = 0;
    inception_c2_1x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x1_3_params.dilation_x = 0;
    inception_c2_1x1_3_params.dilation_y = 0;
    vx_node inception_c2_1x1_3_node;
    inception_c2_1x1_3_node = vxConvolutionLayer(graph, inception_c1_concat, inception_c2_1x1_3_W, NULL, &inception_c2_1x1_3_params, sizeof(inception_c2_1x1_3_params ), inception_c2_1x1_3);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_3_node));

    // inception_c2_1x1_3_bn Layer
    vx_size inception_c2_1x1_3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x1_3_scale;
    inception_c2_1x1_3_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_scale);
    vx_size inception_c2_1x1_3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_1x1_3_bn_eps = 0.001;
    vx_tensor inception_c2_1x1_3_bn_W;
    inception_c2_1x1_3_bn_W = vxCreateTensor(context,1, inception_c2_1x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_3_bn_W, dataFolder + "/weights/inception_c2_1x1_3_bn.f32"));
    vx_size inception_c2_1x1_3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x1_3_bn_B;
    inception_c2_1x1_3_bn_B = vxCreateTensor(context,1, inception_c2_1x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_3_bn_B, dataFolder + "/bias/inception_c2_1x1_3_bn.f32"));
    vx_size inception_c2_1x1_3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_1x1_3_scale_W;
    inception_c2_1x1_3_scale_W = vxCreateTensor(context,1, inception_c2_1x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_3_scale_W, dataFolder + "/weights/inception_c2_1x1_3_scale.f32"));
    vx_size inception_c2_1x1_3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x1_3_scale_B;
    inception_c2_1x1_3_scale_B = vxCreateTensor(context,1, inception_c2_1x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_3_scale_B, dataFolder + "/bias/inception_c2_1x1_3_scale.f32"));
    vx_node inception_c2_1x1_3_bn_node;
    inception_c2_1x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x1_3, inception_c2_1x1_3_bn_W, inception_c2_1x1_3_bn_B, inception_c2_1x1_3_scale_W, inception_c2_1x1_3_scale_B, inception_c2_1x1_3_bn_eps, inception_c2_1x1_3_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_3_bn_node));

    // inception_c2_1x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x1_3_relu Layer
    vx_size inception_c2_1x1_3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x1_3_relu;
    inception_c2_1x1_3_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_relu);
    vx_enum inception_c2_1x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x1_3_relu_param_a = 0;
    vx_float32 inception_c2_1x1_3_relu_param_b = 0;
    vx_node inception_c2_1x1_3_relu_node;
    inception_c2_1x1_3_relu_node = vxActivationLayer(graph, inception_c2_1x1_3_scale, inception_c2_1x1_3_relu_mode, inception_c2_1x1_3_relu_param_a, inception_c2_1x1_3_relu_param_b, inception_c2_1x1_3_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_3_relu_node));

    // inception_c2_1x3 Layer
    vx_size inception_c2_1x3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x3;
    inception_c2_1x3 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3);
    vx_size inception_c2_1x3_W_dims[4] = { 3, 1, 384, 256 };
    vx_tensor inception_c2_1x3_W;
    inception_c2_1x3_W = vxCreateTensor(context,4, inception_c2_1x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_W, dataFolder + "/weights/inception_c2_1x3.f32"));
    vx_nn_convolution_params_t inception_c2_1x3_params;
    inception_c2_1x3_params.padding_x = 1;
    inception_c2_1x3_params.padding_y = 0;
    inception_c2_1x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x3_params.dilation_x = 0;
    inception_c2_1x3_params.dilation_y = 0;
    vx_node inception_c2_1x3_node;
    inception_c2_1x3_node = vxConvolutionLayer(graph, inception_c2_1x1_3_relu, inception_c2_1x3_W, NULL, &inception_c2_1x3_params, sizeof(inception_c2_1x3_params ), inception_c2_1x3);
    ERROR_CHECK_OBJECT(inception_c2_1x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_node));

    // inception_c2_1x3_bn Layer
    vx_size inception_c2_1x3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x3_scale;
    inception_c2_1x3_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_scale);
    vx_size inception_c2_1x3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c2_1x3_bn_eps = 0.001;
    vx_tensor inception_c2_1x3_bn_W;
    inception_c2_1x3_bn_W = vxCreateTensor(context,1, inception_c2_1x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_bn_W, dataFolder + "/weights/inception_c2_1x3_bn.f32"));
    vx_size inception_c2_1x3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x3_bn_B;
    inception_c2_1x3_bn_B = vxCreateTensor(context,1, inception_c2_1x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_bn_B, dataFolder + "/bias/inception_c2_1x3_bn.f32"));
    vx_size inception_c2_1x3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c2_1x3_scale_W;
    inception_c2_1x3_scale_W = vxCreateTensor(context,1, inception_c2_1x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_scale_W, dataFolder + "/weights/inception_c2_1x3_scale.f32"));
    vx_size inception_c2_1x3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x3_scale_B;
    inception_c2_1x3_scale_B = vxCreateTensor(context,1, inception_c2_1x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_scale_B, dataFolder + "/bias/inception_c2_1x3_scale.f32"));
    vx_node inception_c2_1x3_bn_node;
    inception_c2_1x3_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x3, inception_c2_1x3_bn_W, inception_c2_1x3_bn_B, inception_c2_1x3_scale_W, inception_c2_1x3_scale_B, inception_c2_1x3_bn_eps, inception_c2_1x3_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_bn_node));

    // inception_c2_1x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x3_relu Layer
    vx_size inception_c2_1x3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x3_relu;
    inception_c2_1x3_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_relu);
    vx_enum inception_c2_1x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x3_relu_param_a = 0;
    vx_float32 inception_c2_1x3_relu_param_b = 0;
    vx_node inception_c2_1x3_relu_node;
    inception_c2_1x3_relu_node = vxActivationLayer(graph, inception_c2_1x3_scale, inception_c2_1x3_relu_mode, inception_c2_1x3_relu_param_a, inception_c2_1x3_relu_param_b, inception_c2_1x3_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_relu_node));

    // inception_c2_3x1 Layer
    vx_size inception_c2_3x1_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_3x1;
    inception_c2_3x1 = vxCreateVirtualTensor(graph,4, inception_c2_3x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1);
    vx_size inception_c2_3x1_W_dims[4] = { 1, 3, 384, 256 };
    vx_tensor inception_c2_3x1_W;
    inception_c2_3x1_W = vxCreateTensor(context,4, inception_c2_3x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_W, dataFolder + "/weights/inception_c2_3x1.f32"));
    vx_nn_convolution_params_t inception_c2_3x1_params;
    inception_c2_3x1_params.padding_x = 0;
    inception_c2_3x1_params.padding_y = 1;
    inception_c2_3x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x1_params.dilation_x = 0;
    inception_c2_3x1_params.dilation_y = 0;
    vx_node inception_c2_3x1_node;
    inception_c2_3x1_node = vxConvolutionLayer(graph, inception_c2_1x1_3_relu, inception_c2_3x1_W, NULL, &inception_c2_3x1_params, sizeof(inception_c2_3x1_params ), inception_c2_3x1);
    ERROR_CHECK_OBJECT(inception_c2_3x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_node));

    // inception_c2_3x1_bn Layer
    vx_size inception_c2_3x1_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_3x1_scale;
    inception_c2_3x1_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_scale);
    vx_size inception_c2_3x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_c2_3x1_bn_eps = 0.001;
    vx_tensor inception_c2_3x1_bn_W;
    inception_c2_3x1_bn_W = vxCreateTensor(context,1, inception_c2_3x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_bn_W, dataFolder + "/weights/inception_c2_3x1_bn.f32"));
    vx_size inception_c2_3x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_c2_3x1_bn_B;
    inception_c2_3x1_bn_B = vxCreateTensor(context,1, inception_c2_3x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_bn_B, dataFolder + "/bias/inception_c2_3x1_bn.f32"));
    vx_size inception_c2_3x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_c2_3x1_scale_W;
    inception_c2_3x1_scale_W = vxCreateTensor(context,1, inception_c2_3x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_scale_W, dataFolder + "/weights/inception_c2_3x1_scale.f32"));
    vx_size inception_c2_3x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_c2_3x1_scale_B;
    inception_c2_3x1_scale_B = vxCreateTensor(context,1, inception_c2_3x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_scale_B, dataFolder + "/bias/inception_c2_3x1_scale.f32"));
    vx_node inception_c2_3x1_bn_node;
    inception_c2_3x1_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x1, inception_c2_3x1_bn_W, inception_c2_3x1_bn_B, inception_c2_3x1_scale_W, inception_c2_3x1_scale_B, inception_c2_3x1_bn_eps, inception_c2_3x1_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_bn_node));

    // inception_c2_3x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x1_relu Layer
    vx_size inception_c2_3x1_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_3x1_relu;
    inception_c2_3x1_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_relu);
    vx_enum inception_c2_3x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x1_relu_param_a = 0;
    vx_float32 inception_c2_3x1_relu_param_b = 0;
    vx_node inception_c2_3x1_relu_node;
    inception_c2_3x1_relu_node = vxActivationLayer(graph, inception_c2_3x1_scale, inception_c2_3x1_relu_mode, inception_c2_3x1_relu_param_a, inception_c2_3x1_relu_param_b, inception_c2_3x1_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_relu_node));

    // inception_c2_1x1_4 Layer
    vx_size inception_c2_1x1_4_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x1_4;
    inception_c2_1x1_4 = vxCreateVirtualTensor(graph,4, inception_c2_1x1_4_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4);
    vx_size inception_c2_1x1_4_W_dims[4] = { 1, 1, 1536, 384 };
    vx_tensor inception_c2_1x1_4_W;
    inception_c2_1x1_4_W = vxCreateTensor(context,4, inception_c2_1x1_4_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_4_W, dataFolder + "/weights/inception_c2_1x1_4.f32"));
    vx_nn_convolution_params_t inception_c2_1x1_4_params;
    inception_c2_1x1_4_params.padding_x = 0;
    inception_c2_1x1_4_params.padding_y = 0;
    inception_c2_1x1_4_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x1_4_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x1_4_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x1_4_params.dilation_x = 0;
    inception_c2_1x1_4_params.dilation_y = 0;
    vx_node inception_c2_1x1_4_node;
    inception_c2_1x1_4_node = vxConvolutionLayer(graph, inception_c1_concat, inception_c2_1x1_4_W, NULL, &inception_c2_1x1_4_params, sizeof(inception_c2_1x1_4_params ), inception_c2_1x1_4);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_4_node));

    // inception_c2_1x1_4_bn Layer
    vx_size inception_c2_1x1_4_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x1_4_scale;
    inception_c2_1x1_4_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x1_4_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_scale);
    vx_size inception_c2_1x1_4_bn_W_dims[1] = { 384 };
    vx_float32 inception_c2_1x1_4_bn_eps = 0.001;
    vx_tensor inception_c2_1x1_4_bn_W;
    inception_c2_1x1_4_bn_W = vxCreateTensor(context,1, inception_c2_1x1_4_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_4_bn_W, dataFolder + "/weights/inception_c2_1x1_4_bn.f32"));
    vx_size inception_c2_1x1_4_bn_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x1_4_bn_B;
    inception_c2_1x1_4_bn_B = vxCreateTensor(context,1, inception_c2_1x1_4_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_4_bn_B, dataFolder + "/bias/inception_c2_1x1_4_bn.f32"));
    vx_size inception_c2_1x1_4_scale_W_dims[1] = { 384 };
    vx_tensor inception_c2_1x1_4_scale_W;
    inception_c2_1x1_4_scale_W = vxCreateTensor(context,1, inception_c2_1x1_4_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_4_scale_W, dataFolder + "/weights/inception_c2_1x1_4_scale.f32"));
    vx_size inception_c2_1x1_4_scale_B_dims[1] = { 384 };
    vx_tensor inception_c2_1x1_4_scale_B;
    inception_c2_1x1_4_scale_B = vxCreateTensor(context,1, inception_c2_1x1_4_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_4_scale_B, dataFolder + "/bias/inception_c2_1x1_4_scale.f32"));
    vx_node inception_c2_1x1_4_bn_node;
    inception_c2_1x1_4_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x1_4, inception_c2_1x1_4_bn_W, inception_c2_1x1_4_bn_B, inception_c2_1x1_4_scale_W, inception_c2_1x1_4_scale_B, inception_c2_1x1_4_bn_eps, inception_c2_1x1_4_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_4_bn_node));

    // inception_c2_1x1_4_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x1_4_relu Layer
    vx_size inception_c2_1x1_4_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c2_1x1_4_relu;
    inception_c2_1x1_4_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x1_4_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_relu);
    vx_enum inception_c2_1x1_4_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x1_4_relu_param_a = 0;
    vx_float32 inception_c2_1x1_4_relu_param_b = 0;
    vx_node inception_c2_1x1_4_relu_node;
    inception_c2_1x1_4_relu_node = vxActivationLayer(graph, inception_c2_1x1_4_scale, inception_c2_1x1_4_relu_mode, inception_c2_1x1_4_relu_param_a, inception_c2_1x1_4_relu_param_b, inception_c2_1x1_4_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x1_4_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_4_relu_node));

    // inception_c2_3x1_2 Layer
    vx_size inception_c2_3x1_2_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c2_3x1_2;
    inception_c2_3x1_2 = vxCreateVirtualTensor(graph,4, inception_c2_3x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2);
    vx_size inception_c2_3x1_2_W_dims[4] = { 1, 3, 384, 448 };
    vx_tensor inception_c2_3x1_2_W;
    inception_c2_3x1_2_W = vxCreateTensor(context,4, inception_c2_3x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_W, dataFolder + "/weights/inception_c2_3x1_2.f32"));
    vx_nn_convolution_params_t inception_c2_3x1_2_params;
    inception_c2_3x1_2_params.padding_x = 0;
    inception_c2_3x1_2_params.padding_y = 1;
    inception_c2_3x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x1_2_params.dilation_x = 0;
    inception_c2_3x1_2_params.dilation_y = 0;
    vx_node inception_c2_3x1_2_node;
    inception_c2_3x1_2_node = vxConvolutionLayer(graph, inception_c2_1x1_4_relu, inception_c2_3x1_2_W, NULL, &inception_c2_3x1_2_params, sizeof(inception_c2_3x1_2_params ), inception_c2_3x1_2);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_2_node));

    // inception_c2_3x1_2_bn Layer
    vx_size inception_c2_3x1_2_scale_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c2_3x1_2_scale;
    inception_c2_3x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_scale);
    vx_size inception_c2_3x1_2_bn_W_dims[1] = { 448 };
    vx_float32 inception_c2_3x1_2_bn_eps = 0.001;
    vx_tensor inception_c2_3x1_2_bn_W;
    inception_c2_3x1_2_bn_W = vxCreateTensor(context,1, inception_c2_3x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_bn_W, dataFolder + "/weights/inception_c2_3x1_2_bn.f32"));
    vx_size inception_c2_3x1_2_bn_B_dims[1] = { 448 };
    vx_tensor inception_c2_3x1_2_bn_B;
    inception_c2_3x1_2_bn_B = vxCreateTensor(context,1, inception_c2_3x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_bn_B, dataFolder + "/bias/inception_c2_3x1_2_bn.f32"));
    vx_size inception_c2_3x1_2_scale_W_dims[1] = { 448 };
    vx_tensor inception_c2_3x1_2_scale_W;
    inception_c2_3x1_2_scale_W = vxCreateTensor(context,1, inception_c2_3x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_scale_W, dataFolder + "/weights/inception_c2_3x1_2_scale.f32"));
    vx_size inception_c2_3x1_2_scale_B_dims[1] = { 448 };
    vx_tensor inception_c2_3x1_2_scale_B;
    inception_c2_3x1_2_scale_B = vxCreateTensor(context,1, inception_c2_3x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_2_scale_B, dataFolder + "/bias/inception_c2_3x1_2_scale.f32"));
    vx_node inception_c2_3x1_2_bn_node;
    inception_c2_3x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x1_2, inception_c2_3x1_2_bn_W, inception_c2_3x1_2_bn_B, inception_c2_3x1_2_scale_W, inception_c2_3x1_2_scale_B, inception_c2_3x1_2_bn_eps, inception_c2_3x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_2_bn_node));

    // inception_c2_3x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x1_2_relu Layer
    vx_size inception_c2_3x1_2_relu_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c2_3x1_2_relu;
    inception_c2_3x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_relu);
    vx_enum inception_c2_3x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x1_2_relu_param_a = 0;
    vx_float32 inception_c2_3x1_2_relu_param_b = 0;
    vx_node inception_c2_3x1_2_relu_node;
    inception_c2_3x1_2_relu_node = vxActivationLayer(graph, inception_c2_3x1_2_scale, inception_c2_3x1_2_relu_mode, inception_c2_3x1_2_relu_param_a, inception_c2_3x1_2_relu_param_b, inception_c2_3x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_2_relu_node));

    // inception_c2_1x3_2 Layer
    vx_size inception_c2_1x3_2_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c2_1x3_2;
    inception_c2_1x3_2 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2);
    vx_size inception_c2_1x3_2_W_dims[4] = { 3, 1, 448, 512 };
    vx_tensor inception_c2_1x3_2_W;
    inception_c2_1x3_2_W = vxCreateTensor(context,4, inception_c2_1x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_W, dataFolder + "/weights/inception_c2_1x3_2.f32"));
    vx_nn_convolution_params_t inception_c2_1x3_2_params;
    inception_c2_1x3_2_params.padding_x = 1;
    inception_c2_1x3_2_params.padding_y = 0;
    inception_c2_1x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x3_2_params.dilation_x = 0;
    inception_c2_1x3_2_params.dilation_y = 0;
    vx_node inception_c2_1x3_2_node;
    inception_c2_1x3_2_node = vxConvolutionLayer(graph, inception_c2_3x1_2_relu, inception_c2_1x3_2_W, NULL, &inception_c2_1x3_2_params, sizeof(inception_c2_1x3_2_params ), inception_c2_1x3_2);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_2_node));

    // inception_c2_1x3_2_bn Layer
    vx_size inception_c2_1x3_2_scale_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c2_1x3_2_scale;
    inception_c2_1x3_2_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_scale);
    vx_size inception_c2_1x3_2_bn_W_dims[1] = { 512 };
    vx_float32 inception_c2_1x3_2_bn_eps = 0.001;
    vx_tensor inception_c2_1x3_2_bn_W;
    inception_c2_1x3_2_bn_W = vxCreateTensor(context,1, inception_c2_1x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_bn_W, dataFolder + "/weights/inception_c2_1x3_2_bn.f32"));
    vx_size inception_c2_1x3_2_bn_B_dims[1] = { 512 };
    vx_tensor inception_c2_1x3_2_bn_B;
    inception_c2_1x3_2_bn_B = vxCreateTensor(context,1, inception_c2_1x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_bn_B, dataFolder + "/bias/inception_c2_1x3_2_bn.f32"));
    vx_size inception_c2_1x3_2_scale_W_dims[1] = { 512 };
    vx_tensor inception_c2_1x3_2_scale_W;
    inception_c2_1x3_2_scale_W = vxCreateTensor(context,1, inception_c2_1x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_scale_W, dataFolder + "/weights/inception_c2_1x3_2_scale.f32"));
    vx_size inception_c2_1x3_2_scale_B_dims[1] = { 512 };
    vx_tensor inception_c2_1x3_2_scale_B;
    inception_c2_1x3_2_scale_B = vxCreateTensor(context,1, inception_c2_1x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_2_scale_B, dataFolder + "/bias/inception_c2_1x3_2_scale.f32"));
    vx_node inception_c2_1x3_2_bn_node;
    inception_c2_1x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x3_2, inception_c2_1x3_2_bn_W, inception_c2_1x3_2_bn_B, inception_c2_1x3_2_scale_W, inception_c2_1x3_2_scale_B, inception_c2_1x3_2_bn_eps, inception_c2_1x3_2_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_2_bn_node));

    // inception_c2_1x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x3_2_relu Layer
    vx_size inception_c2_1x3_2_relu_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c2_1x3_2_relu;
    inception_c2_1x3_2_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_relu);
    vx_enum inception_c2_1x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x3_2_relu_param_a = 0;
    vx_float32 inception_c2_1x3_2_relu_param_b = 0;
    vx_node inception_c2_1x3_2_relu_node;
    inception_c2_1x3_2_relu_node = vxActivationLayer(graph, inception_c2_1x3_2_scale, inception_c2_1x3_2_relu_mode, inception_c2_1x3_2_relu_param_a, inception_c2_1x3_2_relu_param_b, inception_c2_1x3_2_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_2_relu_node));

    // inception_c2_1x3_3 Layer
    vx_size inception_c2_1x3_3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x3_3;
    inception_c2_1x3_3 = vxCreateVirtualTensor(graph,4, inception_c2_1x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3);
    vx_size inception_c2_1x3_3_W_dims[4] = { 3, 1, 512, 256 };
    vx_tensor inception_c2_1x3_3_W;
    inception_c2_1x3_3_W = vxCreateTensor(context,4, inception_c2_1x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_3_W, dataFolder + "/weights/inception_c2_1x3_3.f32"));
    vx_nn_convolution_params_t inception_c2_1x3_3_params;
    inception_c2_1x3_3_params.padding_x = 1;
    inception_c2_1x3_3_params.padding_y = 0;
    inception_c2_1x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x3_3_params.dilation_x = 0;
    inception_c2_1x3_3_params.dilation_y = 0;
    vx_node inception_c2_1x3_3_node;
    inception_c2_1x3_3_node = vxConvolutionLayer(graph, inception_c2_1x3_2_relu, inception_c2_1x3_3_W, NULL, &inception_c2_1x3_3_params, sizeof(inception_c2_1x3_3_params ), inception_c2_1x3_3);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_3_node));

    // inception_c2_1x3_3_bn Layer
    vx_size inception_c2_1x3_3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x3_3_scale;
    inception_c2_1x3_3_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_scale);
    vx_size inception_c2_1x3_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c2_1x3_3_bn_eps = 0.001;
    vx_tensor inception_c2_1x3_3_bn_W;
    inception_c2_1x3_3_bn_W = vxCreateTensor(context,1, inception_c2_1x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_3_bn_W, dataFolder + "/weights/inception_c2_1x3_3_bn.f32"));
    vx_size inception_c2_1x3_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x3_3_bn_B;
    inception_c2_1x3_3_bn_B = vxCreateTensor(context,1, inception_c2_1x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_3_bn_B, dataFolder + "/bias/inception_c2_1x3_3_bn.f32"));
    vx_size inception_c2_1x3_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c2_1x3_3_scale_W;
    inception_c2_1x3_3_scale_W = vxCreateTensor(context,1, inception_c2_1x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_3_scale_W, dataFolder + "/weights/inception_c2_1x3_3_scale.f32"));
    vx_size inception_c2_1x3_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x3_3_scale_B;
    inception_c2_1x3_3_scale_B = vxCreateTensor(context,1, inception_c2_1x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x3_3_scale_B, dataFolder + "/bias/inception_c2_1x3_3_scale.f32"));
    vx_node inception_c2_1x3_3_bn_node;
    inception_c2_1x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x3_3, inception_c2_1x3_3_bn_W, inception_c2_1x3_3_bn_B, inception_c2_1x3_3_scale_W, inception_c2_1x3_3_scale_B, inception_c2_1x3_3_bn_eps, inception_c2_1x3_3_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_3_bn_node));

    // inception_c2_1x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x3_3_relu Layer
    vx_size inception_c2_1x3_3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x3_3_relu;
    inception_c2_1x3_3_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_relu);
    vx_enum inception_c2_1x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x3_3_relu_param_a = 0;
    vx_float32 inception_c2_1x3_3_relu_param_b = 0;
    vx_node inception_c2_1x3_3_relu_node;
    inception_c2_1x3_3_relu_node = vxActivationLayer(graph, inception_c2_1x3_3_scale, inception_c2_1x3_3_relu_mode, inception_c2_1x3_3_relu_param_a, inception_c2_1x3_3_relu_param_b, inception_c2_1x3_3_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x3_3_relu_node));

    // inception_c2_3x1_3 Layer
    vx_size inception_c2_3x1_3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_3x1_3;
    inception_c2_3x1_3 = vxCreateVirtualTensor(graph,4, inception_c2_3x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3);
    vx_size inception_c2_3x1_3_W_dims[4] = { 1, 3, 512, 256 };
    vx_tensor inception_c2_3x1_3_W;
    inception_c2_3x1_3_W = vxCreateTensor(context,4, inception_c2_3x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_3_W, dataFolder + "/weights/inception_c2_3x1_3.f32"));
    vx_nn_convolution_params_t inception_c2_3x1_3_params;
    inception_c2_3x1_3_params.padding_x = 0;
    inception_c2_3x1_3_params.padding_y = 1;
    inception_c2_3x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_3x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_3x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_3x1_3_params.dilation_x = 0;
    inception_c2_3x1_3_params.dilation_y = 0;
    vx_node inception_c2_3x1_3_node;
    inception_c2_3x1_3_node = vxConvolutionLayer(graph, inception_c2_1x3_2_relu, inception_c2_3x1_3_W, NULL, &inception_c2_3x1_3_params, sizeof(inception_c2_3x1_3_params ), inception_c2_3x1_3);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_3_node));

    // inception_c2_3x1_3_bn Layer
    vx_size inception_c2_3x1_3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_3x1_3_scale;
    inception_c2_3x1_3_scale = vxCreateVirtualTensor(graph,4, inception_c2_3x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_scale);
    vx_size inception_c2_3x1_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c2_3x1_3_bn_eps = 0.001;
    vx_tensor inception_c2_3x1_3_bn_W;
    inception_c2_3x1_3_bn_W = vxCreateTensor(context,1, inception_c2_3x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_3_bn_W, dataFolder + "/weights/inception_c2_3x1_3_bn.f32"));
    vx_size inception_c2_3x1_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c2_3x1_3_bn_B;
    inception_c2_3x1_3_bn_B = vxCreateTensor(context,1, inception_c2_3x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_3_bn_B, dataFolder + "/bias/inception_c2_3x1_3_bn.f32"));
    vx_size inception_c2_3x1_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c2_3x1_3_scale_W;
    inception_c2_3x1_3_scale_W = vxCreateTensor(context,1, inception_c2_3x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_3_scale_W, dataFolder + "/weights/inception_c2_3x1_3_scale.f32"));
    vx_size inception_c2_3x1_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c2_3x1_3_scale_B;
    inception_c2_3x1_3_scale_B = vxCreateTensor(context,1, inception_c2_3x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_3x1_3_scale_B, dataFolder + "/bias/inception_c2_3x1_3_scale.f32"));
    vx_node inception_c2_3x1_3_bn_node;
    inception_c2_3x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_c2_3x1_3, inception_c2_3x1_3_bn_W, inception_c2_3x1_3_bn_B, inception_c2_3x1_3_scale_W, inception_c2_3x1_3_scale_B, inception_c2_3x1_3_bn_eps, inception_c2_3x1_3_scale);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_3_bn_node));

    // inception_c2_3x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_3x1_3_relu Layer
    vx_size inception_c2_3x1_3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_3x1_3_relu;
    inception_c2_3x1_3_relu = vxCreateVirtualTensor(graph,4, inception_c2_3x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_relu);
    vx_enum inception_c2_3x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_3x1_3_relu_param_a = 0;
    vx_float32 inception_c2_3x1_3_relu_param_b = 0;
    vx_node inception_c2_3x1_3_relu_node;
    inception_c2_3x1_3_relu_node = vxActivationLayer(graph, inception_c2_3x1_3_scale, inception_c2_3x1_3_relu_mode, inception_c2_3x1_3_relu_param_a, inception_c2_3x1_3_relu_param_b, inception_c2_3x1_3_relu);
    ERROR_CHECK_OBJECT(inception_c2_3x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_3x1_3_relu_node));

    // inception_c2_pool_ave Layer
    vx_size inception_c2_pool_ave_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor inception_c2_pool_ave;
    inception_c2_pool_ave = vxCreateVirtualTensor(graph,4, inception_c2_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_pool_ave);
    vx_enum inception_c2_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_c2_pool_ave_kernel_w = 3;
    vx_size inception_c2_pool_ave_kernel_h = 3;
    vx_size inception_c2_pool_ave_pad_w = 1;
    vx_size inception_c2_pool_ave_pad_h = 1;
    vx_enum inception_c2_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_c2_pool_ave_node;
    inception_c2_pool_ave_node = vxPoolingLayer(graph, inception_c1_concat, inception_c2_pool_ave_type, inception_c2_pool_ave_kernel_w, inception_c2_pool_ave_kernel_h, inception_c2_pool_ave_pad_w, inception_c2_pool_ave_pad_h, inception_c2_pool_ave_roundPolicy, inception_c2_pool_ave );
    ERROR_CHECK_OBJECT(inception_c2_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_pool_ave_node));

    // inception_c2_1x1 Layer
    vx_size inception_c2_1x1_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x1;
    inception_c2_1x1 = vxCreateVirtualTensor(graph,4, inception_c2_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1);
    vx_size inception_c2_1x1_W_dims[4] = { 1, 1, 1536, 256 };
    vx_tensor inception_c2_1x1_W;
    inception_c2_1x1_W = vxCreateTensor(context,4, inception_c2_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_W, dataFolder + "/weights/inception_c2_1x1.f32"));
    vx_nn_convolution_params_t inception_c2_1x1_params;
    inception_c2_1x1_params.padding_x = 0;
    inception_c2_1x1_params.padding_y = 0;
    inception_c2_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c2_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c2_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c2_1x1_params.dilation_x = 0;
    inception_c2_1x1_params.dilation_y = 0;
    vx_node inception_c2_1x1_node;
    inception_c2_1x1_node = vxConvolutionLayer(graph, inception_c2_pool_ave, inception_c2_1x1_W, NULL, &inception_c2_1x1_params, sizeof(inception_c2_1x1_params ), inception_c2_1x1);
    ERROR_CHECK_OBJECT(inception_c2_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_node));

    // inception_c2_1x1_bn Layer
    vx_size inception_c2_1x1_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x1_scale;
    inception_c2_1x1_scale = vxCreateVirtualTensor(graph,4, inception_c2_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_scale);
    vx_size inception_c2_1x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_c2_1x1_bn_eps = 0.001;
    vx_tensor inception_c2_1x1_bn_W;
    inception_c2_1x1_bn_W = vxCreateTensor(context,1, inception_c2_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_bn_W, dataFolder + "/weights/inception_c2_1x1_bn.f32"));
    vx_size inception_c2_1x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x1_bn_B;
    inception_c2_1x1_bn_B = vxCreateTensor(context,1, inception_c2_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_bn_B, dataFolder + "/bias/inception_c2_1x1_bn.f32"));
    vx_size inception_c2_1x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_c2_1x1_scale_W;
    inception_c2_1x1_scale_W = vxCreateTensor(context,1, inception_c2_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_scale_W, dataFolder + "/weights/inception_c2_1x1_scale.f32"));
    vx_size inception_c2_1x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_c2_1x1_scale_B;
    inception_c2_1x1_scale_B = vxCreateTensor(context,1, inception_c2_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c2_1x1_scale_B, dataFolder + "/bias/inception_c2_1x1_scale.f32"));
    vx_node inception_c2_1x1_bn_node;
    inception_c2_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_c2_1x1, inception_c2_1x1_bn_W, inception_c2_1x1_bn_B, inception_c2_1x1_scale_W, inception_c2_1x1_scale_B, inception_c2_1x1_bn_eps, inception_c2_1x1_scale);
    ERROR_CHECK_OBJECT(inception_c2_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_bn_node));

    // inception_c2_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c2_1x1_relu Layer
    vx_size inception_c2_1x1_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c2_1x1_relu;
    inception_c2_1x1_relu = vxCreateVirtualTensor(graph,4, inception_c2_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_1x1_relu);
    vx_enum inception_c2_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c2_1x1_relu_param_a = 0;
    vx_float32 inception_c2_1x1_relu_param_b = 0;
    vx_node inception_c2_1x1_relu_node;
    inception_c2_1x1_relu_node = vxActivationLayer(graph, inception_c2_1x1_scale, inception_c2_1x1_relu_mode, inception_c2_1x1_relu_param_a, inception_c2_1x1_relu_param_b, inception_c2_1x1_relu);
    ERROR_CHECK_OBJECT(inception_c2_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_1x1_relu_node));

    // inception_c2_concat Layer
    vx_size inception_c2_concat_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor inception_c2_concat;
    inception_c2_concat = vxCreateVirtualTensor(graph,4, inception_c2_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c2_concat);
    vx_node inception_c2_concat_node;
    inception_c2_concat_node = vxConcatLayer(graph, inception_c2_concat, inception_c2_1x1_2_relu, inception_c2_1x3_relu, inception_c2_3x1_relu, inception_c2_1x3_3_relu, inception_c2_3x1_3_relu, inception_c2_1x1_relu, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_c2_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c2_concat_node));

    // inception_c3_1x1_2 Layer
    vx_size inception_c3_1x1_2_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x1_2;
    inception_c3_1x1_2 = vxCreateVirtualTensor(graph,4, inception_c3_1x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2);
    vx_size inception_c3_1x1_2_W_dims[4] = { 1, 1, 1536, 256 };
    vx_tensor inception_c3_1x1_2_W;
    inception_c3_1x1_2_W = vxCreateTensor(context,4, inception_c3_1x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_2_W, dataFolder + "/weights/inception_c3_1x1_2.f32"));
    vx_nn_convolution_params_t inception_c3_1x1_2_params;
    inception_c3_1x1_2_params.padding_x = 0;
    inception_c3_1x1_2_params.padding_y = 0;
    inception_c3_1x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x1_2_params.dilation_x = 0;
    inception_c3_1x1_2_params.dilation_y = 0;
    vx_node inception_c3_1x1_2_node;
    inception_c3_1x1_2_node = vxConvolutionLayer(graph, inception_c2_concat, inception_c3_1x1_2_W, NULL, &inception_c3_1x1_2_params, sizeof(inception_c3_1x1_2_params ), inception_c3_1x1_2);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_2_node));

    // inception_c3_1x1_2_bn Layer
    vx_size inception_c3_1x1_2_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x1_2_scale;
    inception_c3_1x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_scale);
    vx_size inception_c3_1x1_2_bn_W_dims[1] = { 256 };
    vx_float32 inception_c3_1x1_2_bn_eps = 0.001;
    vx_tensor inception_c3_1x1_2_bn_W;
    inception_c3_1x1_2_bn_W = vxCreateTensor(context,1, inception_c3_1x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_2_bn_W, dataFolder + "/weights/inception_c3_1x1_2_bn.f32"));
    vx_size inception_c3_1x1_2_bn_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x1_2_bn_B;
    inception_c3_1x1_2_bn_B = vxCreateTensor(context,1, inception_c3_1x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_2_bn_B, dataFolder + "/bias/inception_c3_1x1_2_bn.f32"));
    vx_size inception_c3_1x1_2_scale_W_dims[1] = { 256 };
    vx_tensor inception_c3_1x1_2_scale_W;
    inception_c3_1x1_2_scale_W = vxCreateTensor(context,1, inception_c3_1x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_2_scale_W, dataFolder + "/weights/inception_c3_1x1_2_scale.f32"));
    vx_size inception_c3_1x1_2_scale_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x1_2_scale_B;
    inception_c3_1x1_2_scale_B = vxCreateTensor(context,1, inception_c3_1x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_2_scale_B, dataFolder + "/bias/inception_c3_1x1_2_scale.f32"));
    vx_node inception_c3_1x1_2_bn_node;
    inception_c3_1x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x1_2, inception_c3_1x1_2_bn_W, inception_c3_1x1_2_bn_B, inception_c3_1x1_2_scale_W, inception_c3_1x1_2_scale_B, inception_c3_1x1_2_bn_eps, inception_c3_1x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_2_bn_node));

    // inception_c3_1x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x1_2_relu Layer
    vx_size inception_c3_1x1_2_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x1_2_relu;
    inception_c3_1x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_relu);
    vx_enum inception_c3_1x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x1_2_relu_param_a = 0;
    vx_float32 inception_c3_1x1_2_relu_param_b = 0;
    vx_node inception_c3_1x1_2_relu_node;
    inception_c3_1x1_2_relu_node = vxActivationLayer(graph, inception_c3_1x1_2_scale, inception_c3_1x1_2_relu_mode, inception_c3_1x1_2_relu_param_a, inception_c3_1x1_2_relu_param_b, inception_c3_1x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_2_relu_node));

    // inception_c3_1x1_3 Layer
    vx_size inception_c3_1x1_3_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c3_1x1_3;
    inception_c3_1x1_3 = vxCreateVirtualTensor(graph,4, inception_c3_1x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3);
    vx_size inception_c3_1x1_3_W_dims[4] = { 1, 1, 1536, 384 };
    vx_tensor inception_c3_1x1_3_W;
    inception_c3_1x1_3_W = vxCreateTensor(context,4, inception_c3_1x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_3_W, dataFolder + "/weights/inception_c3_1x1_3.f32"));
    vx_nn_convolution_params_t inception_c3_1x1_3_params;
    inception_c3_1x1_3_params.padding_x = 0;
    inception_c3_1x1_3_params.padding_y = 0;
    inception_c3_1x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x1_3_params.dilation_x = 0;
    inception_c3_1x1_3_params.dilation_y = 0;
    vx_node inception_c3_1x1_3_node;
    inception_c3_1x1_3_node = vxConvolutionLayer(graph, inception_c2_concat, inception_c3_1x1_3_W, NULL, &inception_c3_1x1_3_params, sizeof(inception_c3_1x1_3_params ), inception_c3_1x1_3);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_3_node));

    // inception_c3_1x1_3_bn Layer
    vx_size inception_c3_1x1_3_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c3_1x1_3_scale;
    inception_c3_1x1_3_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_scale);
    vx_size inception_c3_1x1_3_bn_W_dims[1] = { 384 };
    vx_float32 inception_c3_1x1_3_bn_eps = 0.001;
    vx_tensor inception_c3_1x1_3_bn_W;
    inception_c3_1x1_3_bn_W = vxCreateTensor(context,1, inception_c3_1x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_3_bn_W, dataFolder + "/weights/inception_c3_1x1_3_bn.f32"));
    vx_size inception_c3_1x1_3_bn_B_dims[1] = { 384 };
    vx_tensor inception_c3_1x1_3_bn_B;
    inception_c3_1x1_3_bn_B = vxCreateTensor(context,1, inception_c3_1x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_3_bn_B, dataFolder + "/bias/inception_c3_1x1_3_bn.f32"));
    vx_size inception_c3_1x1_3_scale_W_dims[1] = { 384 };
    vx_tensor inception_c3_1x1_3_scale_W;
    inception_c3_1x1_3_scale_W = vxCreateTensor(context,1, inception_c3_1x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_3_scale_W, dataFolder + "/weights/inception_c3_1x1_3_scale.f32"));
    vx_size inception_c3_1x1_3_scale_B_dims[1] = { 384 };
    vx_tensor inception_c3_1x1_3_scale_B;
    inception_c3_1x1_3_scale_B = vxCreateTensor(context,1, inception_c3_1x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_3_scale_B, dataFolder + "/bias/inception_c3_1x1_3_scale.f32"));
    vx_node inception_c3_1x1_3_bn_node;
    inception_c3_1x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x1_3, inception_c3_1x1_3_bn_W, inception_c3_1x1_3_bn_B, inception_c3_1x1_3_scale_W, inception_c3_1x1_3_scale_B, inception_c3_1x1_3_bn_eps, inception_c3_1x1_3_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_3_bn_node));

    // inception_c3_1x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x1_3_relu Layer
    vx_size inception_c3_1x1_3_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c3_1x1_3_relu;
    inception_c3_1x1_3_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_relu);
    vx_enum inception_c3_1x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x1_3_relu_param_a = 0;
    vx_float32 inception_c3_1x1_3_relu_param_b = 0;
    vx_node inception_c3_1x1_3_relu_node;
    inception_c3_1x1_3_relu_node = vxActivationLayer(graph, inception_c3_1x1_3_scale, inception_c3_1x1_3_relu_mode, inception_c3_1x1_3_relu_param_a, inception_c3_1x1_3_relu_param_b, inception_c3_1x1_3_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_3_relu_node));

    // inception_c3_1x3 Layer
    vx_size inception_c3_1x3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x3;
    inception_c3_1x3 = vxCreateVirtualTensor(graph,4, inception_c3_1x3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3);
    vx_size inception_c3_1x3_W_dims[4] = { 3, 1, 384, 256 };
    vx_tensor inception_c3_1x3_W;
    inception_c3_1x3_W = vxCreateTensor(context,4, inception_c3_1x3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_W, dataFolder + "/weights/inception_c3_1x3.f32"));
    vx_nn_convolution_params_t inception_c3_1x3_params;
    inception_c3_1x3_params.padding_x = 1;
    inception_c3_1x3_params.padding_y = 0;
    inception_c3_1x3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x3_params.dilation_x = 0;
    inception_c3_1x3_params.dilation_y = 0;
    vx_node inception_c3_1x3_node;
    inception_c3_1x3_node = vxConvolutionLayer(graph, inception_c3_1x1_3_relu, inception_c3_1x3_W, NULL, &inception_c3_1x3_params, sizeof(inception_c3_1x3_params ), inception_c3_1x3);
    ERROR_CHECK_OBJECT(inception_c3_1x3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_node));

    // inception_c3_1x3_bn Layer
    vx_size inception_c3_1x3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x3_scale;
    inception_c3_1x3_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_scale);
    vx_size inception_c3_1x3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c3_1x3_bn_eps = 0.001;
    vx_tensor inception_c3_1x3_bn_W;
    inception_c3_1x3_bn_W = vxCreateTensor(context,1, inception_c3_1x3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_bn_W, dataFolder + "/weights/inception_c3_1x3_bn.f32"));
    vx_size inception_c3_1x3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x3_bn_B;
    inception_c3_1x3_bn_B = vxCreateTensor(context,1, inception_c3_1x3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_bn_B, dataFolder + "/bias/inception_c3_1x3_bn.f32"));
    vx_size inception_c3_1x3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c3_1x3_scale_W;
    inception_c3_1x3_scale_W = vxCreateTensor(context,1, inception_c3_1x3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_scale_W, dataFolder + "/weights/inception_c3_1x3_scale.f32"));
    vx_size inception_c3_1x3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x3_scale_B;
    inception_c3_1x3_scale_B = vxCreateTensor(context,1, inception_c3_1x3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_scale_B, dataFolder + "/bias/inception_c3_1x3_scale.f32"));
    vx_node inception_c3_1x3_bn_node;
    inception_c3_1x3_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x3, inception_c3_1x3_bn_W, inception_c3_1x3_bn_B, inception_c3_1x3_scale_W, inception_c3_1x3_scale_B, inception_c3_1x3_bn_eps, inception_c3_1x3_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_bn_node));

    // inception_c3_1x3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x3_relu Layer
    vx_size inception_c3_1x3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x3_relu;
    inception_c3_1x3_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_relu);
    vx_enum inception_c3_1x3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x3_relu_param_a = 0;
    vx_float32 inception_c3_1x3_relu_param_b = 0;
    vx_node inception_c3_1x3_relu_node;
    inception_c3_1x3_relu_node = vxActivationLayer(graph, inception_c3_1x3_scale, inception_c3_1x3_relu_mode, inception_c3_1x3_relu_param_a, inception_c3_1x3_relu_param_b, inception_c3_1x3_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_relu_node));

    // inception_c3_3x1 Layer
    vx_size inception_c3_3x1_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_3x1;
    inception_c3_3x1 = vxCreateVirtualTensor(graph,4, inception_c3_3x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1);
    vx_size inception_c3_3x1_W_dims[4] = { 1, 3, 384, 256 };
    vx_tensor inception_c3_3x1_W;
    inception_c3_3x1_W = vxCreateTensor(context,4, inception_c3_3x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_W, dataFolder + "/weights/inception_c3_3x1.f32"));
    vx_nn_convolution_params_t inception_c3_3x1_params;
    inception_c3_3x1_params.padding_x = 0;
    inception_c3_3x1_params.padding_y = 1;
    inception_c3_3x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_3x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_3x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_3x1_params.dilation_x = 0;
    inception_c3_3x1_params.dilation_y = 0;
    vx_node inception_c3_3x1_node;
    inception_c3_3x1_node = vxConvolutionLayer(graph, inception_c3_1x1_3_relu, inception_c3_3x1_W, NULL, &inception_c3_3x1_params, sizeof(inception_c3_3x1_params ), inception_c3_3x1);
    ERROR_CHECK_OBJECT(inception_c3_3x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_node));

    // inception_c3_3x1_bn Layer
    vx_size inception_c3_3x1_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_3x1_scale;
    inception_c3_3x1_scale = vxCreateVirtualTensor(graph,4, inception_c3_3x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_scale);
    vx_size inception_c3_3x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_c3_3x1_bn_eps = 0.001;
    vx_tensor inception_c3_3x1_bn_W;
    inception_c3_3x1_bn_W = vxCreateTensor(context,1, inception_c3_3x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_bn_W, dataFolder + "/weights/inception_c3_3x1_bn.f32"));
    vx_size inception_c3_3x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_c3_3x1_bn_B;
    inception_c3_3x1_bn_B = vxCreateTensor(context,1, inception_c3_3x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_bn_B, dataFolder + "/bias/inception_c3_3x1_bn.f32"));
    vx_size inception_c3_3x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_c3_3x1_scale_W;
    inception_c3_3x1_scale_W = vxCreateTensor(context,1, inception_c3_3x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_scale_W, dataFolder + "/weights/inception_c3_3x1_scale.f32"));
    vx_size inception_c3_3x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_c3_3x1_scale_B;
    inception_c3_3x1_scale_B = vxCreateTensor(context,1, inception_c3_3x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_scale_B, dataFolder + "/bias/inception_c3_3x1_scale.f32"));
    vx_node inception_c3_3x1_bn_node;
    inception_c3_3x1_bn_node = vxBatchNormalizationLayer(graph, inception_c3_3x1, inception_c3_3x1_bn_W, inception_c3_3x1_bn_B, inception_c3_3x1_scale_W, inception_c3_3x1_scale_B, inception_c3_3x1_bn_eps, inception_c3_3x1_scale);
    ERROR_CHECK_OBJECT(inception_c3_3x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_bn_node));

    // inception_c3_3x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_3x1_relu Layer
    vx_size inception_c3_3x1_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_3x1_relu;
    inception_c3_3x1_relu = vxCreateVirtualTensor(graph,4, inception_c3_3x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_relu);
    vx_enum inception_c3_3x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_3x1_relu_param_a = 0;
    vx_float32 inception_c3_3x1_relu_param_b = 0;
    vx_node inception_c3_3x1_relu_node;
    inception_c3_3x1_relu_node = vxActivationLayer(graph, inception_c3_3x1_scale, inception_c3_3x1_relu_mode, inception_c3_3x1_relu_param_a, inception_c3_3x1_relu_param_b, inception_c3_3x1_relu);
    ERROR_CHECK_OBJECT(inception_c3_3x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_relu_node));

    // inception_c3_1x1_4 Layer
    vx_size inception_c3_1x1_4_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c3_1x1_4;
    inception_c3_1x1_4 = vxCreateVirtualTensor(graph,4, inception_c3_1x1_4_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4);
    vx_size inception_c3_1x1_4_W_dims[4] = { 1, 1, 1536, 384 };
    vx_tensor inception_c3_1x1_4_W;
    inception_c3_1x1_4_W = vxCreateTensor(context,4, inception_c3_1x1_4_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_4_W, dataFolder + "/weights/inception_c3_1x1_4.f32"));
    vx_nn_convolution_params_t inception_c3_1x1_4_params;
    inception_c3_1x1_4_params.padding_x = 0;
    inception_c3_1x1_4_params.padding_y = 0;
    inception_c3_1x1_4_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x1_4_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x1_4_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x1_4_params.dilation_x = 0;
    inception_c3_1x1_4_params.dilation_y = 0;
    vx_node inception_c3_1x1_4_node;
    inception_c3_1x1_4_node = vxConvolutionLayer(graph, inception_c2_concat, inception_c3_1x1_4_W, NULL, &inception_c3_1x1_4_params, sizeof(inception_c3_1x1_4_params ), inception_c3_1x1_4);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_4_node));

    // inception_c3_1x1_4_bn Layer
    vx_size inception_c3_1x1_4_scale_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c3_1x1_4_scale;
    inception_c3_1x1_4_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x1_4_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_scale);
    vx_size inception_c3_1x1_4_bn_W_dims[1] = { 384 };
    vx_float32 inception_c3_1x1_4_bn_eps = 0.001;
    vx_tensor inception_c3_1x1_4_bn_W;
    inception_c3_1x1_4_bn_W = vxCreateTensor(context,1, inception_c3_1x1_4_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_4_bn_W, dataFolder + "/weights/inception_c3_1x1_4_bn.f32"));
    vx_size inception_c3_1x1_4_bn_B_dims[1] = { 384 };
    vx_tensor inception_c3_1x1_4_bn_B;
    inception_c3_1x1_4_bn_B = vxCreateTensor(context,1, inception_c3_1x1_4_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_4_bn_B, dataFolder + "/bias/inception_c3_1x1_4_bn.f32"));
    vx_size inception_c3_1x1_4_scale_W_dims[1] = { 384 };
    vx_tensor inception_c3_1x1_4_scale_W;
    inception_c3_1x1_4_scale_W = vxCreateTensor(context,1, inception_c3_1x1_4_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_4_scale_W, dataFolder + "/weights/inception_c3_1x1_4_scale.f32"));
    vx_size inception_c3_1x1_4_scale_B_dims[1] = { 384 };
    vx_tensor inception_c3_1x1_4_scale_B;
    inception_c3_1x1_4_scale_B = vxCreateTensor(context,1, inception_c3_1x1_4_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_4_scale_B, dataFolder + "/bias/inception_c3_1x1_4_scale.f32"));
    vx_node inception_c3_1x1_4_bn_node;
    inception_c3_1x1_4_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x1_4, inception_c3_1x1_4_bn_W, inception_c3_1x1_4_bn_B, inception_c3_1x1_4_scale_W, inception_c3_1x1_4_scale_B, inception_c3_1x1_4_bn_eps, inception_c3_1x1_4_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_4_bn_node));

    // inception_c3_1x1_4_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x1_4_relu Layer
    vx_size inception_c3_1x1_4_relu_dims[4] = { 8, 8, 384, 1 };
    vx_tensor inception_c3_1x1_4_relu;
    inception_c3_1x1_4_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x1_4_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_relu);
    vx_enum inception_c3_1x1_4_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x1_4_relu_param_a = 0;
    vx_float32 inception_c3_1x1_4_relu_param_b = 0;
    vx_node inception_c3_1x1_4_relu_node;
    inception_c3_1x1_4_relu_node = vxActivationLayer(graph, inception_c3_1x1_4_scale, inception_c3_1x1_4_relu_mode, inception_c3_1x1_4_relu_param_a, inception_c3_1x1_4_relu_param_b, inception_c3_1x1_4_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x1_4_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_4_relu_node));

    // inception_c3_3x1_2 Layer
    vx_size inception_c3_3x1_2_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c3_3x1_2;
    inception_c3_3x1_2 = vxCreateVirtualTensor(graph,4, inception_c3_3x1_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2);
    vx_size inception_c3_3x1_2_W_dims[4] = { 1, 3, 384, 448 };
    vx_tensor inception_c3_3x1_2_W;
    inception_c3_3x1_2_W = vxCreateTensor(context,4, inception_c3_3x1_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_2_W, dataFolder + "/weights/inception_c3_3x1_2.f32"));
    vx_nn_convolution_params_t inception_c3_3x1_2_params;
    inception_c3_3x1_2_params.padding_x = 0;
    inception_c3_3x1_2_params.padding_y = 1;
    inception_c3_3x1_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_3x1_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_3x1_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_3x1_2_params.dilation_x = 0;
    inception_c3_3x1_2_params.dilation_y = 0;
    vx_node inception_c3_3x1_2_node;
    inception_c3_3x1_2_node = vxConvolutionLayer(graph, inception_c3_1x1_4_relu, inception_c3_3x1_2_W, NULL, &inception_c3_3x1_2_params, sizeof(inception_c3_3x1_2_params ), inception_c3_3x1_2);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_2_node));

    // inception_c3_3x1_2_bn Layer
    vx_size inception_c3_3x1_2_scale_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c3_3x1_2_scale;
    inception_c3_3x1_2_scale = vxCreateVirtualTensor(graph,4, inception_c3_3x1_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_scale);
    vx_size inception_c3_3x1_2_bn_W_dims[1] = { 448 };
    vx_float32 inception_c3_3x1_2_bn_eps = 0.001;
    vx_tensor inception_c3_3x1_2_bn_W;
    inception_c3_3x1_2_bn_W = vxCreateTensor(context,1, inception_c3_3x1_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_2_bn_W, dataFolder + "/weights/inception_c3_3x1_2_bn.f32"));
    vx_size inception_c3_3x1_2_bn_B_dims[1] = { 448 };
    vx_tensor inception_c3_3x1_2_bn_B;
    inception_c3_3x1_2_bn_B = vxCreateTensor(context,1, inception_c3_3x1_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_2_bn_B, dataFolder + "/bias/inception_c3_3x1_2_bn.f32"));
    vx_size inception_c3_3x1_2_scale_W_dims[1] = { 448 };
    vx_tensor inception_c3_3x1_2_scale_W;
    inception_c3_3x1_2_scale_W = vxCreateTensor(context,1, inception_c3_3x1_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_2_scale_W, dataFolder + "/weights/inception_c3_3x1_2_scale.f32"));
    vx_size inception_c3_3x1_2_scale_B_dims[1] = { 448 };
    vx_tensor inception_c3_3x1_2_scale_B;
    inception_c3_3x1_2_scale_B = vxCreateTensor(context,1, inception_c3_3x1_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_2_scale_B, dataFolder + "/bias/inception_c3_3x1_2_scale.f32"));
    vx_node inception_c3_3x1_2_bn_node;
    inception_c3_3x1_2_bn_node = vxBatchNormalizationLayer(graph, inception_c3_3x1_2, inception_c3_3x1_2_bn_W, inception_c3_3x1_2_bn_B, inception_c3_3x1_2_scale_W, inception_c3_3x1_2_scale_B, inception_c3_3x1_2_bn_eps, inception_c3_3x1_2_scale);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_2_bn_node));

    // inception_c3_3x1_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_3x1_2_relu Layer
    vx_size inception_c3_3x1_2_relu_dims[4] = { 8, 8, 448, 1 };
    vx_tensor inception_c3_3x1_2_relu;
    inception_c3_3x1_2_relu = vxCreateVirtualTensor(graph,4, inception_c3_3x1_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_relu);
    vx_enum inception_c3_3x1_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_3x1_2_relu_param_a = 0;
    vx_float32 inception_c3_3x1_2_relu_param_b = 0;
    vx_node inception_c3_3x1_2_relu_node;
    inception_c3_3x1_2_relu_node = vxActivationLayer(graph, inception_c3_3x1_2_scale, inception_c3_3x1_2_relu_mode, inception_c3_3x1_2_relu_param_a, inception_c3_3x1_2_relu_param_b, inception_c3_3x1_2_relu);
    ERROR_CHECK_OBJECT(inception_c3_3x1_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_2_relu_node));

    // inception_c3_1x3_2 Layer
    vx_size inception_c3_1x3_2_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c3_1x3_2;
    inception_c3_1x3_2 = vxCreateVirtualTensor(graph,4, inception_c3_1x3_2_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2);
    vx_size inception_c3_1x3_2_W_dims[4] = { 3, 1, 448, 512 };
    vx_tensor inception_c3_1x3_2_W;
    inception_c3_1x3_2_W = vxCreateTensor(context,4, inception_c3_1x3_2_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_2_W, dataFolder + "/weights/inception_c3_1x3_2.f32"));
    vx_nn_convolution_params_t inception_c3_1x3_2_params;
    inception_c3_1x3_2_params.padding_x = 1;
    inception_c3_1x3_2_params.padding_y = 0;
    inception_c3_1x3_2_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x3_2_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x3_2_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x3_2_params.dilation_x = 0;
    inception_c3_1x3_2_params.dilation_y = 0;
    vx_node inception_c3_1x3_2_node;
    inception_c3_1x3_2_node = vxConvolutionLayer(graph, inception_c3_3x1_2_relu, inception_c3_1x3_2_W, NULL, &inception_c3_1x3_2_params, sizeof(inception_c3_1x3_2_params ), inception_c3_1x3_2);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_2_node));

    // inception_c3_1x3_2_bn Layer
    vx_size inception_c3_1x3_2_scale_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c3_1x3_2_scale;
    inception_c3_1x3_2_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x3_2_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_scale);
    vx_size inception_c3_1x3_2_bn_W_dims[1] = { 512 };
    vx_float32 inception_c3_1x3_2_bn_eps = 0.001;
    vx_tensor inception_c3_1x3_2_bn_W;
    inception_c3_1x3_2_bn_W = vxCreateTensor(context,1, inception_c3_1x3_2_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_2_bn_W, dataFolder + "/weights/inception_c3_1x3_2_bn.f32"));
    vx_size inception_c3_1x3_2_bn_B_dims[1] = { 512 };
    vx_tensor inception_c3_1x3_2_bn_B;
    inception_c3_1x3_2_bn_B = vxCreateTensor(context,1, inception_c3_1x3_2_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_2_bn_B, dataFolder + "/bias/inception_c3_1x3_2_bn.f32"));
    vx_size inception_c3_1x3_2_scale_W_dims[1] = { 512 };
    vx_tensor inception_c3_1x3_2_scale_W;
    inception_c3_1x3_2_scale_W = vxCreateTensor(context,1, inception_c3_1x3_2_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_2_scale_W, dataFolder + "/weights/inception_c3_1x3_2_scale.f32"));
    vx_size inception_c3_1x3_2_scale_B_dims[1] = { 512 };
    vx_tensor inception_c3_1x3_2_scale_B;
    inception_c3_1x3_2_scale_B = vxCreateTensor(context,1, inception_c3_1x3_2_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_2_scale_B, dataFolder + "/bias/inception_c3_1x3_2_scale.f32"));
    vx_node inception_c3_1x3_2_bn_node;
    inception_c3_1x3_2_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x3_2, inception_c3_1x3_2_bn_W, inception_c3_1x3_2_bn_B, inception_c3_1x3_2_scale_W, inception_c3_1x3_2_scale_B, inception_c3_1x3_2_bn_eps, inception_c3_1x3_2_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_2_bn_node));

    // inception_c3_1x3_2_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x3_2_relu Layer
    vx_size inception_c3_1x3_2_relu_dims[4] = { 8, 8, 512, 1 };
    vx_tensor inception_c3_1x3_2_relu;
    inception_c3_1x3_2_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x3_2_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_relu);
    vx_enum inception_c3_1x3_2_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x3_2_relu_param_a = 0;
    vx_float32 inception_c3_1x3_2_relu_param_b = 0;
    vx_node inception_c3_1x3_2_relu_node;
    inception_c3_1x3_2_relu_node = vxActivationLayer(graph, inception_c3_1x3_2_scale, inception_c3_1x3_2_relu_mode, inception_c3_1x3_2_relu_param_a, inception_c3_1x3_2_relu_param_b, inception_c3_1x3_2_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x3_2_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_2_relu_node));

    // inception_c3_1x3_3 Layer
    vx_size inception_c3_1x3_3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x3_3;
    inception_c3_1x3_3 = vxCreateVirtualTensor(graph,4, inception_c3_1x3_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3);
    vx_size inception_c3_1x3_3_W_dims[4] = { 3, 1, 512, 256 };
    vx_tensor inception_c3_1x3_3_W;
    inception_c3_1x3_3_W = vxCreateTensor(context,4, inception_c3_1x3_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_3_W, dataFolder + "/weights/inception_c3_1x3_3.f32"));
    vx_nn_convolution_params_t inception_c3_1x3_3_params;
    inception_c3_1x3_3_params.padding_x = 1;
    inception_c3_1x3_3_params.padding_y = 0;
    inception_c3_1x3_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x3_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x3_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x3_3_params.dilation_x = 0;
    inception_c3_1x3_3_params.dilation_y = 0;
    vx_node inception_c3_1x3_3_node;
    inception_c3_1x3_3_node = vxConvolutionLayer(graph, inception_c3_1x3_2_relu, inception_c3_1x3_3_W, NULL, &inception_c3_1x3_3_params, sizeof(inception_c3_1x3_3_params ), inception_c3_1x3_3);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_3_node));

    // inception_c3_1x3_3_bn Layer
    vx_size inception_c3_1x3_3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x3_3_scale;
    inception_c3_1x3_3_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x3_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_scale);
    vx_size inception_c3_1x3_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c3_1x3_3_bn_eps = 0.001;
    vx_tensor inception_c3_1x3_3_bn_W;
    inception_c3_1x3_3_bn_W = vxCreateTensor(context,1, inception_c3_1x3_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_3_bn_W, dataFolder + "/weights/inception_c3_1x3_3_bn.f32"));
    vx_size inception_c3_1x3_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x3_3_bn_B;
    inception_c3_1x3_3_bn_B = vxCreateTensor(context,1, inception_c3_1x3_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_3_bn_B, dataFolder + "/bias/inception_c3_1x3_3_bn.f32"));
    vx_size inception_c3_1x3_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c3_1x3_3_scale_W;
    inception_c3_1x3_3_scale_W = vxCreateTensor(context,1, inception_c3_1x3_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_3_scale_W, dataFolder + "/weights/inception_c3_1x3_3_scale.f32"));
    vx_size inception_c3_1x3_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x3_3_scale_B;
    inception_c3_1x3_3_scale_B = vxCreateTensor(context,1, inception_c3_1x3_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x3_3_scale_B, dataFolder + "/bias/inception_c3_1x3_3_scale.f32"));
    vx_node inception_c3_1x3_3_bn_node;
    inception_c3_1x3_3_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x3_3, inception_c3_1x3_3_bn_W, inception_c3_1x3_3_bn_B, inception_c3_1x3_3_scale_W, inception_c3_1x3_3_scale_B, inception_c3_1x3_3_bn_eps, inception_c3_1x3_3_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_3_bn_node));

    // inception_c3_1x3_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x3_3_relu Layer
    vx_size inception_c3_1x3_3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x3_3_relu;
    inception_c3_1x3_3_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x3_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_relu);
    vx_enum inception_c3_1x3_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x3_3_relu_param_a = 0;
    vx_float32 inception_c3_1x3_3_relu_param_b = 0;
    vx_node inception_c3_1x3_3_relu_node;
    inception_c3_1x3_3_relu_node = vxActivationLayer(graph, inception_c3_1x3_3_scale, inception_c3_1x3_3_relu_mode, inception_c3_1x3_3_relu_param_a, inception_c3_1x3_3_relu_param_b, inception_c3_1x3_3_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x3_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x3_3_relu_node));

    // inception_c3_3x1_3 Layer
    vx_size inception_c3_3x1_3_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_3x1_3;
    inception_c3_3x1_3 = vxCreateVirtualTensor(graph,4, inception_c3_3x1_3_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3);
    vx_size inception_c3_3x1_3_W_dims[4] = { 1, 3, 512, 256 };
    vx_tensor inception_c3_3x1_3_W;
    inception_c3_3x1_3_W = vxCreateTensor(context,4, inception_c3_3x1_3_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_3_W, dataFolder + "/weights/inception_c3_3x1_3.f32"));
    vx_nn_convolution_params_t inception_c3_3x1_3_params;
    inception_c3_3x1_3_params.padding_x = 0;
    inception_c3_3x1_3_params.padding_y = 1;
    inception_c3_3x1_3_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_3x1_3_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_3x1_3_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_3x1_3_params.dilation_x = 0;
    inception_c3_3x1_3_params.dilation_y = 0;
    vx_node inception_c3_3x1_3_node;
    inception_c3_3x1_3_node = vxConvolutionLayer(graph, inception_c3_1x3_2_relu, inception_c3_3x1_3_W, NULL, &inception_c3_3x1_3_params, sizeof(inception_c3_3x1_3_params ), inception_c3_3x1_3);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_3_node));

    // inception_c3_3x1_3_bn Layer
    vx_size inception_c3_3x1_3_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_3x1_3_scale;
    inception_c3_3x1_3_scale = vxCreateVirtualTensor(graph,4, inception_c3_3x1_3_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_scale);
    vx_size inception_c3_3x1_3_bn_W_dims[1] = { 256 };
    vx_float32 inception_c3_3x1_3_bn_eps = 0.001;
    vx_tensor inception_c3_3x1_3_bn_W;
    inception_c3_3x1_3_bn_W = vxCreateTensor(context,1, inception_c3_3x1_3_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_3_bn_W, dataFolder + "/weights/inception_c3_3x1_3_bn.f32"));
    vx_size inception_c3_3x1_3_bn_B_dims[1] = { 256 };
    vx_tensor inception_c3_3x1_3_bn_B;
    inception_c3_3x1_3_bn_B = vxCreateTensor(context,1, inception_c3_3x1_3_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_3_bn_B, dataFolder + "/bias/inception_c3_3x1_3_bn.f32"));
    vx_size inception_c3_3x1_3_scale_W_dims[1] = { 256 };
    vx_tensor inception_c3_3x1_3_scale_W;
    inception_c3_3x1_3_scale_W = vxCreateTensor(context,1, inception_c3_3x1_3_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_3_scale_W, dataFolder + "/weights/inception_c3_3x1_3_scale.f32"));
    vx_size inception_c3_3x1_3_scale_B_dims[1] = { 256 };
    vx_tensor inception_c3_3x1_3_scale_B;
    inception_c3_3x1_3_scale_B = vxCreateTensor(context,1, inception_c3_3x1_3_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_3x1_3_scale_B, dataFolder + "/bias/inception_c3_3x1_3_scale.f32"));
    vx_node inception_c3_3x1_3_bn_node;
    inception_c3_3x1_3_bn_node = vxBatchNormalizationLayer(graph, inception_c3_3x1_3, inception_c3_3x1_3_bn_W, inception_c3_3x1_3_bn_B, inception_c3_3x1_3_scale_W, inception_c3_3x1_3_scale_B, inception_c3_3x1_3_bn_eps, inception_c3_3x1_3_scale);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_3_bn_node));

    // inception_c3_3x1_3_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_3x1_3_relu Layer
    vx_size inception_c3_3x1_3_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_3x1_3_relu;
    inception_c3_3x1_3_relu = vxCreateVirtualTensor(graph,4, inception_c3_3x1_3_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_relu);
    vx_enum inception_c3_3x1_3_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_3x1_3_relu_param_a = 0;
    vx_float32 inception_c3_3x1_3_relu_param_b = 0;
    vx_node inception_c3_3x1_3_relu_node;
    inception_c3_3x1_3_relu_node = vxActivationLayer(graph, inception_c3_3x1_3_scale, inception_c3_3x1_3_relu_mode, inception_c3_3x1_3_relu_param_a, inception_c3_3x1_3_relu_param_b, inception_c3_3x1_3_relu);
    ERROR_CHECK_OBJECT(inception_c3_3x1_3_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_3x1_3_relu_node));

    // inception_c3_pool_ave Layer
    vx_size inception_c3_pool_ave_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor inception_c3_pool_ave;
    inception_c3_pool_ave = vxCreateVirtualTensor(graph,4, inception_c3_pool_ave_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_pool_ave);
    vx_enum inception_c3_pool_ave_type = VX_NN_POOLING_AVG;
    vx_size inception_c3_pool_ave_kernel_w = 3;
    vx_size inception_c3_pool_ave_kernel_h = 3;
    vx_size inception_c3_pool_ave_pad_w = 1;
    vx_size inception_c3_pool_ave_pad_h = 1;
    vx_enum inception_c3_pool_ave_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node inception_c3_pool_ave_node;
    inception_c3_pool_ave_node = vxPoolingLayer(graph, inception_c2_concat, inception_c3_pool_ave_type, inception_c3_pool_ave_kernel_w, inception_c3_pool_ave_kernel_h, inception_c3_pool_ave_pad_w, inception_c3_pool_ave_pad_h, inception_c3_pool_ave_roundPolicy, inception_c3_pool_ave );
    ERROR_CHECK_OBJECT(inception_c3_pool_ave_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_pool_ave_node));

    // inception_c3_1x1 Layer
    vx_size inception_c3_1x1_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x1;
    inception_c3_1x1 = vxCreateVirtualTensor(graph,4, inception_c3_1x1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1);
    vx_size inception_c3_1x1_W_dims[4] = { 1, 1, 1536, 256 };
    vx_tensor inception_c3_1x1_W;
    inception_c3_1x1_W = vxCreateTensor(context,4, inception_c3_1x1_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_W, dataFolder + "/weights/inception_c3_1x1.f32"));
    vx_nn_convolution_params_t inception_c3_1x1_params;
    inception_c3_1x1_params.padding_x = 0;
    inception_c3_1x1_params.padding_y = 0;
    inception_c3_1x1_params.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    inception_c3_1x1_params.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    inception_c3_1x1_params.down_scale_size_rounding = VX_NN_DS_SIZE_ROUNDING_FLOOR ;
    inception_c3_1x1_params.dilation_x = 0;
    inception_c3_1x1_params.dilation_y = 0;
    vx_node inception_c3_1x1_node;
    inception_c3_1x1_node = vxConvolutionLayer(graph, inception_c3_pool_ave, inception_c3_1x1_W, NULL, &inception_c3_1x1_params, sizeof(inception_c3_1x1_params ), inception_c3_1x1);
    ERROR_CHECK_OBJECT(inception_c3_1x1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_node));

    // inception_c3_1x1_bn Layer
    vx_size inception_c3_1x1_scale_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x1_scale;
    inception_c3_1x1_scale = vxCreateVirtualTensor(graph,4, inception_c3_1x1_scale_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_scale);
    vx_size inception_c3_1x1_bn_W_dims[1] = { 256 };
    vx_float32 inception_c3_1x1_bn_eps = 0.001;
    vx_tensor inception_c3_1x1_bn_W;
    inception_c3_1x1_bn_W = vxCreateTensor(context,1, inception_c3_1x1_bn_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_bn_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_bn_W, dataFolder + "/weights/inception_c3_1x1_bn.f32"));
    vx_size inception_c3_1x1_bn_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x1_bn_B;
    inception_c3_1x1_bn_B = vxCreateTensor(context,1, inception_c3_1x1_bn_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_bn_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_bn_B, dataFolder + "/bias/inception_c3_1x1_bn.f32"));
    vx_size inception_c3_1x1_scale_W_dims[1] = { 256 };
    vx_tensor inception_c3_1x1_scale_W;
    inception_c3_1x1_scale_W = vxCreateTensor(context,1, inception_c3_1x1_scale_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_scale_W); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_scale_W, dataFolder + "/weights/inception_c3_1x1_scale.f32"));
    vx_size inception_c3_1x1_scale_B_dims[1] = { 256 };
    vx_tensor inception_c3_1x1_scale_B;
    inception_c3_1x1_scale_B = vxCreateTensor(context,1, inception_c3_1x1_scale_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_scale_B); 
    ERROR_CHECK_STATUS(copyTensor(inception_c3_1x1_scale_B, dataFolder + "/bias/inception_c3_1x1_scale.f32"));
    vx_node inception_c3_1x1_bn_node;
    inception_c3_1x1_bn_node = vxBatchNormalizationLayer(graph, inception_c3_1x1, inception_c3_1x1_bn_W, inception_c3_1x1_bn_B, inception_c3_1x1_scale_W, inception_c3_1x1_scale_B, inception_c3_1x1_bn_eps, inception_c3_1x1_scale);
    ERROR_CHECK_OBJECT(inception_c3_1x1_bn_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_bn_node));

    // inception_c3_1x1_scale Layer
    // [NOTE -- Scale Layer Fused With Batch Norm Layer]

    // inception_c3_1x1_relu Layer
    vx_size inception_c3_1x1_relu_dims[4] = { 8, 8, 256, 1 };
    vx_tensor inception_c3_1x1_relu;
    inception_c3_1x1_relu = vxCreateVirtualTensor(graph,4, inception_c3_1x1_relu_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_1x1_relu);
    vx_enum inception_c3_1x1_relu_mode = VX_NN_ACTIVATION_RELU ; 
    vx_float32 inception_c3_1x1_relu_param_a = 0;
    vx_float32 inception_c3_1x1_relu_param_b = 0;
    vx_node inception_c3_1x1_relu_node;
    inception_c3_1x1_relu_node = vxActivationLayer(graph, inception_c3_1x1_scale, inception_c3_1x1_relu_mode, inception_c3_1x1_relu_param_a, inception_c3_1x1_relu_param_b, inception_c3_1x1_relu);
    ERROR_CHECK_OBJECT(inception_c3_1x1_relu_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_1x1_relu_node));

    // inception_c3_concat Layer
    vx_size inception_c3_concat_dims[4] = { 8, 8, 1536, 1 };
    vx_tensor inception_c3_concat;
    inception_c3_concat = vxCreateVirtualTensor(graph,4, inception_c3_concat_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(inception_c3_concat);
    vx_node inception_c3_concat_node;
    inception_c3_concat_node = vxConcatLayer(graph, inception_c3_concat, inception_c3_1x1_2_relu, inception_c3_1x3_relu, inception_c3_3x1_relu, inception_c3_1x3_3_relu, inception_c3_3x1_3_relu, inception_c3_1x1_relu, NULL, NULL );
    ERROR_CHECK_OBJECT(inception_c3_concat_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&inception_c3_concat_node));

    // pool_8x8_s1 Layer
    vx_size pool_8x8_s1_dims[4] = { 1, 1, 1536, 1 };
    vx_tensor pool_8x8_s1;
    pool_8x8_s1 = vxCreateVirtualTensor(graph,4, pool_8x8_s1_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool_8x8_s1);
    vx_enum pool_8x8_s1_type = VX_NN_POOLING_AVG;
    vx_size pool_8x8_s1_kernel_w = 8;
    vx_size pool_8x8_s1_kernel_h = 8;
    vx_size pool_8x8_s1_pad_w = 0;
    vx_size pool_8x8_s1_pad_h = 0;
    vx_enum pool_8x8_s1_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node pool_8x8_s1_node;
    pool_8x8_s1_node = vxPoolingLayer(graph, inception_c3_concat, pool_8x8_s1_type, pool_8x8_s1_kernel_w, pool_8x8_s1_kernel_h, pool_8x8_s1_pad_w, pool_8x8_s1_pad_h, pool_8x8_s1_roundPolicy, pool_8x8_s1 );
    ERROR_CHECK_OBJECT(pool_8x8_s1_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool_8x8_s1_node));

    // pool_8x8_s1_drop Layer
    vx_size pool_8x8_s1_drop_dims[4] = { 1, 1, 1536, 1 };
    vx_tensor pool_8x8_s1_drop;
    pool_8x8_s1_drop = vxCreateVirtualTensor(graph,4, pool_8x8_s1_drop_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(pool_8x8_s1_drop);
    vx_node pool_8x8_s1_drop_node;
    pool_8x8_s1_drop_node = vxCopyNode( graph, (vx_reference)pool_8x8_s1, (vx_reference)pool_8x8_s1_drop);
    ERROR_CHECK_OBJECT(pool_8x8_s1_drop_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&pool_8x8_s1_drop_node));

    // classifier Layer
    vx_size classifier_dims[4] = { 1, 1, 1000, 1 };
    vx_tensor classifier;
    classifier = vxCreateVirtualTensor(graph,4, classifier_dims, VX_TYPE_FLOAT32,0);
    ERROR_CHECK_OBJECT(classifier);
    vx_size classifier_W_dims[4] = { 1, 1, 1536, 1000 };
    vx_tensor classifier_W;
    classifier_W= vxCreateTensor(context,4,classifier_W_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(classifier_W); 
    ERROR_CHECK_STATUS(copyTensor(classifier_W, dataFolder + "/weights/classifier.f32"));
    vx_size classifier_B_dims[1] = { 1000 };
    vx_tensor classifier_B;
    classifier_B= vxCreateTensor(context,1,classifier_B_dims, VX_TYPE_FLOAT32, 0);
    ERROR_CHECK_OBJECT(classifier_B); 
    ERROR_CHECK_STATUS(copyTensor(classifier_B, dataFolder + "/bias/classifier.f32"));
    vx_enum classifier_convertPolicy = VX_CONVERT_POLICY_SATURATE;
    vx_enum classifier_roundPolicy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    vx_node classifier_node;
    classifier_node = vxFullyConnectedLayer( graph, pool_8x8_s1_drop, classifier_W, classifier_B, classifier_convertPolicy, classifier_roundPolicy, classifier);
    ERROR_CHECK_OBJECT(classifier_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&classifier_node));

    // prob Layer
    vx_node prob_node;
    prob_node = vxSoftmaxLayer(graph, classifier, prob);
    ERROR_CHECK_OBJECT(prob_node);
    ERROR_CHECK_STATUS(vxReleaseNode(&prob_node));

    ////
    // release intermediate objects
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv1_3x3_s2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv2_3x3_s1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&conv3_3x3_s1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_3x3_s2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_3x3_s2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_stem3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_3x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a1_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_3x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a2_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_3x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a3_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_3x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_a4_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_3x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_a_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b1_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b2_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b3_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b4_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b5_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b6_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_7x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x7_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_b7_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_reduce_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_1x7_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_7x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_3x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_pool));
    ERROR_CHECK_STATUS(vxReleaseTensor(&reduction_b_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_4_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_3x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c1_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_4_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_3x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c2_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_4_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_2_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x3_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_3x1_3_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_pool_ave));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_scale));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_bn_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_bn_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_scale_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_scale_B));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_1x1_relu));
    ERROR_CHECK_STATUS(vxReleaseTensor(&inception_c3_concat));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool_8x8_s1));
    ERROR_CHECK_STATUS(vxReleaseTensor(&pool_8x8_s1_drop));
    ERROR_CHECK_STATUS(vxReleaseTensor(&classifier));
    ERROR_CHECK_STATUS(vxReleaseTensor(&classifier_W));
    ERROR_CHECK_STATUS(vxReleaseTensor(&classifier_B));

    ////
    // verify the built graph
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    return graph;
}

vx_status copyTensorBuffer(vx_tensor tensor, void *tensor_buf, vx_size buffer_len, vx_enum usage = VX_WRITE_ONLY)
{
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    vx_size itemsize = sizeof(float);
    if(data_type == VX_TYPE_UINT8 || data_type == VX_TYPE_INT8) {
        itemsize = sizeof(vx_uint8);
    }
    else if(data_type == VX_TYPE_UINT16 || data_type == VX_TYPE_INT16 || data_type == VX_TYPE_FLOAT16) {
        itemsize = sizeof(vx_uint16);
    }
    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_map_id map_id;
    float * ptr;
    vx_status status = vxMapTensorPatch(tensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for " << tensor << std::endl;
        return -1;
    }
    if (usage == VX_WRITE_ONLY)
    {
        float *img = (float *)tensor_buf;
        if (buffer_len < itemsize*count)
        {
            std::cerr << "ERROR: buffer_size: " << buffer_len << (itemsize*count) << std::endl;
            return -1;
        }
        vx_size length = dims[0] * dims[1];     
        float * B_buf = ptr;
        float * G_buf = B_buf + length;
        float * R_buf = G_buf + length;
        for (int i=0; i < length; i++, img += 3) {
            *B_buf++ = img[0];
            *G_buf++ = img[1];
            *R_buf++ = img[2];
        }
        //FILE *fp = fopen("input_dump.f32", "wb");
        //size_t ret = fwrite(ptr, sizeof(float), count, fp);
        //fclose(fp);
        //memcpy(ptr, tensor_buf, itemsize*count);
    }else
    { 
        //FILE *fp = fopen("output_dump.f32", "wb");
        //fwrite(ptr, sizeof(float), count, fp);
        //fclose(fp);    
        if (buffer_len < itemsize*count)
        {
            printf("bufferlen: %d req: %d\n", (int)buffer_len, (int)(itemsize*count));
            std::cerr << "ERROR: buffer_size: " << buffer_len << (itemsize*count) << std::endl;
            return -1;
        }
        memcpy(tensor_buf, ptr, itemsize*count);
    }
    status = vxUnmapTensorPatch(tensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << tensor << std::endl;
        return -1;
    }
    return 0;
}

VX_API_ENTRY INFHANDLE VX_API_CALL annCreateContext(const char * dataFolder_)
{
    inf_context inf_ctx = new inference_context_t;
    if (inf_ctx) {
        memset(inf_ctx, 0, sizeof(inference_context_t));
        annGetTensorDimensions(inf_ctx->dimInput, inf_ctx->dimOutput);
        // create context, input, output, and graph
        vxRegisterLogCallback(NULL, log_callback, vx_false_e);
        inf_ctx->context = vxCreateContext();
        if(vxGetStatus((vx_reference)inf_ctx->context)) {
            //printf("ERROR: vxCreateContext() failed\n");
            return nullptr;
        }
        vx_context context = inf_ctx->context;
        inf_ctx->input = vxCreateTensor(inf_ctx->context, 4, inf_ctx->dimInput, VX_TYPE_FLOAT32, 0);
        ERROR_CHECK_OBJECT(inf_ctx->input);
        inf_ctx->output = vxCreateTensor(inf_ctx->context, 4, inf_ctx->dimOutput, VX_TYPE_FLOAT32, 0);
        ERROR_CHECK_OBJECT(inf_ctx->output);

        inf_ctx->graph = annCreateGraph(inf_ctx->context, inf_ctx->input, inf_ctx->output, dataFolder_);
    }
    return (INFHANDLE)inf_ctx;

}

VX_API_ENTRY int VX_API_CALL annRunInference(INFHANDLE ctx, float *in_tensor_buff, vx_size in_buffer_size, float *out_tensor_buff, vx_size out_buffer_size)
{
    inf_context ictx = (inf_context)ctx;
    if(ictx) {
        if(copyTensorBuffer(ictx->input, in_tensor_buff, in_buffer_size, VX_WRITE_ONLY) < 0) {
            return -1;
        }
        vx_status status = vxProcessGraph(ictx->graph);
        if(status != VX_SUCCESS) {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        // copy output tensor buffer
        if(copyTensorBuffer(ictx->output, out_tensor_buff, out_buffer_size, VX_READ_ONLY) < 0) {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        return (int)VX_SUCCESS;
    }
    else
        return (int)VX_ERROR_INVALID_PARAMETERS;
}

VX_API_ENTRY int VX_API_CALL annReleaseContext(INFHANDLE ctx)
{
    inf_context ictx = (inf_context)ctx;
    vx_context context = ictx->context;
    // release resources
    ERROR_CHECK_STATUS1(vxReleaseGraph(&ictx->graph));
    ERROR_CHECK_STATUS1(vxReleaseTensor(&ictx->input));
    ERROR_CHECK_STATUS1(vxReleaseTensor(&ictx->output));
    ERROR_CHECK_STATUS1(vxReleaseContext(&ictx->context));
    return 0;
}