#ifndef _TOP_H_
#define _TOP_H_
#include "hls_video.h"
#include "ap_int.h"

#define SRC_WIDTH  640
#define SRC_HEIGHT 360

#define DST_WIDTH  320
#define DST_HEIGHT 160
//#define INPUT_IMAGE           "lena_gray.jpg"

typedef hls::stream<ap_axiu<24,1,1,1> >               AXI_STREAM;

typedef hls::Mat<SRC_HEIGHT, SRC_WIDTH, HLS_8UC3>     SRC_IMAGE;
typedef hls::Mat<DST_HEIGHT, DST_WIDTH, HLS_8UC3>     DST_IMAGE;

//void my_hls_resize(AXI_STREAM& src_axi, AXI_STREAM& dst_axi, int src_rows, int src_cols,int dst_rows,int dst_cols);

#endif


#define HLS_INTER_LINEAR 1
void my_hls_resize(AXI_STREAM& src_axi, AXI_STREAM& dst_axi, int src_rows, int src_cols,int dst_rows,int dst_cols)
{
#pragma HLS INTERFACE axis port=src_axi
#pragma HLS INTERFACE axis port=dst_axi

#pragma HLS RESOURCE core=AXI_SLAVE variable=src_rows metadata="-bus_bundle CONTROL_BUS"
#pragma HLS RESOURCE core=AXI_SLAVE variable=src_cols metadata="-bus_bundle CONTROL_BUS"

#pragma HLS RESOURCE core=AXI_SLAVE variable=dst_rows metadata="-bus_bundle CONTROL_BUS"
#pragma HLS RESOURCE core=AXI_SLAVE variable=dst_cols metadata="-bus_bundle CONTROL_BUS"

#pragma HLS RESOURCE core=AXI_SLAVE variable=return metadata="-bus_bundle CONTROL_BUS"

#pragma HLS INTERFACE ap_stable port=src_rows
#pragma HLS INTERFACE ap_stable port=src_cols
#pragma HLS INTERFACE ap_stable port=dst_rows
#pragma HLS INTERFACE ap_stable port=dst_cols

        int interpolation;
            SRC_IMAGE       imag_0(src_rows,src_cols);
            DST_IMAGE       imag_1(dst_rows, dst_cols);
            #pragma HLS dataflow
            hls::AXIvideo2Mat(src_axi, imag_0);
            hls::Resize(imag_0,imag_1,interpolation=HLS_INTER_LINEAR );
            hls::Mat2AXIvideo(imag_1, dst_axi);
}
