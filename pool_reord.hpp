
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "sliding_window_unit.h"
#include "stream_tools.hpp"

template <unsigned IN_BIT, unsigned PE>
ap_uint<IN_BIT * PE> max2_PE(ap_uint<IN_BIT * PE> data0,
                             ap_uint<IN_BIT * PE> data1) {
  ap_uint<IN_BIT * PE> ret;

  for (int i = 0; i < PE; i++) {
    ap_uint<IN_BIT> d0 = data0(IN_BIT * (i + 1) - 1, IN_BIT * i);
    ap_uint<IN_BIT> d1 = data1(IN_BIT * (i + 1) - 1, IN_BIT * i);
    ap_uint<IN_BIT> dret = d1 > d0 ? d1 : d0;
    ret(IN_BIT * (i + 1) - 1, IN_BIT * i) = dret;
  }
  return ret;
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned PE>
void max_pool2x2(stream<ap_uint<PE * IN_BIT * 2> > &vec,
                 stream<ap_uint<PE * IN_BIT * 2> > &out,
                 const unsigned reps = 1) {

  ap_uint<PE * IN_BIT> row_store[IN_W / 2 * IN_CH / PE];

  bool load_flag;
  ap_uint<IN_BIT * PE> dataOut0;
  ap_uint<IN_BIT * PE> dataOut1;
  for (unsigned h = 0; h < IN_H * reps; h++)
    for (unsigned peIdx = 0; peIdx < IN_CH / PE; peIdx++)
      for (unsigned w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline II = 1
        ap_uint<IN_BIT * PE> data0;
        ap_uint<IN_BIT * PE> data1;
        (data1, data0) = vec.read();
        ap_uint<IN_BIT *PE> dataMax2 = max2_PE<IN_BIT, PE>(data0, data1);
        int addr = w * (IN_CH / PE) + peIdx;
        if (h % 2) {
          ap_uint<IN_BIT *PE> dataRes = row_store[addr];
          dataOut0 = max2_PE<IN_BIT, PE>(dataMax2, dataRes);

        } else {
          row_store[addr] = dataMax2;
        }
        if (w % 2 && h % 2) {
          out.write((dataOut0, dataOut1));

        } else {
          dataOut1 = dataOut0;
        }
      }
}



template <unsigned IN_BIT, unsigned PE, unsigned INC_BIT, unsigned IN_CH>
ap_uint<IN_BIT * PE> max2min_PE(ap_uint<IN_BIT * PE> data0,
                             ap_uint<IN_BIT * PE> data1, const ap_int<INC_BIT> inc[PE][IN_CH/PE], unsigned peIdx) {
  ap_uint<IN_BIT * PE> ret;

  for (int i = 0; i < PE; i++) {
    ap_uint<IN_BIT> d0 = data0(IN_BIT * (i + 1) - 1, IN_BIT * i);
    ap_uint<IN_BIT> d1 = data1(IN_BIT * (i + 1) - 1, IN_BIT * i);
    ap_uint<IN_BIT> dret = (inc[i][peIdx]>0)?(d1 > d0 ? d1 : d0):(d1 < d0 ? d1 : d0);
    ret(IN_BIT * (i + 1) - 1, IN_BIT * i) = dret;
  }
  return ret;
}

/// two stage
template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned M_BIT, unsigned OUT_BIT,
          unsigned PE, unsigned INC_BIT, unsigned BIAS_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned IN_BIT>
void max_pool2x2_stage1(//stream<ap_uint<PE * IN_BIT * 2> > &vec,
		         stream<ap_uint<PE * M_BIT * 2> > &vec,
                 stream<ap_uint<PE * M_BIT * 2> > &out,
				 const ap_int<INC_BIT> inc[PE][IN_CH / PE],
				 const ap_int<BIAS_BIT> bias[PE][IN_CH / PE],
                 const unsigned reps = 1) {

//  ap_uint<PE * IN_BIT> row_store[IN_W / 2 * IN_CH / PE];
  ap_uint<PE * M_BIT> row_store[IN_W / 2 * IN_CH / PE];        ///// 最终是存在了BRAM中   160*416  相比原版640*16
  bool load_flag;
  ap_uint<M_BIT * PE> dataOut0;
  ap_uint<M_BIT * PE> dataOut1;
  ap_uint<OUT_BIT * PE> databn0, databn1;
  ap_uint<M_BIT> dataOut0_vec[PE];
  ap_uint<M_BIT> dataOut1_vec[PE];
#pragma HLS array_partition variable = dataOut0_vec dim = 1 complete
#pragma HLS array_partition variable = dataOut1_vec dim = 1 complete
  for (unsigned h = 0; h < IN_H * reps; h++)
    for (unsigned peIdx = 0; peIdx < IN_CH / PE; peIdx++)
      for (unsigned w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline II = 1
        ap_uint<M_BIT * PE> data0;
        ap_uint<M_BIT * PE> data1;
        (data1, data0) = vec.read();
        ap_uint<M_BIT * PE> dataMax2 = max2min_PE<M_BIT, PE, INC_BIT, IN_CH>(data0, data1, inc, peIdx);
        int addr = w * (IN_CH / PE) + peIdx;
        if (h % 2) {
          ap_uint<OUT_BIT *PE> dataRes = row_store[addr];
          dataOut0 = max2min_PE<OUT_BIT, PE, INC_BIT, IN_CH>(dataMax2, dataRes, inc, peIdx);

        } else {
          row_store[addr] = dataMax2;
        }
        if (w % 2 && h % 2) {
          out.write((dataOut1, dataOut0));
        } else {
          dataOut1 = dataOut0;
        }
      }
}
//////////////pool0
template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned M_BIT, unsigned OUT_BIT,
          unsigned PE, unsigned INC_BIT, unsigned BIAS_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned IN_BIT>
void max_pool2x2_stage20(//stream<ap_uint<PE * IN_BIT * 2> > &vec,
		         stream<ap_uint<PE * M_BIT * 2> > &vec,
                 stream<ap_uint<PE * OUT_BIT * 2> > &out,
				 const ap_int<INC_BIT> inc[PE][IN_CH / PE],
				 const ap_int<BIAS_BIT> bias[PE][IN_CH / PE],
                 const unsigned reps = 1) {

//  ap_uint<PE * IN_BIT> row_store[IN_W / 2 * IN_CH / PE];
  ap_uint<PE * M_BIT> row_store[IN_W / 2 * IN_CH / PE];
  bool load_flag;
  ap_uint<M_BIT * PE> dataOut0;
  ap_uint<M_BIT * PE> dataOut1;
  ap_uint<OUT_BIT * PE> databn0, databn1;
  for (unsigned h = 0; h < (IN_H / 2* reps); h++)
    for (unsigned peIdx = 0; peIdx < IN_CH / PE; peIdx++)
      for (unsigned w = 0; w < IN_W / 4; w++) {
#pragma HLS pipeline II = 18   //16+2 pool0 2dsp
//#pragma HLS pipeline II = 10   //8+2 pool1  1dsp
//#pragma HLS pipeline II = 18   //16+2 pool2  1dsp
//#pragma HLS pipeline II = 10   //8+2 pool3  1dsp
//        ap_uint<M_BIT * PE> data0;
//        ap_uint<M_BIT * PE> data1;
        (dataOut1, dataOut0) = vec.read();

          for (unsigned q=0; q<PE; q++)
          {
        	  databn0((q + 1) * OUT_BIT - 1, q * OUT_BIT) =
        			  bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, L_SHIFT, IN_BIT,
			  	  	  	  W_BIT>(dataOut0((q + 1) * M_BIT - 1, q * M_BIT), inc[q][peIdx], bias[q][peIdx]);
           	  databn1((q + 1) * OUT_BIT - 1, q * OUT_BIT) =
            			  bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, L_SHIFT, IN_BIT,
    			  	  	  	  W_BIT>(dataOut1((q + 1) * M_BIT - 1, q * M_BIT), inc[q][peIdx], bias[q][peIdx]);
          }
          out.write((databn1, databn0));

        }
}


template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned M_BIT, unsigned OUT_BIT,
          unsigned PE, unsigned INC_BIT, unsigned BIAS_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned IN_BIT>
void max_pool_gen0(//stream<ap_uint<PE * IN_BIT * 2> > &vec,
		         stream<ap_uint<PE * M_BIT * 2> > &vec,
                 stream<ap_uint<PE * OUT_BIT * 2> > &out,
				 const ap_int<INC_BIT> inc[PE][IN_CH / PE],
				 const ap_int<BIAS_BIT> bias[PE][IN_CH / PE],
                 const unsigned reps = 1){
#pragma HLS DATAFLOW
	stream<ap_uint<PE * M_BIT * 2> > pool_out("pool_out");
	max_pool2x2_stage1<IN_H, IN_W, IN_CH, M_BIT, OUT_BIT, PE, INC_BIT, BIAS_BIT, W_BIT, L_SHIFT, IN_BIT>
			(vec, pool_out, inc, bias,  reps);
	max_pool2x2_stage20<IN_H, IN_W, IN_CH, M_BIT, OUT_BIT, PE, INC_BIT, BIAS_BIT, W_BIT, L_SHIFT, IN_BIT>
	(pool_out, out, inc, bias, reps);
}


///////////// no dataflow
template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned M_BIT, unsigned OUT_BIT,
          unsigned PE, unsigned INC_BIT, unsigned BIAS_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned IN_BIT>
void max_pool2x2_onestep(//stream<ap_uint<PE * IN_BIT * 2> > &vec,
		         stream<ap_uint<PE * M_BIT * 2> > &vec,
                 stream<ap_uint<PE * OUT_BIT * 2> > &out,
				 const ap_int<INC_BIT> inc[PE][IN_CH / PE],
				 const ap_int<BIAS_BIT> bias[PE][IN_CH / PE],
                 const unsigned reps = 1) {

//  ap_uint<PE * IN_BIT> row_store[IN_W / 2 * IN_CH / PE];
  ap_uint<PE * M_BIT> row_store[IN_W / 2 * IN_CH / PE];        ///// 最终是存在了BRAM中   160*416  相比原版640*16
  bool load_flag;
  ap_uint<M_BIT * PE> dataOut0;
  ap_uint<M_BIT * PE> dataOut1;
  ap_uint<OUT_BIT * PE> databn0, databn1;
  ap_uint<M_BIT> dataOut0_vec[PE];
  ap_uint<M_BIT> dataOut1_vec[PE];
#pragma HLS array_partition variable = dataOut0_vec dim = 1 complete
#pragma HLS array_partition variable = dataOut1_vec dim = 1 complete
  for (unsigned h = 0; h < IN_H * reps; h++)
    for (unsigned peIdx = 0; peIdx < IN_CH / PE; peIdx++)
      for (unsigned w = 0; w < IN_W / 2; w++) {
//#pragma HLS pipeline II = 1
#pragma HLS pipeline II = 8
        ap_uint<M_BIT * PE> data0;
        ap_uint<M_BIT * PE> data1;
        (data1, data0) = vec.read();
        ap_uint<M_BIT * PE> dataMax2 = max2min_PE<M_BIT, PE, INC_BIT, IN_CH>(data0, data1, inc, peIdx);
        int addr = w * (IN_CH / PE) + peIdx;
        if (h % 2) {
          ap_uint<OUT_BIT *PE> dataRes = row_store[addr];
          dataOut0 = max2min_PE<OUT_BIT, PE, INC_BIT, IN_CH>(dataMax2, dataRes, inc, peIdx);

        } else {
          row_store[addr] = dataMax2;
        }
        if (w % 2 && h % 2) {
            for (unsigned q=0; q<PE; q++)
            {
          	  databn0((q + 1) * OUT_BIT - 1, q * OUT_BIT) =
          			  bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, L_SHIFT, IN_BIT,
  			  	  	  	  W_BIT>(dataOut0((q + 1) * M_BIT - 1, q * M_BIT), inc[q][peIdx], bias[q][peIdx]);
             	  databn1((q + 1) * OUT_BIT - 1, q * OUT_BIT) =
              			  bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, L_SHIFT, IN_BIT,
      			  	  	  	  W_BIT>(dataOut1((q + 1) * M_BIT - 1, q * M_BIT), inc[q][peIdx], bias[q][peIdx]);
            }
            out.write((databn1, databn0));
//          out.write((dataOut1, dataOut0));
        } else {
          dataOut1 = dataOut0;
        }
      }
}
