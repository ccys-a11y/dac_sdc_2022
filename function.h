#pragma once
#include <ap_int.h>
#include <hls_stream.h>
// using namespace hls;
// #include <iostream>
using namespace std;
#include "stream_tools.hpp"
#include <assert.h>

/**
 *  padding 鍑芥暟
 *  鏁版嵁瀹藉害涓� IN_BIT * SIMD
 *
 */
template <unsigned IN_BIT, unsigned SIMD, unsigned P>
void padding_var(
    // 灏嗘瘡涓�鏁扮珫鐪嬫垚涓�涓厓绱�
    stream<ap_uint<IN_BIT * SIMD> > &in, stream<ap_uint<IN_BIT * SIMD> > &out,
    const unsigned in_row,         //
    const unsigned in_col,         //
    const unsigned in_simd_pre_ch, // ch / simd
    const unsigned reps = 1) {
  // const unsigned OUT_ROW = in_row + 2 * P;
  const unsigned OUT_COL = in_col + 2 * P;
  // const unsigned DATA_NUM_PRE_CH = in_ch / SIMD;

  for (unsigned rep = 0; rep < reps; rep++) {

    for (unsigned h = 0; h < P; h++) {
      for (unsigned s = 0; s < OUT_COL; s++) {
        // 灏嗕竴 ch 鐨勬暟鎹疆闆�
        append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
      }
    }

    for (unsigned h = 0; h < in_row; h++) {

      for (unsigned s = 0; s < OUT_COL; s++) {
        // #pragma HLS PIPELINE II=1

        if ((s < P) || (s >= OUT_COL - P)) {
          // temp_out = 0;
          append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
        } else {
          // cout << "in size :" << in.size() << endl;
          stream_move<IN_BIT * SIMD>(in, out, in_simd_pre_ch);
        }
        // out.write(temp_out);
      }
    }

    for (unsigned h = 0; h < P; h++) {
      for (unsigned i = 0; i < OUT_COL; i++) {
        append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
      }
    }
  }
}

/**
 *  padding 鍑芥暟
 */
template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned P>
void padding(
    // 灏嗘瘡涓�鏁扮珫鐪嬫垚涓�涓厓绱�
    stream<ap_uint<IN_CH * IN_BIT> > &in, stream<ap_uint<IN_CH * IN_BIT> > &out,
    const unsigned reps = 1) {
  const unsigned OUT_ROW = IN_ROW + 2 * P;
  const unsigned OUT_COL = IN_COL + 2 * P;

  ap_uint<IN_CH *IN_BIT> temp_out = 0;

  for (unsigned rep = 0; rep < reps; rep++) {

    for (unsigned h = 0; h < P; h++) {
      for (unsigned s = 0; s < OUT_COL; s++) {
        out.write(0);
      }
    }

    for (unsigned h = 0; h < IN_ROW; h++) {

      for (unsigned s = 0; s < OUT_COL; s++) {
#pragma HLS PIPELINE II = 1

        if ((s < P) || (s >= OUT_COL - P)) {
          temp_out = 0;
        } else {
          temp_out = in.read();
        }

        out.write(temp_out);
      }
    }

    for (unsigned h = 0; h < P; h++) {
      for (unsigned i = 0; i < OUT_COL; i++) {
        out.write(0);
      }
    }
  }
}

/**
 * 瀹炵幇閲忓寲婵�娲荤畻娉�
 * 浣跨敤浜屽垎鏌ユ壘
 * TODO
 * 涓㈠け绮惧害 鏆傛椂寮冪敤
 */
// int d = 0;
// template <	unsigned IN_BIT,
// 			unsigned OUT_BIT,
// 			unsigned INC_BIT,
// 			unsigned BIAS_BIT>
// ap_uint<OUT_BIT> bn_qurelu( ap_int<IN_BIT> in,
//                 ap_int<INC_BIT> inc,
//                 ap_int<BIAS_BIT> bias ) {

// 	if (d < 16) {
// 		cout << d << " in " << in << " inc " << inc << " bias " << bias
// << endl; 		d ++;
// 	}
//     ap_int<IN_BIT> target = in + bias;
//     ap_uint<OUT_BIT> index = 1 << (OUT_BIT - 1);

// 	// 璁＄畻鎵�浣跨敤鐨勬暟鎹搴� INC_BIT+OUT_BIT
// 	ap_int<INC_BIT+OUT_BIT> inc_exp = inc;
// 	// 鐩存帴瀵筰nc绉讳綅浼氭孩鍑� 鎵�浠ュ垵濮嬪寲鏃堕渶瑕� 浣嶅鎵╁睍
//     ap_int<INC_BIT+OUT_BIT + 1> mid = inc_exp << (OUT_BIT - 1);

//     for(int i=OUT_BIT-2; i >= 0; i --) {
// #pragma HLS UNROLL
//         // TODO
//         // 鍥犱负涓嶈兘鐗瑰埆纭畾 IN_BIT 鍜� inc_BIT 鍏崇郴 鎵�浠ヨ繖閲屽彲鑳芥湁绮惧害鎹熷け
//         ap_int<INC_BIT+OUT_BIT> inc_shift = inc_exp << i;
//         ap_uint<INC_BIT+OUT_BIT> one_shift = 1 << i;
//         if(target < mid) {
//             mid -= inc_shift;
//             index -= one_shift;
//         } else if(mid < target){
//             mid += inc_shift;
//             index += one_shift;
//         }
//     }
//     if(target < mid) {
//         index --;
//     }
//     return index;
// }

// int d = 0;
template <unsigned IN_BIT, unsigned OUT_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT,

          unsigned DATA_BIT, unsigned W_BIT, unsigned L_SHIFT>
ap_uint<OUT_BIT> bn_qurelu(ap_int<IN_BIT> in, ap_int<INC_BIT> inc,
                           ap_int<BIAS_BIT> bias) {

  const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

  ap_int<IN_BIT> bn_res = in * inc + bias;
  ap_uint<OUT_BIT> res;

  if (bn_res > 0) {
    bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
    if (bn_res > 15) {
      res = 15;
    } else {
      res = bn_res;
    }
  } else {
    res = 0;
  }
  return res;
}

template <unsigned IN_BIT, unsigned OUT_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT,

          unsigned DATA_BIT, unsigned W_BIT, unsigned L_SHIFT>
ap_uint<OUT_BIT> bn_qurelu_fixed(ap_int<IN_BIT> in, ap_int<INC_BIT> inc,
                                 ap_int<BIAS_BIT> bias) {

  const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

  ap_int<IN_BIT + INC_BIT + 1> bn_res = in * inc + bias;
  ap_uint<OUT_BIT> res;

  if (bn_res > 0) {
    bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
    if (bn_res > 15) {
      res = 15;
    } else {
      res = bn_res;
    }
  } else {
    res = 0;
  }
  return res;
}

/**
 * 鎵规鍒欏寲 鍜� 閲忓寲婵�娲诲嚱鏁�
 */
// template <	unsigned IN_BIT,
// 			unsigned OUT_BIT,
// 			unsigned INC_BIT,
// 			unsigned BIAS_BIT,
//             unsigned SHIFT>
// void bn_qurelu( ap_int<IN_BIT> in,
//                 ap_uint<INC_BIT> inc,
//                 ap_int<BIAS_BIT> bias )
// {
//     target = target + bias;
//     int index = 1 << (BIT - 1);
//     int mid = inc << (BIT - 1);
//     for(int i=BIT-2; i >= 0; i --) {
//         int inc_shift = inc << i;
//         int one_shift = 1 << i;
//         if(target < mid) {
//             mid -= inc_shift;
//             index -= one_shift;
//         } else if(mid < target){
//             mid += inc_shift;
//             index += one_shift;
//         }
//     }
//     if(target < mid) {
//         index --;
//     }
//     return index;
// }

// template<int IN_CH, int IN_ROW, int IN_COL, int IN_BIT, int OUT_BIT>
// void conv_bn_qrelu(int in[IN_CH][IN_ROW][IN_COL], int
// out[IN_CH][IN_ROW][IN_COL], int w[IN_CH], int b[IN_CH]) {

//     for(int ch=0; ch < IN_CH; ch ++) {
//         for(int row=0; row < IN_ROW; row ++) {
//             for(int col=0; col < IN_COL; col ++) {
//                 out[ch][row][col] = qrelu_search<OUT_BIT>(in[ch][row][col],
//                 w[ch], b[ch]);
//             }
//         }
//     }
// }

// template<int LEN, int IN_BIT, int OUT_BIT>
// void linear_bn_qrelu(int in[LEN], int out[LEN], int w[LEN], int b[LEN]) {
//     for(int i=0; i < LEN; i ++) {

//         out[i] = qrelu_search<OUT_BIT>(in[i], w[i], b[i]);
//     }
// }
