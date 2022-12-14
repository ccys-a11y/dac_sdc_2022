#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "sliding_window_unit.h"
#include "stream_tools.hpp"

/**
 * max pool 妯″潡
 * TODO 杩樹笉鑳戒繚璇佹瘡涓椂閽熷懆鏈熼兘鏈夋暟鎹緭鍑�
 *
 */
// template <unsigned K, unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH,
//           unsigned BIT>
// void max_pool2d(stream<ap_uint<IN_CH * BIT>> &in,
//                 stream<ap_uint<IN_CH * BIT>> &out, unsigned int reps = 1) {
//   // need buffer space for a single maxpooled row of the image
//   ap_uint<BIT> buf[IN_COL / K][IN_CH];
// #pragma HLS ARRAY_PARTITION variable = buf complete dim = 2
//   for (unsigned int i = 0; i < IN_COL / K; i++) {
//     for (unsigned int ch = 0; ch < IN_CH; ch++) {
//       buf[i][ch] = 0;
//     }
//   }
//   for (unsigned int rep = 0; rep < reps; rep++) {
//     for (unsigned int yp = 0; yp < IN_ROW / K; yp++) {
//       for (unsigned int ky = 0; ky < K; ky++) {
//         for (unsigned int xp = 0; xp < IN_COL / K; xp++) {
//           // Change to comparator
//           for (unsigned int kx = 0; kx < K; kx++) {
//             // 姣忎釜 ch
// #pragma HLS PIPELINE II = 1
//             ap_uint<IN_CH *BIT> input_data = in.read();
//             for (unsigned int ch = 0; ch < IN_CH; ch++) {
// #pragma HLS UNROLL
//               unsigned int lowBit = ch * BIT;
//               unsigned int highBit = (ch + 1) * BIT - 1;
//               ap_uint<BIT> channeldata = input_data(highBit, lowBit);
//               ap_uint<BIT> oldMax = buf[xp][ch];
//               if (channeldata > oldMax) {
//                 buf[xp][ch] = channeldata;
//               }
//             }
//           }
//         }
//         for (unsigned int outpix = 0; outpix < IN_COL / K; outpix++) {
//           // ch
//           ap_uint<IN_CH * BIT> output_data;
//           for (unsigned int ch = 0; ch < IN_CH; ch++) {
// #pragma HLS UNROLL
//             unsigned int lowBit = ch * BIT;
//             unsigned int highBit = (ch + 1) * BIT - 1;
//             output_data(highBit, lowBit) = buf[outpix][ch];
//             // get buffer ready for next use
//             buf[outpix][ch] = 0;
//           }
//           out.write(output_data);
//         }
//       }
//     }
//   }
// }

/**
 * pool灞傝绠楀鐞嗗嚱鏁�
 */
template <unsigned K, unsigned IN_CH, unsigned IN_BIT, unsigned VEC_NUMS>
void pool_cal(stream<ap_uint<IN_CH * IN_BIT> > &vec,
              stream<ap_uint<IN_CH * IN_BIT> > &out, const unsigned reps = 1) {
  ap_uint<IN_CH *IN_BIT> result = 0;
  unsigned k_cnt = 0;

  for (unsigned rep = 0; rep < reps * VEC_NUMS; rep++) {
#pragma HLS PIPELINE II = 1

    // 杩欓噷鐨則emp_vec搴旇鏄瘎瀛樺櫒锛坮eg锛夌被鍨�
    ap_uint<IN_CH *IN_BIT> temp_vec = vec.read();

    for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL
      // if(temp_vec((c+1)*IN_BIT-1, c*IN_BIT) > result( (c+1)*IN_BIT-1,
      // c*IN_BIT)) {
      //     result( (c+1)*IN_BIT-1, c*IN_BIT) = temp_vec((c+1)*IN_BIT-1,
      //     c*IN_BIT);
      // }

      ap_uint<IN_BIT> temp = temp_vec((c + 1) * IN_BIT - 1, c * IN_BIT);

      result((c + 1) * IN_BIT - 1, c * IN_BIT) =
          (temp > result((c + 1) * IN_BIT - 1, c * IN_BIT))
              ? temp
              : result((c + 1) * IN_BIT - 1, c * IN_BIT);
    }

    if (++k_cnt == K * K) {
      out.write(result);
      result = 0;
      k_cnt = 0;
    }
  }
}

/**
 * 姹犲寲灞�
 * TODO 褰撳墠鍙粰 K = 2, S = 2鍋氫紭鍖�
 */
template <unsigned K, // kernel
                      // unsigned S,                 // stride
          unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT>
void max_pool2d(stream<ap_uint<IN_CH * IN_BIT> > &in,
                stream<ap_uint<IN_CH * IN_BIT> > &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW
  // TODO IN_ROW % S != 0
  // 鏆傛椂鍙鐞嗙壒娈婃儏鍐�
  const unsigned OUT_ROW = IN_ROW / 2;
  const unsigned OUT_COL = IN_COL / 2;
  const unsigned S = 2;

  // 浜х敓婊戝姩绐楀彛鏁版嵁
  hls::stream<ap_uint<IN_CH * IN_BIT> > swu_out("swu_out");
  SWU<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT>(in, swu_out, reps);

  // 澶勭悊鏁版嵁
  // POOL<IN_ROW*IN_COL, Ibit, K, Cin, 1>(swu_out, out, reps);
  pool_cal<K, IN_CH, IN_BIT, OUT_ROW * OUT_COL * K * K>(swu_out, out, reps);
}
