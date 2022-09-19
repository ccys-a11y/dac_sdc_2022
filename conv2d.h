#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

// #include "debug.hpp"
#include "function.h"
#include "matrix_vector_unit.hpp"
#include "sliding_window_unit.h"
#include "stream_tools.hpp"
/**
 * 鍗风Н璁＄畻鍗曞厓 鍚屾椂璁＄畻bn_灞備笌婵�娲诲眰
 * 鍦ㄧ煩闃靛悜閲忚绠楀悗绔嬪嵆璁＄畻寰楀埌婵�娲昏緭鍑哄��
 * 鍙绠� 3x3 鐨勫嵎绉� K = 3, P = 1 S = 1
 * 杈撳叆鏁版嵁瀹藉害 涓� IN_STREAM_BIT
 * 杈撳嚭鏁版嵁瀹藉害涓� PE * OUT_BIT
 */
template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, // 閲忓寲婵�娲诲悗鐨勪綅瀹�

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned PE, unsigned L_SHIFT>
void conv3x3_bn_act(stream<ap_uint<IN_BIT * IN_CH> > &in,
                    const ap_uint<SIMD * W_BIT>
                        weights[PE][((IN_CH * 9) / SIMD) * (OUT_CH / PE)],
                    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                    stream<ap_uint<OUT_BIT * OUT_CH> > &out,
                    const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  // 鏆傛椂璁や负杈撳叆 杈撳嚭缁村害涓嶅彉
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
  // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj,
  // reps); pading
  stream<ap_uint<IN_CH * IN_BIT> > padding_out("samepad_out");
  // stream<ap_uint<IN_CH*IN_BIT> > padding_out_print("samepad_out_print");
  padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

  stream<ap_uint<IN_CH * IN_BIT> > swu_out("swu_out");
  SWU<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);
  // 浣嶅璋冩暣
  stream<ap_uint<SIMD * IN_BIT> > adj_out("adj_out");
  StreamingDataWidthConverter_Batch<IN_CH * IN_BIT, SIMD * IN_BIT,
                                    9 * OUT_ROW * OUT_COL>(swu_out, adj_out,
                                                           reps);

  // cout << "adj_out size " << adj_out.size() << endl;
  // 鐭╅樀鍚戦噺璁＄畻
  //   stream<ap_uint<PE * OUT_BIT> > mvau_out("mvau_out");
  stream<ap_uint<PE * OUT_BIT> > mvau_out("mvau_out");
  matrix_vector_act_unit<IN_CH * 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT,
                         INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT,
                         OUT_ROW * OUT_COL>(adj_out, weights, inc, bias,
                                            mvau_out, reps);
  //   print_mavu_stream_through<OUT_ROW, OUT_COL, OUT_CH, PE, M_BIT>(
  //       mvau_out, "conv_l0_gold.txt");
  // cout << "mvau_out size " << mvau_out.size() << endl;
  StreamingDataWidthConverter_Batch<PE * OUT_BIT, OUT_CH * OUT_BIT,
                                    OUT_ROW * OUT_COL * OUT_CH / PE>(mvau_out,
                                                                     out, reps);
}

/**
 * 鍗风Н璁＄畻鍗曞厓 鍚屾椂璁＄畻bn_灞備笌婵�娲诲眰
 * 鍦ㄧ煩闃靛悜閲忚绠楀悗绔嬪嵆璁＄畻寰楀埌婵�娲昏緭鍑哄��
 * 鍙绠� 3x3 鐨勫嵎绉� K = 3, P = 1 S = 1
 * 杈撳叆鏁版嵁瀹藉害 涓� IN_STREAM_BIT
 * 杈撳嚭鏁版嵁瀹藉害涓� PE * OUT_BIT
 * 浣跨敤 lut 璁＄畻涔樻硶
 */
template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, // 閲忓寲婵�娲诲悗鐨勪綅瀹�

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned PE, unsigned L_SHIFT>
void conv3x3_bn_act_lut(stream<ap_uint<IN_BIT * IN_CH> > &in,
                        const ap_uint<SIMD * W_BIT>
                            weights[PE][((IN_CH * 9) / SIMD) * (OUT_CH / PE)],
                        const ap_uint<INC_BIT> inc[PE][OUT_CH / PE],
                        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                        stream<ap_uint<OUT_BIT * OUT_CH> > &out,
                        const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  // 鏆傛椂璁や负杈撳叆 杈撳嚭缁村害涓嶅彉
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
  // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in,
  // in_adj, reps); pading
  stream<ap_uint<IN_CH * IN_BIT> > padding_out("samepad_out");
  padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

  // 婊戝姩绐楀彛
  stream<ap_uint<IN_CH * IN_BIT> > swu_out("swu_out");
  SWU<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);
  // 浣嶅璋冩暣
  stream<ap_uint<SIMD * IN_BIT> > adj_out("adj_out");
  StreamingDataWidthConverter_Batch<IN_CH * IN_BIT, SIMD * IN_BIT,
                                    9 * OUT_ROW * OUT_COL>(swu_out, adj_out,
                                                           reps);
  // 鐭╅樀鍚戦噺璁＄畻
  stream<ap_uint<PE * OUT_BIT> > mvau_out("mvau_out");
  matrix_vector_act_unit_lut<IN_CH * 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT,
                             M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT,
                             OUT_ROW * OUT_COL>(adj_out, weights, inc, bias,
                                                mvau_out, reps);

  StreamingDataWidthConverter_Batch<PE * OUT_BIT, OUT_CH * OUT_BIT,
                                    OUT_ROW * OUT_COL>(mvau_out, out, reps);
}

/**
 * 鍗风Н璁＄畻鍗曞厓 鍚屾椂璁＄畻bn_灞備笌婵�娲诲眰
 * 鍦ㄧ煩闃靛悜閲忚绠楀悗绔嬪嵆璁＄畻寰楀埌婵�娲昏緭鍑哄��
 * 鍙绠� 1x1 鐨勫嵎绉� K = 1, P = 1 S = 1
 */
template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, // 閲忓寲婵�娲诲悗鐨勪綅瀹�

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned PE, unsigned L_SHIFT>
void conv1x1_bn_act(
    stream<ap_uint<IN_BIT * IN_CH> > &in,
    const ap_uint<SIMD * W_BIT> weights[PE][((IN_CH) / SIMD) * (OUT_CH / PE)],
    const ap_uint<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * OUT_CH> > &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;
  stream<ap_uint<SIMD * IN_BIT> > adj_out("adj_out");
  StreamingDataWidthConverter_Batch<IN_CH * IN_BIT, SIMD * IN_BIT,
                                    OUT_ROW * OUT_COL>(in, adj_out, reps);
  // 鐭╅樀鍚戦噺璁＄畻
  stream<ap_uint<PE * OUT_BIT> > mvau_out("mvau_out");
  matrix_vector_act_unit<IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT,
                         BIAS_BIT, SIMD, PE, L_SHIFT, OUT_ROW * OUT_COL>(
      adj_out, weights, inc, bias, mvau_out, reps);

  StreamingDataWidthConverter_Batch<PE * OUT_BIT, OUT_CH * OUT_BIT,
                                    OUT_ROW * OUT_COL * OUT_CH / PE>(mvau_out,
                                                                     out, reps);
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned OUT_CH,

          unsigned W_BIT, unsigned M_BIT,

          unsigned SIMD, unsigned PE>
void conv1x1(stream<ap_uint<IN_BIT * SIMD> > &in,
             const ap_uint<SIMD * W_BIT>
                 weights[PE][((IN_CH * 1) / SIMD) * (OUT_CH / PE)],
             stream<ap_uint<PE * M_BIT> > &out, const unsigned reps = 1) {
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  matrix_vector_unit<IN_CH, OUT_CH, IN_BIT, W_BIT, M_BIT, SIMD, PE,
                     OUT_ROW * OUT_COL>(in, weights, out, reps);
}
