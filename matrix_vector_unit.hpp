#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "function.h"

/**
 *  simd 涔�
 * 浣跨敤 閫昏緫鏌ユ壘琛�
 */
template <unsigned W_BIT, unsigned IN_BIT, unsigned M_BIT, unsigned SIMD>
ap_int<M_BIT> simd_mul_lut(ap_uint<SIMD * W_BIT> weights,
                           ap_uint<SIMD * IN_BIT> in) {
  ap_int<M_BIT> accumulation = 0;

  for (unsigned p = 0; p < SIMD; p++) {
#pragma HLS UNROLL
    ap_int<W_BIT> temp_w = weights((p + 1) * W_BIT - 1, p * W_BIT);
    ap_uint<IN_BIT> temp_in = in((p + 1) * IN_BIT - 1, p * IN_BIT);
    ap_int<W_BIT + IN_BIT> result = temp_w * temp_in;
#pragma HLS RESOURCE variable = result core = Mul_LUT
    accumulation += result;
  }
  return accumulation;
}

/**
 *  simd 涔�
 *  鐢� 缂栬瘧鍣ㄨ嚜鍔ㄩ�夋嫨浣跨敤 dsp 鎴栬�� lut
 */
template <unsigned W_BIT, unsigned IN_BIT, unsigned M_BIT, unsigned SIMD>
ap_int<M_BIT> simd_mul(ap_uint<SIMD * W_BIT> weights,
                       ap_uint<SIMD * IN_BIT> in) {
  ap_int<M_BIT> accumulation = 0;

  for (unsigned p = 0; p < SIMD; p++) {
#pragma HLS UNROLL
    ap_int<W_BIT> temp_w = weights((p + 1) * W_BIT - 1, p * W_BIT);
    ap_uint<IN_BIT> temp_in = in((p + 1) * IN_BIT - 1, p * IN_BIT);
    ap_int<W_BIT + IN_BIT> result = temp_w * temp_in;
    // #pragma HLS RESOURCE variable=result core=Mul_LUT
    accumulation += result;
  }
  return accumulation;
}

/**
 * 鐭╅樀鍚戦噺璁＄畻鍗曞厓
 *
 */
template <unsigned MAT_ROW, // 灞曞紑鍚庣殑k 脳 k 脳 in_ch
          unsigned MAT_COL, // 灞曞紑鍚庣殑out_ch
          unsigned IN_BIT, unsigned W_BIT,
          unsigned M_BIT, // 涔樼疮鍔犲悗鐨勮绠楃粨鏋滅殑鍊�
          unsigned SIMD, unsigned PE, unsigned VECT_NUMS>
void matrix_vector_unit(
    stream<ap_uint<SIMD * IN_BIT> > &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
    stream<ap_uint<PE * M_BIT> > &out, const unsigned reps = 1) {
//  static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
//  static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

  const unsigned INPUT_FOLD = MAT_ROW / SIMD;
  const unsigned OUTPUT_FOLD = MAT_COL / PE;

  const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
  // const unsigned total_reps = 18;
  // 闇�瑕佷繚瀛樹竴琛屾暟鎹�
  ap_uint<SIMD * IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable = row_store core = RAM_2P_BRAM

  // 鐢ㄦ潵淇濆瓨绱姞缁撴灉
  // 	ap_uint<M_BIT> result_vec[PE];
  // #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
  unsigned in_fold_cnt = 0;  // 杈撳叆鎶樺彔璁℃暟
  unsigned out_fold_cnt = 0; // 杈撳嚭鎶樺彔璁℃暟
  unsigned tile = 0;

  // 涓�娆� 璇诲叆鐨勬暟鎹� 闇�瑕佷繚瀛� in_ch * k * k闀垮害鐨勬暟鎹�
  ap_uint<SIMD * IN_BIT> temp_vec;
  // 绱姞缁撴灉 杩欓噷闇�瑕佸垵濮嬪寲涓�0

  // TODO
  ap_int<M_BIT> acc[PE];

  // cout << "acc init value \n";
  // for(unsigned i=0; i < PE; i ++) {
  // 	cout << acc[i] << "  ";
  // }
  // static ap_uint<M_BIT> acc1[PE] = {0};

  // cout << "acc1 init value \n";
  // for(unsigned i=0; i < PE; i ++) {
  // 	cout << acc1[i] << "  ";
  // }

  // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
  for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II = 1

    // 杩欓噷鏄湪绗竴娆¤緭鍑轰箣鍓� 灏卞害瀹屼簡鏁版嵁锛屼箣鍚庝竴鐩寸敤
    // 鍦ㄨ緭鍑烘姌鍙犵涓�娆¤绠楁椂璇�
    if (out_fold_cnt == 0) {
      temp_vec = vec.read();
      row_store[in_fold_cnt] = temp_vec;
    } else {
      temp_vec = row_store[in_fold_cnt];
    }

    // index = wVec*OutputFold+wMat;

    // 鍒濆鍖栫疮鍔犵粨鏋�
    if (in_fold_cnt == 0) {
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        acc[p] = 0;
      }
    }

    // 涓昏璁＄畻鍗曞厓 杩欓噷鐢║NROLL灞曞紑 鏈熸湜鐢ㄥ崟鍛ㄦ湡瀹炵幇璁＄畻
    // PE 骞惰璁＄畻
    for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
      // 璇� W 瀛愬潡
      ap_uint<SIMD *W_BIT> temp_mat = weights[p][tile];
      // SIMD 骞惰
      acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
    }

    // 璁℃暟閫昏緫 鍜岃緭鍑哄鐞�
    tile++;
    if (++in_fold_cnt == INPUT_FOLD) {
      in_fold_cnt = 0;
      ap_uint<PE * M_BIT> out_buf;
      // PE 鍒楄绠楀畬鎴� 鍙互杈撳嚭
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        out_buf((p + 1) * M_BIT - 1, p * M_BIT) = acc[p];
        // acc[p] = 0;
      }
      out.write(out_buf);
      // 瀹屾暣鐨勪竴娆＄煩闃靛悜閲忚绠�
      if (++out_fold_cnt == OUTPUT_FOLD) {
        out_fold_cnt = 0;
        tile = 0;
      }
    }
  } // end for
}

/**
 * 鐭╅樀鍚戦噺璁＄畻鍗曞厓
 * 鍚屾椂杩涜閲忓寲婵�娲诲鐞�
 */
template <unsigned MAT_ROW, // 灞曞紑鍚庣殑k 脳 k 脳 in_ch
          unsigned MAT_COL, // 灞曞紑鍚庣殑out_ch

          unsigned IN_BIT,
          unsigned OUT_BIT, //

          unsigned W_BIT,
          unsigned M_BIT, // 涔樼疮鍔犲悗鐨勮绠楃粨鏋滅殑鍊�

          unsigned INC_BIT,  // 婵�娲荤瓑宸暟鍒� 鐨勬闀�
          unsigned BIAS_BIT, //

          unsigned SIMD, unsigned PE, unsigned L_SHIFT, unsigned VECT_NUMS>
void matrix_vector_act_unit(
    stream<ap_uint<SIMD * IN_BIT> > &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
    const ap_int<INC_BIT> inc[PE][MAT_COL / PE],
    const ap_int<BIAS_BIT> bias[PE][MAT_COL / PE],
    // stream<ap_uint<PE * OUT_BIT> > &out,
    stream<ap_uint<PE * OUT_BIT> > &out, const unsigned reps = 1) {
//  static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
//  static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

  const unsigned INPUT_FOLD = MAT_ROW / SIMD;
  const unsigned OUTPUT_FOLD = MAT_COL / PE;

  const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

  // 闇�瑕佷繚瀛樹竴琛屾暟鎹�
  ap_uint<SIMD * IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable = row_store core = RAM_2P_BRAM

  // 鐢ㄦ潵淇濆瓨绱姞缁撴灉
  // ap_uint<M_BIT> result_vec[PE];
  // #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
  unsigned in_fold_cnt = 0;  // 杈撳叆鎶樺彔璁℃暟
  unsigned out_fold_cnt = 0; // 杈撳嚭鎶樺彔璁℃暟
  unsigned tile = 0;

  // 涓�娆� 璇诲叆鐨勬暟鎹� 闇�瑕佷繚瀛� in_ch * k * k闀垮害鐨勬暟鎹�
  ap_uint<SIMD * IN_BIT> temp_vec;
  // 绱姞缁撴灉 杩欓噷闇�瑕佸垵濮嬪寲涓�0
  ap_int<M_BIT> acc[PE];

  // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
  for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II = 1

    // 杩欓噷鏄湪绗竴娆¤緭鍑轰箣鍓� 灏卞害瀹屼簡鏁版嵁锛屼箣鍚庝竴鐩寸敤
    // 鍦ㄨ緭鍑烘姌鍙犵涓�娆¤绠楁椂璇�
    if (out_fold_cnt == 0) {
      temp_vec = vec.read();
      row_store[in_fold_cnt] = temp_vec;
    } else {
      temp_vec = row_store[in_fold_cnt];
    }

    // index = wVec*OutputFold+wMat;

    // 鍒濆鍖栫疮鍔犵粨鏋�
    if (in_fold_cnt == 0) {
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        acc[p] = 0;
      }
    }

    // 涓昏璁＄畻鍗曞厓 杩欓噷鐢║NROLL灞曞紑 鏈熸湜鐢ㄥ崟鍛ㄦ湡瀹炵幇璁＄畻
    // PE 骞惰璁＄畻
    for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
      // 璇� W 瀛愬潡
      ap_uint<SIMD *W_BIT> temp_mat = weights[p][tile];
      // SIMD 骞惰
      acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
      // if (p == 0)
      // 	cout << temp_vec(7, 0) << " " <<  temp_vec(15, 8) << " " <<
      // temp_vec(23, 16) << endl;
    }

    // 璁℃暟閫昏緫 鍜岃緭鍑哄鐞�
    tile++;
    if (++in_fold_cnt == INPUT_FOLD) {
      in_fold_cnt = 0;
      ap_uint<PE * OUT_BIT> out_buf;
      // ap_uint<PE * M_BIT> out_buf;
      // PE 鍒楄绠楀畬鎴� 鍙互杈撳嚭
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        out_buf((p + 1) * OUT_BIT - 1, p * OUT_BIT) =
            bn_qurelu<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                      L_SHIFT>(acc[p], inc[p][out_fold_cnt],
                               bias[p][out_fold_cnt]);
        // out_buf((p + 1) * M_BIT - 1, p * M_BIT) = acc[p];
        // cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " "
        // << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
        // acc[p] = 0;
      }
      out.write(out_buf);
      // 瀹屾暣鐨勪竴娆＄煩闃靛悜閲忚绠�
      if (++out_fold_cnt == OUTPUT_FOLD) {
        out_fold_cnt = 0;
        tile = 0;
      }
    }
  } // end for
}

/**
 * 鐭╅樀鍚戦噺璁＄畻鍗曞厓
 * 浣跨敤 lut 璁＄畻
 *
 */
template <unsigned MAT_ROW, // 灞曞紑鍚庣殑k 脳 k 脳 in_ch
          unsigned MAT_COL, // 灞曞紑鍚庣殑out_ch
          unsigned IN_BIT, unsigned W_BIT,
          unsigned M_BIT, // 涔樼疮鍔犲悗鐨勮绠楃粨鏋滅殑鍊�
          unsigned SIMD, unsigned PE, unsigned VECT_NUMS>
void matrix_vector_unit_lut(
    stream<ap_uint<SIMD * IN_BIT> > &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
    stream<ap_uint<PE * M_BIT> > &out, const unsigned reps = 1) {
//  static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
//  static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

  const unsigned INPUT_FOLD = MAT_ROW / SIMD;
  const unsigned OUTPUT_FOLD = MAT_COL / PE;

  const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
  // const unsigned total_reps = 18;
  // 闇�瑕佷繚瀛樹竴琛屾暟鎹�
  ap_uint<SIMD * IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable = row_store core = RAM_2P_BRAM

  // 鐢ㄦ潵淇濆瓨绱姞缁撴灉
  // 	ap_uint<M_BIT> result_vec[PE];
  // #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
  unsigned in_fold_cnt = 0;  // 杈撳叆鎶樺彔璁℃暟
  unsigned out_fold_cnt = 0; // 杈撳嚭鎶樺彔璁℃暟
  unsigned tile = 0;

  // 涓�娆� 璇诲叆鐨勬暟鎹� 闇�瑕佷繚瀛� in_ch * k * k闀垮害鐨勬暟鎹�
  ap_uint<SIMD * IN_BIT> temp_vec;
  // 绱姞缁撴灉 杩欓噷闇�瑕佸垵濮嬪寲涓�0

  // TODO
  ap_int<M_BIT> acc[PE];

  // cout << "acc init value \n";
  // for(unsigned i=0; i < PE; i ++) {
  // 	cout << acc[i] << "  ";
  // }
  // static ap_uint<M_BIT> acc1[PE] = {0};

  // cout << "acc1 init value \n";
  // for(unsigned i=0; i < PE; i ++) {
  // 	cout << acc1[i] << "  ";
  // }

  // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
  for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II = 1

    // 杩欓噷鏄湪绗竴娆¤緭鍑轰箣鍓� 灏卞害瀹屼簡鏁版嵁锛屼箣鍚庝竴鐩寸敤
    // 鍦ㄨ緭鍑烘姌鍙犵涓�娆¤绠楁椂璇�
    if (out_fold_cnt == 0) {
      temp_vec = vec.read();
      row_store[in_fold_cnt] = temp_vec;
    } else {
      temp_vec = row_store[in_fold_cnt];
    }

    // index = wVec*OutputFold+wMat;

    // 鍒濆鍖栫疮鍔犵粨鏋�
    if (in_fold_cnt == 0) {
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        acc[p] = 0;
      }
    }

    // 涓昏璁＄畻鍗曞厓 杩欓噷鐢║NROLL灞曞紑 鏈熸湜鐢ㄥ崟鍛ㄦ湡瀹炵幇璁＄畻
    // PE 骞惰璁＄畻
    for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
      // 璇� W 瀛愬潡
      ap_uint<SIMD *W_BIT> temp_mat = weights[p][tile];
      // SIMD 骞惰
      acc[p] += simd_mul_lut<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
    }

    // 璁℃暟閫昏緫 鍜岃緭鍑哄鐞�
    tile++;
    if (++in_fold_cnt == INPUT_FOLD) {
      in_fold_cnt = 0;
      ap_uint<PE * M_BIT> out_buf;
      // PE 鍒楄绠楀畬鎴� 鍙互杈撳嚭
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        out_buf((p + 1) * M_BIT - 1, p * M_BIT) = acc[p];
        // acc[p] = 0;
      }
      out.write(out_buf);
      // 瀹屾暣鐨勪竴娆＄煩闃靛悜閲忚绠�
      if (++out_fold_cnt == OUTPUT_FOLD) {
        out_fold_cnt = 0;
        tile = 0;
      }
    }
  } // end for
}

/**
 * 鐭╅樀鍚戦噺璁＄畻鍗曞厓
 * 鍚屾椂杩涜閲忓寲婵�娲诲鐞�
 * 浣跨敤 lut 璁＄畻
 */
template <unsigned MAT_ROW, // 灞曞紑鍚庣殑k 脳 k 脳 in_ch
          unsigned MAT_COL, // 灞曞紑鍚庣殑out_ch

          unsigned IN_BIT,
          unsigned OUT_BIT, //

          unsigned W_BIT,
          unsigned M_BIT, // 涔樼疮鍔犲悗鐨勮绠楃粨鏋滅殑鍊�

          unsigned INC_BIT,  // 婵�娲荤瓑宸暟鍒� 鐨勬闀�
          unsigned BIAS_BIT, //

          unsigned SIMD, unsigned PE, unsigned L_SHIFT, unsigned VECT_NUMS>
void matrix_vector_act_unit_lut(
    stream<ap_uint<SIMD * IN_BIT> > &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
    const ap_uint<INC_BIT> inc[PE][MAT_COL / PE],
    const ap_int<BIAS_BIT> bias[PE][MAT_COL / PE],
    stream<ap_uint<PE * OUT_BIT> > &out, const unsigned reps = 1) {
//  static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
//  static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

  const unsigned INPUT_FOLD = MAT_ROW / SIMD;
  const unsigned OUTPUT_FOLD = MAT_COL / PE;

  const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

  // 闇�瑕佷繚瀛樹竴琛屾暟鎹�
  ap_uint<SIMD * IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable = row_store core = RAM_2P_BRAM

  // 鐢ㄦ潵淇濆瓨绱姞缁撴灉
  // ap_uint<M_BIT> result_vec[PE];
  // #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
  unsigned in_fold_cnt = 0;  // 杈撳叆鎶樺彔璁℃暟
  unsigned out_fold_cnt = 0; // 杈撳嚭鎶樺彔璁℃暟
  unsigned tile = 0;

  // 涓�娆� 璇诲叆鐨勬暟鎹� 闇�瑕佷繚瀛� in_ch * k * k闀垮害鐨勬暟鎹�
  ap_uint<SIMD * IN_BIT> temp_vec;
  // 绱姞缁撴灉 杩欓噷闇�瑕佸垵濮嬪寲涓�0
  ap_int<M_BIT> acc[PE];

  // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
  for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II = 1

    // 杩欓噷鏄湪绗竴娆¤緭鍑轰箣鍓� 灏卞害瀹屼簡鏁版嵁锛屼箣鍚庝竴鐩寸敤
    // 鍦ㄨ緭鍑烘姌鍙犵涓�娆¤绠楁椂璇�
    if (out_fold_cnt == 0) {
      temp_vec = vec.read();
      row_store[in_fold_cnt] = temp_vec;
    } else {
      temp_vec = row_store[in_fold_cnt];
    }

    // index = wVec*OutputFold+wMat;

    // 鍒濆鍖栫疮鍔犵粨鏋�
    if (in_fold_cnt == 0) {
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        acc[p] = 0;
      }
    }

    // 涓昏璁＄畻鍗曞厓 杩欓噷鐢║NROLL灞曞紑 鏈熸湜鐢ㄥ崟鍛ㄦ湡瀹炵幇璁＄畻
    // PE 骞惰璁＄畻
    for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
      // 璇� W 瀛愬潡
      ap_uint<SIMD *W_BIT> temp_mat = weights[p][tile];
      // SIMD 骞惰
      acc[p] += simd_mul_lut<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
    }

    // 璁℃暟閫昏緫 鍜岃緭鍑哄鐞�
    tile++;
    if (++in_fold_cnt == INPUT_FOLD) {
      in_fold_cnt = 0;
      ap_uint<PE * M_BIT> out_buf;
      // PE 鍒楄绠楀畬鎴� 鍙互杈撳嚭
      for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
        out_buf((p + 1) * OUT_BIT - 1, p * OUT_BIT) =
            bn_qurelu<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                      L_SHIFT>(acc[p], inc[p][out_fold_cnt],
                               bias[p][out_fold_cnt]);
        // cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " "
        // << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
        // acc[p] = 0;
      }
      out.write(out_buf);
      // 瀹屾暣鐨勪竴娆＄煩闃靛悜閲忚绠�
      if (++out_fold_cnt == OUTPUT_FOLD) {
        out_fold_cnt = 0;
        tile = 0;
      }
    }
  } // end for
}
