#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "stream_tools.hpp"

template <unsigned K, unsigned S, unsigned Din_H, unsigned Din_W, unsigned Cin,
          unsigned Ibit>
void SWU(stream<ap_uint<Cin * Ibit> > &in, stream<ap_uint<Cin * Ibit> > &out,
         const unsigned reps = 1) {
//  static_assert((Din_W - K) % S == 0, "(Din_W-K) mod S is not 0");
//  static_assert((Din_H - K) % S == 0, "(Din_H-K) mod S is not 0");
//  static_assert(K >= S, "K is not >= than S");

  const unsigned steps = (Din_W - K) / S + 1;
  const unsigned line_buffer_size = K * Din_W;
#ifdef SWU_DEBUG
  cout << "steps: " << steps << endl;
  cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

  ap_uint<Cin * Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core = RAM_2P

  ap_uint<Cin * Ibit> temp_in;

  ap_uint<1> initial_fill = 0;
  unsigned stride = 0;
  unsigned pointer = 0;
  unsigned h = 0;

  for (unsigned rep = 0; rep < reps * Din_H; rep++) {

    if (h == Din_H) {
      initial_fill = 0;
      stride = 0;
      pointer = 0;
      h = 0;
    }
    h += 1;

#ifdef SWU_DEBUG
    cout << "wpointer: " << pointer << endl;
#endif

    for (unsigned w = 0; w < Din_W; w++) {
#pragma HLS PIPELINE II = 1
      temp_in = in.read();

      unsigned line_buffer_pointer = pointer + w;
      if (line_buffer_pointer >= line_buffer_size) {
        line_buffer_pointer = line_buffer_pointer - line_buffer_size;
      }
#ifdef SWU_DEBUG
      cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
      line_buffer[line_buffer_pointer] = temp_in;
    }

    stride += 1;
    pointer += Din_W;
    if (pointer >= line_buffer_size) {
      pointer = pointer - line_buffer_size;
      initial_fill = 1;
#ifdef SWU_DEBUG
      cout << "initial_fill set to 1!" << endl;
#endif
    }

#ifdef SWU_DEBUG
    cout << "stride: " << stride << endl;
    cout << "rpointer: " << pointer << endl;
    cout << "line_buffer for out: ";
    for (unsigned j = 0; j < line_buffer_size; j++) {
      cout << line_buffer[j] << " ";
    }
    cout << endl;
#endif
    if (initial_fill == 1 && stride >= S) {
      stride = 0;

      unsigned s = 0;
      unsigned x = 0;
      unsigned y = 0;

      for (unsigned i = 0; i < steps * (K * K); i++) {
#pragma HLS PIPELINE II = 1
        unsigned read_address = (pointer + s * S) + y * Din_W + x;

        if (read_address >= line_buffer_size)
          read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
        cout << "read_address: " << read_address << endl;
#endif
        ap_uint<Cin *Ibit> temp_out = line_buffer[read_address];
        out.write(temp_out);

        if (x == K - 1) {
          x = 0;
          if (y == K - 1) {
            y = 0;
            if (s == steps - 1)
              s = 0;
            else
              s++;
          } else
            y++;
        } else
          x++;
      }
    }
  }
}

template <unsigned K, unsigned S, unsigned IN_ROW, unsigned IN_COL,
          unsigned IN_CH, unsigned IN_BIT>
void sliding_window_unit(stream<ap_uint<IN_CH * IN_BIT> > &in,
                         stream<ap_uint<IN_CH * IN_BIT> > &out,
                         const unsigned reps = 1) {
//  static_assert((IN_ROW - K) % S == 0, "(IN_ROW-K) mod S is not 0");
//  static_assert((IN_COL - K) % S == 0, "(IN_COL-K) mod S is not 0");
//  static_assert(K >= S, "K is not >= than S");

  // 琛屾柟鍚戜笂闇�瑕佺Щ鍔ㄥ灏戞 鍚戜笅绉诲姩娆℃暟
  const unsigned ROW_STEPS = (IN_ROW - K) / S + 1;
  // 鎯冲彸绉诲姩娆℃暟
  const unsigned COL_STEPS = (IN_COL - K) / S + 1;

  // TODO buf搴旇杩樺彲浠ヤ紭鍖�
  // 褰撳浘鍍忓昂瀵镐笉涓�鑷存椂 閫夌敤 row浼樺厛 or col浼樺厛搴旇瀵� 杩欓噷鐨刡uff娑堣�楁湁褰卞搷
  // 鏋勫缓涓�涓惊鐜垪闃�
  // 渚嬪褰� K = 3鏃� 瀹為檯涓婁笉闇�瑕� 瀹屾暣鐨� 3琛屾潵缂撳瓨 鑰屾槸鍙渶瑕� 2 脳 IN_COL +
  // 3灏卞彲浠ヨВ闄や緷璧�
  const unsigned BUF_SIZE = (K - 1) * IN_COL + K;
  ap_uint<IN_CH * IN_BIT> line_buffer[BUF_SIZE];
#pragma HLS RESOURCE variable line_buffer core = RAM_2P
  unsigned buf_len = 0;
  unsigned buf_pointer = 0;
  ap_uint<IN_CH * IN_BIT> temp_in;

  // 婊戝姩璁℃暟
  unsigned right_slid = 0;
  unsigned down_slid = 0;
  // 涓�鍏卞惊鐜殑娆℃暟
  for (unsigned rep = 0; rep < IN_ROW * IN_COL * reps; rep++) {
    // 鍐欐暟鎹埌 buf
    // buf 涓嶆弧鐨勬椂鍊欎竴鐩村啓鏁版嵁
    if (buf_len < BUF_SIZE) {
      // TODO
      temp_in = in.read();
      line_buffer[buf_pointer++] = temp_in;
      if (buf_pointer == BUF_SIZE) {
        buf_pointer = 0;
      }
      buf_len++;
    }

    // 缂撳啿鍖烘弧 鍙互杈撳嚭鏁版嵁
    if (buf_len == BUF_SIZE) {
      // 杈撳嚭绐楀彛鏁版嵁
      // 缂撳啿鍖哄鍧� pointer 鎸囧悜鐨勬槸涓嬩竴涓綅缃�
      // 濡傛灉瑙勫畾姣忔潵涓�涓厓绱犻兘鏄斁鍦ㄩ槦澶达紝褰搃=0鏃�
      // pointer瀹為檯鎸囧悜鐨勫厓绱犳槸鏈�鍚庝竴涓厓绱� 鑰岃繖涓厓绱犳鏄繖閲岃鏈�鍏堣緭鍑虹殑
      for (unsigned i = 0; i < K; i++) {
        for (unsigned j = 0; j < K; j++) {
          // 瀵诲潃
          unsigned temp_pointer = (buf_pointer + (i * IN_COL) + j);
          // 杩欓噷temp_pointer 涓嶅彲鑳藉ぇ浜� 2 脳 BUF_SIZE
          if (temp_pointer > BUF_SIZE) {
            temp_pointer -= BUF_SIZE;
          }

          ap_uint<IN_CH *IN_BIT> temp_out = line_buffer[temp_pointer];
          out.write(temp_out);
        }
      }
      // 杈撳嚭鍚庣獥鍙ｅ悜鍙虫粦鍔�
      // 婊戝埌澶翠簡
      if (++right_slid == COL_STEPS) {
        right_slid = 0;
        // 鍙虫粦鍒板ご 涓嬫粦
        if (++down_slid == ROW_STEPS) {
          down_slid = 0;
          // 涓�甯ф暟鎹畬
          buf_len = 0;
        } else {
          // 涓嬫粦娌℃湁鍒板ご
          buf_len = buf_len - (S - 1) * IN_COL - K;
        }
      } else {
        // 鍙虫粦娌″埌澶�
        // S 涓暟鎹� 鍑虹紦鍐�
        buf_len -= S;
      }
    }
  }
}
