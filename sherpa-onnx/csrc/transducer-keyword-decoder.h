// sherpa-onnx/csrc/transducer-keywords-decoder.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_TRANSDUCER_KEYWORD_DECODER_H_
#define SHERPA_ONNX_CSRC_TRANSDUCER_KEYWORD_DECODER_H_

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/keyword-spotter-state.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

struct TransducerKeywordResult {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs for keywords
  std::vector<int64_t> tokens;

  /// The triggered keyword
  std::string keyword;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;

  // used only in modified beam_search
  Hypotheses hyps;

  // The last frames of this stream's search, oldest first. Bounded by the
  // decoder so a long stream cannot grow the snapshot without limit.
  std::deque<KeywordFrameState> recent_frames;
};

class TransducerKeywordDecoder {
 public:
  TransducerKeywordDecoder(OnlineTransducerModel *model,
                           int32_t max_active_paths,
                           int32_t num_trailing_blanks, int32_t unk_id)
      : model_(model),
        max_active_paths_(max_active_paths),
        num_trailing_blanks_(num_trailing_blanks),
        unk_id_(unk_id) {}

  TransducerKeywordResult GetEmptyResult() const;

  // frame_energy_db[i][t] is the input energy of stream i's encoder frame t in
  // this chunk, measured from the same feature block the encoder saw.
  void Decode(Ort::Value encoder_out, OnlineStream **ss,
              const std::vector<std::vector<float>> &frame_energy_db,
              std::vector<TransducerKeywordResult> *result);

 private:
  OnlineTransducerModel *model_;  // Not owned

  int32_t max_active_paths_;
  int32_t num_trailing_blanks_;
  int32_t unk_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TRANSDUCER_KEYWORD_DECODER_H_
