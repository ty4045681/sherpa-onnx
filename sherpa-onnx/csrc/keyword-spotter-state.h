// sherpa-onnx/csrc/keyword-spotter-state.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_STATE_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_STATE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

// One node of the keyword context graph. Nodes are numbered by a breadth-first
// walk from the root, so an id stays stable for the life of one snapshot.
struct ContextGraphNodeState {
  int32_t id = 0;
  // The token this node consumes. The root uses -1.
  int32_t token = -1;
  std::string piece;
  // How many keyword units have been consumed to reach this node.
  int32_t level = 0;
  bool is_end = false;
  std::string phrase;
  int32_t fail = 0;
  std::vector<int32_t> next;
};

struct ContextGraphState {
  int32_t root = 0;
  std::vector<ContextGraphNodeState> nodes;
};

// One token and the joiner's log probability for it at a frame.
struct KeywordTokenScore {
  int32_t token = -1;
  std::string piece;
  float log_prob = 0;
};

// One decoder path and the context graph node it currently sits on.
struct KeywordHypothesisState {
  int32_t level = 0;
  int32_t token = -1;
  std::string piece;
  bool is_end = false;
  std::string phrase;
  double score = 0;
  // The token history since the last reset, in decoder order.
  std::vector<int32_t> tokens;
  std::vector<std::string> pieces;
  // timestamps[i] is the subsampled frame where tokens[i] was emitted.
  std::vector<int32_t> timestamps;
  // The acoustic probability recorded for each emitted token.
  std::vector<float> token_probs;
};

// What the decoder held after one encoder frame.
struct KeywordFrameState {
  int32_t frame_index = 0;
  int32_t level = 0;
  double score = 0;
  // Input energy of the frame's feature block, and the L2 norm of its encoder
  // output, both measured by the decoder.
  float energy_db = -100.0f;
  float encoder_norm = 0;
  // The joiner's best tokens for this frame, before the context score.
  std::vector<KeywordTokenScore> top_tokens;
  std::vector<KeywordHypothesisState> hypotheses;
};

// The live keyword search of one stream.
struct KeywordSearchState {
  int32_t frame_offset = 0;
  KeywordHypothesisState best;
  std::vector<KeywordFrameState> frames;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_STATE_H_
