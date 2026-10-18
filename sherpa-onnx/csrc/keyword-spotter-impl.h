// sherpa-onnx/csrc/keyword-spotter-impl.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/keyword-spotter-state.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

class KeywordSpotterImpl {
 public:
  static std::unique_ptr<KeywordSpotterImpl> Create(
      const KeywordSpotterConfig &config);

  template <typename Manager>
  static std::unique_ptr<KeywordSpotterImpl> Create(
      Manager *mgr, const KeywordSpotterConfig &config);

  virtual ~KeywordSpotterImpl() = default;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream(
      const std::string &keywords) const = 0;

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual void Reset(OnlineStream *s) const = 0;

  virtual void DecodeStreams(OnlineStream **ss, int32_t n) const = 0;

  virtual KeywordResult GetResult(OnlineStream *s) const = 0;

  // The keyword context graph this backend searches. A backend that cannot
  // observe its search state returns an empty graph instead of a guess.
  virtual ContextGraphState GetContextGraph(OnlineStream * /*s*/) const {
    return {};
  }

  // The live search state of one stream, empty on backends that cannot expose
  // it. The state is read-only; it never feeds the decision.
  virtual KeywordSearchState GetSearchState(OnlineStream * /*s*/) const {
    return {};
  }
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_
