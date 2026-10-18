// sherpa-onnx/python/csrc/keyword-spotter.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/keyword-spotter.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/keyword-spotter.h"

namespace sherpa_onnx {

static void PybindKeywordResult(py::module *m) {
  using PyClass = KeywordResult;
  py::class_<PyClass>(*m, "KeywordResult")
      .def_property_readonly(
          "keyword",
          [](PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.keyword.c_str(),
                                                self.keyword.size(), "ignore"));
          })
      .def_property_readonly(
          "tokens",
          [](PyClass &self) -> std::vector<std::string> { return self.tokens; })
      .def_property_readonly(
          "timestamps",
          [](PyClass &self) -> std::vector<float> { return self.timestamps; });
}

static void PybindKeywordSpotterConfig(py::module *m) {
  using PyClass = KeywordSpotterConfig;
  py::class_<PyClass>(*m, "KeywordSpotterConfig")
      .def(py::init<const FeatureExtractorConfig &, const OnlineModelConfig &,
                    int32_t, int32_t, float, float, const std::string &>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("max_active_paths") = 4, py::arg("num_trailing_blanks") = 1,
           py::arg("keywords_score") = 1.0,
           py::arg("keywords_threshold") = 0.25, py::arg("keywords_file") = "")
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("num_trailing_blanks", &PyClass::num_trailing_blanks)
      .def_readwrite("keywords_score", &PyClass::keywords_score)
      .def_readwrite("keywords_threshold", &PyClass::keywords_threshold)
      .def_readwrite("keywords_file", &PyClass::keywords_file)
      .def("__str__", &PyClass::ToString);
}

static constexpr const char *kKeywordSpotterDoc = R"doc(
Keyword spotter engine.

Args:
  config:
    The configuration for the keyword spotter.
)doc";

static constexpr const char *kCreateStreamDoc = R"doc(
Create a new streaming recognition instance using the keywords from the
configuration.

Returns:
  An ``OnlineStream`` object.
)doc";

static constexpr const char *kCreateStreamKeywordsDoc = R"doc(
Create a new streaming recognition instance with custom keywords.

Args:
  keywords:
    A string of keywords to spot, separated by ``/``.

Returns:
  An ``OnlineStream`` object.
)doc";

static constexpr const char *kIsReadyDoc = R"doc(
Return True if the stream has enough frames for decoding.

Args:
  stream:
    The stream to check.
)doc";

static constexpr const char *kDecodeStreamDoc = R"doc(
Run one decoding step on the given stream.

Args:
  stream:
    The stream to decode.
)doc";

static constexpr const char *kGetResultDoc = R"doc(
Get the current keyword result for the given stream.

Args:
  stream:
    The stream to query.

Returns:
  A ``KeywordResult`` object.
)doc";

static constexpr const char *kResetStreamDoc = R"doc(
Reset the given stream for the next detection round.

Args:
  stream:
    The stream to reset.
)doc";

static py::str Utf8(const std::string &text) {
  return py::str(PyUnicode_DecodeUTF8(text.c_str(), text.size(), "ignore"));
}

static py::dict ToPy(const ContextGraphNodeState &node) {
  py::dict out;
  out["id"] = node.id;
  out["token"] = node.token;
  out["piece"] = Utf8(node.piece);
  out["level"] = node.level;
  out["is_end"] = node.is_end;
  out["phrase"] = Utf8(node.phrase);
  out["fail"] = node.fail;
  out["next"] = py::cast(node.next);
  return out;
}

static py::dict ToPy(const ContextGraphState &graph) {
  py::dict out;
  out["root"] = graph.root;
  py::list nodes;
  for (const auto &node : graph.nodes) {
    nodes.append(ToPy(node));
  }
  out["nodes"] = nodes;
  return out;
}

static py::dict ToPy(const KeywordTokenScore &token) {
  py::dict out;
  out["token"] = token.token;
  out["piece"] = Utf8(token.piece);
  out["log_prob"] = token.log_prob;
  return out;
}

static py::dict ToPy(const KeywordHypothesisState &hypothesis) {
  py::dict out;
  out["level"] = hypothesis.level;
  out["token"] = hypothesis.token;
  out["piece"] = Utf8(hypothesis.piece);
  out["is_end"] = hypothesis.is_end;
  out["phrase"] = Utf8(hypothesis.phrase);
  out["score"] = hypothesis.score;
  out["tokens"] = py::cast(hypothesis.tokens);
  out["pieces"] = py::cast(hypothesis.pieces);
  out["timestamps"] = py::cast(hypothesis.timestamps);
  out["token_probs"] = py::cast(hypothesis.token_probs);
  return out;
}

static py::dict ToPy(const KeywordFrameState &frame) {
  py::dict out;
  out["frame_index"] = frame.frame_index;
  out["level"] = frame.level;
  out["score"] = frame.score;
  out["energy_db"] = frame.energy_db;
  out["encoder_norm"] = frame.encoder_norm;
  py::list top_tokens;
  for (const auto &token : frame.top_tokens) {
    top_tokens.append(ToPy(token));
  }
  out["top_tokens"] = top_tokens;
  py::list hypotheses;
  for (const auto &hypothesis : frame.hypotheses) {
    hypotheses.append(ToPy(hypothesis));
  }
  out["hypotheses"] = hypotheses;
  return out;
}

static py::dict ToPy(const KeywordSearchState &state) {
  py::dict out;
  out["frame_offset"] = state.frame_offset;
  out["best"] = ToPy(state.best);
  py::list frames;
  for (const auto &frame : state.frames) {
    frames.append(ToPy(frame));
  }
  out["frames"] = frames;
  return out;
}

static constexpr const char *kGetContextGraphDoc = R"doc(
Return the keyword context graph of a stream.

The graph is a flat node list walked breadth-first from the root. Each node
carries its token and piece, its level (how many keyword units it consumed),
its fail link, and the ids of its next nodes.

Args:
  stream:
    The stream whose context graph to return.

Returns:
  A dict with a root id and a nodes list. A backend that cannot expose its
  graph returns an empty node list.
)doc";

static constexpr const char *kGetSearchStateDoc = R"doc(
Return the live keyword search state of a stream.

The state is the decoder's own state: the best hypothesis and the last
decoded frames, each with every active hypothesis (context level, token,
score, token history and timestamps). It is read-only and never feeds the
decision.

Args:
  stream:
    The stream whose search state to return.

Returns:
  A dict with a frame offset, the best hypothesis and a frames list. A backend
  that cannot expose its state returns an empty frames list.
)doc";

void PybindKeywordSpotter(py::module *m) {
  PybindKeywordResult(m);
  PybindKeywordSpotterConfig(m);

  using PyClass = KeywordSpotter;
  py::class_<PyClass>(*m, "KeywordSpotter", kKeywordSpotterDoc)
      .def(py::init<const KeywordSpotterConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          kCreateStreamDoc, py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &keywords) {
            return self.CreateStream(keywords);
          },
          py::arg("keywords"), kCreateStreamKeywordsDoc,
          py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &PyClass::IsReady, kIsReadyDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, kResetStreamDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, kDecodeStreamDoc,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](PyClass &self, std::vector<OnlineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_result", &PyClass::GetResult, kGetResultDoc,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "get_context_graph",
          [](const PyClass &self, OnlineStream *s) {
            return ToPy(self.GetContextGraph(s));
          },
          py::arg("stream"), kGetContextGraphDoc)
      .def(
          "get_search_state",
          [](const PyClass &self, OnlineStream *s) {
            return ToPy(self.GetSearchState(s));
          },
          py::arg("stream"), kGetSearchStateDoc);
}

}  // namespace sherpa_onnx
