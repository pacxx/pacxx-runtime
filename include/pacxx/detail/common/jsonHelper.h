#pragma once
#include "json.hpp"
#include "../KernelConfiguration.h"
using nlohmann::json;

namespace pacxx {
namespace v2 {
  void to_json(json& j, const KernelConfiguration& kc) {
    j = json{"KernelConfiguration", {
                                      {"blocks", {{"x", kc.blocks.x}, {"y", kc.blocks.y}, {"z", kc.blocks.z}}},
                                      {"threads", {{"x", kc.threads.x}, {"y", kc.threads.y}, {"z", kc.threads.z}}},
                                      {"shmem", kc.sm_size}
                                    }};
  }

  void from_json(const json& j, KernelConfiguration& kc) {
    json inside = j.at("KernelConfiguration");
    kc.blocks.x = inside.at("blocks").at("x").get<std::size_t>();
    kc.blocks.y = inside.at("blocks").at("y").get<std::size_t>();
    kc.blocks.z = inside.at("blocks").at("z").get<std::size_t>();
    kc.threads.x = inside.at("threads").at("x").get<std::size_t>();
    kc.threads.y = inside.at("threads").at("y").get<std::size_t>();
    kc.threads.z = inside.at("threads").at("z").get<std::size_t>();
    kc.sm_size = inside.at("shmem").get<std::size_t>();
  }
}
}
