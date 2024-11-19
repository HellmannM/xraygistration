// Copyright 2022 Matthias Hellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// glad
#include <glad/glad.h>
// glm
#include <anari/anari_cpp/ext/glm.h>
// anari
#include "anari_viewer/windows/Window.h"
// std
#include <functional>
#include <string>
#include <vector>

namespace windows {

using ColorPoint = glm::vec4;
using OpacityPoint = glm::vec2;

using SettingsUpdateCallback =
    std::function<void(const float &)>;

class SettingsEditor : public anari_viewer::Window
{
 public:
  SettingsEditor(const char *name = "Settings Editor");
  ~SettingsEditor() = default;

  void buildUI() override;

  void setUpdateCallback(SettingsUpdateCallback cb);
  void triggerUpdateCallback();

 private:
  // callback called whenever settings are updated
  SettingsUpdateCallback m_updateCallback;

  // flag indicating transfer function has changed in UI
  bool m_settingsChanged{true};

  // photon energy
  float m_photonEnergy{13000.f};
  float m_defaultPhotonEnergy{13000.f};
};

} // namespace windows
