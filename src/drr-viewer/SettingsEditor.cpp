// Copyright 2022 Matthias Hellmann
// SPDX-License-Identifier: Apache-2.0

#include "SettingsEditor.h"
// std
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace windows {

SettingsEditor::SettingsEditor(const char *name)
    : Window(name, true) {}

void SettingsEditor::buildUI()
{
  if (m_settingsChanged) {
    triggerUpdateCallback();
    m_settingsChanged = false;
  }

//  //combo box
//  std::vector<const char *> names(m_tfnsNames.size(), nullptr);
//  std::transform(m_tfnsNames.begin(),
//      m_tfnsNames.end(),
//      names.begin(),
//      [](const std::string &t) { return t.c_str(); });
//
//  int newMap = m_currentMap;
//  if (ImGui::Combo("color map", &newMap, names.data(), names.size()))
//    setMap(newMap);
//
//  ImGui::Separator();

//  drawEditor();
//
//  ImGui::Separator();

  m_settingsChanged |=
      ImGui::SliderFloat("tube potential [keV]", &m_photonEnergy, 0.f, 30000.f);

  if (ImGui::Button("reset##energy")) {
    m_photonEnergy = m_defaultPhotonEnergy;
    m_settingsChanged = true;
  }

  ImGui::Separator();

//  //DragFloatRange2
//  m_tfnChanged |= ImGui::DragFloatRange2("value range",
//      &m_valueRange.x,
//      &m_valueRange.y,
//      0.1f,
//      -10000.f,
//      10000.0f,
//      "Min: %.7f",
//      "Max: %.7f");
//
//  if (ImGui::Button("reset##valueRange")) {
//    m_valueRange = m_defaultValueRange;
//    m_tfnChanged = true;
//  }
}

void SettingsEditor::setUpdateCallback(SettingsUpdateCallback cb)
{
  m_updateCallback = cb;
  triggerUpdateCallback();
}

void SettingsEditor::triggerUpdateCallback()
{
  if (m_updateCallback)
    m_updateCallback(m_photonEnergy);
}

} // namespace windows
