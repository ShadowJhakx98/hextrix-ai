# CHANGELOG

## [Unreleased]
### Added
- Initial setup for all core modules, including:
  - `main.py`: Central Flask app management.
  - `planner_agent.py`: Task planning and intent recognition.
  - `mem_drive.py`: Memory and context management.
  - `specialized_sub_agent.py`: OCR, screen capture, and multimodal tasks.
  - `ui_automator.py`: Voice handling and UI automation.
  - `gemini_api_doc_reference.py`: Gemini API documentation and examples.
  - `emotions.py`: Sentiment and emotion detection.
  - `discord_bot.py`: Discord bot integration.
  - `ocr_module.py`: Optical character recognition module.
  - `tts_module.py`: Text-to-speech module.
  - `stt_module.py`: Speech-to-text module.
  - `image_processing.py`: Image handling and descriptions.
  - `gesture_recognition.py`: Real-time gesture recognition.
  - `lighting_control.py`: Dynamic RGB lighting control for ASUS Aura Sync and Razer Chroma.
  - `multimodal_manager.py`: Multimodal input/output management.
  - `audio_streaming.py`: Real-time audio streaming.
  - `config.py`: Configuration and environment management.
  - `logging_utils.py`: Centralized logging utilities.

### Changed
- Updated `planner_agent.py` to include support for more complex task sequencing.
- Enhanced `emotions.py` with more granular emotional detection mapping to RGB lighting.

### Fixed
- Resolved memory overflow issue in `mem_drive.py` during extended conversations.
- Fixed inconsistent TTS playback in `tts_module.py`.
- Addressed latency issues in `audio_streaming.py` during live mode.

---

## [1.0.0] - YYYY-MM-DD
### Added
- Full integration of Whisper-1 for advanced speech-to-text capabilities.
- Support for DALL·E 3 image generation in `image_processing.py`.
- Gesture-based command controls in `gesture_recognition.py`.
- Dynamic RGB lighting integration with ASUS Aura Sync and Razer Chroma.
- API endpoints for live mode and multimodal queries in `main.py`.

### Changed
- Refactored `ui_automator.py` for better compatibility with diverse voice commands.
- Improved error handling across all modules.
- Optimized task planner in `planner_agent.py` for real-time performance.

### Fixed
- Resolved issues with inconsistent emotional state mapping in `emotions.py`.
- Fixed incorrect OCR processing in `ocr_module.py` for high-resolution images.

---

## [0.9.0] - YYYY-MM-DD
### Added
- Initial prototypes for modules, including task planner, memory management, and multimodal processing.
- Basic Discord bot commands and integration in `discord_bot.py`.
- Screen capture and OCR functionality in `specialized_sub_agent.py`.
- Sentiment analysis module in `emotions.py`.
- Core Flask app structure in `main.py`.

### Changed
- Adjusted configuration structure in `config.py` for better environment management.

### Fixed
- Minor bugs in logging utilities (`logging_utils.py`).



---

## [0.0.1] - 2025-1-18
### Added
- TODO.md


### Changed
- Adjusted configuration structure in `config.py` for better environment management.

### Fixed
- Minor bugs in logging utilities (`logging_utils.py`).

