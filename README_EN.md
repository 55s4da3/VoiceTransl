
<p align="center">
	<img src="icon.png" alt="Logo" width="160" />
</p>

<h1><p align='center' >VoiceTransl</p></h1>
<div align=center><img src="https://img.shields.io/github/v/release/shinnpuru/VoiceTransl"/>   <img src="https://img.shields.io/github/license/shinnpuru/VoiceTransl"/>   <img src="https://img.shields.io/github/stars/shinnpuru/VoiceTransl"/></div>

[简体中文](README.md) | **English**

VoiceTransl is an all-in-one offline AI subtitle generation and translation software for videos, supporting macOS and Windows. It simplifies every step for translators: video download, audio extraction, speech-to-text with timestamps, subtitle translation, video synthesis, and subtitle summarization. This project is based on [Galtransl](https://github.com/xd2333/GalTransl) and is licensed under GPLv3.

## Features

* Supports multiple translation models, including online models (any OpenAI-compatible API) and local models (Sakura, Galtransl, Ollama, Llamacpp).
* Supports AMD/NVIDIA/Intel GPU acceleration, and the translation engine supports adjusting VRAM usage.
* Supports multiple input formats, including audio, video, and SRT subtitles.
* Supports multiple output formats, including SRT and LRC subtitles.
* Supports multiple languages, including Japanese, English, Korean, Russian, and French.
* Uses CrispASR's Qwen3-ASR with forced alignment to generate timestamped subtitles.
* Supports VAD (Voice Activity Detection) to automatically detect speech segments in audio.
* Supports a dictionary feature to customize translation dictionaries for input/output replacement.
* Supports world book / script input to customize translation reference materials.
* Supports direct video download from YouTube/Bilibili and media links.
* Supports batch processing of files and links with automatic file type detection.
* Supports audio segmentation, subtitle merging, and video synthesis.
* Supports video summarization, generating concise timestamped text summaries of video content.
* Supports vocal separation, separating vocals from accompaniment, with multiple models.

## Modes

The software supports five modes: Download, Translate, Transcribe, Full, and Tools.

1. Download mode: Download videos directly from YouTube/Bilibili. Enter video links, set speech recognition to "no transcription" and subtitle translation to "no translation", then click Run.
2. Translate mode: Translate subtitles with multiple translation models. Enter subtitle files, set speech recognition to "no transcription", choose a translation model, then click Run.
3. Transcribe mode: Transcribe audio with multiple ASR models. Enter audio/video files or video links, choose an ASR model, set subtitle translation to "no translation", then click Run.
4. Full mode: Run the complete pipeline from download to translation. Enter audio/video files or video links, choose an ASR model and a translation model, then click Run.
5. Tools mode: Perform audio separation, audio segmentation, subtitle merging, video synthesis, and video summarization. Fill in the corresponding inputs, choose a tool, then click Run.

<div align=center><img src="title.jpg" alt="title" style="width:512px;"/></div>

## Download

Download the latest release of [VoiceTransl](https://github.com/shinnpuru/VoiceTransl/releases/), unzip it, and run `VoiceTransl.exe`.

## Usage Guide

See the [video tutorial](https://www.bilibili.com/video/BV1koZ6YuE1x) for usage instructions.

## Disclaimer

This software is provided for learning and communication purposes only and must not be used for commercial purposes. This software is not responsible for the actions of any user and does not guarantee the accuracy of translation results. By using this software you agree to assume all risks associated with its use, including but not limited to copyright risks and legal risks. Please comply with local laws and regulations and do not use this software for any illegal activities.

## If this project helps you, please give it a Star!

![Star History Chart](https://api.star-history.com/svg?repos=shinnpuru/VoiceTransl&type=Date)
