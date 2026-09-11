#!/usr/bin/env bash
# Claude Code Stop hook: 有未提交改动时提醒把摘要写进 CLAUDE.md 工作日志
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "[提醒] 本轮若完成了阶段性改动，请把它写进根目录 CLAUDE.md 的【工作日志】顶部并刷新【当前状态】（同一条别重复写），让下一个 AI 不用重新整理。"
fi
