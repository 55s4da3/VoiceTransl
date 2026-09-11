#!/usr/bin/env bash
# Claude Code SessionEnd hook: 结束时会话自动留一条 git 机械记录(兜底)，供追踪
TS=$(date "+%Y-%m-%d %H:%M")
mkdir -p .claude
B=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "无git")
C=$(git status --porcelain 2>/dev/null | wc -l | tr -d " ")
M=$(git log -1 --format="%h %s" 2>/dev/null || echo "-")
printf -- "- %s | 分支 %s | 未提交 %s 个 | HEAD %s\n" "$TS" "$B" "$C" "$M" >> .claude/session_history.md
echo "[已记录] .claude/session_history.md"
