#!/bin/bash
# Wrapper script to run openclaw from the repo via pnpm
# Uses isolated dev environment to avoid interfering with main setup

# Set isolated state directory for dev/testing
export OPENCLAW_STATE_DIR="$HOME/.openclaw/workspace/skills-marketplace/.dev-env"

# Create the directory if it doesn't exist
mkdir -p "$OPENCLAW_STATE_DIR"

cd ~/openclaw
exec pnpm openclaw "$@"
