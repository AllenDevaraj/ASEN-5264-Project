#!/usr/bin/env bash
# Run the v2 venv Python isolated from ROS: the shell's ROS PYTHONPATH would otherwise leak in.
exec env -u PYTHONPATH "$(dirname "$0")/.venv/bin/python" "$@"
