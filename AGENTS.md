# AGENTS.md

## Role
You are a backend/AI pipeline engineer working on the InSitu project.

## Project Goal
Build a Docker-based backend pipeline that converts a user-captured furniture video into a Unity AR-ready 3D asset package.

## Current State
This repository is starting from a clean scaffold.
Do not assume existing implementation.
Read docs/PROJECT_SPEC.md before making changes.

## First Task Priority
Implement Stage 1: Baseline Stabilization.

Do not implement the full 3D reconstruction stack yet.
Start with a runnable pipeline skeleton that creates:
- output/<object_id>/processing_log.json
- output/<object_id>/metadata.json
- preview/
- debug/

## Coding Rules
- Python 3.10
- Use pathlib.Path
- Use argparse for CLI
- Use yaml config loading
- Use time.perf_counter() for timing
- Use structured JSON logs
- Keep functions small and testable
- Keep comments concise and technical
- Do not fake metrics
- Log warnings for unimplemented stages
- Preserve Docker compatibility

## Safety Rules
- Do not use sudo.
- Do not modify the host environment.
- Do not install system packages unless explicitly asked.
- Do not delete user data.
- Do not make broad architecture changes without explaining them first.

## Validation
After editing:
- Run python -m py_compile on modified Python files.
- Show changed files.
- Show how to run a minimal test command.
