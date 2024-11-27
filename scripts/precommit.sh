#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/utils.sh"

log "Running black..."
black src/

log "Running ruff..."
ruff check . --fix

log "Running isort..."
isort .

log "Running pyright..."
pyright .