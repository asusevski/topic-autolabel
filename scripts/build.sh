#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
source "$SCRIPT_DIR/utils.sh"

# Move to project root to ensure proper build context
cd "$PROJECT_ROOT"

# Clean up any previous builds
log "Cleaning up previous builds..."
rm -rf dist/ build/ *.egg-info/

# Run pre-commit checks
log "Running pre-commit checks..."
"$SCRIPT_DIR/precommit.sh"

# Build the package
log "Building package..."
python -m build

log "Build complete! Check the dist/ directory for your package files."