#!/usr/bin/env bash
# Update version in Cargo.toml workspace and constants.rs
#
# Usage: update-version.sh <version>
# Example: update-version.sh 0.4.4

set -euo pipefail

VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: update-version.sh <version>" >&2
  exit 1
fi

echo "Updating version to: $VERSION"

# Parse semver components
IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

# Always create VERSION file (used by semantic-release to detect if release happened)
echo "$VERSION" > VERSION
echo "  Updated: VERSION"

# Update workspace version in root Cargo.toml
if [[ -f "Cargo.toml" ]]; then
  awk -v ver="$VERSION" '
    /^\[workspace\.package\]/ { in_workspace=1 }
    /^\[/ && !/^\[workspace\.package\]/ { in_workspace=0 }
    in_workspace && /^version\s*=/ {
      sub(/^version\s*=\s*"[^"]*"/, "version = \"" ver "\"")
    }
    { print }
  ' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml
  echo "  Updated: Cargo.toml (workspace version)"
fi

# Update version constants in constants.rs
CONSTANTS_FILE="clickhouse-arrow/src/constants.rs"
if [[ -f "$CONSTANTS_FILE" ]]; then
  sed -i "s/pub(super) const VERSION_MAJOR: u64 = [0-9]*;/pub(super) const VERSION_MAJOR: u64 = $MAJOR;/" "$CONSTANTS_FILE"
  sed -i "s/pub(super) const VERSION_MINOR: u64 = [0-9]*;/pub(super) const VERSION_MINOR: u64 = $MINOR;/" "$CONSTANTS_FILE"
  sed -i "s/pub(super) const VERSION_PATCH: u64 = [0-9]*;/pub(super) const VERSION_PATCH: u64 = $PATCH;/" "$CONSTANTS_FILE"
  echo "  Updated: $CONSTANTS_FILE"
fi

# Update clickhouse-arrow-derive dependency version
CRATE_TOML="clickhouse-arrow/Cargo.toml"
if [[ -f "$CRATE_TOML" ]]; then
  sed -i "s/clickhouse-arrow-derive = { version = \"[^\"]*\"/clickhouse-arrow-derive = { version = \"$VERSION\"/" "$CRATE_TOML"
  echo "  Updated: $CRATE_TOML (derive dependency)"
fi

echo "Version update complete: $VERSION"
