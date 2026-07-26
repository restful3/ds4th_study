#!/bin/bash
# Install the guarded browser runner owner-only, and detect source/install drift.
set -Eeuo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

readonly PROGRAM_NAME="${0##*/}"
readonly RUNNER_DIR_NAME="ds4th-safe-browser-shot"
readonly WRAPPER_NAME="safe-browser-shot.sh"
readonly GUARD_NAME="safe_browser_guard.py"
readonly MANIFEST_NAME="manifest.sha256"
readonly WRAPPER_MODE="700"
readonly GUARD_MODE="600"

mode="install"
prefix=""

usage() {
  cat <<'EOF'
Usage:
  install-safe-browser-shot.sh [--prefix DIR]
  install-safe-browser-shot.sh --verify [--prefix DIR]

The runner refuses to execute from the repository, so the repository copy and
the installed copy can drift apart silently. Install through this script and
verify with --verify instead of comparing hashes by hand.

Options:
  --prefix DIR   parent of the runner directory (default: $HOME/.local/libexec)
  --verify       compare repository source, recorded manifest and installed
                 bytes; exit non-zero on any mismatch
  -h, --help     show this help

--verify reports two distinct problems:
  stale    the repository source changed and was never reinstalled
  altered  an installed file no longer matches the manifest recorded at install
EOF
}

die() {
  printf '%s: %s\n' "$PROGRAM_NAME" "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --prefix)
      (($# >= 2)) || die "--prefix requires a value"
      prefix="$2"
      shift 2
      ;;
    --verify)
      mode="verify"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

for command_name in chmod cp date dirname install mktemp realpath rm sha256sum stat; do
  command -v "$command_name" >/dev/null 2>&1 ||
    die "required command not found: $command_name"
done

script_path="$(realpath -e -- "$0")"
source_dir="$(dirname -- "$script_path")"
source_wrapper="$source_dir/$WRAPPER_NAME"
source_guard="$source_dir/$GUARD_NAME"
[[ -f "$source_wrapper" ]] || die "missing source file: $source_wrapper"
[[ -f "$source_guard" ]] || die "missing source file: $source_guard"

using_default_prefix=0
if [[ -z "$prefix" ]]; then
  [[ -n "${HOME:-}" && -d "$HOME" ]] || die "HOME is not a usable directory"
  prefix="$HOME/.local/libexec"
  using_default_prefix=1
fi

target="$prefix/$RUNNER_DIR_NAME"
installed_wrapper="$target/$WRAPPER_NAME"
installed_guard="$target/$GUARD_NAME"
manifest="$target/$MANIFEST_NAME"

hash_of() {
  sha256sum -- "$1" | awk '{ print $1 }'
}

manifest_hash_of() {
  awk -v name="$1" '$2 == name || $2 == "*" name { print $1; exit }' "$manifest"
}

if [[ "$mode" == "verify" ]]; then
  [[ -d "$target" ]] || die "nothing is installed at $target"
  [[ -f "$manifest" ]] || die "missing manifest: $manifest (reinstall to recreate it)"

  problems=0
  for name in "$WRAPPER_NAME" "$GUARD_NAME"; do
    source_file="$source_dir/$name"
    installed_file="$target/$name"

    if [[ ! -f "$installed_file" ]]; then
      printf '%s: missing installed file: %s\n' "$PROGRAM_NAME" "$installed_file" >&2
      problems=$((problems + 1))
      continue
    fi

    recorded="$(manifest_hash_of "$name")"
    if [[ -z "$recorded" ]]; then
      printf '%s: %s is not recorded in the manifest\n' "$PROGRAM_NAME" "$name" >&2
      problems=$((problems + 1))
      continue
    fi

    installed_hash="$(hash_of "$installed_file")"
    if [[ "$installed_hash" != "$recorded" ]]; then
      printf '%s: altered %s: installed %s, manifest %s\n' \
        "$PROGRAM_NAME" "$name" "${installed_hash:0:16}" "${recorded:0:16}" >&2
      problems=$((problems + 1))
      continue
    fi

    source_hash="$(hash_of "$source_file")"
    if [[ "$source_hash" != "$recorded" ]]; then
      printf '%s: stale %s: source %s, installed %s\n' \
        "$PROGRAM_NAME" "$name" "${source_hash:0:16}" "${recorded:0:16}" >&2
      problems=$((problems + 1))
      continue
    fi

    printf 'ok %s %s\n' "$name" "${recorded:0:16}"
  done

  ((problems == 0)) ||
    die "$problems file(s) drifted; reinstall with $PROGRAM_NAME"
  printf 'status=verified\ntarget=%s\n' "$target"
  exit 0
fi

if ((using_default_prefix)); then
  install -d -m 700 -- "$HOME/.local"
  chmod 700 -- "$HOME/.local"
fi
install -d -m 700 -- "$prefix"
chmod 700 -- "$prefix"
install -d -m 700 -- "$target"
chmod 700 -- "$target"

backup=""
if [[ -f "$installed_wrapper" || -f "$installed_guard" ]]; then
  backup="$(mktemp -d "$target/rollback-$(date +%Y%m%dT%H%M%S)-XXXXXX")"
  chmod 700 -- "$backup"
  for name in "$WRAPPER_NAME" "$GUARD_NAME" "$MANIFEST_NAME"; do
    [[ -f "$target/$name" ]] || continue
    cp -p -- "$target/$name" "$backup/$name"
  done
fi

install -m "$WRAPPER_MODE" -- "$source_wrapper" "$installed_wrapper"
install -m "$GUARD_MODE" -- "$source_guard" "$installed_guard"

wrapper_hash="$(hash_of "$installed_wrapper")"
guard_hash="$(hash_of "$installed_guard")"
[[ "$wrapper_hash" == "$(hash_of "$source_wrapper")" ]] ||
  die "installed wrapper does not match the source"
[[ "$guard_hash" == "$(hash_of "$source_guard")" ]] ||
  die "installed guard does not match the source"

manifest_tmp="$(mktemp "$target/.manifest.XXXXXX")"
printf '%s  %s\n%s  %s\n' \
  "$wrapper_hash" "$WRAPPER_NAME" \
  "$guard_hash" "$GUARD_NAME" >"$manifest_tmp"
chmod 600 -- "$manifest_tmp"
mv -fT -- "$manifest_tmp" "$manifest"

printf 'status=installed\ntarget=%s\nwrapper=%s\nguard=%s\n' \
  "$target" "${wrapper_hash:0:16}" "${guard_hash:0:16}"
[[ -z "$backup" ]] || printf 'backup=%s\n' "$backup"
