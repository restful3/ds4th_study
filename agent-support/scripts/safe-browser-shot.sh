#!/bin/bash
set -Eeuo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

readonly PROGRAM_NAME="${0##*/}"
readonly MEMORY_HIGH="3G"
readonly MEMORY_MAX="4G"
readonly MEMORY_SWAP_MAX="0"
readonly TASKS_MAX="1024"
readonly CPU_QUOTA="300%"
readonly MIN_AVAILABLE_KIB=$((10 * 1024 * 1024))
readonly MAX_MEMORY_FULL_PSI_AVG10="1.00"
readonly UNIT_NAME="ds4th-safe-browser-shot"

url=""
output=""
width="1600"
height="900"
virtual_time_budget_ms="6000"
runtime_max_sec="120"
check_only=0

usage() {
  cat <<'EOF'
Usage:
  safe-browser-shot.sh --url URL --output /absolute/path.png [options]
  safe-browser-shot.sh --check

Options:
  --url URL                       http:// or https:// page to capture
  --output ABSOLUTE.png           final PNG path; published atomically
  --width PIXELS                  viewport width (320..4096; default 1600)
  --height PIXELS                 viewport height (180..32000; default 900)
  --virtual-time-budget MS        Chrome virtual-time budget (1..60000; default 6000)
  --timeout SECONDS               hard cgroup runtime (5..120; default 120)
  --check                         check dependencies, host headroom, and policy only
  -h, --help                      show this help

Safety policy (not configurable from the command line):
  one concurrent render; app.slice transient service; MemoryHigh=3G;
  MemoryMax=4G; MemorySwapMax=0; TasksMax=1024; CPUQuota=300%;
  KillMode=control-group; unique temporary profile; validated atomic PNG.

The command fails closed. It never falls back to an unguarded browser launch.
EOF
}

die() {
  printf '%s: %s\n' "$PROGRAM_NAME" "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_integer_range() {
  local label="$1"
  local value="$2"
  local minimum="$3"
  local maximum="$4"

  [[ "$value" =~ ^[0-9]+$ ]] ||
    die "$label must be an integer: $value"
  ((value >= minimum && value <= maximum)) ||
    die "$label must be between $minimum and $maximum: $value"
}

while (($#)); do
  case "$1" in
    --url)
      (($# >= 2)) || die "--url requires a value"
      url="$2"
      shift 2
      ;;
    --output)
      (($# >= 2)) || die "--output requires a value"
      output="$2"
      shift 2
      ;;
    --width)
      (($# >= 2)) || die "--width requires a value"
      width="$2"
      shift 2
      ;;
    --height)
      (($# >= 2)) || die "--height requires a value"
      height="$2"
      shift 2
      ;;
    --virtual-time-budget)
      (($# >= 2)) || die "--virtual-time-budget requires a value"
      virtual_time_budget_ms="$2"
      shift 2
      ;;
    --timeout)
      (($# >= 2)) || die "--timeout requires a value"
      runtime_max_sec="$2"
      shift 2
      ;;
    --check)
      check_only=1
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

require_integer_range "--width" "$width" 320 4096
require_integer_range "--height" "$height" 180 32000
require_integer_range "--virtual-time-budget" "$virtual_time_budget_ms" 1 60000
require_integer_range "--timeout" "$runtime_max_sec" 5 120

for command_name in awk basename chmod dirname flock getconf getent grep id \
  mktemp mv od python3 realpath rm sleep stat systemctl systemd-run tr; do
  require_command "$command_name"
done

systemctl --user show-environment >/dev/null 2>&1 ||
  die "systemd user manager is unavailable"

systemd-run --help 2>&1 | grep -q -- '--wait' ||
  die "systemd-run does not support --wait"

[[ -f /sys/fs/cgroup/cgroup.controllers ]] ||
  die "cgroup v2 unified hierarchy is unavailable"
grep -qw memory /sys/fs/cgroup/cgroup.controllers ||
  die "cgroup v2 memory controller is unavailable"

browser=""
for candidate in google-chrome-stable google-chrome chromium chromium-browser; do
  if candidate_path="$(command -v "$candidate" 2>/dev/null)"; then
    if [[ "$candidate_path" == /* ]]; then
      browser="$candidate_path"
    else
      browser="$(realpath -e -- "$candidate_path")"
    fi
    break
  fi
done
[[ -n "$browser" ]] || die "no supported Chrome/Chromium executable found"

current_uid="$(id -u)"
account_home="$(
  getent passwd "$current_uid" |
    awk -F: -v uid="$current_uid" '$3 == uid { print $6; exit }'
)"
[[ "$account_home" == /* && -d "$account_home" ]] ||
  die "could not resolve the current account home directory"
trusted_root="$account_home/.local/libexec/ds4th-safe-browser-shot"
expected_script="$trusted_root/safe-browser-shot.sh"
expected_guard="$trusted_root/safe_browser_guard.py"
script_path="$(realpath -e -- "$0")"
guard_path="$(realpath -e -- "$(dirname -- "$script_path")/safe_browser_guard.py")"
[[ "$script_path" == "$expected_script" ]] ||
  die "run only the owner-installed copy: $expected_script"
[[ "$guard_path" == "$expected_guard" ]] ||
  die "unexpected browser guard path: $guard_path"

for trusted_dir in \
  "$account_home" \
  "$account_home/.local" \
  "$account_home/.local/libexec" \
  "$trusted_root"; do
  [[ -d "$trusted_dir" && ! -L "$trusted_dir" ]] ||
    die "trusted directory is absent, not a directory, or a symlink: $trusted_dir"
  [[ "$(stat -c '%u' -- "$trusted_dir")" == "$current_uid" ]] ||
    die "trusted directory has an unexpected owner: $trusted_dir"
  trusted_mode="$(stat -c '%a' -- "$trusted_dir")"
  (( (8#$trusted_mode & 0022) == 0 )) ||
    die "trusted directory is writable by group or others: $trusted_dir"
done
[[ "$(stat -c '%a' -- "$account_home/.local")" == "700" ]] ||
  die "trusted .local directory must have mode 0700"
[[ "$(stat -c '%a' -- "$account_home/.local/libexec")" == "700" ]] ||
  die "trusted libexec directory must have mode 0700"
[[ "$(stat -c '%a' -- "$trusted_root")" == "700" ]] ||
  die "trusted runner directory must have mode 0700"

[[ -f "$script_path" && ! -L "$script_path" &&
  "$(stat -c '%u:%a' -- "$script_path")" == "$current_uid:700" ]] ||
  die "installed wrapper must be a current-user-owned regular file with mode 0700"
[[ -f "$guard_path" && ! -L "$guard_path" &&
  "$(stat -c '%u:%a' -- "$guard_path")" == "$current_uid:600" ]] ||
  die "installed guard must be a current-user-owned regular file with mode 0600"

python_runner="$(command -v python3)"
[[ "$python_runner" == /* ]] ||
  python_runner="$(realpath -e -- "$python_runner")"
[[ -f "$guard_path" && -r "$guard_path" ]] ||
  die "browser cgroup guard is unavailable: $guard_path"

runtime_root="/run/user/$(id -u)"
[[ -d "$runtime_root" && -O "$runtime_root" && -w "$runtime_root" ]] ||
  die "private writable runtime directory unavailable: $runtime_root"
runtime_root="$(realpath -e -- "$runtime_root")"

mem_available_kib="$(
  awk '$1 == "MemAvailable:" { print $2; exit }' /proc/meminfo
)"
[[ "$mem_available_kib" =~ ^[0-9]+$ ]] ||
  die "could not read MemAvailable from /proc/meminfo"
((mem_available_kib >= MIN_AVAILABLE_KIB)) ||
  die "insufficient host headroom: MemAvailable=${mem_available_kib}KiB, required>=${MIN_AVAILABLE_KIB}KiB"

[[ -r /proc/pressure/memory ]] ||
  die "memory PSI is unavailable"
memory_full_avg10="$(
  awk '
    $1 == "full" {
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^avg10=/) {
          split($i, value, "=")
          print value[2]
          exit
        }
      }
    }
  ' /proc/pressure/memory
)"
[[ "$memory_full_avg10" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "could not read memory full PSI avg10"
awk -v actual="$memory_full_avg10" -v maximum="$MAX_MEMORY_FULL_PSI_AVG10" \
  'BEGIN { exit !(actual <= maximum) }' ||
  die "host is under memory pressure: full PSI avg10=${memory_full_avg10}% (max ${MAX_MEMORY_FULL_PSI_AVG10}%)"

cpu_count="$(getconf _NPROCESSORS_ONLN)"
[[ "$cpu_count" =~ ^[1-9][0-9]*$ ]] ||
  die "could not determine online CPU count"
load_one="$(awk '{ print $1 }' /proc/loadavg)"
[[ "$load_one" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "could not read load average"
max_load=$((cpu_count * 2))
awk -v actual="$load_one" -v maximum="$max_load" \
  'BEGIN { exit !(actual <= maximum) }' ||
  die "host load is too high: load1=$load_one (max $max_load)"

swap_total_kib="$(awk '$1 == "SwapTotal:" { print $2; exit }' /proc/meminfo)"
swap_free_kib="$(awk '$1 == "SwapFree:" { print $2; exit }' /proc/meminfo)"
if [[ "$swap_total_kib" =~ ^[0-9]+$ && "$swap_free_kib" =~ ^[0-9]+$ ]] &&
  ((swap_total_kib > 0 && swap_free_kib < 512 * 1024)); then
  printf '%s: warning: host swap is nearly full (%sKiB free); workload swap remains disabled\n' \
    "$PROGRAM_NAME" "$swap_free_kib" >&2
fi

if ((check_only)); then
  cat <<EOF
status=ready
browser=$browser
runtime_root=$runtime_root
mem_available_kib=$mem_available_kib
memory_full_psi_avg10=$memory_full_avg10
load1=$load_one
load1_max=$max_load
memory_high=$MEMORY_HIGH
memory_max=$MEMORY_MAX
memory_swap_max=$MEMORY_SWAP_MAX
tasks_max=$TASKS_MAX
cpu_quota=$CPU_QUOTA
runtime_max_sec=$runtime_max_sec
EOF
  exit 0
fi

[[ -n "$url" ]] || die "--url is required"
[[ "$url" =~ ^https?:// ]] ||
  die "--url must use http:// or https://"
[[ -n "$output" ]] || die "--output is required"
[[ "$output" == /* ]] || die "--output must be an absolute path"
[[ "${output,,}" == *.png ]] || die "--output must end in .png"

output_leaf="$(basename -- "$output")"
output_parent="$(realpath -e -- "$(dirname -- "$output")")"
[[ -n "$output_leaf" && "$output_leaf" != "." && "$output_leaf" != ".." ]] ||
  die "invalid output filename"
[[ "${output_leaf,,}" == *.png ]] || die "--output must end in .png"
[[ -d "$output_parent" && -w "$output_parent" ]] ||
  die "output directory must already exist and be writable: $output_parent"
output="$output_parent/$output_leaf"
[[ ! -L "$output" ]] || die "output path must not be a symbolic link"
[[ ! -e "$output" || -f "$output" ]] ||
  die "existing output must be a regular file"

lock_path="$runtime_root/ds4th-safe-browser-shot.lock"
exec {lock_fd}>"$lock_path"
if ! flock -n "$lock_fd"; then
  die "another guarded browser render is already running"
fi

profile_dir=""
publish_dir=""
unit_name="$UNIT_NAME"
unit_may_exist=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP

  if ((unit_may_exist)) && [[ "$unit_name" == "$UNIT_NAME" ]]; then
    systemctl --user stop "${unit_name}.service" >/dev/null 2>&1 || true
  fi

  if [[ -n "$profile_dir" && "$profile_dir" == "$runtime_root"/ds4th-browser-profile.* &&
    -d "$profile_dir" ]]; then
    rm -rf -- "$profile_dir"
  fi

  if [[ -n "$publish_dir" && "$publish_dir" == "$output_parent"/.ds4th-browser-shot.* &&
    -d "$publish_dir" ]]; then
    rm -rf -- "$publish_dir"
  fi

  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

profile_dir="$(mktemp -d "$runtime_root/ds4th-browser-profile.XXXXXX")"
publish_dir="$(mktemp -d "$output_parent/.ds4th-browser-shot.XXXXXX")"
temporary_png="$publish_dir/render.png"

unit_may_exist=1
run_status=0
systemd-run \
  --user \
  --unit="$unit_name" \
  --description="Guarded ds4th browser screenshot" \
  --slice=app.slice \
  --service-type=exec \
  --wait \
  --collect \
  --pipe \
  --quiet \
  --property="MemoryHigh=$MEMORY_HIGH" \
  --property="MemoryMax=$MEMORY_MAX" \
  --property="MemorySwapMax=$MEMORY_SWAP_MAX" \
  --property="TasksMax=$TASKS_MAX" \
  --property="CPUQuota=$CPU_QUOTA" \
  --property="RuntimeMaxSec=${runtime_max_sec}s" \
  --property="TimeoutStopSec=5s" \
  --property="KillMode=control-group" \
  --property="OOMPolicy=kill" \
  --property="SendSIGKILL=yes" \
  --property="UMask=0077" \
  --property="Nice=5" \
  --setenv="PYTHONDONTWRITEBYTECODE=1" \
  "$python_runner" \
  "$guard_path" \
  --browser "$browser" \
  --profile "$profile_dir" \
  --output "$temporary_png" \
  --viewport "$width,$height" \
  --virtual-time-budget "$virtual_time_budget_ms" \
  --url "$url" ||
  run_status=$?
unit_may_exist=0

((run_status == 0)) ||
  die "guarded browser service failed with exit status $run_status"
[[ -s "$temporary_png" ]] ||
  die "browser exited without a non-empty PNG"

png_signature="$(
  od -An -N8 -tx1 "$temporary_png" | tr -d '[:space:]'
)"
[[ "$png_signature" == "89504e470d0a1a0a" ]] ||
  die "browser output does not have a valid PNG signature"

chmod 0600 "$temporary_png"
mv -fT -- "$temporary_png" "$output"
printf '%s\n' "$output"
