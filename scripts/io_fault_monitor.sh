#!/usr/bin/env bash
# io_fault_monitor.sh
# 在大量下载/高 IO 时监控 NVMe、EXT4 只读、SSH 可用性。
# 日志写在 /mnt/data1（独立 HDD /dev/sda1），系统盘变只读后仍可能保留诊断信息。
#
# 用法:
#   sudo ./io_fault_monitor.sh start              # 启动监控
#   sudo ./io_fault_monitor.sh mark "开始下载"     # 标记事件（如下载开始）
#   sudo ./io_fault_monitor.sh status             # 查看当前状态
#   sudo ./io_fault_monitor.sh stop               # 停止监控
#   sudo ./io_fault_monitor.sh analyze            # 分析最近一次运行

set -u

BASE="${IO_MONITOR_BASE:-/mnt/data1/crash_watch}"
PID_FILE="$BASE/io_monitor.pid"
LATEST_RUN="$BASE/io_monitor_latest.txt"

INTERVAL="${IO_MONITOR_INTERVAL:-10}"
SMART_INTERVAL="${IO_MONITOR_SMART_INTERVAL:-120}"

KERNEL_PATTERN='I/O error|Buffer I/O error|EXT4-fs.*(error|warning|Remounting)|read-only|Journal has aborted|lost sync page write|nvme.*(timeout|reset|abort|error|not ready)|Device not ready|blk_update_request|AER:|PCIe Bus Error|machine check|mce:|EDAC|hung task|soft lockup|hard LOCKUP|watchdog|Out of memory|oom-kill'

RUN_DIR=""

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_DIR/monitor.log"
}

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    if [ "${IO_MONITOR_ALLOW_USER:-0}" = "1" ]; then
      echo "警告: 非 root 模式，部分 SMART/内核日志可能不完整" >&2
      return 0
    fi
    echo "请使用 root 运行: sudo $0 $*" >&2
    exit 1
  fi
}

is_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

root_mount_mode() {
  awk '$2=="/"{print $4}' /proc/mounts | cut -d, -f1
}

root_is_readonly() {
  root_mount_mode | grep -q 'ro'
}

sshd_listening() {
  ss -tlnp 2>/dev/null | grep -q ':22 '
}

write_probe() {
  local f="/tmp/.io_monitor_probe_$$"
  timeout 3s touch "$f" 2>/dev/null && rm -f "$f" 2>/dev/null
}

get_smart_field() {
  local field="$1"
  smartctl -A -d nvme /dev/nvme0n1 2>/dev/null | awk -F: -v k="$field" '
    $1 ~ k { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }
  '
}

safe_run() {
  local title="$1"
  local cmd="$2"
  echo "===== $title ====="
  timeout 30s bash -c "$cmd" 2>&1 || echo "[timeout or error: $title]"
  echo
}

take_snapshot() {
  local tag="$1"
  local reason="${2:-manual}"
  local ts snap

  ts="$(date +%F_%H%M%S)"
  snap="$RUN_DIR/snap_${ts}_${tag}"
  mkdir -p "$snap"

  log "SNAPSHOT [$tag] reason=$reason -> $snap"

  {
    echo "tag=$tag"
    echo "reason=$reason"
    echo "time=$(date -Is)"
    echo "root_mount=$(root_mount_mode)"
    echo "root_readonly=$(root_is_readonly && echo yes || echo no)"
    echo "sshd_listening=$(sshd_listening && echo yes || echo no)"
    echo "write_probe=$(write_probe && echo ok || echo fail)"
  } > "$snap/meta.txt"

  {
    safe_run "date/uptime" "date; uptime; uname -a"
    safe_run "cmdline" "cat /proc/cmdline"
    safe_run "mounts" "cat /proc/mounts"
    safe_run "mount root detail" "mount | grep -E ' on / |nvme'"
    safe_run "df" "df -h; echo '---'; df -i / /mnt/data1 2>/dev/null"
    safe_run "free/vmstat" "free -h; vmstat 1 3"
    safe_run "load top cpu" "ps -eo pid,user,stat,pcpu,pmem,comm,args --sort=-pcpu | head -40"
    safe_run "load top mem" "ps -eo pid,user,stat,pcpu,pmem,comm,args --sort=-pmem | head -40"
    safe_run "io top" "command -v pidstat >/dev/null && pidstat -d 1 3 || ps -eo pid,comm,args | grep -iE 'modelscope|wget|curl|python|download' | head -20"
  } > "$snap/system.txt" 2>&1

  {
    safe_run "sshd status" "systemctl status ssh sshd --no-pager 2>&1; ss -tlnp | grep -E ':22|sshd'"
    safe_run "ssh auth recent" "grep -i sshd /var/log/auth.log 2>/dev/null | tail -30"
    safe_run "systemd failed" "systemctl --failed --no-pager 2>/dev/null"
    safe_run "journal write test" "journalctl --disk-usage 2>/dev/null; ls -la /var/log/journal 2>/dev/null | head -5"
  } > "$snap/ssh_services.txt" 2>&1

  {
    safe_run "diskstats" "cat /proc/diskstats | grep -E 'nvme|sda'"
    safe_run "iostat" "command -v iostat >/dev/null && iostat -xz 1 3 || echo no_iostat"
    safe_run "nvme list" "nvme list 2>/dev/null"
    safe_run "nvme smart-log" "nvme smart-log -H /dev/nvme0n1 2>/dev/null"
    safe_run "nvme error-log" "nvme error-log /dev/nvme0n1 2>/dev/null"
    safe_run "smartctl full" "smartctl -a -d nvme /dev/nvme0n1 2>/dev/null"
    safe_run "pcie nvme link" "lspci -vvv -s 81:00.0 2>/dev/null | grep -iE 'LnkCap|LnkSta|Err|AER|CESta|UESta'"
  } > "$snap/disk.txt" 2>&1

  {
    safe_run "dmesg tail" "dmesg -T 2>/dev/null | tail -300"
    safe_run "journal kernel 30min" "journalctl -k --since '30 min ago' --no-pager 2>/dev/null | tail -500"
    safe_run "journal kernel keywords" "journalctl -k --since '2 hours ago' --no-pager 2>/dev/null | grep -iE \"$KERNEL_PATTERN\" | tail -200"
    safe_run "syslog keywords" "grep -iE \"$KERNEL_PATTERN\" /var/log/syslog /var/log/kern.log 2>/dev/null | tail -100"
  } > "$snap/kernel.txt" 2>&1

  echo "$ts $tag $reason" >> "$RUN_DIR/alerts.log"
  sync
}

watch_kernel() {
  local last_alert=0 now

  journalctl -k -f --no-pager 2>/dev/null | while read -r line; do
    echo "[$(date '+%F %T')] $line" >> "$RUN_DIR/kernel_live.log"
    if echo "$line" | grep -iE "$KERNEL_PATTERN" >/dev/null 2>&1; then
      now=$(date +%s)
      log "KERNEL_ALERT: $line"
      if [ $((now - last_alert)) -ge 30 ]; then
        last_alert=$now
        take_snapshot "KERNEL_ALERT" "$line"
      fi
    fi
  done
}

watch_dmesg() {
  dmesg -wT 2>/dev/null | while read -r line; do
    echo "[$(date '+%F %T')] $line" >> "$RUN_DIR/dmesg_live.log"
  done
}

periodic_check() {
  local last_smart=0 last_ro_alert=0 last_ssh_alert=0 last_write_alert=0
  local prev_unsafe="" prev_media="" now

  prev_unsafe="$(get_smart_field 'Unsafe Shutdowns' || echo 0)"
  prev_media="$(get_smart_field 'Media and Data Integrity Errors' || echo 0)"
  log "SMART baseline: unsafe_shutdowns=$prev_unsafe media_errors=$prev_media"

  while true; do
    now=$(date +%s)

    {
      echo -n "[$(date '+%F %T')]"
      echo -n " root=$(root_mount_mode)"
      echo -n " sshd=$(sshd_listening && echo up || echo down)"
      echo -n " write=$(write_probe && echo ok || echo FAIL)"
      uptime | awk -F'load average:' '{print " load="$2}'
    } >> "$RUN_DIR/heartbeat.log"

    if root_is_readonly; then
      log "CRITICAL: root filesystem is READ-ONLY"
      if [ $((now - last_ro_alert)) -ge 15 ]; then
        last_ro_alert=$now
        take_snapshot "ROOT_READONLY" "ext4 remounted ro"
      fi
    fi

    if ! write_probe; then
      log "CRITICAL: write probe to /tmp failed"
      if [ $((now - last_write_alert)) -ge 15 ]; then
        last_write_alert=$now
        take_snapshot "WRITE_FAIL" "cannot write to /tmp"
      fi
    fi

    if ! sshd_listening; then
      log "CRITICAL: sshd not listening on port 22"
      if [ $((now - last_ssh_alert)) -ge 30 ]; then
        last_ssh_alert=$now
        take_snapshot "SSHD_DOWN" "port 22 not listening"
      fi
    fi

    if [ $((now - last_smart)) -ge "$SMART_INTERVAL" ]; then
      last_smart=$now
      local cur_unsafe cur_media
      cur_unsafe="$(get_smart_field 'Unsafe Shutdowns' || echo 0)"
      cur_media="$(get_smart_field 'Media and Data Integrity Errors' || echo 0)"
      log "SMART: unsafe=$cur_unsafe media=$cur_media temp=$(get_smart_field 'Temperature' || echo ?)"
      if [ "$cur_unsafe" != "$prev_unsafe" ] || [ "$cur_media" != "$prev_media" ]; then
        log "SMART_CHANGED: unsafe $prev_unsafe->$cur_unsafe, media $prev_media->$cur_media"
        take_snapshot "SMART_CHANGE" "unsafe=$cur_unsafe media=$cur_media"
        prev_unsafe="$cur_unsafe"
        prev_media="$cur_media"
      fi
    fi

    {
      echo "===== $(date '+%F %T') ====="
      cat /proc/diskstats | grep -E 'nvme0|sda ' || true
      command -v iostat >/dev/null && iostat -x /dev/nvme0n1 /dev/sda 1 1 2>/dev/null || true
      df -h / /mnt/data1 2>/dev/null
      echo
    } >> "$RUN_DIR/io_sample.log"

    sleep "$INTERVAL"
  done
}

cmd_start() {
  need_root start
  if is_running; then
    echo "监控已在运行, PID=$(cat "$PID_FILE")"
    echo "运行目录: $(cat "$LATEST_RUN" 2>/dev/null || echo unknown)"
    exit 0
  fi

  RUN_DIR="$BASE/io_run_$(date +%F_%H%M%S)"
  mkdir -p "$RUN_DIR"
  echo "$RUN_DIR" > "$LATEST_RUN"

  log "===== io_fault_monitor START ====="
  log "RUN_DIR=$RUN_DIR interval=${INTERVAL}s smart_interval=${SMART_INTERVAL}s"
  log "kernel=$(uname -r) root_disk=$(lsblk -no PKNAME /dev/nvme0n1p2 2>/dev/null)"

  take_snapshot "START" "monitor started"

  watch_kernel &
  echo $! > "$RUN_DIR/pid_kernel.log"
  watch_dmesg &
  echo $! > "$RUN_DIR/pid_dmesg.log"

  periodic_check &
  local periodic_pid=$!
  echo "$periodic_pid" > "$RUN_DIR/pid_periodic.log"
  echo "$periodic_pid" > "$PID_FILE"
  echo "$RUN_DIR" >> "$BASE/io_monitor_history.txt"

  log "PIDs: kernel=$(cat "$RUN_DIR/pid_kernel.log") dmesg=$(cat "$RUN_DIR/pid_dmesg.log") periodic=$periodic_pid"
  log "监控已启动。下载前请执行: sudo $0 mark '开始下载 xxx'"

  wait "$periodic_pid"
}

cmd_stop() {
  need_root stop
  if ! is_running; then
    echo "监控未在运行"
    rm -f "$PID_FILE"
    exit 0
  fi

  local main_pid run_dir
  main_pid="$(cat "$PID_FILE")"
  run_dir="$(cat "$LATEST_RUN" 2>/dev/null || echo "")"

  if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    RUN_DIR="$run_dir"
    for f in "$run_dir"/pid_*.log; do
      [ -f "$f" ] || continue
      kill "$(cat "$f")" 2>/dev/null || true
    done
    echo "[$(date '+%F %T')] ===== io_fault_monitor STOP =====" >> "$run_dir/monitor.log"
    take_snapshot "STOP" "monitor stopped" 2>/dev/null || true
  fi

  kill "$main_pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "监控已停止 (was PID=$main_pid)"
  [ -n "$run_dir" ] && echo "日志目录: $run_dir" && echo "分析: sudo $0 analyze"
}

cmd_mark() {
  need_root mark
  local msg="${*:2}"
  [ -z "$msg" ] && msg="manual mark"
  local run_dir
  run_dir="$(cat "$LATEST_RUN" 2>/dev/null || echo "")"
  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    echo "没有正在进行的监控运行。请先: sudo $0 start" >&2
    exit 1
  fi
  RUN_DIR="$run_dir"
  log "USER_MARK: $msg"
  echo "[$(date '+%F %T')] $msg" >> "$RUN_DIR/markers.log"
  take_snapshot "USER_MARK" "$msg"
  echo "已标记: $msg"
}

cmd_status() {
  local run_dir
  echo "=== io_fault_monitor 状态 ==="
  if is_running; then
    echo "状态: 运行中 (PID=$(cat "$PID_FILE"))"
  else
    echo "状态: 未运行"
  fi
  run_dir="$(cat "$LATEST_RUN" 2>/dev/null || echo "")"
  echo "最近运行目录: ${run_dir:-无}"
  echo ""
  echo "=== 当前系统 ==="
  echo "根分区挂载: $(root_mount_mode)"
  echo "根分区只读: $(root_is_readonly && echo 是 || echo 否)"
  echo "sshd :22:    $(sshd_listening && echo 监听中 || echo 未监听)"
  echo "写探测 /tmp: $(write_probe && echo 正常 || echo 失败)"
  echo ""
  if command -v smartctl >/dev/null; then
    echo "NVMe SMART 摘要:"
    smartctl -A -d nvme /dev/nvme0n1 2>/dev/null | grep -iE 'overall-health|Temperature|Available Spare|Percentage Used|Unsafe Shutdowns|Media and Data|Error Information'
  fi
  if [ -n "$run_dir" ] && [ -f "$run_dir/heartbeat.log" ]; then
    echo ""
    echo "最近心跳:"
    tail -3 "$run_dir/heartbeat.log"
  fi
  if [ -n "$run_dir" ] && [ -f "$run_dir/alerts.log" ]; then
    echo ""
    echo "告警记录:"
    cat "$run_dir/alerts.log"
  fi
}

cmd_analyze() {
  local run_dir
  run_dir="$(cat "$LATEST_RUN" 2>/dev/null || echo "")"
  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    echo "找不到最近运行目录" >&2
    exit 1
  fi

  echo "===== 故障分析报告 ====="
  echo "运行目录: $run_dir"
  echo ""

  if [ -f "$run_dir/alerts.log" ]; then
    echo "--- 触发的告警快照 ---"
    cat "$run_dir/alerts.log"
    echo ""
  else
    echo "--- 无告警快照（可能硬死机未来得及记录）---"
    echo ""
  fi

  if [ -f "$run_dir/markers.log" ]; then
    echo "--- 用户标记 ---"
    cat "$run_dir/markers.log"
    echo ""
  fi

  echo "--- 内核关键词 (live) ---"
  if [ -f "$run_dir/kernel_live.log" ]; then
    grep -iE "$KERNEL_PATTERN" "$run_dir/kernel_live.log" | tail -30 || echo "(无匹配)"
  else
    echo "(无 kernel_live.log)"
  fi
  echo ""

  echo "--- 根分区只读证据 ---"
  grep -h -iE 'read-only|Remounting|EXT4-fs.*error|error -5|I/O error|nvme.*timeout|Device not ready' \
    "$run_dir"/kernel_live.log "$run_dir"/dmesg_live.log "$run_dir"/kernel.txt 2>/dev/null | tail -20 || echo "(无)"
  echo ""

  echo "--- 心跳末尾（最后存活时间）---"
  tail -5 "$run_dir/heartbeat.log" 2>/dev/null || echo "(无)"
  echo ""

  echo "--- 建议判读 ---"
  if grep -qi 'ROOT_READONLY\|Remounting filesystem read-only' "$run_dir"/alerts.log "$run_dir"/kernel_live.log 2>/dev/null; then
    echo "=> 根因倾向: EXT4 因底层 I/O 错误被强制 remount-ro"
  fi
  if grep -qi 'nvme.*timeout\|Device not ready\|I/O Error' "$run_dir"/kernel_live.log "$run_dir"/dmesg_live.log 2>/dev/null; then
    echo "=> 根因倾向: NVMe 控制器/磁盘 I/O 超时或不可用"
  fi
  if grep -qi 'SSHD_DOWN' "$run_dir"/alerts.log 2>/dev/null; then
    echo "=> SSH 不可用: sshd 未监听（可能是系统盘只读导致服务无法启动）"
  fi
  if [ ! -f "$run_dir/alerts.log" ] || [ ! -s "$run_dir/alerts.log" ]; then
    echo "=> 若 SSH 断开但无告警: 可能是硬死机/网络断开，查看 heartbeat 最后时间戳"
  fi
  echo ""
  echo "详细快照目录: $run_dir/snap_*"
  echo "完整日志: $run_dir/monitor.log"
}

main() {
  local cmd="${1:-help}"
  case "$cmd" in
    start)   shift; cmd_start "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    mark)    shift; cmd_mark "$@" ;;
    status)  shift; cmd_status "$@" ;;
    analyze) shift; cmd_analyze "$@" ;;
    help|-h|--help)
      head -12 "$0" | tail -10
      echo ""
      echo "推荐流程:"
      echo "  1. sudo $0 start"
      echo "  2. sudo $0 mark '开始 modelscope download ...'"
      echo "  3. 执行下载命令"
      echo "  4. 出问题时（或恢复后）sudo $0 analyze"
      echo "  5. sudo $0 stop"
      ;;
    *)
      echo "未知命令: $cmd (使用 help)" >&2
      exit 1
      ;;
  esac
}

main "$@"
