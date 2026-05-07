#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — Automated compression benchmark
#
# Usage:
#   ./benchmark.sh [OPTIONS]
#
# Options:
#   -d DIR        Data directory (default: data)
#   -f FILES      Comma-separated file names to test (default: A,B,C,D,E,F,G,H)
#   -c            Test on concatenated file (all files joined — matches prof table)
#   -m            Per-file aggregate mode: compress each file individually, then
#                 show one summary table with total sizes and summed times.
#                 bits/byte = total_compressed_bits / total_original_bytes.
#                 Lossless = YES only if every file round-tripped correctly.
#   -o TOOL_CMD   Add your own tool. Format: "name:compress_cmd:decompress_cmd"
#                 Use %i/%o placeholders for file-argument tools:
#                   -o "myc:./compress %i %o:./decompress %i %o"
#                 Without placeholders, stdin/stdout is used (gzip-style -c tools).
#   -r RUNS       Number of timing runs per compressor (default: 1)
#   -q            Quiet — suppress progress output, only show final table
#   -h            Show this help
#
# Examples:
#   ./benchmark.sh                          # test all files individually (one table each)
#   ./benchmark.sh -c                       # concatenate A-H into one file, one table
#   ./benchmark.sh -m                       # per-file runs, single aggregate summary table
#   ./benchmark.sh -m -o "ox:./compress %i %o:./decompress %i %o"
#   ./benchmark.sh -f C,D -r 3             # test files C and D, 3 timing runs
# =============================================================================

set -euo pipefail

# ---------- defaults ---------------------------------------------------------
DATA_DIR="data"
FILES_ARG="A,B,C,D,E,F,G,H"
CONCAT_MODE=false
MULTI_MODE=false
OWN_TOOLS=()
RUNS=1
QUIET=false

# ---------- ANSI colours -----------------------------------------------------
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
B='\033[0;34m'; C='\033[0;36m'; W='\033[1;37m'; N='\033[0m'

log()  { $QUIET || printf "${C}[bench]${N} %s\n" "$*"; }
warn() { printf "${Y}[warn] ${N}%s\n" "$*" >&2; }
err()  { printf "${R}[error]${N} %s\n" "$*" >&2; exit 1; }

# ---------- arg parsing ------------------------------------------------------
while getopts "d:f:cmo:r:qh" opt; do
    case $opt in
        d) DATA_DIR="$OPTARG" ;;
        f) FILES_ARG="$OPTARG" ;;
        c) CONCAT_MODE=true ;;
        m) MULTI_MODE=true ;;
        o) OWN_TOOLS+=("$OPTARG") ;;
        r) RUNS="$OPTARG" ;;
        q) QUIET=true ;;
        h) sed -n '2,32p' "$0"; exit 0 ;;
        *) err "Unknown option -$OPTARG. Use -h for help." ;;
    esac
done

# ---------- dependencies check -----------------------------------------------
for bin in gzip bzip2 xz zstd bc md5sum; do
    command -v "$bin" &>/dev/null || err "Required tool not found: $bin"
done
# /usr/bin/time (not the shell builtin) needed for -f flag
TIME_CMD="/usr/bin/time"
[[ -x "$TIME_CMD" ]] || err "/usr/bin/time not found (needed for timing)"

# ---------- compressor definitions -------------------------------------------
# Arrays: NAMES, COMP_CMDS, DECOMP_CMDS
NAMES=()
COMP_CMDS=()
DECOMP_CMDS=()

add_compressor() {
    NAMES+=("$1")
    COMP_CMDS+=("$2")
    DECOMP_CMDS+=("$3")
}

add_compressor "gzip"     "gzip -6 -c"          "gzip -d -c"
add_compressor "bzip2"    "bzip2 -9 -c"          "bzip2 -d -c"
add_compressor "lzma-1"   "xz -1 --format=lzma -c" "xz -d -c"
add_compressor "lzma-5"   "xz -5 --format=lzma -c" "xz -d -c"
add_compressor "lzma-9"   "xz -9 --format=lzma -c" "xz -d -c"
add_compressor "xz-6"     "xz -6 -c"             "xz -d -c"
add_compressor "zstd-1"   "zstd -1 -q -c"        "zstd -d -q -c"
add_compressor "zstd-3"   "zstd -3 -q -c"        "zstd -d -q -c"
add_compressor "zstd-19"  "zstd -19 -q -c"       "zstd -d -q -c"

# Append user's own tools if provided
for OWN_TOOL in "${OWN_TOOLS[@]}"; do
    IFS=':' read -r own_name own_comp own_decomp <<< "$OWN_TOOL"
    add_compressor "$own_name" "$own_comp" "$own_decomp"
done

# ---------- helpers ----------------------------------------------------------
# All floating-point formatting goes through awk with OFMT to stay locale-safe.
bytes_to_mb()  { awk "BEGIN{printf \"%.2f\", $1/1048576}"; }
bits_per_sym() { awk "BEGIN{printf \"%.3f\", ($2*8)/$1}"; }
ratio_pct()    { awk "BEGIN{printf \"%.1f\", $2*100/$1}"; }
fmt3()         { awk "BEGIN{printf \"%.3f\", $1}"; }   # format to 3 dp, leading zero

# ---------- benchmark one file with one compressor ---------------------------
# Supports two calling conventions:
#   - stdin/stdout (default): comp_cmd < input > output  (gzip -c style)
#   - file-argument mode:     comp_cmd input output       (use %i/%o placeholders)
# Outputs one result row: comp_bytes t_comp t_decomp lossless
bench_one() {
    local input="$1" name="$2" comp_cmd="$3" decomp_cmd="$4"
    local tmpcomp tmpdecomp tfile
    tmpcomp=$(mktemp)
    tmpdecomp=$(mktemp)
    tfile=$(mktemp)

    # Detect file-argument mode via %i/%o placeholders
    local file_args_c=false file_args_d=false
    [[ "$comp_cmd"   == *"%i"* || "$comp_cmd"   == *"%o"* ]] && file_args_c=true
    [[ "$decomp_cmd" == *"%i"* || "$decomp_cmd" == *"%o"* ]] && file_args_d=true

    # Substitute placeholders with actual paths
    local actual_comp actual_decomp
    actual_comp="${comp_cmd//%i/$input}";    actual_comp="${actual_comp//%o/$tmpcomp}"
    actual_decomp="${decomp_cmd//%i/$tmpcomp}"; actual_decomp="${actual_decomp//%o/$tmpdecomp}"

    # --- compress (exactly RUNS runs, averaged) ---
    local t_comp="0.000"
    local comp_ok=true
    local sum_c=0 i t
    for (( i=0; i<RUNS; i++ )); do
        if $file_args_c; then
            "$TIME_CMD" -f "%e" -o "$tfile" bash -c "$actual_comp" > /dev/null 2>/dev/null \
                || { comp_ok=false; break; }
        else
            "$TIME_CMD" -f "%e" -o "$tfile" bash -c "$actual_comp < \"\$0\" > \"\$1\"" \
                "$input" "$tmpcomp" 2>/dev/null \
                || { comp_ok=false; break; }
        fi
        t=$(cat "$tfile")
        sum_c=$(echo "$sum_c + $t" | bc)
    done
    if $comp_ok; then
        t_comp=$(echo "scale=3; $sum_c / $RUNS" | bc)
    fi

    local comp_bytes
    comp_bytes=$(stat -c%s "$tmpcomp" 2>/dev/null || echo 0)

    # --- decompress (exactly RUNS runs, averaged) ---
    local t_decomp="0.000"
    local lossless="NO"
    if $comp_ok && [[ "$comp_bytes" -gt 0 ]]; then
        local sum_d=0
        local decomp_ok=true
        for (( i=0; i<RUNS; i++ )); do
            if $file_args_d; then
                "$TIME_CMD" -f "%e" -o "$tfile" bash -c "$actual_decomp" > /dev/null 2>/dev/null \
                    || { decomp_ok=false; break; }
            else
                "$TIME_CMD" -f "%e" -o "$tfile" bash -c "$actual_decomp < \"\$0\" > \"\$1\"" \
                    "$tmpcomp" "$tmpdecomp" 2>/dev/null \
                    || { decomp_ok=false; break; }
            fi
            t=$(cat "$tfile")
            sum_d=$(echo "$sum_d + $t" | bc)
        done
        if $decomp_ok; then
            t_decomp=$(echo "scale=3; $sum_d / $RUNS" | bc)
            # Lossless check via md5
            local md5_orig md5_decomp
            md5_orig=$(md5sum "$input" | cut -d' ' -f1)
            md5_decomp=$(md5sum "$tmpdecomp" | cut -d' ' -f1)
            [[ "$md5_orig" == "$md5_decomp" ]] && lossless="YES" || lossless="NO"
        fi
    fi

    rm -f "$tmpcomp" "$tmpdecomp" "$tfile"
    echo "$comp_bytes $t_comp $t_decomp $lossless"
}

# ---------- print a separator ------------------------------------------------
separator() {
    printf '+------+%-16s+%-14s+%-14s+--------+-----------+-----------+-----------+-----------+-----------+\n' \
        "----------------" "--------------" "--------------"
}

# ---------- print table header -----------------------------------------------
print_header() {
    local label="$1"
    echo ""
    printf "${W}%s${N}\n" "$label"
    separator
    printf '| %-4s | %-14s | %-12s | %-12s | %-6s | %-9s | %-9s | %-9s | %-9s | %-9s |\n' \
        "Rank" "Compressor" "Original(MB)" "Comprssd(MB)" "Ratio%" \
        "bits/byte" "t_comp(s)" "t_dcp(s)" "t_total(s)" "Lossless"
    separator
}

# ---------- collect results for one input file and print table ---------------
bench_file() {
    local input="$1" label="$2"

    local orig_bytes
    orig_bytes=$(stat -c%s "$input")
    local orig_mb
    orig_mb=$(bytes_to_mb "$orig_bytes")

    log "Benchmarking: $label  ($(printf '%s' "$orig_mb") MB, $orig_bytes bytes)"

    # Collect rows: name comp_bytes t_comp t_decomp lossless
    local -a rows_name rows_comp rows_tc rows_td rows_lossless

    local i
    for (( i=0; i<${#NAMES[@]}; i++ )); do
        local name="${NAMES[$i]}"
        local cc="${COMP_CMDS[$i]}"
        local dc="${DECOMP_CMDS[$i]}"
        log "  -> $name ..."
        read -r cb tc td ls <<< "$(bench_one "$input" "$name" "$cc" "$dc")"
        rows_name+=("$name")
        rows_comp+=("$cb")
        rows_tc+=("$tc")
        rows_td+=("$td")
        rows_lossless+=("$ls")
    done

    # Sort by compressed size (ascending) to get rank
    local -a order
    # Create sortable list: "comp_bytes index"
    local sortlist=""
    for (( i=0; i<${#rows_name[@]}; i++ )); do
        sortlist+="${rows_comp[$i]} $i"$'\n'
    done
    mapfile -t sorted <<< "$(echo "$sortlist" | sort -n)"

    print_header "$label"

    local rank=1
    for entry in "${sorted[@]}"; do
        [[ -z "$entry" ]] && continue
        local idx
        idx=$(echo "$entry" | awk '{print $2}')
        local name="${rows_name[$idx]}"
        local cb="${rows_comp[$idx]}"
        local tc="${rows_tc[$idx]}"
        local td="${rows_td[$idx]}"
        local ls="${rows_lossless[$idx]}"

        local comp_mb ratio bps tt
        comp_mb=$(bytes_to_mb "$cb")
        ratio=$(ratio_pct "$orig_bytes" "$cb")
        bps=$(bits_per_sym "$orig_bytes" "$cb")
        tc=$(fmt3 "$tc")
        td=$(fmt3 "$td")
        tt=$(awk "BEGIN{printf \"%.3f\", $tc+$td}")

        local lossless_mark
        [[ "$ls" == "YES" ]] && lossless_mark="${G}YES${N}" || lossless_mark="${R}NO ${N}"

        printf "| %-4s | %-14s | %12s | %12s | %6s | %9s | %9s | %9s | %9s | %-9b |\n" \
            "$rank" "$name" "$orig_mb" "$comp_mb" "${ratio}%" \
            "$bps" "$tc" "$td" "$tt" "$lossless_mark"

        (( rank++ ))
    done

    separator
    echo ""
}

# ---------- per-file aggregate mode (-m) -------------------------------------
# Run every compressor on each file individually, accumulate totals, print one
# summary table.
#   bits/byte  = total_compressed_bits / total_original_bytes  (weighted)
#   t_comp/dcp = mean across files  (sum / num_files)
#   Lossless   = YES only if every file round-tripped correctly
bench_files_aggregate() {
    local total_orig=0
    local num_files="${#FILE_LIST[@]}"

    # Associative arrays keyed by compressor name
    declare -A acc_comp   # total compressed bytes
    declare -A acc_tc     # total t_comp (bc string, accumulated)
    declare -A acc_td     # total t_decomp
    declare -A acc_ok     # "true" if ALL files lossless so far

    # Initialise accumulators
    local i
    for (( i=0; i<${#NAMES[@]}; i++ )); do
        local n="${NAMES[$i]}"
        acc_comp[$n]=0
        acc_tc[$n]="0"
        acc_td[$n]="0"
        acc_ok[$n]="true"
    done

    # --- iterate over files ---
    for f in "${FILE_LIST[@]}"; do
        local fp="$DATA_DIR/$f"
        local orig_bytes
        orig_bytes=$(stat -c%s "$fp")
        total_orig=$(( total_orig + orig_bytes ))
        local orig_mb
        orig_mb=$(bytes_to_mb "$orig_bytes")
        log "File $f  ($orig_mb MB, $orig_bytes bytes)"

        for (( i=0; i<${#NAMES[@]}; i++ )); do
            local n="${NAMES[$i]}"
            log "  -> $n ..."
            read -r cb tc td ls <<< "$(bench_one "$fp" "$n" "${COMP_CMDS[$i]}" "${DECOMP_CMDS[$i]}")"
            acc_comp[$n]=$(( acc_comp[$n] + cb ))
            acc_tc[$n]=$(echo "${acc_tc[$n]} + $tc" | bc)
            acc_td[$n]=$(echo "${acc_td[$n]} + $td" | bc)
            [[ "$ls" != "YES" ]] && acc_ok[$n]="false"
        done
    done

    # --- build sorted list by total compressed bytes ---
    local sortlist=""
    for (( i=0; i<${#NAMES[@]}; i++ )); do
        sortlist+="${acc_comp[${NAMES[$i]}]} $i"$'\n'
    done
    mapfile -t sorted <<< "$(echo "$sortlist" | sort -n)"

    local total_orig_mb
    total_orig_mb=$(bytes_to_mb "$total_orig")
    local label
    label="Per-file aggregate ($(IFS=+; echo "${FILE_LIST[*]}"))  —  ${num_files} files, ${total_orig_mb} MB total  [times = mean per file]"
    print_header "$label"

    local rank=1
    for entry in "${sorted[@]}"; do
        [[ -z "$entry" ]] && continue
        local idx
        idx=$(echo "$entry" | awk '{print $2}')
        local n="${NAMES[$idx]}"
        local cb="${acc_comp[$n]}"
        local tc td tt comp_mb ratio bps
        comp_mb=$(bytes_to_mb "$cb")
        ratio=$(ratio_pct "$total_orig" "$cb")
        bps=$(bits_per_sym "$total_orig" "$cb")
        tc=$(awk "BEGIN{printf \"%.3f\", ${acc_tc[$n]} / $num_files}")
        td=$(awk "BEGIN{printf \"%.3f\", ${acc_td[$n]} / $num_files}")
        tt=$(awk "BEGIN{printf \"%.3f\", $tc+$td}")

        local lossless_mark
        [[ "${acc_ok[$n]}" == "true" ]] \
            && lossless_mark="${G}YES${N}" \
            || lossless_mark="${R}NO ${N}"

        printf "| %-4s | %-14s | %12s | %12s | %6s | %9s | %9s | %9s | %9s | %-9b |\n" \
            "$rank" "$n" "$total_orig_mb" "$comp_mb" "${ratio}%" \
            "$bps" "$tc" "$td" "$tt" "$lossless_mark"
        (( rank++ ))
    done

    separator
    echo ""
}

# ---------- main -------------------------------------------------------------
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

IFS=',' read -ra FILE_LIST <<< "$FILES_ARG"

# Validate files exist
for f in "${FILE_LIST[@]}"; do
    fp="$DATA_DIR/$f"
    [[ -f "$fp" ]] || err "File not found: $fp"
done

echo ""
printf "${W}========================================================${N}\n"
printf "${W}  TAI Project 1 — Compression Benchmark${N}\n"
printf "${W}  Runs per compressor: $RUNS${N}\n"
printf "${W}========================================================${N}\n"

if $CONCAT_MODE; then
    # Concatenate all selected files into one
    cat_file="$WORK_DIR/concat"
    log "Concatenating files: ${FILE_LIST[*]}"
    for f in "${FILE_LIST[@]}"; do
        cat "$DATA_DIR/$f" >> "$cat_file"
    done
    label="Concatenated ($(IFS=+; echo "${FILE_LIST[*]}"))"
    bench_file "$cat_file" "$label"
elif $MULTI_MODE; then
    bench_files_aggregate
else
    for f in "${FILE_LIST[@]}"; do
        bench_file "$DATA_DIR/$f" "File $f"
    done
fi

printf "${W}Done.${N}\n\n"
