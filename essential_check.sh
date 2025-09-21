#!/usr/bin/env bash
cd "$(dirname "$0")"
cat <<'TXT' | while read path expect _; do
checkpoints/latentsync_unet.pt           3400000000  # 3.4 GB
checkpoints/whisper/tiny.pt                76000000  # 76 MB
checkpoints/face-parse-bisent-79999.pth   167000000  # 167 MB
TXT
    if [[ -f "$path" ]]; then
        real=$(stat -c%s "$path")
        delta=$(( 100 * (real - expect) / expect ))
        if (( delta > -5 && delta < 5 )); then
            printf "✅ %-40s %9d 字节\n" "$path" "$real"
        else
            printf "⚠️  %-40s %9d 字节（期望 %9d）\n" "$path" "$real" "$expect"
        fi
    else
        printf "❌ %-40s 缺失\n" "$path"
    fi
done
