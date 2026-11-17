#!/bin/bash
#!/bin/bash
set -euo pipefail

#output="outputs/p1_pred_without_VCD.json"
extra_args=()

if [[ "${4:-}" == "--use_cd" ]]; then
    output="outputs/p1_pred_with_VCD.json"
    extra_args+=(--use_cd)
fi

python3 p1_inference.py \
    --images_root "$1" \
    --annotation_file "$2" \
    --model_path "$3" \
    --output "$$4" \
    "${extra_args[@]}"

# run your inference Python3 code
