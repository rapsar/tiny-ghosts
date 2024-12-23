#!/bin/bash
BASE_DIR=${LLAVA_PROJECT_DIR:-/path/to/default/directory}
cd "$BASE_DIR"
"$BASE_DIR/llava-cli" \
-m "$BASE_DIR/models/llava/ggml-model-q4_k.gguf" \
--mmproj "$BASE_DIR/models/llava/mmproj-model-f16.gguf" \
--image "$1" \
--temp 0.001 \
-p "Answer by yes or no: is there any firefly flash in the image?" \
2> /dev/null
