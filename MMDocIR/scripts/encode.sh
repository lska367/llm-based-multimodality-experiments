if [ ! -d "MMDocIR/encode" ]; then
    mkdir MMDocIR/encode
fi
cd MMDocIR && uv run encode.py BGE --bs 256 --mode vlm_text --encode query,page,layout