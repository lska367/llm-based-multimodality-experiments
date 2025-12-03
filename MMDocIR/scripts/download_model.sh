if [ ! -d "MMDocIR/checkpoint" ]; then
    mkdir MMDocIR/checkpoint
fi
git clone https://huggingface.co/BAAI/bge-large-en-v1.5.git MMDocIR/checkpoint/bge-large-en-v1.5