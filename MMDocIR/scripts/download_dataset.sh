if [ ! -d "MMDocIR/Dataset" ]; then
    mkdir MMDocIR/dataset
fi
git clone https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset.git && mv MMDocIR_Evaluation_Dataset/*.parquet MMDocIR/dataset/
mv MMDocIR_Evaluation_Dataset/*.jsonl MMDocIR/dataset/
rm -rf MMDocIR_Evaluation_Dataset