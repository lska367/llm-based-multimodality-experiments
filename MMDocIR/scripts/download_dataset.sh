if [ ! -d "MMDocIR/Dataset" ]; then
    mkdir MMDocIR/Dataset
fi
git clone https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset.git && mv MMDocIR_Evaluation_Dataset/*.parquet MMDocIR/Dataset/
mv MMDocIR_Evaluation_Dataset/*.jsonl MMDocIR/dataset/
rm -rf MMDocIR_Evaluation_Dataset