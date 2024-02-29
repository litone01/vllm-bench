PATH_PREFIX='results'
for input_len in 25 ; do
  for output_len in 25 ; do
    for req_rate in 1 2 4 8 16; do

      # Can modify req_rate and TOTAL based on input/output length as needed

      # Number of requests
      TOTAL = 100
      OUTPUT_FILE="${PATH_PREFIX}/${input_len}-${output_len}-${req_rate}-${TOTAL}.txt"
      python benchmark_serving.py --backend vllm --tokenizer ~/data/models/llama-2-7b-hf/ --dataset ~/data/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate $req_rate --num-prompts $TOTAL --input_len $input_len --output_len $output_len > ${OUTPUT_FILE}

    done
  done
done