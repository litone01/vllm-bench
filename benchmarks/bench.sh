PATH_PREFIX='results'
for input_len in 100 ; do
  for output_len in 100 ; do
    for req_rate in 4 8 16 32; do

      # Can modify req_rate and TOTAL based on input/output length as needed

      # Number of requests
      OUTPUT_FILE="${PATH_PREFIX}/${input_len}-${output_len}-${req_rate}-100.txt"
      python benchmark_serving.py --backend vllm --tokenizer ~/data/models/llama-2-7b-hf/ --dataset ~/data/ShareGPT_V3_unfiltered_cleaned_split.json --request-rate $req_rate --input_len $input_len --output_len $output_len > ${OUTPUT_FILE}

    done
  done
done
