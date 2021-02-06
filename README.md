I. TASK TOKEN CLASSIFICATION:
  - Code finetune Electra this task are base on https://github.com/huggingface/transformers/tree/master/examples/token-classification .
  - We use gpu which has been supported by colaboratory
    - Requirements:
        - We use https://github.com/huggingface/transformers version 4.1.1.
        - pyarrow version 0.17.1 .
        - Another stuffs was package into ./transformers/examples/_tests_requirements.txt .
    - Guide:
        - Download code: 
            - !git clone https://github.com/huggingface/transformers.git -b v4.1.1
        - Install the environment:
            - !pip install transformers==4.1.1
            - !pip install -r ./transformers/examples/_tests_requirements.txt
            - !pip install pyarrow==0.17.1
        - Use our code to be able to run with the ner dataset (because the format of dataset is .txt when transform to .csv have a host of issue from special characters and            quota):
            - !rm -f  ./transformers/examples/token-classification/run_ner.py
            - %cp run_ner.py ./transformers/examples/token-classification/
        - Adjust the parameters conform with finetune model Pho-BERT:
                -- %cd ./transformers/examples/token-classification
                -- !python run_ner.py --model_name_or_path $MODEL \
                --output_dir $OUTPUT_DIR \
                --overwrite_output_dir \
                --learning_rate 1e-5 \
                --per_device_train_batch_size <small + base: 32, large: 16> \
                --per_device_eval_batch_size <small + base: 32, large: 16> \
                --save_steps 16000 \
                --num_train_epochs 30 \
                --seed 42 \
                --test_file $DIR_TO_FILE_test.txt \
                --do_train \
                --do_predict \
                --validation_file $DIR_TO_FILE_train.txt \
                --train_file $DIR_TO_FILE_train.txt \
                --task_name NER \
                --label_all_tokens \
                --evaluation_strategy epoch \
                --greater_is_better True
   - Result
        - briverse/vi-electra-small-cased :
            eval_loss = 0.07347527146339417
            eval_accuracy_score = 0.9813233724653149
            eval_precision = 0.8174927113702624
            eval_recall = 0.797724039829303
            eval_f1 = 0.8074874010079193
        -  briverse/vi-electra-base-cased :
                
