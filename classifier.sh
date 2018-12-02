export CUDA_VISIBLE_DEVICES='0'
export BERT_BASE_DIR=uncased_L-12_H-768_A-12
export GLUE_DIR=glue_data

python run_classifier.py \
  --task_name=SST-2 \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/SST-2 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=64 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --output_dir=/tmp/sst-2_output-$BERT_BASE_DIR/
