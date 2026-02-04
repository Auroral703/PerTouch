export CACHE_DIR="./model"
export MODEL_SAVED_DIR="model/trained_model"
export TRAIN_DATA_DIR="data/train"

accelerate launch train.py \
  --cache_dir=$CACHE_DIR \
  --model_saved_dir=$MODEL_SAVED_DIR \
  --resolution=512 \
  --learning_rate=5e-5 \
  --train_batch_size=16 \
  --train_data_dir=$TRAIN_DATA_DIR \
  --seed=42 \
  --resume_from_checkpoint="latest" \
  --num_train_epochs=30 \
  --alpha=0.05 \
  --mixed_precision="no" \
  --report_to="wandb" \
  --tracker_project_name="PerTouch" \
  --run_id="train_v1" \
  --checkpointing_steps=1000 \
  --gradient_accumulation_steps=2 \
