export MODEL_NAME="./model/stable-diffusion-2-1"
export MODEL_SAVED_DIR="model/trained_model"
export TRAIN_DATA_DIR="data/train"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --model_saved_dir=$MODEL_SAVED_DIR \
  --resolution=512 \
  --learning_rate=5e-5 \
  --train_batch_size=6 \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resume_from_checkpoint="latest" \
  --num_train_epochs=80 \
  --alpha=0.05 \
  --mixed_precision="no" \
  --report_to="wandb" \
  --tracker_project_name="PerTouch" \
  --run_id="PerTouch_train_v1" \
  --checkpointing_steps=5000 \
  --gradient_accumulation_steps=2 \
