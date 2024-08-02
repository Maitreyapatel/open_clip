# enter the src folder of the open_clip repository
cd ./src

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# set the training args

TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 1 -m open_clip_train.main \
    --batch-size 500 \
    --precision amp \
    --workers 16 \
    --report-to tensorboard \
    --save-frequency 1 \
    --dataset-type custom \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=5 \
    --model ViT-B-32 \
    --pretrained openai \
    --negclip