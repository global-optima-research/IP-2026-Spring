CUDA_VISIBLE_DEVICES=0 python app.py \
    --model_path ../ckpt/CogVideoX-5b-I2V \
    --inpainting_branch ../ckpt/VideoPainter/checkpoints/branch \
    --id_adapter ../ckpt/VideoPainterID/checkpoints \
    --img_inpainting_model ../ckpt/flux_inp
