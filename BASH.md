# BASH Commands

## Vao thu muc train
```bash
cd /workspace/source/Conformal_Risk_Tube_Prediction
```
Muc dich: chuyen vao module CRTP de chay script train.

## Chay nen train_cls (tu dong resume neu co ckpt)
```bash
bash Conformal_Risk_Tube_Prediction/run_auto.sh \
  --entry train_cls \
  --run-name cls_exp01 \
  --mode overwrite \
  --device cuda:0 \
  --run-root /workspace/source/Conformal_Risk_Tube_Prediction/runs \
  --input-dir /workspace/source/datasets \
  --gpus 1 \
  --num-workers 15
```
Muc dich: train classifier va chay nen theo mode auto.

## Chay nen train CRTP (co pretrain ckpt)
```bash
bash run_auto.sh \
  --entry train \
  --run-name crtp_exp01 \
  --mode auto \
  --device cuda:0 \
  --run-root /workspace/source/Conformal_Risk_Tube_Prediction/runs \
  --input-dir /workspace/source/datasets \
  --pretrain-ckpt /workspace/source/Conformal_Risk_Tube_Prediction/runs/cls_exp01/checkpoints/xxx.ckpt \
  --gpus 1
```
Muc dich: train model chinh CRTP va chay nen theo mode auto.

## Xem trang thai job nen
```bash
tmux ls
```
Muc dich: kiem tra session train dang chay.

## Xem log train theo thoi gian thuc
```bash
tail -f /workspace/source/Conformal_Risk_Tube_Prediction/runs/<run_name>/logs/train.log
```
Muc dich: theo doi tien trinh train.
