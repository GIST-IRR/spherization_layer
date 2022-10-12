# Few-shot Learning

This experiment has been done based on the official implementation of Self-Supervised Learning For Few-shot Image Classification. \[[paper](https://arxiv.org/abs/1911.06045), [code](https://github.com/GabirYousef/SSL-FEW-SHOT)\]


## Requirements

To create an environment, please go to the above github link and do **Prerequisites**.

## Dataset

The datasets used in our experiments can be downloaded from the direction in the reference. Please see the README.md in [SSL-FEW-SHOT/data](https://github.com/GabirYousef/SSL-FEW-SHOT/tree/master/data) directory.


## Training and Evaluation

To train and test the model(s) in this task, run this command:

```
python train_protonet.py --lr 0.0001 --temperature 128   \
--max_epoch 100 --model_type {ConvNet, ResNet, SphConvNet, SphResNet} --dataset MiniImageNet \
--save_path ./MINI_ProtoNet_MINI_1shot_5way/ \
--shot 5 --way 5 --step_size 10 --gamma 0.5 \
--ndf 192 --rkhs 1536 --nd 8 \
--metric {euclidean, cos} \
--scaling 5.0 --radius 5.0 --lrable TF
```

If you want to train fine-tuned models, use the argument ```--init_weights``` like as below:

```
--init_weights ./saves/initialization/miniimagenet/{con-pre, res-pre-old}.pth
```

## Pre-Trained Models

### MiniImagenet

| Model Name   |  Model URL |
|:----------:|:-------------:|
| ConvNet |  [con-pre.pth](https://drive.google.com/file/d/1T0K0gYZLX467z-TW57_jAuIIRzg9UrRz/view?usp=sharing)    |
| ResNet |  [res-pre-old.pth](https://drive.google.com/file/d/17dl48EqzdXGOcPT_7nbPuG9x_DcykLe7/view?usp=sharing)    |

## Results

Our model achieves the following performance on :

#### Performance of ConvNet on Mini-ImageNet

Test Accuracy (%): Mean(Std.)

| Model Name | Euclidean | Cosine |
|:----------:|:---------:|:------:|
| ConvNet  | 50.29(0.18) | 52.87(0.18) |
| Spherized ConvNet  | 43.41(0.16) | **53.74(0.16)** |

#### Performance of ResNet on Mini-ImageNet

Test Accuracy (%): Mean(Std.)

| Model Name | Euclidean | Cosine |
|:----------:|:---------:|:------:|
| ResNet  | 37.63(0.15) | 33.41(0.15) |
| Spherized ResNet  | 31.77(0.13) | **38.71(0.16)** |

#### Performance of ConvNet on Mini-ImageNet (fine-tuned)

Test Accuracy (%): Mean(Std.)

| Model Name | Euclidean | Cosine |
|:----------:|:---------:|:------:|
| ConvNet  | **67.80(0.17)** | 66.60(0.17) |
| Spherized ConvNet  | 51.56(0.17) | 61.77(0.17) |

#### Performance of ResNet on Mini-ImageNet (fine-tuned)

Test Accuracy (%): Mean(Std.)

| Model Name | Euclidean | Cosine |
|:----------:|:---------:|:------:|
| ResNet  | 77.73(0.15) | **78.86(0.15)** |
| Spherized ResNet  | 72.18(0.15) | 74.57(0.15) |
