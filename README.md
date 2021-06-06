# hair_classification

## Description

This is shufflenet based hairstyle classifier. It uses face detector to get face regions and takes expanded regions as features.

To train model run

`python main.py --data_dir <path-to-data> [--epoch_num <training-epoches>]`

Model is saved as `result_model.pth`

For inference run

`python inference.py --data_dir <path-to-img-folder> --model_path <path-to-model>`

Results are saved as `result.csv`

## How to improve

- Use better model for face detection
- Apply training hyperparameter search 
- Try to use bigger model with static quantisation or distilation
- Find optimal face bbox expansion rate
