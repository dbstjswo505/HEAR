# HEAR
[STARLAB] Audio Enhancement Module for video-dialogue system

# Dataset setting
The annotation file is in 'data' folder

The features are in the [link](https://drive.google.com/drive/u/2/folders/1JGE4eeelA0QBA7BwYvj89kSClE3f9k65)

# Settings
Use conda environment 'environment.yaml' for training and generation

Use conda environment 'eval_environment.yaml' for evaluation

## Training
```
python train.py
```

## Response Generate
```
python generate.py
```

## Evaluation
result.json file should be in './dstc7avsd_eval/sample' folder
```
bash dstc7avsd_eval/dstc7avsd_eval.sh
```
