# HEAR: Hearing Enhanced Audio Response for Video-grounded Dialogue, EMNLP 2023 (long, findings)

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

result.json file should be in './dstc8avsd_eval/sample' folder
```
bash dstc7avsd_eval/dstc7avsd_eval.sh
bash dstc8avsd_eval/dstc8avsd_eval.sh
```

# Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by a grant of the KAIST-KT joint research project through AI2XL Laboratory, Institute of convergence Technology, funded by KT [Project No. G01220646, Visual Dialogue System: Developing Visual and Language Capabilities for AI-Based Dialogue Systems]

# Citation
@inproceedings{yoon2023hear,
  title={HEAR: Hearing Enhanced Audio Response for Video-grounded Dialogue},
  author={Yoon, Sunjae and Kim, Dahyun and Yoon, Eunseop and Yoon, Hee and Kim, Junyeong and Yoo, Chang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={11911--11924},
  year={2023}
}

#nfo
