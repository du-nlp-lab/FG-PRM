<h1 style="text-align: center;">Fine-grained Hallucination Mitigation and Detection in Language Model Reasoning</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2410.06304">Paper</a> ｜ 
  <a href="https://huggingface.co/du-lab">Data</a>
</p>

The repo is built based on the [FineGrainedRLHF GitHub repo](https://github.com/allenai/FineGrainedRLHF).

## Getting Started
To set up an environment, please follow the following commands:
```bash
conda create -n fgprm python==3.10
pip install -r requirements.txt
```

## Prepare for all the following steps
To run the the following experiments, please step into the folder `src`.
```bash
cd src
```
Then, please confirm all the parameters in the file `config.yml`.

## Synthetic Data Generation
To automatically generate data, please run the following commands:
```bash
./sample_synthetic_data.sh
```
If you want to use other models to generate synthetic data, please find the corresponding files in the `generation` folder and activate the corresponding line in the shell script.

## Train Reward Models
To train a model, please run:
```bash
bash train_reward_models.sh
```
If you want to change the model or switch to a specific type of hallucination, please modify the corresponding file mentioned in the shell script.

## Evaluate Reward Models
To test a checkpoint, the procedure is similar to the training part, but the file is `run_evaluation.sh`.

## LLM Tests
To test the availability of LLMs, please run:
```bash
bash sample_answers_generation.sh
```
To change the target model, please open the file and activate the model you want to test.

## Citation

```bibtex
@article{li2024fg,
  title={FG-PRM: Fine-grained Hallucination Detection and Mitigation in Language Model Mathematical Reasoning},
  author={Li, Ruosen and Luo, Ziming and Du, Xinya},
  journal={arXiv preprint arXiv:2410.06304},
  year={2024}
}
