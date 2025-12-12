*This is the code repository for our paper "Explainable deep learning framework incorporating medical knowledge for insulin titration in diabetes".*

## Installation

To run this project, you will need **Conda** (Anaconda or Miniconda) installed.

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo
```

### 2. Set up the environment

Create the Conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

3. Activate the environment

```bash
conda activate insulin
```

## Train the interpretation model


The command of running the interpretation algorithm:

```bash
CUDA_VISIBLE_DEVICES=0 python lt_mix_shapley.py --gpus=1 --epochs=800 --ckpt_name="CKPT_NAME" --loss_name=mae --optimizer_name=adam --lr_scheduler_name=cosine --batch_size=1 --patient_limit_start 0 --patient_limit_end 500 --save_dir assess_save_20230101_all 
```

Note: 
1. Access to data and models requires authorization from the corresponding author.
2. The files that are highly relevant to the core interpretation algorithm include lt_mix_shapley.py, shapley_taylor.py, and shapley_utils.py.

## 💡 Citation

If you use this repository or any part of the methodology in your research, please consider citing our work.

[//]: # (If you use this repository or any part of the methodology in your research, please consider citing our work:)

[//]: # (```bibtex)

[//]: # (@article{YourLastName_Year_ProjectTitle,)

[//]: # (  title={Your Full Project Title},)

[//]: # (  author={Your Name&#40;s&#41; and Collaborator&#40;s&#41;},)

[//]: # (  journal={Journal Name or Pre-print Server &#40;e.g., arXiv:2401.12345&#41;},)

[//]: # (  year={2024})

[//]: # (})
## 📧 Support and Contact
If you have any questions, suggestions, or encounter any issues, please feel free to:

1. Open an Issue on this GitHub repository.

2. Contact the authors via the email address listed in our paper. 