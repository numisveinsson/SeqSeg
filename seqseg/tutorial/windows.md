# SeqSeg on Windows

## 1. Install Python

Install Python 3.11 (3.9 or newer also works; 3.11 matches the tutorial) from [python.org](https://www.python.org/downloads/). Check **Add python.exe to PATH** in the installer.

```bash
python --version
```

## 2. Install Git

Install Git from [git-scm.com](https://git-scm.com/downloads). The tutorial data lives in the SeqSeg repository.

```bash
git --version
```

## 3. Create a virtual environment

```bash
python -m venv C:\seqseg_env
C:\seqseg_env\Scripts\activate
```

## 4. Install SeqSeg

```bash
pip install seqseg
seqseg --help
seqseg --version
```

`seqseg --help` lists subcommands (`run`, `post`, `config`, `doctor`, …). `seqseg --version` prints the installed version.

## 5. Clone the repository

The clone is how you get the tutorial image and `seeds.json`. Pick a directory such as `C:\Documents` and run:

```bash
git clone https://github.com/numisveinsson/SeqSeg.git
cd SeqSeg
```

## 6. Run the tutorial

Download the aorta weights and follow [tutorial.md](tutorial.md) from the repository root. The command there includes `-max_n_steps 10`, `-max_n_branches 3`, and `-max_n_steps_per_branch 5` so the example finishes in a few minutes.

The same run in one line, once `nnUNet_results` is extracted next to the clone:

```powershell
seqseg run batch -data_dir seqseg\tutorial\data\ -nnunet_results_path ..\nnUNet_results\ -outdir tutorial_output\ -img_ext .mha -train_dataset Dataset005_SEQAORTANDFEMOMR -config_name aorta_tutorial -max_n_steps 10 -max_n_branches 3 -max_n_steps_per_branch 5 -simvascular 1
```

Drop those three `-max_n_*` flags to use the defaults (1000 steps, 100 branches, 100 steps per branch) and trace more of the tree.
