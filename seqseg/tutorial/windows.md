# SeqSeg Windows Install ✨

## 1. Install Python (if not already installed) 🐍

Download and install Python 3.11 from the official Python website: [Python Downloads](https://www.python.org/downloads/). Make sure to check the box to add Python to your PATH during installation.

You can verify the installation by opening a command prompt and running:

```bash
python --version
```

## 2. Install Git (if not already installed)

Download and install Git from the official Git website: [Git Downloads](https://git-scm.com/downloads). This is necessary to clone the SeqSeg repository.

You can verify the installation by opening a command prompt and running:

```bash
git --version
```

## 3. Create virtual environment (optional but recommended) 🌱

1. Choose a location for your virtual environment, e.g., `C:\seqseg_env`.
2. Open a command prompt and run the following commands:

```bash
python -m venv C:\seqseg_env
```

3. Activate the virtual environment:

```bash
C:\seqseg_env\Scripts\activate
```

## 4. Install SeqSeg using pip 📦
```bash
pip install seqseg
```