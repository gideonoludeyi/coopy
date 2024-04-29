# coopy

## Getting Started
0. Prerequisite(s)
    - [Python 3.12+](https://www.python.org/)

1. Install program (macOS, Linux)
```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

2. Install program (Windows)
```sh
$ python -m venv .venv
$ .venv\Scripts\activate
$ pip install -e .
```

3. Run the training
```sh
$ coopy run --mapfile data/map.txt
```

4. Generate experiments
```
$ coopy experiment --mapfile data/map.txt
```

