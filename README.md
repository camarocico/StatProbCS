# Introduction

In order to run the apps in this repository, your system needs to have a recent Python interpreter. In each app folder you should create its own virtual environment, e.g. with `uv`:

```shell
uv init
```

Once initialized, you should install the necessary Python modules:

```shell
uv add -r requirements.txt
```

activate the virtual environment:

```shell
source .venv/bin/activate
```

and run the app:

```shell
streamlit run main.py
```
