# RouardJax

## Installation

Start by cloning this repo, either by just downloading it here or by running (in your shell):

```bash
git clone https://github.com/xkstein/RouardJax/
```

I recommend installing the required python packages in a python virtual environment (learn more about them [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)), to set this up go into the downloaded folder and run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This makes a virtual environment in this folder (under `venv/`), sources it (meaning that you're now using the virtual environment), and installs the required python packages to that virtual environment.

Now you can access the `gratings` package normally, with `import gratings`
