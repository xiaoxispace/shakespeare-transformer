# Shakespeare-transformer

# Example output

Hyper parameters:

```
epoch = 100
block_size = 64
batch_size = 256
dmodel = 256 
learning_rate = 0.00001
cuda_available = torch.cuda.is_available()
num_of_encoder_layers = 3
num_of_heads = 4
decoder_only = True
```

```
(shakespeare) ubuntu@193-122-144-209:~/dev/shakespeare-transformer$ python eval.py -m "data/model-final.pth"
Loading vocab ...
Vocab size:  83
Done loading vocab ...

Construct and load model ...
Construct and load model done.

================================
|| Begin generating contents: ||
================================

 puppets me falls, and down so in his cries
For thy fleet all edge me commended
With other to school of him thou dost quick crries.
If very doubt, Nature my father's feather.

SIR HUGH  Now, is is this from thus?

POLONIUS
I thank you faith-wilted, be dismission, get the other.
If thou best thou whoreson, with all appears
Blots, go with speak stock, or God's name
Doth she lie it it dower threater king?

GLOUCESTER
We'll said she, their gaze was fair Tito's vow.

TIMON
After did me than thou mean and children on so far
We have may blood) too.

[Enter Talbot.]


MESSENGER
Have you to York. Nurse! If he be marriages thing.
He loves the brother's name? Why, sir?
What means depart I'll like thee preserve.

POET  Where, bring him. By guess that you bear,
Not would say bondman.

BEROWNE  Not a pax the for who, of mother?

BOATSWAIN  Nay, if my lord of greater sake,
What is order,
And yet forgive shine own part. Therefore.

ULYSSES
That feather they offerent Caesar. I shall
Believe me soon pla(shakespeare) ubuntu@193-122-144-209:~/dev/shakespeare-transformer$ 
```

# Setup

## Install Python Environment

1. Install pyenv from [the pyenv public repo](https://github.com/pyenv/pyenv#a-getting-pyenv). Specifically, for linux, 
    1. Update and install system dependencies
        ```bash
        sudo apt update
        sudo apt install -y make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev curl git \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev
        ```
    2. Run `curl -fsSL https://pyenv.run | bash` to install pyenv
    3. Setup pyenv for shell with command
        ```bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        ```
    4. Restart shell with command `exec "$SHELL"`
2. Install Python and the required dependencies
    1. Run `pyenv install 3.12.15`
    2. Create the virtual environment `pyenv virtualenv 3.12.15 shakespeare`
    3. Install the required dependencies `python -m pip install -r requirements.txt`
3. Execute training, eval or generation scripts. 
    - To train model, run `python train.py -e <epoch number> --enable-mixed-precision`. `--enable-wandb` is optional provided `` environment variable is available to upload training statistics to WandB.
    - To evaluate model on the test dataset, run `python eval.py -m <path to the model>`.
    - To genearate response, run `python gen.py`.


# Reference
The complete text of Shakespeare's work can be found in [Folger Shakespeare Library](https://www.folger.edu/explore/shakespeares-works/all-works/). 