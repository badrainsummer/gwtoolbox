# GWToolbox

_A Python-based toolbox for multi-messenger gravitational-wave astrophysics._

## Getting started

Before start to use these package, there are few step you have to do:

### Create a environment

We recommond you to create a virtual environment to install GWToolbox. If you don't have conda, install it first. Then create a new environment:

```
conda create -n gwtoolbox_enc python=3.9
conda activate gwtoolbox_enc
```

Then install some basic package:

```
pip install numpy scipy astropy request tqdm h5py matplotlib pandas
```

### Download data step

You can manually download the data [here](https://zenodo.org/records/17962656), unzip the file and copy or the folders in _Supplementary Materials_ into gwtoolbox folder. Or just run:

```
python download.py
```

### Download and install package

- _Ground-base_: Just usr conda to install [finesse](https://finesse.ifosim.org/docs/latest/index.html):

```
conda install -c conda-forge finesse
```

- _Space-borne_: This module depends on the codes inside the Mock LISA data challenge (MLDC). The relevant parts of MLDC is already included in the GWToolbox. Please follow these steps in sequence:

```
cd /MLDC-master/Packages
python setup_lisaxml2.py install

cd /MLDC-master/Packages/common
python setup.py install

cd /MLDC-master/Waveforms/MBH_IMR/IMRPhenomD
make

cd /MLDC-master/Waveforms/fastGB
python setup.py install
``` 

The simulation on the waveform of EMRIs depends on an other tool [_EMRI_Kludge_Suite_](https://github.com/alvincjk/EMRI\_Kludge\_Suite). In the original _EMRI_Kludge_Suite_, the arm-length of LISA is hard-coded. I revised the source code to enable a different arm-length. The revised _EMRI_Kludge_Suite_ is included in the GWToolbox.

To build the binaries. The _GSL_ and _FFTW_ libraries are required for compilation. After this step, you will need to install the python wrapper. Make sure that in the setup.py, the variables _gsl_ and _gslcblas_ are set correctly to the path of the libraries, and make sure that your environment variable _LD_LIBRARY_PATH_ is set properly so that the required libraries can be found. Then run:

```
cd /EMRI_Kludge_Suite
make
python setup.py install
```

- _PTA_: You only need to install _hasasia_:

```
pip install hasasia
```

## AI assistant

We develop an program that can create code based on _GWToolbox_ and run automatically and return result using large language mode like deepseek or chatgpt. You only have to install:

```
pip install gradio==5.38.2 openai
```

Then run:

```
cd AI_assistant/
python AI_assistant_local.py
```

Then go to your browser and visit http://127.0.0.1:7860 or http://localhost:7860.

Due to the computational requirements for running large language models locally, we currently only provide access to remote large model services. Users must apply for the corresponding model's API key independently.

## Website

If you don't want to install _GWToolbox_ but still want to use it, we provide a [website](http://gw-universe.ihep.ac.cn/GWtoolbox.html)for some simple task.
