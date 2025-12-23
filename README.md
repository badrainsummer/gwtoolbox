# GWToolbox

**GWToolbox** is a Python-based toolbox for **multi-messenger gravitational-wave astrophysics**, supporting simulations and analyses for ground-based detectors, space-borne detectors, and pulsar timing arrays (PTAs).

## Getting Started

Before using **GWToolbox**, please follow the steps below to set up the environment, download required data, and install dependencies.

## 1. Create a Virtual Environment

We strongly recommend using a virtual environment. If you do not have **conda** installed, please install it first.

Create and activate a new environment:

```bash
conda create -n gwtoolbox_enc python=3.9
conda activate gwtoolbox_enc
```

Then install the basic Python dependencies:

```bash
pip install numpy scipy astropy requests tqdm h5py matplotlib pandas
```

## 2. Download Data

You can manually download the required data from Zenodo:

👉 [https://zenodo.org/records/17962656](https://zenodo.org/records/17962656)

After downloading, unzip the file and copy the folders under ***Supplementary Materials*** into the `GWToolbox` directory.

Alternatively, you can download and extract the data automatically by running:

```bash
python download.py
```

## 3. Install GWToolbox Dependencies

GWToolbox supports **ground-based detectors**, **space-borne detectors**, and **PTA analyses**. Please install the corresponding dependencies as needed.

### 3.1 Ground-Based Detectors

For ground-based simulations, install **FINESSE** using conda:

```bash
conda install -c conda-forge finesse
```

Documentation:
[https://finesse.ifosim.org/docs/latest/index.html](https://finesse.ifosim.org/docs/latest/index.html)

### 3.2 Space-Borne Detectors (LISA)

The space-borne module depends on components from the **Mock LISA Data Challenge (MLDC)**. The required parts of MLDC are already included in GWToolbox.

Please install them **in the following order**:

```bash
cd MLDC-master/Packages
python setup_lisaxml2.py install

cd ../common
python setup.py install

cd ../../Waveforms/MBH_IMR/IMRPhenomD
make

cd ../../fastGB
python setup.py install
```

#### EMRI Waveforms

The simulation of EMRI waveforms relies on the
**EMRI_Kludge_Suite**:

[https://github.com/alvincjk/EMRI_Kludge_Suite](https://github.com/alvincjk/EMRI_Kludge_Suite)

In the original implementation, the LISA arm length is hard-coded.
We provide a **modified version** that allows for arbitrary arm lengths, which is already included in GWToolbox.

To compile the EMRI binaries:

* Make sure **GSL** and **FFTW** are installed on your system
* Check that `gsl` and `gslcblas` paths are correctly specified in `setup.py`
* Ensure the environment variable `LD_LIBRARY_PATH` includes the required library paths

Then run:

```bash
cd EMRI_Kludge_Suite
make
python setup.py install
```

---

### 3.3 Pulsar Timing Array (PTA)

For PTA-related analyses, only **hasasia** is required:

```bash
pip install hasasia
```

---

## 4. AI Assistant

GWToolbox includes an **AI assistant** that can automatically generate and execute GWToolbox-based code using large language models (e.g. ChatGPT or DeepSeek).

To enable the AI assistant, install:

```bash
pip install gradio==5.38.2 openai
```

Then launch the assistant:

```bash
cd AI_assistant/
python AI_assistant_local.py
```

Open your browser and visit:

* [http://127.0.0.1:7860](http://127.0.0.1:7860)
* [http://localhost:7860](http://localhost:7860)

> **Note**
> Due to the computational requirements of running large language models locally, the AI assistant currently supports **remote model services only**.
> Users must obtain and configure the corresponding API keys independently.

## 5. Online Version

If you prefer not to install GWToolbox locally, we provide a **web-based interface** for simple tasks:

🌐 [http://gw-universe.ihep.ac.cn/GWtoolbox.html](http://gw-universe.ihep.ac.cn/GWtoolbox.html)

## Citation

If you use **GWToolbox** in your research, please consider citing the relevant papers or acknowledging this software in your publications.
