- This folder contains the training code (in Python Jupyter notebooks) for current uni-modal (text-only or audio-only) models trained on either scripted or improvised data from the IEMOCAP dataset (Busso et al., 2008)

- We also use the dialog acts (DAs) annotated by (Bothe et al., 2020) and publically available on their [GitHub repo](https://github.com/bothe/EDAs). The labels have already been downloaded and can be found in `multimodal-dac/eda_iemocap_no_utts_dataset.csv` and `multimodal-dac/EDA_Dataset.csv`. You will also require the full IEMOCAP dataset in the root folder of the script with the path `IEMOCAP_full_release/` to access transcripts, audio files, video files, etc.

# Instructions to run training scripts for unimodal results
1. To run the training scripts, you can create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from the `requirements.txt` file. Or, if you come across complications, I recommend creating a new conda environment and downloading packages are you require them. You will definitely need a compatible recent version of PyTorch and HuggingFace transformers library.

2. The main training scripts are in Python Jupyter notebooks

    2a. *iemocap_dataset_preprocessing.ipynb*\
        This script looks through the IEMOCAP dataset and splits it into improvised and scripted subsets. There are some interesting (to say kindly) heuristics that I took to decide how to split into train/val/test so that there is not any data leakage between the splits (you don't want data you've already seen in the training set to be in the validation dataset). I definitely recommend looking at this code carefully and potential refine it. 

    2b. *train_audio.ipynb*\
        This script takes in the audio .wav files from IEMOCAP, extracts [GeMAPs features](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf) (think features such as pitch that may hold important semantic meaning in audio data). These have already been extracted and found in files `gemaps_xxx.csv`. You can see at the bottom how this is done, and feel free to experiment with different feature sets. The rest of the cells pretty much can be run in order to take the pre-processing steps, training, and evaluation. 

    2c. *train_text.ipynb*\
        This script is very similar to `train_audio.ipynb` script, but with textual features (using [RoBERTa transformer encoder model](https://huggingface.co/FacebookAI/roberta-large) to get these features). 
    



