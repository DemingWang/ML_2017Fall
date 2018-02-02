1. Install Anaconda:
https://docs.anaconda.com/anaconda/install/linux

2. Add anaconda bin to the PATH envorinment variable:
> setenv PATH /home/razvan/tools/others/anaconda3/bin:${PATH}

3. Create deeprlbootcamp environment:
> conda env create -f environment.yml

4. Activate deeprlbootcamp environment:
> source activate deeprlbootcamp
  OR
> source /home/razvan/tools/others/anaconda3/bin/activate deeprlbootcamp
  OR
> bash
> source /home/razvan/tools/others/anaconda3/bin/activate deeprlbootcamp

5. Launch IPython Notebook from this directory.
> jupyter notebook
