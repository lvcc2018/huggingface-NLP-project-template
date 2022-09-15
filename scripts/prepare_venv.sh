### Set environment variables ###
if [ x"$1" = x ]
then 
    export VENV_NAME="template_environment"
else
    export VENV_NAME=$1
fi

if [ x"$2" = x ]
then 
    export PROJ_NAME="template_project"
else
    export PROJ_NAME=$2
fi

### Install Anaconda ###
mkdir ~/tmp
cd ~/tmp
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
source ~/anaconda3/etc/profile.d/conda.sh
cd ../
rm -r tmp

### Create Venv ###
echo "Conda Pytorch Env Setup"
conda create --name ${VENV_NAME}

conda activate ${VENV_NAME}
conda install pip
pip3 install numpy scipy sklearn
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  # make sure to install the right torch
pip3 install transformers datasets
pip3 install -r requirements.txt

### Create Dir ###
mkdir ${PROJ_NAME}

### Define Eliases ###
conda deactivate
echo "alias 'conda_env'='source \$HOME/anaconda3/etc/profile.d/conda.sh'" >> ~/.bash_profile
echo "alias '${VENV_NAME}'='conda_env && conda activate ${VENV_NAME} && export PYTHONPATH=\$HOME/${PROJ_NAME}/:\$PYTHONPATH && cd \$HOME/${PROJ_NAME}/'" >> ~/.bash_profile
source ~/.bash_profile
