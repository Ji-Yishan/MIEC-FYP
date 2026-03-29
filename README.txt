The current version of code is developed with MacBook Pro Apple Silicon M5, and uses MPS to accelerate training

Project structure:
-- RTE # contain processed(remove unbalance quotation marks) RTE dataset used for training and evaluation
-- scripts # contain scrips used to run the code
    -- baseline.sh # used to run baseline.py with parameters
    -- innovation.sh # used to run innovation.py with parameters
    -- train_teacher.sh # used to fine-tune the teacher model
-- ckpts # store checkpoints after training
-- baseline.py # baseline method used in report
-- innovation.py # Tri-LossKD method proposed in report
-- requirements.txt # provide package versions used in this project

Environment setting:
-- python version:3.12.12
-- package version and installation: use requirements.txt provided in the folder

 How to compile:
 1. use python 3.12 ( preferably use miniconda to create an environment named FYP)
    Note: if not using FYP environments, should update the conda activate command in scripts
 2. download all required package in requirements.txt
 3. use chmod +x [scriptName].sh  command in terminal to make the script executable
 4. make sure able to access huggingface.co website
 5. use ./scripts/[scriptName].sh command in terminal to run desired python code
    Note: baseline.py and innovation.py requires  fine-tuned teacher model existed in ckpts folder
          if wanted to use provided python file to train or fine-tune model, make sure delete previous
          checkpoints in ckpts file, checkpoint folder name can be seen in corresponding script file
 6. to change passing parameters, simply change parameter values in scrip files
