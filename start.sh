# 1
cd /vc_data/users/taoli1/VL-T5-OKVQA

conda create -n vlt5 python=3.7




# 2
conda init bash

source /home/aiscuser/.bashrc

conda activate vlt5

pip install -r requirements.txt

cd VL-T5



# 3
bash nlvr_eval.sh 1

bash VQA_VLT5.sh 1