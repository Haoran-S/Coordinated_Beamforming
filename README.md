## DeepMIMO for Power Allocation

This is a MATLAB / Python code package modified from [1-2] to generate data and perform continual learning training [3-4] for coordinated beamforming. The code is based on the publicly available DeepMIMO dataset published for deep learning applications in mmWave and massive MIMO systems. 

This MATLAB / Python code package is related to the following article:

[1] Alkhateeb, Ahmed, Sam Alex, Paul Varkey, Ying Li, Qi Qu, and Djordje Tujkovic. "Deep learning coordinated beamforming for highly-mobile millimeter wave systems." IEEE Access 6 (2018): 37328-37348.

[2] Ahmed Alkhateeb, “DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications,” in Proc. of Information Theory and Applications Workshop (ITA), San Diego, CA, Feb. 2019.

[3] Haoran Sun, Wenqiang Pu, Minghe Zhu,  Xiao Fu, Tsung-Hui Chang, and Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In Episodically Dynamic Environment." arXiv preprint arXiv:2011.07782 (2020).

[4] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu and Nikos D. Sidiropoulos, “Learning to Optimize: Training Deep Neural Networks for Wireless Resource Management”, IEEE Transactions on Signal Processing 66.20 (2018): 5438-5453. 
 
 
## Setup

- pip install -r requirements.txt


## Steps

- Download source data 'O1_60' under ‘O1’ Ray-Tracing Scenario from https://www.deepmimo.net/ray_tracing.html then put it into the folder: Data_Generation/RayTracing Scenarios/O1

'''
matlab main_data_part1.m
'''
'''
python3 main_data_part2.py
'''
'''
sh main_train_part3.sh
'''
'''
matlab main_figure_part4.m
'''
