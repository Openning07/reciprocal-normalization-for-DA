This is the official implementation of:
Reciprocal Normalization for domain adaptation

Prerequisites:
- python == 3.6.2
- pytorch == 0.4.0
- torchvision == 0.2.2
- numpy == 1.18.1
- CUDA == 10.1.105

Dataset:
- Office-Home
- visda

Training:
1) change the args.root in option.py,  or change the paths in ./data/xxx.txt, to set the correct directory for Office-Home or Visda in your computer;

2) To train CDAN + RN on the transfer tasks of "X domain to Y domain"  and "Y domain  to X domain", run "sh run_X-Y.sh $GUP_ID". GPU_ID is the index of GPU you want to run the experiment.
ps: we record and provide the random seeds of the experiments we conducted

3) If everything runs well, you will see
- iter:   loss: 
- ...
- ...
- ...

Test: 
iter:    precision:  




