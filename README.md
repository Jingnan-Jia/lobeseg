
# Multi-task Semi-supervised Learning for Lobe Segmentation
## Introduction for several directories with their specific functions:

- data  (training datasets are saved here)
- futils  (common used functions and models including building models, compute metrics)
- logs  (save monitor metrics during training)
- models  (save trained models)
- results  (save the training/validation/testing results including Dice, MSD, Hausdorff distance, false positive, etfc.)

## Introduction for each files
- Use `python train_ori_fit_rec_epoch.py` to train model.  
- Use `write_preds_save_dice.py` to evaluate the trained model.  
- Modify `set_parameters.py` to set custom parameters.   
- Scipts files `script*` are used to submit job to HPC cluster.  
- `plot_curve*` are used to plot training loss curve.  
  
