# Original Authors:

Jungyoon Song(song0326@snu.ac.kr) - Seoul National University

Woojin Chang - Seoul National University

Jae Wook Song - Hanyang University

Reference:

https://doi.org/10.1007/s10489-024-06077-7

# Data Availability

Due to data size limitations, only the M4-hourly dataset has been uploaded.

(https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) − electricity

(https://archive.ics.uci.edu/dataset/204/pems+sf) − traffic

(https://www.nrel.gov/grid/solar-power-data.html) − solar

(https://github.com/Mcompetitions/M4-methods/tree/master) − M4 hourly

(https://robjhyndman.com/publications/the-tourism-forecasting-competition) - tourism-monthly, tourism-quarterly

## To run:
Install all dependencies listed in requirements.txt. 

1. Start model train :
  `python hyperparameter.py`
2. Model test:
  `python test.py`
3. Evaluate the model performance:
   check the result folder

# Citation
If this repository is useful for your research, please consider citing it (example format):
```bash
@article{song2025nqf,
  title={NQF-RNN: probabilistic forecasting via neural quantile function-based recurrent neural networks},
  author={Song, Jungyoon and Chang, Woojin and Song, Jae Wook},
  journal={Applied Intelligence},
  volume={55},
  number={3},
  pages={183},
  year={2025},
  publisher={Springer}
}
