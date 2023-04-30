#### Generate final submission result
```
sh 310581027.sh
```
#### Data pre-processing
* Data Structure
```
data --- train
      |    |--- {image}.jpg
      |    |--- {ground_truth}.txt 
      |
      |
      -- test
           |--- {image}.jpg
```
* {image} and {ground} like:
```
0001.jpg 
0002.jpg
    .
    .
    .
9999.jpg
---------
0001.txt
0002.txt
    .
    .
    .
9999.txt
```
* Run the data processing 
```
python data_process.py
```
* Or
```
python data_process.py --origin-dir data --data-dir data_processed
```
* After processing
```
img_0001.jpg 
img_0002.jpg
    .
    .
    .
img_9999.jpg
---------
img_0001.npy
img_0002.npy
    .
    .
    .
img_9999.npy
```
* They will split into train and val directory
```
data_processed 
      ------------ train
      |              |--- {image}.jpg
      |              |--- {ground_truth}.npy 
      |
      |
      --------------val
      |              |--- {image}.jpg
      |
      |
      --------------test
```
#### Reference by "Bayesian Loss for Crowd Count Estimation with Point Supervision"
```
@inproceedings{ma2019bayesian,
  title={Bayesian loss for crowd count estimation with point supervision},
  author={Ma, Zhiheng and Wei, Xing and Hong, Xiaopeng and Gong, Yihong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={6142--6151},
  year={2019}
}
```