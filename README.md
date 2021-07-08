# SAN Project
Test task 


## Install env

```conda env create --name san_env --file conda_env```

or

```conda env create --file env.yml```

## Download pretrained model
Download pretrained model checkpoint_49.pth.tar from [here](https://drive.google.com/drive/folders/1SZVJHl8tM0G5MOFQmCrx5mB6vFj-aAFu)
to the checkpoints directory 


## Run tests
```
cd tests
py.test test_landmark_detector.py
```

## Run demo
```
cd landmark_detection
python demo.py
```