### Football - Player and ball posession detection using YOLOV5

ByteTrack unfortunately has a lot of dependencies that only work with older version of python.
This code uses yolov5 and ByteTrack on custom trained weights to detect player, ball, Ref, and assign them ids.

All the dependencies needed by both of them to work together along with their versions are frozen in `football-tracker/requirements.txt`

The dependencies require `python==3.8` to run properly.

In case of any dependency failing to build or run, try uninstalling and reinstalling it.

Use of a virtual environment whether venv, or virutalenv is _Highly Recommended_.

1. First run: `pip install -r requirements.txt` in the root of project
2. Then `cd ByteTrack && pip python setup.py develop`

The entry point of code is in `main.py`

The custom weights should be saved in a folder `./weights/<weight_file_name.pt>`. 

Download from:
[Google drive link](https://docs.google.com/uc?export=download&confirm=t&id=1OYwrlRti4cieuvVr8ERaJhTQdFJXWT4I)

### Issues

1. Fix imports of constants
2. 