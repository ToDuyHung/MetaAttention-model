from roboflow import Roboflow
rf = Roboflow(model_format="voc", notebook="yolox")
rf = Roboflow(api_key="LfHL0lqVOBBxF8ze6m73")
project = rf.workspace("new-workspace-mfcqt").project("house-vd1mh")
dataset = project.version(7).download("voc")
print(dataset.location)
# /AIHCM/ComputerVision/hungtd/house-7