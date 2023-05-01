#!/bin/bash

# mkdir FlyingThings3D_release
cd FlyingThings3D_release

# wget --no-check-certificate http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar
tar xvf flyingthings3d__frames_cleanpass.tar
wget --no-check-certificate http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2

tar xvf flyingthings3d__optical_flow.tar.bz2

# cd ..
# wget --no-check-certificate http://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip
# unzip FlyingChairs.zip

# wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/FlowNet2/ChairsSDHom/ChairsSDHom.tar.gz
# tar xvzf ChairsSDHom.tar.gz