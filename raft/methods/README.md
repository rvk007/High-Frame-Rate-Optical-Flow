# Optical Flow Methods

## Traditional Methods
Done:
- Lucas-Kanade

To find:
- Horn Schunk
- Dense Pyramid Lucas-Kanade: Shi-Tomasi https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
- Farneback: https://nanonets.com/blog/optical-flow/
- Robust Local Optical Flow (RLOF)
https://learnopencv.com/optical-flow-in-opencv/

## Data Driven
Done:
- RAFT

To Find:
- Neural Scene Flow Prior
- Optical Flow Perceiver [DeepMind] https://huggingface.co/deepmind/optical-flow-perceiver
- FlowNet2 [NVIDIA] https://github.com/NVIDIA/flownet2-pytorch
- SPyNet https://github.com/anuragranj/spynet


https://link.springer.com/article/10.1007/s42452-021-04227-x#Sec2

## Output dimension
- Lucas-Kanade: (1024, 1024, 3)
- Dense Pyramid Lucas-Kanade: (1024, 1024, 3)
- Horn Schunck: (480, 640, 3)
- Farneback: (1024, 1024, 3)
- Robust Local Optical Flow (RLOF): (1024, 1024, 3)

- RAFT: (1000, 1000, 3)
- Optical Flow Perceiver: (480, 640, 3)
