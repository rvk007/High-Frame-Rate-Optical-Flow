python dd_raft.py --model /scratch/rr3937/optical_flow/raft/models/raft-things.pth --path /scratch/rr3937/optical_flow/BlenderProc/examples/advanced/optical_flow/output/$val
python ocv_lucas_kanade.py --path /scratch/rr3937/optical_flow/BlenderProc/examples/advanced/optical_flow/output/$val

# dd: data driven
# ocv: OpenCV
