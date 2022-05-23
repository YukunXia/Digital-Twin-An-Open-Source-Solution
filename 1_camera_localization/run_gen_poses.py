import sys
sys.path.append("..")
from llff_poses.pose_utils import gen_poses

scene_folder = "sample_data_subfolder"
gen_poses(scene_folder)