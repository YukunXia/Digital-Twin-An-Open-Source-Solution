import bpy
from mathutils import Vector
from mathutils import Matrix
import mathutils
import math
import json
import numpy as np

POSE_JSON_FILE_PATH = NONE
POSE_OFFSET = 300

pose_bone_name_lists = [
    [
        "spine", "head", "shoulder_l", "upper_arm_l", "shoulder_r", "upper_arm_r", "hip_l", "thigh_l",
        "hip_r", "thigh_r", "chest_l", "chest_r"
    ], [
        "lower_arm_r", "lower_arm_l", "shin_r", "shin_l",
    ], [
        "foot_r", "foot_l", "hand_r", "hand_l"
    ]
]

## TODO: clean up the following function
##armature_obj.pose.bones[pose_bone_1_name]
#def GetPoseBoneAxisPitch(pose_bone_0, axis_index_0, pose_bone_1):
#    pose_bone_0_axis__world = pose_bone_0.matrix.to_3x3().col[axis_index_0]
#    pose_bone_0_axis__pose_bone_1 = pose_bone_1.matrix.to_3x3().inverted() @ pose_bone_0_axis__world
#    pitch__pose_bone_0_axis__pose_bone_1 = math.atan2(pose_bone_0_axis__pose_bone_1[0], pose_bone_0_axis__pose_bone_1[2])
#    return pitch__pose_bone_0_axis__pose_bone_1



if __name__ == "__main__":
    blender_data = []
    with open(POSE_JSON_FILE_PATH, "r") as blender_data_file:  
        blender_data = json.load(blender_data_file)

    armature_obj = bpy.data.objects["armature"]
    # reset armature frame to world frame
    armature_obj.matrix_world = armature_obj.matrix_world.inverted() @ armature_obj.matrix_world
#    world__T__armature = armature_obj.matrix_world.inverted()
#    armature__T__world = armature_obj.matrix_world

    # reset rotations
    for pose_bone_name_list in pose_bone_name_lists:
        for pose_bone_name in pose_bone_name_list:
            pose_bone = armature_obj.pose.bones[pose_bone_name]
            pose_bone.rotation_quaternion = (1, 0, 0, 0)
            pose_bone.keyframe_insert('rotation_quaternion', frame=max(1, POSE_OFFSET - 20))
    pose_bone = armature_obj.pose.bones["base"]
    pose_bone.rotation_quaternion = (1, 0, 0, 0)
    pose_bone.keyframe_insert('rotation_quaternion', frame=max(1, POSE_OFFSET - 20))
    
        
    for i, blender_data_entry in enumerate(blender_data):
        # update base orientation, base local z axis should align with the vector pointing from chest_l to chest_r
        bpy.context.view_layer.update()
        pose_bone = armature_obj.pose.bones['base']
        base_direction__world = Vector(blender_data_entry["chest_r"]) - Vector(blender_data_entry["chest_l"])
        base_direction__base = pose_bone.matrix.to_3x3().inverted() @ base_direction__world
        pitch = math.atan2(base_direction__base[0], base_direction__base[2])
        R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
        pose_bone.matrix = R @ pose_bone.matrix
        pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
        
        
        # update the directions of the first-level bones
        for pose_bone_name in pose_bone_name_lists[0]:
            pose_bone = armature_obj.pose.bones[pose_bone_name]
            
            bpy.context.view_layer.update()
            pose_bone_dir_vec = pose_bone.tail - pose_bone.head
            pose_bone_tgt_vec = Vector(blender_data_entry[pose_bone_name])
            R = pose_bone_dir_vec.rotation_difference(pose_bone_tgt_vec).to_matrix().to_4x4()
            pose_bone.matrix = R @ pose_bone.matrix
            
            if pose_bone_name not in ["upper_arm_l", "upper_arm_r"]:
                pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
            

        # update head orientation, spine_to_head local z axis should align with the vector pointing from eye_l to eye_r
        # and from ear_l to ear_r
        bpy.context.view_layer.update()
        pose_bone = armature_obj.pose.bones['head']
        head_direction__world = Vector(blender_data_entry["head_direction"])
        head_direction__head = pose_bone.matrix.to_3x3().inverted() @ head_direction__world
        pitch = math.atan2(head_direction__head[0], head_direction__head[2])
        R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
        pose_bone.matrix = R @ pose_bone.matrix
        pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
        
        
        # update upper arm orientation, upper_arm local x axis should align with the lower arm direction vector
        for side in ["r", "l"]:
            bpy.context.view_layer.update()
            pose_bone = armature_obj.pose.bones[f'upper_arm_{side}']
            lower_arm_direction__world = Vector(blender_data_entry[f"lower_arm_{side}"])
            lower_arm_direction__upper_arm = pose_bone.matrix.to_3x3().inverted() @ lower_arm_direction__world
            lower_arm_direction__upper_arm.normalize()
            
            # only when the direction vector of the lower arm deviates from upper arm y axis for a certain degree, do we need 
            # to take it into account. eg. the thershold here is 0.975, and arccos(0.975) \approx 12.8 deg
            # besides, plesae note that the y component could be negative
            if abs(lower_arm_direction__upper_arm[1]) < 0.975:
                # align x axis with that direction. negative pitch because y_axis = - x_axis \cross z_axis
                pitch = -math.atan2(lower_arm_direction__upper_arm[2], lower_arm_direction__upper_arm[0])
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
            # in other cases, the upper and lower arms on one side of the body are in a line, and their in-place rotations should 
            # be highly correlated. And that requires good estimations of the hand direction, but here, those estimations can be
            #  untrustworthy (TO BE VERIFIED)


        ## TODO: clean up the following code
        ## NOTE: thigh orientation here is chosen to be fixed, because the direction of the subsequent shin or foot can be unstable
        ## NOTE: thigh's x and z axis were rotated along the y axis for 90 deg. Previously x axis points forward in the canonical pose.
        ##       Therefore, some of the code need to be modified.
#        # update thigh orientation, thigh local x axis should align with the shin direction vector
#        for side in ["r", "l"]:
#            bpy.context.view_layer.update()
#            pose_bone = armature_obj.pose.bones[f'thigh_{side}']
#            shin_direction__world = Vector(blender_data_entry[f"shin_{side}"])
#            # negative sign here because initially shin direction is at the local negative x direction
#            shin_direction__thigh = -(pose_bone.matrix.to_3x3().inverted() @ shin_direction__world)
#            shin_direction__thigh.normalize()
#            
#            # only when the direction vector of the lower arm deviates from upper arm y axis for a certain degree, do we need 
#            # to take it into account. eg. the thershold here is 0.975, and arccos(0.975) \approx 12.8 deg
#            # besides, plesae note that the y component could be negative, and given the physical constraints of normal human
#            # bodies, if the negatively normalized shin_direction__thigh[1] > 0, it should never go down to values around 1.0
#            if abs(shin_direction__thigh[1]) < 0.975:
#                # align x axis with that direction. negative pitch because y_axis = - x_axis \cross z_axis
#                pitch = -math.atan2(shin_direction__thigh[2], shin_direction__thigh[0])
#                if abs(pitch) < np.pi/2:
#                    R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
#                    pose_bone.matrix = R @ pose_bone.matrix
#                    pitch__thigh_x_axis__base = GetPoseBoneAxisPitch(pose_bone, 0, "base")
##                    print("shin, pitch__thigh_x_axis__base = ", pitch__thigh_x_axis__base, abs(pitch__thigh_x_axis__base - np.pi/2) > np.pi/3)
#                    if abs(pitch__thigh_x_axis__base - np.pi/2) > np.pi/3:
#                        pose_bone.matrix = R.inverted() @ pose_bone.matrix
#                    pose_bone.keyframe_insert('rotation_quaternion', frame=i+20)
##            # else, use foot direction
#             # the following code is commented out, because foot estimations from mediapipe seems to be quite noisy sometimes
#            else:
#                foot_direction__world = Vector(blender_data_entry[f"foot_{side}"])
#                foot_direction__thigh = pose_bone.matrix.to_3x3().inverted() @ foot_direction__world
#                foot_direction__thigh.normalize()
#                foot_direction__base = armature_obj.pose.bones['base'].matrix.to_3x3().inverted() @ foot_direction__world
#                foot_direction__base.normalize()
#                foot_pitch__base = math.atan2(foot_direction__base[0], foot_direction__base[2])
#                if abs(foot_direction__thigh[1]) < 0.975 and abs(foot_pitch__base - np.pi/2) < np.pi/3:
#                    # align x axis with that direction. negative pitch because y_axis = - x_axis \cross z_axis
#                    pitch = -math.atan2(foot_direction__thigh[2], foot_direction__thigh[0])
#                    if abs(pitch) < np.pi/2:
#                        R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
#                        pose_bone.matrix = R @ pose_bone.matrix
#                        pitch__thigh_x_axis__base = GetPoseBoneAxisPitch(pose_bone, 0, "base")
##                        print("foot, pitch__thigh_x_axis__base = ", pitch__thigh_x_axis__base, abs(pitch__thigh_x_axis__base - np.pi/2) > np.pi/2)
#                        if abs(pitch__thigh_x_axis__base - np.pi/2) > np.pi/3:
#                            pose_bone.matrix = R.inverted() @ pose_bone.matrix
#                        pose_bone.keyframe_insert('rotation_quaternion', frame=i+20)
#                # else: both foot and shin align with thigh direction, which shouldn't happen in real world
                
                
        # update the directions of the second-level bones
        for pose_bone_name in pose_bone_name_lists[1]:
            pose_bone = armature_obj.pose.bones[pose_bone_name]
            
            bpy.context.view_layer.update()
            pose_bone_dir_vec = pose_bone.tail - pose_bone.head
            pose_bone_tgt_vec = Vector(blender_data_entry[pose_bone_name])
            R = pose_bone_dir_vec.rotation_difference(pose_bone_tgt_vec).to_matrix().to_4x4()
            pose_bone.matrix = R @ pose_bone.matrix
#            
#            if pose_bone_name not in ["shin_l", "shin_r"]:
#                pose_bone.keyframe_insert('rotation_quaternion', frame=i+20)


        for side in ["r", "l"]:
            # update shin in-place rotation
            # shin x/z should align with thigh's x/z axis
            bpy.context.view_layer.update()
            pose_bone = armature_obj.pose.bones[f'shin_{side}']
            
            thigh_x_axis__world = armature_obj.pose.bones[f'thigh_{side}'].matrix.to_3x3().col[0]
            thigh_x_axis__shin = pose_bone.matrix.to_3x3().inverted() @ thigh_x_axis__world
            thigh_x_axis__shin.normalize()
            
            if abs(thigh_x_axis__shin[1]) < 0.975:
                pitch = math.atan2(-thigh_x_axis__shin[2], thigh_x_axis__shin[0])
#                if abs(pitch) < np.pi/2:
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            else:
                thigh_z_axis__world = armature_obj.pose.bones[f'thigh_{side}'].matrix.to_3x3().col[2]
                thigh_z_axis__shin = pose_bone.matrix.to_3x3().inverted() @ thigh_z_axis__world
                thigh_z_axis__shin.normalize()
                # only one of x and z axis could possibly align with the y axis
                assert(abs(thigh_z_axis__shin[1]) < 0.975)
                
                pitch = math.atan2(thigh_z_axis__shin[0], thigh_z_axis__shin[2])
#                if abs(pitch) < np.pi/2:
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            
            pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
            
            
            # update lower arm in-place rotation
            # lower arm x/z should align with upper arm's x/z axis
            bpy.context.view_layer.update()
            pose_bone = armature_obj.pose.bones[f'lower_arm_{side}']
            
            upper_arm_x_axis__world = armature_obj.pose.bones[f'upper_arm_{side}'].matrix.to_3x3().col[0]
            upper_arm_x_axis__lower_arm = pose_bone.matrix.to_3x3().inverted() @ upper_arm_x_axis__world
            upper_arm_x_axis__lower_arm.normalize()
            
            if abs(upper_arm_x_axis__lower_arm[1]) < 0.975:
                pitch = math.atan2(-upper_arm_x_axis__lower_arm[2], upper_arm_x_axis__lower_arm[0])
#                if abs(pitch) < np.pi/2:
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            else:
                upper_arm_z_axis__world = armature_obj.pose.bones[f'upper_arm_{side}'].matrix.to_3x3().col[2]
                upper_arm_z_axis__lower_arm = pose_bone.matrix.to_3x3().inverted() @ upper_arm_z_axis__world
                upper_arm_z_axis__lower_arm.normalize()
                # only one of x and z axis could possibly align with the y axis
                assert(abs(upper_arm_z_axis__lower_arm[1]) < 0.975)
                
                pitch = math.atan2(upper_arm_z_axis__lower_arm[0], upper_arm_z_axis__lower_arm[2])
#                if abs(pitch) < np.pi/2:
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            
            pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
            
#                
                
        # update the directions of the third-level bones
        for pose_bone_name in pose_bone_name_lists[2]:
            bpy.context.view_layer.update()
            pose_bone = armature_obj.pose.bones[pose_bone_name]
            
            pose_bone_dir_vec = pose_bone.tail - pose_bone.head
            pose_bone_tgt_vec = Vector(blender_data_entry[pose_bone_name])
            R = pose_bone_dir_vec.rotation_difference(pose_bone_tgt_vec).to_matrix().to_4x4()
#            print("before pose_bone.matrix = ", pose_bone.matrix)
            pose_bone.matrix = R @ pose_bone.matrix
            ## BUG? There's no change after matrix multiplication, but under the hood, that multiplication was effective
#            print("after pose_bone.matrix = ", pose_bone.matrix)
            
            if pose_bone_name in ["foot_l", "foot_r"]:
                # verify the possibility of the foot orientation
                # (1) the angle between foot y axis and shin y axis should not be smaller than 60 deg or greater than 120 deg
                foot_y__world = pose_bone_tgt_vec.normalized()
                foot_y__shin = armature_obj.pose.bones[f'shin_{pose_bone_name[-1]}'].matrix.to_3x3().inverted() @ foot_y__world
#                # assert that the vector is still normalized
#                assert(abs(foot_y__shin[0]**2 + foot_y__shin[1]**2 + foot_y__shin[2]**2 - 1) < 0.001)
                # if roll of the foot is greater than 30 deg
                if abs(foot_y__shin[1]) > 0.5:
                    # revert the rotation
                    pose_bone.matrix = R.inverted() @ pose_bone.matrix
                else:
                    # absolute angle between foot y axis and shin z axis in shin's x-z plane
                    abs_pitch_of_foot_y__shin = abs(math.atan2(foot_y__shin[0], foot_y__shin[2]))
                    # maximum tolerable `abs_pitch_of_foot_y__shin` is 45 deg. Typically, it should be within 30 deg.
                    # revert the rotation
                    if abs_pitch_of_foot_y__shin > np.pi/4:
                        pose_bone.matrix = R.inverted() @ pose_bone.matrix

                # DON'T `bpy.context.view_layer.update()` here, otherwise weird rotations of bones in the previous level show up
                # Reasons need to be figured out 
                
                # update foot in-place rotation
                # foot z should align with shin direction
                shin_direction__world = Vector(blender_data_entry[f"shin_{pose_bone_name[-1]}"])
                shin_direction__foot = pose_bone.matrix.to_3x3().inverted() @ shin_direction__world
                shin_direction__foot.normalize()
                
                if abs(shin_direction__foot[1]) < 0.975:
                    pitch = math.atan2(shin_direction__foot[0], shin_direction__foot[2])
    #                if abs(pitch) < np.pi/2:
                    R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                    pose_bone.matrix = R @ pose_bone.matrix
                    
            elif pose_bone_name in ["hand_l", "hand_r"]:
                # verify the possibility of the hand orientation
                # (1) the angle between foot y axis and shin y axis should not be smaller than 60 deg or greater than 120 deg
                hand_y__world = pose_bone_tgt_vec.normalized()
                hand_y__lower_arm = armature_obj.pose.bones[f'lower_arm_{pose_bone_name[-1]}'].matrix.to_3x3().inverted() @ hand_y__world
                # there should be a continuous threshold function for hand rotation, 
                # but for simplicity a rough average 0.2 is chosen to be the universal threshold
                if abs(hand_y__lower_arm[1]) < 0.2:
                    # revert the rotation
                    pose_bone.matrix = R.inverted() @ pose_bone.matrix


                # DON'T `bpy.context.view_layer.update()` here, otherwise weird rotations of bones in the previous level show up
                # Reasons need to be figured out 
                
                # update hand in-place rotation
                # hand z should align with lower arm z direction
                lower_arm_z__world = armature_obj.pose.bones[f'lower_arm_{pose_bone_name[-1]}'].matrix.to_3x3().col[2]
                lower_arm_z__hand = pose_bone.matrix.to_3x3().inverted() @ lower_arm_z__world
                lower_arm_z__hand.normalize()
                
                if abs(lower_arm_z__hand[1]) < 0.975:
                    pitch = math.atan2(lower_arm_z__hand[0], lower_arm_z__hand[2])
    #                if abs(pitch) < np.pi/2:
                    R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                    pose_bone.matrix = R @ pose_bone.matrix
            
            pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
                
                
                    
        # update hand in-place rotation
        # hand x/z should align with lower arm x/z axis
        for side in ["r", "l"]:
            bpy.context.view_layer.update()
            pose_bone = armature_obj.pose.bones[f'hand_{side}']
#            lower_arm_z_axis__world = armature_obj.pose.bones[f'lower_arm_{side}'].matrix.to_3x3().col;[2]
#            lower_arm_z_axis__hand = pose_bone.matrix.to_3x3().inverted() @ lower_arm_z_axis__world
#            lower_arm_z_axis__hand.normalize()
#            if abs(lower_arm_z_axis__hand[1]) < 0.975:
#                pitch = math.atan2(lower_arm_z_axis__hand[0], lower_arm_z_axis__hand[2])
#                if abs(pitch) < np.pi/2:
#                    R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
#                    pose_bone.matrix = R @ pose_bone.matrix
#                    pose_bone.keyframe_insert('rotation_quaternion', frame=i+20)

            upper_arm_x_axis__world = armature_obj.pose.bones[f'upper_arm_{side}'].matrix.to_3x3().col[0]
            upper_arm_x_axis__lower_arm = pose_bone.matrix.to_3x3().inverted() @ upper_arm_x_axis__world
            upper_arm_x_axis__lower_arm.normalize()
            
            if abs(upper_arm_x_axis__lower_arm[1]) < 0.975:
                pitch = math.atan2(-upper_arm_x_axis__lower_arm[2], upper_arm_x_axis__lower_arm[0])
#                if abs(pitch) < np.pi/2:
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            else:
                upper_arm_z_axis__world = armature_obj.pose.bones[f'upper_arm_{side}'].matrix.to_3x3().col[2]
                upper_arm_z_axis__lower_arm = pose_bone.matrix.to_3x3().inverted() @ upper_arm_z_axis__world
                upper_arm_z_axis__lower_arm.normalize()
                # only one of x and z axis could possibly align with the y axis
                assert(abs(upper_arm_z_axis__lower_arm[1]) < 0.975)
                
                pitch = math.atan2(upper_arm_z_axis__lower_arm[0], upper_arm_z_axis__lower_arm[2])
#                if abs(pitch) < np.pi/2:
                R = Matrix.Rotation(pitch, 4, pose_bone.tail - pose_bone.head)
                pose_bone.matrix = R @ pose_bone.matrix
            
            pose_bone.keyframe_insert('rotation_quaternion', frame=i+POSE_OFFSET)
#            
    
        # update absolute location offsets of the armature
        pose_bone = armature_obj.pose.bones['base']
        
        bpy.context.view_layer.update()
        bl_height = abs(
            armature_obj.pose.bones['head'].tail.z - (
                armature_obj.pose.bones['foot_l'].tail.z + armature_obj.pose.bones['foot_r'].tail.z
            )/2
        )
        mp_height = blender_data_entry["center_xyh"][2]
        bl_mp_ratio = bl_height / mp_height
        mp_x_offset = blender_data_entry["center_xyh"][0]
        mp_y_offset = blender_data_entry["center_xyh"][1]
        bl_x_offset = mp_x_offset * bl_mp_ratio
        bl_y_offset = mp_y_offset * bl_mp_ratio
        
        foot_l_head_world_location = armature_obj.location + armature_obj.pose.bones["foot_l"].head
        foot_l_tail_world_location = armature_obj.location + armature_obj.pose.bones["foot_l"].tail
        foot_r_head_world_location = armature_obj.location + armature_obj.pose.bones["foot_r"].head
        foot_r_tail_world_location = armature_obj.location + armature_obj.pose.bones["foot_r"].tail
        
        pose_bone.location.x = bl_x_offset
        pose_bone.location.y += -min(
            foot_l_head_world_location[2], foot_l_tail_world_location[2],
            foot_r_head_world_location[2], foot_r_tail_world_location[2],
        )
        pose_bone.location.z = bl_y_offset
        pose_bone.keyframe_insert('location', frame=i+POSE_OFFSET)
        
        print(i, "/", len(blender_data))
        
