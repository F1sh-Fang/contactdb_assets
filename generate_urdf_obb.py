import os
import trimesh
import numpy as np
import glob

# --- 配置区域 ---
BASE_ASSET_PATH = "contactdb_assets"
OUTPUT_NPZ_FILE = "contactdb_obb_extents.npz"

# --- 3D打印质量计算参数 (Sim2Real) ---
PLA_DENSITY_KG_M3 = 1300.0  # PLA 密度 ~1.3 g/cm^3 = 1300 kg/m^3
# 15% 填充率通常意味着整体密度约为实心的 25% - 30% (加上了实心的外壳和顶底)
# 如果想要更轻，可以设为 0.20；想要更重设为 0.30
PRINT_EFFECTIVE_DENSITY_RATIO = 0.25 

# --- 配置结束 ---

# URDF模板
URDF_TEMPLATE_BASE_ROOT = """<?xml version="1.0"?>
<robot name="{object_name}">
  
  <link name="base_link">
    <inertial>
      <origin xyz="{com_origin_in_base_str}" rpy="0 0 0"/>
      <mass value="{mass_value}"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="{visual_origin_xyz_str}" rpy="{visual_origin_rpy_str}"/>
      <geometry>
        <mesh filename="{visual_mesh}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="{visual_origin_xyz_str}" rpy="{visual_origin_rpy_str}"/>
      <geometry>
        <mesh filename="{collision_mesh}"/>
      </geometry>
    </collision>
  </link>

  <link name="center_of_mass"/>

  <link name="obb_center"/>

  <joint name="base_to_com" type="fixed">
    <parent link="base_link"/>
    <child link="center_of_mass"/>
    <origin xyz="{com_origin_in_base_str}" rpy="0 0 0"/>
  </joint>

  <joint name="base_to_obb" type="fixed">
    <parent link="base_link"/>
    <child link="obb_center"/>
    <origin xyz="{obb_origin_in_base_str}" rpy="0 0 0"/>
  </joint>

</robot>
"""

def calculate_mass_from_mesh(mesh, object_name):
    """
    根据体积和PLA参数计算质量。
    包含单位检测（mm vs m）和水密性检查。
    """
    # 1. 计算体积
    if mesh.is_watertight:
        volume = mesh.volume
        method = "Mesh Volume"
    else:
        # 如果模型有破洞（扫描件常见），原始体积计算可能出错，使用凸包体积近似
        volume = mesh.convex_hull.volume
        method = "Convex Hull Volume"

    # 2. 单位检测与转换
    # 经验法则：如果体积大于 0.1 (10cm x 100cm x 100cm)，这对于手持物体来说太大了，肯定是 mm^3
    # 1 m^3 = 10^9 mm^3
    # 即使是大物体（如 0.5m x 0.5m x 0.5m），体积也是 0.125
    is_mm = False
    if volume > 1.0: 
        # 肯定是 mm，转换为 m^3
        volume_m3 = volume / 1e9
        is_mm = True
    else:
        # 肯定是 m
        volume_m3 = volume
        is_mm = False

    # 3. 计算质量
    # Mass = Vol * Density * Effective_Ratio
    mass = volume_m3 * PLA_DENSITY_KG_M3 * PRINT_EFFECTIVE_DENSITY_RATIO
    
    # 4. 安全限制：防止质量过小或过大导致仿真不稳定
    mass = max(0.01, mass) # 最小 10g

    print(f"  -> [{method}] 原始体积: {volume:.4f} ({'mm^3' if is_mm else 'm^3'})")
    print(f"  -> 计算质量 (PLA 15% infill): {mass:.4f} kg")
    
    return mass

def get_orientation_and_extents_from_projection(mesh):
    """
    通过PCA分析物体在XY平面的投影，计算旋转矩阵和[长,宽,高]尺寸。
    """
    obb = mesh.bounding_box_oriented
    obb_vertices_3d = obb.vertices
    points_2d = obb_vertices_3d[:, :2]

    centered_points = points_2d - np.mean(points_2d, axis=0)
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    y_direction_2d = eigenvectors[:, -1]
    y_final = np.array([y_direction_2d[0], y_direction_2d[1], 0.0])
    z_final = np.array([0.0, 0.0, 1.0])
    x_final = np.cross(y_final, z_final)
    
    x_final /= np.linalg.norm(x_final)
    y_final /= np.linalg.norm(y_final)
    z_final /= np.linalg.norm(z_final)
    
    R_final = np.column_stack([x_final, y_final, z_final])

    transformed_vertices = obb_vertices_3d @ R_final
    min_coords = np.min(transformed_vertices, axis=0)
    max_coords = np.max(transformed_vertices, axis=0)
    aabb_extents = max_coords - min_coords
    
    final_extents = np.array([aabb_extents[1], aabb_extents[0], aabb_extents[2]])
    
    return R_final, final_extents

def generate_urdf_files_final():
    print(f"--- 开始生成URDF文件 (自动计算质量: PLA {PLA_DENSITY_KG_M3}kg/m3, Ratio {PRINT_EFFECTIVE_DENSITY_RATIO}) ---")
    
    target_base_path = os.path.join(BASE_ASSET_PATH, "contactdb_assets")
    if not os.path.isdir(target_base_path):
        if os.path.isdir(os.path.join(BASE_ASSET_PATH)):
             target_base_path = BASE_ASSET_PATH
        else:
            print(f"错误: 目录 '{target_base_path}' 不存在。")
            return

    all_obb_extents = {}
    object_folders = sorted([d for d in os.listdir(target_base_path) if os.path.isdir(os.path.join(target_base_path, d))])
    
    count = 0
    for folder_name in object_folders:
        object_name = folder_name
        folder_path = os.path.join(target_base_path, folder_name)
        stl_filename = f"{object_name}.stl"
        mesh_path = os.path.join(folder_path, stl_filename)

        if not os.path.exists(mesh_path):
            stls = glob.glob(os.path.join(folder_path, "*.stl"))
            if stls:
                mesh_path = stls[0]
                stl_filename = os.path.basename(mesh_path)
            else:
                print(f"  -> 跳过 {object_name}: 找不到 .stl 文件")
                continue

        print(f"正在处理: {object_name}")

        try:
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
        except Exception as e:
            print(f"  -> 错误: 加载模型失败: {e}")
            continue

        # 1. 自动计算质量
        mass_val = calculate_mass_from_mesh(mesh, object_name)

        # 2. 计算对齐和尺寸
        aligned_rotation, final_extents = get_orientation_and_extents_from_projection(mesh)
        all_obb_extents[object_name] = final_extents
        
        # 3. 计算 Base Link 位置 (投影质心到最低面)
        com_original = mesh.centroid
        
        vertices = mesh.vertices
        z_values = vertices[:, 2]
        min_z = np.min(z_values)
        
        base_link_origin_world = np.array([com_original[0], com_original[1], min_z])
        
        T_base_world = np.eye(4)
        T_base_world[:3, :3] = aligned_rotation
        T_base_world[:3, 3] = base_link_origin_world
        
        T_visual_collision = np.linalg.inv(T_base_world)
        
        com_in_base = T_visual_collision @ np.append(com_original, 1)
        obb_original_center = mesh.bounding_box_oriented.centroid
        obb_in_base = T_visual_collision @ np.append(obb_original_center, 1)

        com_str = " ".join(f"{x:.6f}" for x in com_in_base[:3])
        obb_str = " ".join(f"{x:.6f}" for x in obb_in_base[:3])
        vis_xyz = T_visual_collision[:3, 3]
        vis_rpy = trimesh.transformations.euler_from_matrix(T_visual_collision[:3, :3], 'sxyz')
        vis_xyz_str = " ".join(f"{x:.6f}" for x in vis_xyz)
        vis_rpy_str = " ".join(f"{x:.6f}" for x in vis_rpy)

        urdf_content = URDF_TEMPLATE_BASE_ROOT.format(
            object_name=object_name,
            visual_mesh=stl_filename,
            collision_mesh=stl_filename,
            com_origin_in_base_str=com_str,
            obb_origin_in_base_str=obb_str,
            visual_origin_xyz_str=vis_xyz_str,
            visual_origin_rpy_str=vis_rpy_str,
            mass_value=f"{mass_val:.4f}"
        )
        
        output_urdf_path = os.path.join(folder_path, f"{object_name}.urdf")
        
        try:
            with open(output_urdf_path, 'w') as f:
                f.write(urdf_content)
            count += 1
        except IOError as e:
            print(f"  -> 错误: 写入 URDF 失败: {e}")

    if all_obb_extents:
        try:
            np.savez_compressed(OUTPUT_NPZ_FILE, **all_obb_extents)
            print(f"\n成功处理 {count} 个物体。OBB尺寸已保存到: '{OUTPUT_NPZ_FILE}'")
        except Exception as e:
            print(f"\n保存 .npz 失败: {e}")
    else:
        print("\n未处理任何物体。")

if __name__ == "__main__":
    generate_urdf_files_final()