import mujoco_py
import numpy as np

def create_maze_xml(maze_matrix, maze_size, wall_size, map_size):
    xml_str = f'''
    <mujoco>
        <default>
            <joint armature="1" damping="1" limited="true"/>
            <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
        </default>

        <asset>
            <texture name="texplane" type="2d" builtin="flat" rgb1="1 1 1" width="4096" height="4096"/>
            <material name="MatPlane" reflectance="0" texture="texplane" texrepeat="1 1" texuniform="true"/>~
        </asset>    
        <worldbody>
        <camera name="top_view" mode="trackcom" pos="20 20 30" xyaxes="1 0 0 0 1 0" fovy = "50"/>
        <geom conaffinity="0" name="Goal" type="box" pos="0 0 0.5" size="0.5 0.5 0.5" rgba="0.8 0.0 0.0 1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="75 75 40" type="plane"/>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>

    '''
    
    col_size = len(maze_matrix)
    row_size = len(maze_matrix[0])

    for i in range(col_size):
        for j in range(row_size):
            if maze_matrix[i][j] == 1:  # 1인 경우 벽을 생성합니다.

                pos_x =  -(map_size-1)/2*maze_size + maze_size * i
                pos_y =  -(map_size-1)/2*maze_size + maze_size * j

                xml_str += f'''
                <geom conaffinity="1" type="cylinder" name="wall_{i}_{j}" size="{maze_size/2} {maze_size} {wall_size}" pos="{pos_x} {pos_y} {wall_size/2}" rgba="0.5 0.75 0.25 1" />
                '''

    xml_str += '''
        </worldbody>
    </mujoco>
    '''
    return xml_str

# 행렬 상에서의 (3,3) -> Mujoco (0,0)
# 행렬 상에서의 (7,7) -> Mujoco (20,20)
# 행렬 상에서의 (i,j) -> Mujoco (5j-15, 5i-15)

# 미로 매트릭스
#maze_matrix = np.array([
#    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#    [1, 0, 0, 0, 0, 0, 0, 1 ,0, 1, 1],
#    [1, 0, 1, 1, 1, 0, 1, 0 ,0, 0, 1],
#    [1, 0, 0, 0, 1, 0, 1, 0 ,1, 0, 1],
#    [1, 0, 1, 0, 1, 0, 1, 0 ,1, 0, 1],
#    [1, 0, 1, 0, 0, 0, 0, 0 ,1, 0, 1],
#    [1, 0, 1, 1, 1, 1, 1, 1 ,1, 0, 1],
#    [1, 0, 0, 0, 0, 0, 0, 0 ,1, 0, 1],
#    [1, 0, 1, 0, 0, 0 ,1, 1, 1, 0, 1],
#    [1, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 1],
#    [1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1]
#])

# 4 Rooms
# maze_matrix = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 1, 0, 0, 1, 1, 1, 0 ,0, 1, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1]
# ])

# HW2 maze
# maze_matrix = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 1, 0, 0, 1, 1, 1, 0 ,0, 1, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 1, 0, 0 ,0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1]
# ])

maze_matrix = np.array([
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 7, 1, 0, 1, 0, 0, 7, 1, 7, 1, 1],
     [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
     [1, 1, 0, 1, 7, 1, 0, 1, 7, 0, 0, 1, 1],
     [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
     [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 7, 1, 0, 0, 7, 1, 7, 1, 1],
     [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
     [1, 1, 7, 1, 7, 0, 0, 0, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1]
 ])

# maze_matrix = np.array([
#      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   
#      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#      [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#      [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#      [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#      [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#      [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#      [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#      [1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1],
#      [1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1]
#  ])


# 미로 설정을 위한 매개변수
wall_size = 1  # 벽의 크기
maze_size = 5  # 셀의 크기
map_size = 30

maze_matrix = np.zeros([map_size, map_size])

for i in range(map_size):
    for j in range(map_size):
        if np.random.uniform(low=0, high=1, size=1) > 0.10:
            maze_matrix[i][j] = 0
        else:
            maze_matrix[i][j] = 1

for i in range(map_size):
    for j in range(map_size):
        if (i * j) % 4 == 0:
            maze_matrix[i][j] = 1
        else:
            maze_matrix[i][j] = 0
        if (i % 4 ==0) or (j % 4 ==0):
            maze_matrix[i][j] = 0
        


# MuJoCo XML 파일 생성
maze_xml = create_maze_xml(maze_matrix, maze_size, wall_size , map_size)
with open('maze.xml', 'w') as f:
    f.write(maze_xml)

# MuJoCo XML 파일 로드
model = mujoco_py.load_model_from_path('maze.xml')

sim = mujoco_py.MjSim(model)

# 시뮬레이션 실행
viewer = mujoco_py.MjViewer(sim)
while True:
    sim.step()
    viewer.render()