1. 先将原始数据转换到LabelMaker的格式
根据scrips下面的脚本
- 过滤一下图片的个数
2.生成2D语义图 （有GT语义的话，可以跳过）

3.生成3D语义Mesh/Point

4. CAD模型检索
代码在HOC-Search那边
5.生成全局OCC/单帧OCC
根据`mesh.ply`和`label.txt`



###



# 输出wn11 . 输出文件： consensus_wn199
python labelmaker/consensus.py --workspace /home/chm/datasets/arkitscenes/labelmaker/47333462
# 3D Lifting wn199
python -m labelmaker.lifting_3d.lifting_points --workspace /home/chm/datasets/arkitscenes/labelmaker/47333462 --output_mesh point_lifted_mesh_wn199.ply --label_folder intermediate/consensus_wn199

# 输出scannnet200   输出文件：  consensus_scannet200
python labelmaker/consensus_scannet200.py --workspace /home/chm/datasets/arkitscenes/labelmaker/47333462
# 3D Lifting scannet200
python -m labelmaker.lifting_3d.lifting_points --workspace /home/chm/datasets/arkitscenes/labelmaker/47333462 --output_mesh point_lifted_mesh_scannet200.ply --label_folder intermediate/consensus_scannet200