
# 批量脚本
python scripts/batch_process_arkit.py  --scan_base_dir /data1/chm/datasets/arkitscenes/raw/Training/  --target_base_dir /data1/chm/datasets/arkitscenes/LabelMaker/Training   --max_scans 600   --verbose



2. 先将原始数据转换到LabelMaker的格式
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


语义标注的时间
- 一个116张图片的场景，做完语义标注需要24分钟


# 自动化标注的流程

1. 将原始数据转换成LabelMaker的格式
2. 语义标注
3. CAD标注检索
4. 生成Occupancy真值


## 语义标注
### 标注单个场景
bash ./scripts/semantic.sh

### 标注多个场景
bash scripts/semantic.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data
bash scripts/semantic.sh /data1/chm/datasets/arkitscenes/LabelMaker/Training

bash scripts/semantic_batch.sh /data1/chm/datasets/arkitscenes/LabelMaker/Training

bash scripts/semantic_batch.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3

## CAD标注检索
TODO

## 3D Lifting
bash scripts/point_3d.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data
bash scripts/point_3d.sh /data1/chm/datasets/arkitscenes/LabelMaker/Training
bash scripts/point_3d.sh /data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3

## 不同服务器之间数据同步
rsync -avzL --progress --exclude='intermediate' chm@49.52.10.241:/data1/chm/datasets/arkitscenes/LabelMaker/mini_data/ .
rsync -avzL --progress --exclude='intermediate' chm@49.52.10.241:/data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v2 .
rsync -avzL --progress --exclude='intermediate' --exclude='point_lifted_mesh.ply' --exclude='mesh.ply' chm@49.52.10.241:/data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3 .

rsync -avzL --progress --exclude='intermediate' chm@10.1.200.4:/data1/share/NYU_dataset .
rsync -avzL --progress --exclude='mnt' chm@10.1.200.4:/data1/share/occscannet .

## 数据迁移脚本

## 列出已经处理好的场景
python find_valid_parent_folders.py



# 数据的存储位置
ECNU 62 - /home/chm/workspaces/chm_data/datasets/arkitscenes/LabelMaker
ECNU 241 - /data1/chm/datasets/arkitscenes/LabelMaker


python scripts/batch_process_arkit.py --scan_base_dir /data1/chm/datasets/arkitscenes/raw/Training --target_base_dir /data1/chm/datasets/arkitscenes/LabelMaker/Training --verbose


# 数据移动
bash move_processed_raw_data.sh
