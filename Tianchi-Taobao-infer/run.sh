python3 /yolo_master/data_prepare/json2txt/dataset.py
python3 /yolo_master/data_prepare/json2txt/maketxt.py
python3 /yolo_master/data_prepare/json2txt/voc_label.py
#训练全部视频和图片数据
#python3 -m torch.distributed.launch --nproc_per_node=4 /yolo_master/train.py  --multi-scale --output_model_dir='/yolo_master/myspace/yolo_model.pt'
#仅训练图片数据
python3 -m torch.distributed.launch --nproc_per_node=4 /yolo_master/train.py  --data='/yolo_master/data/clothes_retrieval_img.yaml' --multi-scale --output_model_dir='/yolo_master/myspace/yolo_model_img.pt'
#仅训练视频数据
python3 -m torch.distributed.launch --nproc_per_node=4 /yolo_master/train.py  --data='/yolo_master/data/clothes_retrieval_vid.yaml' --multi-scale --single-cls  --output_model_dir='/yolo_master/myspace/yolo_model_vid.pt'
python3 /metric_sub/src_train/preprocess.py
{
python3 /metric_sub/src_train/train_net_b5.py
}&
{
python3  /metric_sub/src_train/train_net_deit.py
}&
{
python3  /metric_sub/src_train/train_net_b6.py
}&
wait
python3 /yolo_master/detect_img.py --save-txt --nosave --weights='/yolo_master/myspace/yolo_model_img.pt'
python3 /yolo_master/detect_vid.py --save-txt --nosave --weights='/yolo_master/myspace/yolo_model_vid.pt'
python3 /metric_sub/src_infer/run_fusion_net.py
