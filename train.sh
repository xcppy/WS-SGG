export CUDA_VISIBLE_DEVICES=1,3
export CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=$1



##############################################################################################


if [ $MODEL_TYPE == "VG_Caption_Ground_Motif" ]
then
    # Visual Genome Caption supervision + offline detector + motif as relation predictor
    python -m torch.distributed.launch --master_port 10023 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/vg_caption_ground_motifs \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER False \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/ws_caption_labels/vg-151-53/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/ws_caption_labels/vg-151-53/vg_triplet_labels.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/ws_caption_labels/vg-151-53/vg2VG_word_map.npy \
    WSVL.USE_GROUND True \
    WSVL.GROUND_FILE ./datasets/ground_files/vg_caption.pkl \
    WSVL.SKIP_TRAIN False
fi

##############################################################################################


if [ $MODEL_TYPE == "VG_Caption_SGNLS_Motif" ]
then
    # Visual Genome Caption supervision + offline detector + motif as relation predictor
    python -m torch.distributed.launch --master_port 10023 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/vg_caption_sgnls_motifs \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER False \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/ws_caption_labels/vg-151-53/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/ws_caption_labels/vg-151-53/vg_triplet_labels.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/ws_caption_labels/vg-151-53/vg2VG_word_map.npy \
    WSVL.USE_GROUND False \
    WSVL.SKIP_TRAIN False
fi


##############################################################################################


if [ $MODEL_TYPE == "VG_Caption_Ground_Uniter" ]
then
    # Visual Genome Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/vg_caption_ground_uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/ws_caption_labels/vg-151-53/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/ws_caption_labels/vg-151-53/vg_triplet_labels.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/ws_caption_labels/vg-151-53/vg2VG_word_map.npy \
    WSVL.USE_GROUND True \
    WSVL.GROUND_FILE ./datasets/ground_files/vg_caption.pkl \
    WSVL.SKIP_TRAIN False 
fi

##############################################################################################

if [ $MODEL_TYPE == "Unlocal_VG_Ground_Motif" ]
then
    # Unlocalized scene graph supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=1 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 128 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Unlocal_VG_ground_Motif \
    TEST.IMS_PER_BATCH 1 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER False \
    WSVL.USE_CAP_TRIP False \
    WSVL.HARD_DICT_MATCH ./datasets/vg/oid_word_map_synset.npy \
    WSVL.USE_GROUND True \
    WSVL.GROUND_FILE ./datasets/ground_files/vg_unlocal.pkl \
    WSVL.SKIP_TRAIN False 
fi

##############################################################################################

if [ $MODEL_TYPE == "Unlocal_VG_SGNLS_Uniter" ]
then
    # Unlocalized scene graph supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 128 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/unlocal_vg_sgnls_uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD False \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP False \
    WSVL.HARD_DICT_MATCH ./datasets/vg/oid_word_map_synset.npy \
    WSVL.SKIP_TRAIN False 
fi

##############################################################################################


if [ $MODEL_TYPE == "Caption_COCO_Uniter" ]
then
    # COCO Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/coco_caption_uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/COCO-144-57/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/COCO-144-57/COCO_triplet_labels.npy \
    WSVL.LOSS_CLS_WEIGHTS ./datasets/vg/caption_labels/COCO-144-57/COCO2vg_weights.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/COCO-144-57/COCO2VG_word_map.npy \
    WSVL.SKIP_TRAIN False 
fi
