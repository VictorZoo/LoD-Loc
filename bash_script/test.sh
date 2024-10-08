## UAVD4L-LoD ##
# inTraj
python -m maploc.evaluation.evaluation_UAVD4L \
  --experiment reproduce/UAV.ckpt \
  model.name='LoD_Loc' \
  data.split=./split/UAV_inTraj_test.json \
  data.name='UAVD4L-LoD' \
  data.scenes='['inTraj']'\
  model.num_sample_val='[[8, 10, 10, 30],[8, 10, 10, 30],[8, 10, 10, 30]]'\
  model.lamb_val='[0.8,0.8,0.8]'\
  data.loading.val.interval=1 \
  --output_name inTraj

# outTraj
python -m maploc.evaluation.evaluation_UAVD4L \
  --experiment reproduce/UAV.ckpt \
  model.name='LoD_Loc' \
  data.split=./split/UAV_outTraj_test.json \
  data.name='UAVD4L-LoD' \
  data.scenes='['outTraj']' \
  model.num_sample_val='[[8, 10, 10, 30],[8, 10, 10, 30],[8, 10, 10, 30]]'\
  model.lamb_val='[0.8,0.8,0.8]'\
  data.loading.val.interval=1 \
  --output_name outTraj

## Swiss-EPFL ##
# inPlace
python -m maploc.evaluation.evaluation_Swiss \
  --experiment reproduce/Swiss.ckpt \
  model.name='LoD_Loc' \
  data.split=./split/Swiss_inPlace_test.json \
  data.name='Siwss-EPFL' \
  data.scenes='['inPlace']'\
  model.num_sample_val='[[8, 10, 10, 30],[8, 10, 10, 30],[8, 10, 10, 30]]'\
  model.lamb_val='[0.8,0.8,0.8]'\
  data.loading.val.interval=1 \
  --output_name inPlace
  
# outPlace
python -m maploc.evaluation.evaluation_Swiss \
  --experiment reproduce/Swiss.ckpt \
  model.name='LoD_Loc' \
  data.split=./split/Swiss_outPlace_test.json \
  data.name='Siwss-EPFL' \
  data.scenes='['outPlace']' \
  model.num_sample_val='[[8, 10, 10, 30],[8, 10, 10, 30],[8, 10, 10, 30]]'\
  model.lamb_val='[0.8,0.8,0.8]'\
  data.loading.val.interval=1 \
  --output_name outPlace
