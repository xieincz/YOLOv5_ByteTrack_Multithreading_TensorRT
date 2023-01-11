echo -e "\nAttention! This is only design for online coding evnerviment.\n"
echo -e "\nIf you want to use it in local, please change the path in this file.\n"

model_dir=/project/ev_sdk/src/cppmodels/
src_dir=/project/ev_sdk/src/
#最后面的斜杠不能少

model_type=l6
#可以是 n6 s6 m6 l6 x6之一

model_name=yolov5${model_type}
yolov5_cpp=yolov5_cpp_6
############################################################################################################
#/project/ev_sdk/model/yolov5l.pt -> /project/ev_sdk/model/yolov5l.engine
cp ${src_dir}${yolov5_cpp}/gen_wts.py ${src_dir}yolov5/
echo -e "\nConverting ${model_dir}${model_name}.pt to ${model_dir}${model_name}.wts\n"
python ${src_dir}yolov5/gen_wts.py -w ${model_dir}${model_name}.pt -o ${model_dir}${model_name}.wts
# update CLASS_NUM in yololayer.h if your model is trained on custom dataset
echo -e "\nAttention! Please update CLASS_NUM in ${src_dir}${yolov5_cpp}/yololayer.h if your model is trained on custom dataset.\n"
echo -e "\nConverting ${model_dir}${model_name}.wts to ${model_dir}${model_name}.engine\n"
${src_dir}${yolov5_cpp}/build/yolov5 -s ${model_dir}${model_name}.wts ${model_dir}${model_name}.engine ${model_type}
rm ${model_dir}${model_name}.wts
echo -e "\nDone.\n"