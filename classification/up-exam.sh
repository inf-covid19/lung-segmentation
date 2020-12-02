#! /bin/bash


exam_id='NEG-011'
study_id='1.3.6.1.4.1.5962.99.1.2172677760.153238374.1595605577344.22135.0'
meta_exam='NEG-011'

# Copy files...

# command="scp -r -P 9090 chicobentojr@127.0.0.1:/ssd/share/CT-Original/NII-HMV/exame-pulmao/$exam_id/*/* ."
command="scp -r -P 9090 chicobentojr@127.0.0.1:/ssd/share/CT-Original/NII-HCPA/*/$exam_id* ."
printf "copying exam ... $command\n"
$command

command="scp -r -P 9090 chicobentojr@127.0.0.1:/ssd/share/CT-Segmentado/03-P-HNN/MP4-HMV-HCPA/$exam_id.mp4 ."
printf "copying video ... $command\n"
$command

command="scp -r -P 9090 chicobentojr@127.0.0.1:/ssd/share/CT-Segmentado/03-P-HNN/NII-HCPA/exame-pulmao/$exam_id/*/*  ."
printf "copying files ... $command\n"
$command
 

# Uploading files...

file="$exam_id.mp4"
tag='3d-video'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="crop_by_mask_$meta_exam.nii.gz"
tag='nii-crop-by-mask'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="ggo_$meta_exam.nii.gz"
tag='nii-ggo'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="lungs_$meta_exam.nii.gz"
tag='nii-lungs'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="mask_$meta_exam.nii.gz"
tag='nii-mask'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="prob_map_$meta_exam.nii.gz"
tag='nii-prob-map'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="prob_mask_$meta_exam.nii.gz"
tag='nii-prob-mask'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

file="$meta_exam.nii.gz"
tag='nii-exam'
command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
printf "uploading $tag ... $command\n"
$command

for f in $(seq 1); do
    file="$(ls $meta_exam\_$f*.png)"
    tag="top-lesion-$f"
    command="curl --request PUT --data-binary @$file https://cidia.ufrgs.dev/storage/v1/exam-media/$study_id/$tag/$file"
    printf "uploading $tag ... $command\n"
    $command
done;

