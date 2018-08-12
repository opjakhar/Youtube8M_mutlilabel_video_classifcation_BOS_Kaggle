Here we have uploaded our youtube8M mutlilabel video classification challenge on Kaggle.we are using starter code provided by the team(kaggle or Google).
We ran this code on google cloud plateform using 4Gpus(even can use only one Gpus but it was taking lots of time)
python version: 3.5
runtime version: 1.8
below is the script and some important commmand required for run this code.for more understadning of this challege and code you can read README1.md file.
******************************************************************************************************************************************************
-------------------***************script for training and generating pridction using  LstmModel_late******************************** -----------------
******************************************************************************************************************************************************


total number of epoch: 3 or 4
python verion=3.5
runtime version=1.8

--------------------training model-----------------------------------------------------------------------------
BUCKET_NAME=gs://jakhar95op_bucket12
gsutil mb -l asia-east1 $BUCKET_NAME 


 JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME --package-path=youtubeMix --module-name=youtubeMix.train --staging-bucket=gs://jakhar95op_bucket12 --region=asia-east1 --config=youtubeMix/cloudml-4gpu.yaml -- --train_data_pattern='gs://youtube8m-ml/2/frame/*i*/*i*.tfrecord' --model=LstmModel_late --train_dir=gs://jakhar95op_bucket12/model1 --batch_size=256 --num_epochs=1  --frame_features=True --feature_names="rgb, audio" --feature_sizes="1024, 128" --video_level_classifier_model=MoeModel --crop=True --crop_interval=5 --start_new_model=False --base_learning_rate=0.001




	------------------generating checkpoint for inference-------------------------------------------------

 JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME --package-path=youtubeMix --module-name=youtubeMix.eval --staging-bucket=gs://jakhar95op_bucket12 --region=asia-east1 --config=youtubeMix/cloudml-4gpu.yaml -- --eval_data_pattern='gs://youtube8m-ml/2/frame/validate/validate*.tfrecord' --batch_size=1024 --num_epochs=1--frame_features=True --feature_names="rgb, audio" --feature_sizes="1024, 128" --model=LstmModel_late --crop=True --crop_interval=5 --train_dir=gs://jakhar95op_bucket12/model1 --run_once=True

-----------------------generating prediction---------------------------------------------------

JOB_TO_EVAL=model1
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME --package-path=youtubeMix --module-name=youtubeMix.inference --staging-bucket=gs://jakhar95op_bucket12  --region=asia-east1 --config=youtubeMix/cloudml-4gpu.yaml -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' --train_dir=gs://jakhar95op_bucket12/${JOB_TO_EVAL} --frame_features=True --feature_names="rgb, audio" --feature_sizes="1024, 128" --batch_size=1024 --crop=True --crop_interval=5 --output_file=gs://jakhar95op_bucket12/${JOB_TO_EVAL}/fifth_predictions.csv






******************************************************************************************************************************************************
-------------------***************script for training and generating pridction using  simple LSTM****************** ----------------------------------
******************************************************************************************************************************************************

BUCKET_NAME=gs://jakhar95op_demo_checkk
gsutil mb -l asia-east1 $BUCKET_NAME
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtubeMix --module-name=youtubeMix.train \
--staging-bucket=gs://cs17m062_bidir_lstm_mlp_bucket \
--region=asia-east1 \
--config=youtubeMix/cloudml-4gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml/2/frame/*i*/*i*.tfrecord' \
--model=BiLstmModel_sep \
--train_dir=gs://cs17m062_bidir_lstm_mlp_bucket/model1 \
--batch_size=256 --num_epochs=1  \
--frame_features=True --feature_names="rgb, audio" \
--feature_sizes="1024, 128" \
--start_new_model=False \
--video_level_classifier_model=MLPModel \
--crop=True --crop_interval=5 \
-base_learning_rate=0.001


-------------------------------------attention eval---------------------------------------------------
JOB_TO_EVAL=model1
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtubeMix --module-name=youtubeMix.eval \
--staging-bucket=gs://cs17m062_bidir_lstm_mlp_bucket --region=asia-east1 \
--config=youtubeMix/cloudml-4gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml/2/frame/validate/validate*.tfrecord' \
--batch_size=1024 --num_epochs=1\
--frame_features=True --feature_names="rgb, audio" \
--feature_sizes="1024, 128" \
--model=BiLstmModel_sep \
--crop=True --crop_interval=5 \
--train_dir=gs://cs17m062_bidir_lstm_mlp_bucket/model1 \
--run_once=True

----------------------------------attention inference------------------------------------------------------------
JOB_TO_EVAL=model1
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtubeMix --module-name=youtubeMix.inference1 \
--staging-bucket=gs://cs17m062_bidir_lstm_mlp_bucket  --region=asia-east1 \
--config=youtubeMix/cloudml-4gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
--train_dir=gs://cs17m062_bidir_lstm_mlp_bucket/${JOB_TO_EVAL} \
--frame_features=True --feature_names="rgb, audio" \
--feature_sizes="1024, 128" \
--batch_size=1024 \
--crop=True --crop_interval=5 \
--output_file=gs://cs17m062_bidir_lstm_mlp_bucket/${JOB_TO_EVAL}/predictions.csv



#---other important commands---
(1) transfer files from google cloud to  local machine 
				gcloud compute scp jakhar95op@instance-1:~/youtubeMix/*  "/home/omprakashjakhar/Downloads/youtubeMix/"

(2) transfer files from local machine to google cloud 
				gcloud compute scp --recurse /home/omprakashjakhar/youtube-zhs/inference1.py jakhar95op@instance-1:~/youtubeMix/

(3) cancle running job($JOB_NBAME) on google cloud
				gcloud ml-engine jobs cancel $JOB_NAME

(4) view job($JOB_NBAME) info running on google cloud 
				gcloud ml-engine jobs stream-logs $JOB_NAME
