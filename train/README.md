## Train Jigsaw model  
Training CFN model from scratch  
`$user train_jigsaw.sh`  
## Train classifier model  
Using CFN weights to initialize all conv leayers of the standard model  
`$user train_classifier.sh`  
## Train Jigsaw model  
Using pretrained weight of standard model on ImageNet  
`$user train_jigsaw.sh`  
