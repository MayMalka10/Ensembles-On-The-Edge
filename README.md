## Decentralized Low-Latency Collaborative Inference on the Edge

Performing Edge Ensembles via Bagging & Random Noise Initializations:

1. The Config_files folder comprised of all the setting for running the code:
  
   - To select number of participating users, go to Ensemble/ensemble_params.yaml, change the value in:
     n_ensemble: num_of_users
     
   - To select number of quantization bits, go to Quantization/quantization_params.yaml, change the value in: 
     n_embed: num_of_vectors
     n_parts: num_of_parts (which we split each vector)

   - To define all other hyperparameters, go to Training/training_params.yaml, change the value in: 
     lr: learning_rate
     batch_size: size_of_batch
     
     
2. The Archs folder comprised of all the setting for running the code:

   - In the folder can be found two different architectures: MobileNetV2 and ResNet18.
   - Each one can be selected in the main.py file via the imports section.
