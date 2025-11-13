### Model Descriptions
1. **ModelSequence**: A GCN block is added before each spatial module. The processing order follows "spatial processing first, then temporal processing".
2. **MODELfusion** : Integrates GCN with the Mamba model. Specifically, after projecting the input \( x \), graph convolutional operations are performed on both forward and backward streams separately, followed by causal convolutions, and finally the State Space Model (SSM) is applied.
3. **modelGcnFull** (Ineffective): Bidirectional GCN is introduced at the initial input stage of the model. The input is constructed by concatenating coordinates, vectors, and angles, which are then fused for model feeding. Notably, multiple experimental iterations of this design failed to yield satisfactory performance.
4. **model_forecastangle**: Focuses on predicting joint angles. To switch between predicting 6 angle categories or 4 angle categories, only the final output dimension of the model needs to be adjusted.
5. **model_forecast_multi_angle**: A multi-task learning model fixed for predicting both 6 types and 7 types of joint angles.
6. **model_concatangle**: The model used for prediction in the second stage of the training pipeline.
7. **model_concatangle_wo_fusionhead**: A variant of the second-stage model without the fusion head.  