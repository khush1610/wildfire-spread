# Wildfire Spread Prediction

## Introduction
This repository contains an implementation of the "Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data" paper by Huot et al. (2021). The project uses a CNN Autoencoder to predict wildfire spread over a 64×64 grid, leveraging the Next Day Wildfire Spread Dataset. The model forecasts a 64×64×1 FireMask for the next day based on a 64×64×12 input, achieving an AUC-PR of 28.53%.

## Dataset
- **Source**: Derived from the Next Day Wildfire Spread Dataset, available on Kaggle and originally from Google Research's repository (https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread).
- **Components**:
  - **PrevFireMask (time t)**: Binary 64×64 grid (1 = fire, 0 = no fire) showing current fire locations.
  - **FireMask (time t+1)**: Target 64×64 grid for next-day fire prediction.
  - **Features**: Elevation, NDVI, Precipitation, Temperature, Wind Speed, and more, forming the 12-channel input.
- **Preprocessing**: Includes normalization, feature selection, and handling of uncertain pixels (e.g., clouds, smoke).

## Model Architecture
- **Encoder**: Convolutional layers (32 to 256 filters) with batch normalization, dropout, and residual blocks to compress 64×64×12 to 8×8×256.
- **Bottleneck**: Residual blocks for feature extraction, maintaining 8×8×256.
- **Decoder**: Upsampling layers with residual blocks to reconstruct 64×64×1 FireMask.
- **Details**: Uses weighted binary cross-entropy loss, Adam optimizer (learning rate 0.0001), and 450 training epochs.

## Acknowledgements
- Inspired by the work of Huot et al. (2021), available at arXiv.
- Dataset and initial concepts adapted from Google Research's Next Day Wildfire Spread project.

## Notes
- The model faces challenges with rapid fire spread, small fires, cloudy conditions, new ignitions, and extreme weather due to data limitations (1 km resolution, daily averages).
