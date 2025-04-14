# ECCV 2024: Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation

Official PyTorch implementation of the paper: [**ECCV2024: Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00589.pdf)

## Overview

This repository contains the code for our paper on instance-dependent noisy-label learning, which introduces a novel approach to estimate the noise rate from the training data and use it for sample selection. The key insight of our work is that most state-of-the-art (SOTA) noisy-label learning methods use sample selection techniques that rely on a pre-defined curriculum for selecting clean/noisy samples, which is often sub-optimal. Our approach estimates the actual noise rate present in the dataset, leading to a more effective and dynamically adapted sample selection.

## Method

Our method is based on a novel probabilistic graphical model that captures the generation of noisy labels conditioned on the clean labels and image features. The core components include:

1. **Graphical Model**: We model the relationship between observed variables (data X and noisy labels Ŷ) and latent variables (clean labels Y) to estimate the noise rate ε.

2. **Variational EM Algorithm**: We use a variational expectation-maximization approach to:
   - **E-step**: Estimate the posterior distribution of clean labels given noisy data
   - **M-step**: Update the parameters of the model, including the noise rate ε

3. **Sample Selection Strategy**: We introduce a curriculum R(t) = 1-ε, where ε is the estimated noise rate, to dynamically select clean and noisy samples during training.

4. **Integration with SOTA Methods**: Our approach can be seamlessly integrated with existing noisy-label learning methods, enhancing their performance.

## Code Structure

The main implementation is in the `NoiseRateEstimator` class, which includes:

- **Expectation Step**: Updates the posterior model parameters
- **Maximization Step**: Updates the noise model parameters and noise rate
- **Sample Selection**: Divides samples into clean and noisy sets based on the estimated noise rate

```python
# Example of how to use the NoiseRateEstimator
estimator = NoiseRateEstimator(feature_dim, num_classes)

# Perform E-step
q_y = estimator.expectation_step(features, y_hat, clean_model)

# Perform M-step
loss, epsilon = estimator.maximization_step(features, y_hat, q_y, clean_model)

# Get clean and noisy indices
criterion_scores = compute_criterion_scores(features, y_hat, clean_model)
clean_indices, noisy_indices = estimator.get_clean_noisy_indices(criterion_scores)
```

## Integration with SOTA Methods

Our approach can be easily integrated with various state-of-the-art noisy-label learning methods. Here's how to integrate it with existing methods:

### Integration with DivideMix

```python
# Initialize the noise rate estimator
estimator = NoiseRateEstimator(feature_dim, num_classes)

# During training
for epoch in range(num_epochs):
    # Extract features and compute loss values
    features = feature_extractor(x_batch)
    criterion_scores = compute_loss(clean_model, x_batch, y_hat_batch)
    
    # Perform EM steps
    q_y = estimator.expectation_step(features, y_hat_batch, clean_model)
    loss, epsilon = estimator.maximization_step(features, y_hat_batch, q_y, clean_model)
    
    # Get clean and noisy indices using estimated noise rate
    clean_indices, noisy_indices = estimator.get_clean_noisy_indices(criterion_scores)
    
    # Train DivideMix using the obtained clean/noisy split
    train_dividemix(clean_indices, noisy_indices, ...)
```

### Integration with Other Methods (C2D, SSR, etc.)

The integration follows a similar pattern for other methods. The key is to replace their sample selection strategy with our noise-rate-based approach:

1. Initialize the `NoiseRateEstimator`
2. Run the E-step and M-step to estimate the noise rate
3. Use `get_clean_noisy_indices()` to obtain clean and noisy sample indices
4. Feed these indices to the original method's training procedure

## Experimental Results

Our method consistently improves the performance of various SOTA methods on multiple benchmarks:

- **CIFAR-100 with IDN**: Improved accuracy by up to 6% for various noise rates
- **Red Mini-ImageNet**: Enhanced performance across different noise rates
- **Clothing1M and WebVision**: Better performance on real-world noisy datasets

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{garg2024instance,
  title={Instance-Dependent Noisy-Label Learning with Graphical Model Based Noise-Rate Estimation},
  author={Garg, Arpit and Nguyen, Cuong and Felix, Rafael and Do, Thanh-Toan and Carneiro, Gustavo},
  booktitle={European Conference on Computer Vision},
  pages={372--389},
  year={2024},
  organization={Springer}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank the reviewers for their valuable feedback. This work was supported by the Australian Research Council through grant FT190100525.