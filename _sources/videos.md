# The Video Lectures

Every chapter of this book grew out of a set of short lecture videos — **89 videos, about 9 hours 27 minutes** in the Fall 2026 edition of OPIM 5509. Most run five to nine minutes, and each one walks through a notebook or a set of slides you can open yourself.

:::{admonition} Where to watch, and where to read
:class: tip
The videos themselves live in the course site (HuskyCT), so they are for enrolled students. The **transcripts are public**: every video below links to its cleaned-up transcript, with timings, so you can search for the moment a topic is covered. Videos are numbered in the order they were recorded.
:::

## At a glance

| Chapter | Videos | Runtime (h:mm) |
| :-- | --: | --: |
| {doc}`Chapter 1 — Refresher: EDA and machine learning <m1_refresher/index>` | 9 | 0:51 |
| {doc}`Chapter 2 — Dense neural networks <m2_dense/index>` | 24 | 2:30 |
| {doc}`Chapter 3 — Convolutional neural networks <m3_cnn/index>` | 17 | 1:50 |
| {doc}`Chapter 4 — Recurrent networks for numeric sequences <m4_rnn_numeric/index>` | 26 | 2:54 |
| {doc}`Chapter 5 — Recurrent networks for text <m5_rnn_text/index>` | 13 | 1:19 |
| **All five chapters** | **89** | **9:27** |

## Chapter 1 — Refresher: EDA and machine learning

*9 videos, 0:51. Read the chapter: {doc}`m1_refresher/index`.*

| # | Video | Length | |
| --: | :-- | --: | :-- |
| 1 | Welcome to IDL from Dr. Dave! | 2:55 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_01_Welcome_to_IDL_from_Dr_Dave.srt) |
| 2 | Google Colaboratory and the UConn Library | 2:42 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_02_Google_Colaboratory_and_the_UConn_Library.srt) |
| 3 | Welcome, Colab and GitHub | 7:03 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_03_Welcome_Colab_and_GitHub.srt) |
| 4 | Introduction to EDA on CA Housing | 5:11 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_04_Introduction_to_EDA_on_CA_Housing.srt) |
| 5 | Outliers, censored data, univariate and bivariate plots | 7:00 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_05_Outliers_censored_data_univariate_and_bivariate_plots.srt) |
| 6 | Geographic EDA and final data cleaning | 2:33 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_06_Geographic_EDA_and_final_data_cleaning.srt) |
| 7 | Intro to end-to-end ML for regression | 5:50 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_07_Intro_to_end-to-end_ML_for_regression.srt) |
| 8 | Fitting and evaluating ML models for regression | 9:40 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_08_Fitting_and_evaluating_ML_models_for_regression.srt) |
| 9 | End-to-end ML for classification | 8:39 | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module1/transcripts/5509IDL_M1_09_End-to-end_ML_for_classification.srt) |

## Chapter 2 — Dense neural networks

*24 videos, 2:30. Read the chapter: {doc}`m2_dense/index`.*

### M2.1 — Theory, by hand

*10 videos, about 64 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 1 | Introduction to forward propagation | 6:51 | `1_ForwardPropagation` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_01_IDL_Introduction_to_forward_propagation.srt) |
| 2 | Golden rule of dot products | 7:25 | `1_ForwardPropagation` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_02_IDL_Golden_rule_of_dot_products.srt) |
| 3 | Trainable Parameters in a dense neural network | 7:51 | `1_ForwardPropagation` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_03_IDL_Trainable_Parameters_in_a_dense_neural_network.srt) |
| 4 | Hot and cold learning | 5:35 | `2_HotAndCold` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_04_IDL_Hot_and_cold_learning.srt) |
| 5 | Hot and cold learning, direction and amount (derivatives) | 6:45 | `2_HotAndCold` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_05_IDL_Hot_and_cold_learning_direction_and_amount_derivatives.srt) |
| 6 | Divergence: scale your data or use a learning rate! | 2:10 | `2_HotAndCold` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_06_IDL_Divergence_scale_your_data_or_use_a_learning_rate.srt) |
| 7 | Learning a full dataset with stochastic gradient descent | 7:47 | `3_BackProp_and_ReLU` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_07_IDL_Learning_a_full_dataset_with_stochastic_gradient_descent.srt) |
| 8 | Gradient descent and an intro to activation functions (ReLU) | 6:18 | `3_BackProp_and_ReLU` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_08_IDL_Gradient_descent_and_an_intro_to_activation_functions_ReLU.srt) |
| 9 | One full iteration for a single row (forward and back propagation) | 9:04 | `3_BackProp_and_ReLU` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_09_IDL_One_full_iteration_for_a_single_row_forward_and_back_propagation.srt) |
| 10 | Learn the entire dataset! | 3:47 | `3_BackProp_and_ReLU` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_10_IDL_Learn_the_entire_dataset.srt) |

### M2.2 — Regression in Keras

*5 videos, about 32 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 11 | Implementation of neural networks for regression (EDA) | 5:36 | `CA_Housing_Regression` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_11_IDL_Implementation_of_neural_networks_for_regression_EDA.srt) |
| 12 | The sequential API, compile, early stopping callback | 8:36 | `CA_Housing_Regression` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_12_IDL_The_sequential_API_compile_early_stopping_callback.srt) |
| 13 | Fitting and evaluating a neural network (regression) | 6:48 | `CA_Housing_Regression` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_13_IDL_Fitting_and_evaluating_a_neural_network_regression.srt) |
| 14 | Dropout | 5:22 | `CA_Housing_Regression` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_14_IDL_Dropout.srt) |
| 15 | Dropout bake-off and strategies for building NN architectures | 5:34 | `CA_Housing_Regression` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_15_IDL_Dropout_bake-off_and_strategies_for_building_NN_architectures.srt) |

### M2.3 — Classification in Keras

*9 videos, about 55 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 16 | Introduction to NNs for Classification | 6:33 | `0_BinaryClassification_Titanic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_16_IDL_Introduction_to_NNs_for_Classification.srt) |
| 17 | Building the NN classification model and fitting it with and without callbacks | 7:52 | `0_BinaryClassification_Titanic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_17_IDL_Building_the_NN_classification_model_and_fitting_it_with_and_witho.srt) |
| 18 | NN classification metrics and evaluation | 4:00 | `0_BinaryClassification_Titanic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_18_IDL_NN_classification_metrics_and_evaluation.srt) |
| 19 | Cross-validation exercises for classification NNs | 7:42 | `0b_TrainValTest + 0c_KFold` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_19_IDL_Cross-validation_exercises_for_classification_NNs.srt) |
| 20 | Introduction to multiclass classification NNs | 5:18 | `1_MulticlassClassification_Iris` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_20_IDL_Introduction_to_multiclass_classification_NNs.srt) |
| 21 | Multiclass metrics and evaluation/conclusion | 4:37 | `1_MulticlassClassification_Iris` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_21_IDL_Multiclass_metrics_and_evaluationconclusion.srt) |
| 22 | Introduction to the MNIST dataset | 5:36 | `2_A_FirstLook_atNN_with_MNIST_images` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_22_Introduction_to_the_MNIST_dataset.srt) |
| 23 | Evaluating our NN (dense layers) for MNIST | 7:53 | `2_A_FirstLook_atNN_with_MNIST_images` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_23_Evaluating_our_NN_dense_layers_for_MNIST.srt) |
| 24 | Fashion MNIST | 5:49 | `3_Fashion_MNIST_with_fashion_images` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module2/transcripts/5509IDL_M2_24_Fashion_MNIST.srt) |

## Chapter 3 — Convolutional neural networks

*17 videos, 1:50. Read the chapter: {doc}`m3_cnn/index`.*

### M3.1 — ConvNet theory and the math

*5 videos, about 39 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 1 | Introduction to ConvNets with Setosa and Kernels | 8:26 | slides | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_01_IDL_Introduction_to_ConvNets_with_Setosa_and_Kernels.srt) |
| 2 | Convolutional layers / light downsampling | 7:56 | slides | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_02_IDL_Convolutional_layerslight_downsampling.srt) |
| 3 | Pooling (no trainable parameters) | 6:26 | slides | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_03_IDL_Pooling_no_trainable_parameters.srt) |
| 4 | Where we are going with ConvNets | 7:50 | slides | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_04_IDL_Where_we_are_going_with_ConvNets.srt) |
| 5 | Math for a ConvNet (Pt 1, updated) | 8:34 | `Simple_Size_and_Param` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_05_IDL_Math_for_a_ConvNet_Pt_1_updated.srt) |

### M3.2 — Cats vs. dogs, and what the model is looking at

*6 videos, about 36 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 6 | Welcome to cats vs dogs! Pt 1: Intro and prepping your data | 5:59 | `ConvNets_on_Small_Datasets_Cats_vs_Dogs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_06_IDL_Welcome_to_cats_vs_dogs_Pt_1_Intro_and_prepping_your_data.srt) |
| 7 | Cats and Dogs Pt 2: Data generators and flow from directory | 7:56 | `ConvNets_on_Small_Datasets_Cats_vs_Dogs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_07_IDL_Cats_and_Dogs_Pt_2_Data_generators_and_flow_from_directory.srt) |
| 8 | Cats and Dogs Pt 3: Steps per epoch, model fit and initial results | 4:45 | `ConvNets_on_Small_Datasets_Cats_vs_Dogs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_08_IDL_Cats_and_Dogs_Pt_3_Steps_per_epoch_model_fit_and_initial_results.srt) |
| 9 | Cats and Dogs Pt 4: Using data augmentation and getting a better, more stable model | 5:37 | `ConvNets_on_Small_Datasets_Cats_vs_Dogs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_09_IDL_Cats_and_Dogs_Pt_4_Data_augmentation_and_a_better_more_stable_model.srt) |
| 10 | Wrapping up cats and dogs: evaluating with a classification report and confusion matrix | 5:33 | `ConvNets_on_Small_Datasets_Cats_vs_Dogs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_10_IDL_Wrapping_up_cats_and_dogs_classification_report_and_confusion_matrix.srt) |
| 11 | Interpreting Cats and Dogs with Grad-CAM | 6:11 | `Inside_the_ConvNet_Activations_and_GradCAM` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_11_IDL_Interpreting_Cats_and_Dogs_with_Grad-CAM.srt) |

### M3.3 — Transfer learning and autoencoders

*6 videos, about 35 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 12 | Introduction to transfer learning (feature extraction vs. fine-tuning) | 4:30 | `Transfer_Learning_with_a_Pretrained_ConvNet` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_12_IDL_Introduction_to_transfer_learning_feature_extraction_vs_fine-tuning.srt) |
| 13 | Feature extraction: freeze the ConvNet, then train the classifier | 6:05 | `Transfer_Learning_with_a_Pretrained_ConvNet` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_13_IDL_Feature_extraction_freeze_the_ConvNet_then_train_the_classifier.srt) |
| 14 | Fine-tuning the conv layer AND the dense layer | 6:27 | `Transfer_Learning_with_a_Pretrained_ConvNet` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_14_IDL_Fine-tuning_the_conv_layer_and_the_dense_layer.srt) |
| 15 | Wrapping up transfer learning | 1:59 | `Transfer_Learning_with_a_Pretrained_ConvNet` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_15_IDL_Wrapping_up_transfer_learning.srt) |
| 16 | Introduction to autoencoders | 7:55 | `Autoencoders_for_Images` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_16_IDL_Introduction_to_autoencoders.srt) |
| 17 | Applications of autoencoders | 7:57 | `What_Autoencoders_Can_Actually_Do` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module3/transcripts/5509IDL_M3_17_IDL_Applications_of_autoencoders.srt) |

## Chapter 4 — Recurrent networks for numeric sequences

*26 videos, 2:54. Read the chapter: {doc}`m4_rnn_numeric/index`.*

### M4.1 — Theory: the window method, SimpleRNN, LSTM, GRU

*9 videos, about 58 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 1 | Introduction to the Window Method | 4:04 | `Univariate_Temperature_Lags` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_01_Introduction_to_the_Window_Method.srt) |
| 2 | Series to supervised function (univariate) | 5:16 | `Univariate_Temperature_Lags` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_02_Series_to_supervised_function_univariate.srt) |
| 3 | Fitting model and evaluating the window method | 8:36 | `Univariate_Temperature_Lags` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_03_Fitting_model_and_evaluating_the_window_method.srt) |
| 4 | Prepping data as samples and a SimpleRNN | 9:20 | `RNN_Samples_and_the_Hidden_State` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_04_Prepping_data_as_samples_and_a_SimpleRNN.srt) |
| 5 | Appreciating that RNNs make an intermediate sequence | 7:52 | `RNN_Samples_and_the_Hidden_State` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_05_Appreciating_that_RNNs_make_an_intermediate_sequence.srt) |
| 6 | Intro to trainable parameters | 3:49 | `RNNs_By_Hand_basic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_06_Intro_to_trainable_parameters.srt) |
| 7 | SimpleRNN trainable parameters and theory | 9:11 | `RNNs_By_Hand_basic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_07_SimpleRNN_trainable_parameters_and_theory.srt) |
| 8 | A 'scary' SimpleRNN and intro to the LSTM | 5:35 | `RNNs_By_Hand_basic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_08_A_scary_SimpleRNN_and_intro_to_the_LSTM.srt) |
| 9 | GRU and closing | 4:33 | `RNNs_By_Hand_basic` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_09_GRU_and_closing.srt) |

### M4.2 — Implementation: temperature and room occupancy

*9 videos, about 67 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 10 | Intro to fitting our first univariate RNN | 5:21 | `Univariate_Temperature_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_10_Intro_to_fitting_our_first_univariate_RNN.srt) |
| 11 | Fitting our first SimpleRNN on univariate temperature data | 6:16 | `Univariate_Temperature_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_11_Fitting_our_first_SimpleRNN_on_univariate_temperature_data.srt) |
| 12 | Evaluating RNNs vs baseline models | 8:37 | `Univariate_Temperature_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_12_Evaluating_RNNs_vs_baseline_models.srt) |
| 13 | Occlusion and autoregressive treatment of the RNN | 7:42 | `Univariate_Temperature_RNN_pt2` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_13_Occlusion_and_autoregressive_treatment_of_the_RNN.srt) |
| 14 | Intro to the occupancy data for RNN classification | 7:17 | `Multivariate_Occupancy_Lags` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_14_Intro_to_the_occupancy_data_for_RNN_classification.srt) |
| 15 | Wrapping up occupancy with lags | 8:31 | `Multivariate_Occupancy_Lags` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_15_Wrapping_up_occupancy_with_lags.srt) |
| 16 | Intro to SimpleRNN for occupancy data | 8:11 | `Multivariate_Occupancy_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_16_Intro_to_SimpleRNN_for_occupancy_data.srt) |
| 17 | Wrapping up room occupancy with stacked RNN layers | 8:55 | `Multivariate_Occupancy_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_17_Wrapping_up_room_occupancy_with_stacked_RNN_layers.srt) |
| 18 | xAI for RNNs - permutation importance, occlusion and what-if | 6:19 | `Multivariate_Occupancy_RNN_pt2` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_18_xAI_for_RNNs_permutation_importance_occlusion_and_whatif.srt) |

### M4.3 — Advanced topics

*8 videos, about 50 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 19 | Treatment of time, Conv1D and pooling | 4:44 | `A1_Primer_for_Advanced_Topics` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_19_Treatment_of_time_Conv1D_and_pooling.srt) |
| 20 | Dropout, recurrent dropout, and bidirectional layers | 8:40 | `A1_Primer_for_Advanced_Topics` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_20_Dropout_recurrent_dropout_and_bidirectional_layers.srt) |
| 21 | Our first ConvLSTM on univariate data (early models) | 5:44 | `A2_Univariate_Temperature_Advanced` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_21_Our_first_ConvLSTM_on_univariate_data_early_models.srt) |
| 22 | Bidirectional layers, stacking convolutional layers, baselines | 8:10 | `A2_Univariate_Temperature_Advanced` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_22_Bidirectional_layers_stacking_convolutional_layers_baselines.srt) |
| 23 | Intro to the multivariate problem and first model | 6:44 | `A3_Multivariate_Energy_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_23_Intro_to_the_multivariate_problem_and_first_model.srt) |
| 24 | ConvLSTM, dropout and bidirectional for multivariate | 4:00 | `A3_Multivariate_Energy_RNN` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_24_ConvLSTM_dropout_and_bidirectional_for_multivariate.srt) |
| 25 | One shared model for two outputs | 3:40 | `A4_Many_to_Many_Energy` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_25_One_shared_model_for_two_outputs.srt) |
| 26 | Many to many and recursive time series forecasting | 7:52 | `A5_Multi_Step_Forecasting` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module4/transcripts/5509IDL_M4_26_Many_to_many_and_recursive_time_series_forecasting.srt) |

## Chapter 5 — Recurrent networks for text

*13 videos, 1:19. Read the chapter: {doc}`m5_rnn_text/index`.*

### M5.1 — Text as a bag of words

*6 videos, about 40 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 1 | Intro to Storm Classification dataset | 6:15 | `EDA_ML_Storms` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_01_Intro_to_Storm_Classification_dataset.srt) |
| 2 | Your ML Text processing methodology | 8:52 | `EDA_ML_Storms` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_02_Your_ML_Text_processing_methodology.srt) |
| 3 | Bag of Words | 6:49 | `EDA_ML_Storms` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_03_Bag_of_Words.srt) |
| 4 | Building a model, TF-IDF and closing ML approaches to text | 5:13 | `EDA_ML_Storms` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_04_Building_a_model_TFIDF_and_closing_ML_approaches_to_text.srt) |
| 5 | Using Keras to do basic text processing | 6:25 | `Tokenizer_FFNN_Storms` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_05_Using_Keras_to_do_basic_text_processing.srt) |
| 6 | Text sequence data wrap-up with FFNNs | 6:25 | `Tokenizer_FFNN_Storms` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_06_Text_sequence_data_wrapup_with_FFNNs.srt) |

### M5.2 — Text as a sequence: embeddings and RNNs

*7 videos, about 40 minutes.*

| # | Video | Length | Notebook | |
| --: | :-- | --: | :-- | :-- |
| 7 | Welcome to Module 5.2 Advanced Topics | 4:44 | `M5_2a_GloVe_Geometry` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_07_Welcome_to_Module_52_Advanced_Topics.srt) |
| 8 | The Geometry of an Embedding | 7:26 | `M5_2a_GloVe_Geometry` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_08_The_Geometry_of_an_Embedding.srt) |
| 9 | Wrapping up geometry of embeddings | 4:45 | `M5_2a_GloVe_Geometry` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_09_Wrapping_up_geometry_of_embeddings.srt) |
| 10 | Embeddings and one-hot encoding | 6:57 | `M5_2b_Word_Embeddings` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_10_Embeddings_and_one_hot_encoding_v2.srt) |
| 11 | Embeddings and SimpleRNN and the sports gambling analogy (real-time odds) | 6:20 | `M5_2c_Understanding_RNNs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_11_Embeddings_and_SimpleRNN_and_the_sports_gambling_analogy_realtime_odds.srt) |
| 12 | "Monster" LSTMs for text data and wrapping up class | 2:59 | `M5_2c_Understanding_RNNs` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_12_Monster_LSTMs_for_text_data_and_wrapping_up_class.srt) |
| 13 | A real-world capstone of RNNs and text with Bluesky | 6:23 | `Who_Posted_It_Bluesky` | [transcript](https://github.com/drdave-teaching/opim5509-transcripts/blob/main/fall2026_idl/module5/transcripts/5509IDL_M5_13_A_realword_capstone_of_RNNstext_with_Bluesky.srt) |

---

*Chapter 6 (special topics) has no Fall 2026 videos yet. The notebooks named above are linked, with Colab buttons, at the top of each chapter.*
