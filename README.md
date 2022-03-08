# MAP583 - Deep learning project
- Based on the **backgrounds challenge**, we use several new datasets (mixup, cutmix) to improve the robustness of networks to background changes. On the "original" dataset in IN-9, we use mixup and cutmix data augmentation tricks to generate new datasets called "mixup" and "cutmix". On the pretrained Resnet50, we fine-tune it respectively with these two datasets. Then, test the fine-tuned networks on validation set in IN-9 (Only-background-b, no foreground, mixed-same, mixed-rand, mixed-next). We also fine-tuned the Resnet50 on "original" dataset in IN-9, to compare the model robustness.
- Moreover, we use [Stochastic Weight Averaging](https://github.com/timgaripov/swa) to improve generalization. 
- You can find our codes in folder "Modification MAP583" and our models on Google drive: [Resnet50 Trained on Original](https://drive.google.com/file/d/1iGlUQNj-uGG8UE5LAaveBOr5_91z4Cpn/view?usp=sharing), [Resnet50 Trained on Mixup](https://drive.google.com/file/d/1ktUgIab0Tslkl_HJ1GOHv66Wy4OQIEpZ/view?usp=sharing) and [Resnet50 Trained on MixCut](https://drive.google.com/file/d/17myBFhDjed6LLdVT1V7KO9IqwK5-N_mQ/view?usp=sharing).


## Our code on Google colab:
https://colab.research.google.com/drive/1oYp8CwsTCoSe_ULJtVvvE0MPXVN5gJeM#scrollTo=JQ6lT16zwGUj

## Our result:
Accuracy of Resnet50 on different test set:
|Test on |    Trained on Mixup	| Trained on Cutmix	| Trained on Original |
|  :----:  | :----:  | :----: | :----: |
|Only-bg-b | 20.5% | 22.1% | 20.1% |
|No fg 	| 30.6% | 36.3% | 26.5% |
|Mixed-next | 38.0% | 32.5% | 21.1% |
|Mixed-same | 59.3% | 60.0% | 44.6% |
|Mixed-rand	| 42.6% | 38.7% | 26.6% |
|BG-gap  | 16.7% | 21.3% | 18.0% |

The BG-GAP is defined as the difference in test accuracy between Mixed-same and Mixed-rand and helps assess the tendency of such models to rely on background signal. Bigger the BG-Gap, more the model depends on the background. 

We could find that the model 'mixup' relies less on background (smallest BG-Gap among the three). This is because there is no clear interface between foreground and background in training set, the model tends to depend less on background. On the contrary, the clear dividing line between background and foreground in dataset Cutmix makes the corresponding model more dependent on background(biggest BG-Gap among the three).

## Comparison with the paper:
The last two columns (Pre-trained on ImageNet & Pre-trained on IN-9L) are from paper **"Noise or Signal: The Role of Image Backgrounds in Object Recognition"** ([preprint](https://arxiv.org/abs/2006.09994)).
|Test on |    Trained on Mixup	| Trained on Cutmix	| Trained on Original | Pre-trained on ImageNet | Trained on IN-9L |
|  :----:  | :----:  | :----: | :----: | :----: | :----: | 
|Mixed-same | 59.3% | 60.0% | 44.6% | 82.3% | 89.9% |
|Mixed-rand	| 42.6% | 38.7% | 26.6% | 76.3% | 75.6% | 
|BG-gap  | 16.7% | 21.3% | 18.0% | 6% | 14.3%|

We find that the two models in the paper are in average more accurate than our models. Moreover, their BG-gap are smaller.

Below is the original README in Project-backgrounds-challenge repository.

----------------------


# Backgrounds Challenge
The **backgrounds challenge** is a public dataset challenge for creating more background-robust models. This repository contains test datasets of ImageNet-9 (IN-9) with different amounts of background and foreground signal, which you can use to measure the extent to which your models rely on image backgrounds. These are described further in the paper: **"Noise or Signal: The Role of Image Backgrounds in Object Recognition"** ([preprint](https://arxiv.org/abs/2006.09994), [blog](http://gradsci.org/background)).

Deep computer vision models rely on both foreground objects and image backgrounds. Even when the correct foreground object is present, such models often make incorrect predictions when the image background is changed, and they are especially vulnerable to **adversarially chosen backgrounds**. For example, the [the official pre-trained PyTorch ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html) has an accuracy of 22% when evaluated against adversarial backgrounds on ImageNet-9 (for reference, a model that always predicts "dog" has an accuracy of 11%).

Thus, the goal of this challenge is to understand how background-robust models can be. Specifically, we assess models by their accuracy on images containing foregrounds superimposed on backgrounds which are adversarially chosen from the test set. We encourage researchers to use this challenge to benchmark progress on background-robustness, which can be important for determining models' out of distribution performance. We will maintain a leaderboard of top submissions.

<img align="center" src="assets/adversarial_backgrounds_insect.png" width="750">
<sub><sup>Examples from the insect class of the most adversarial backgrounds for a model. The number above each image represents the proportion of non-insect foregrounds that can be fooled by these backgrounds.</sup></sub>


## Backgrounds Challenge Leaderboard

| Model                     | Reference                     | Challenge <br> Accuracy        | Clean Accuracy <br> (on IN-9) | Download Link
|---------------------------|-------------------------------|---------------------------|----------------|------------------------|
| ResNet-50                 | (initial entry)               | 22.3%                     | 95.6%          | [Official Pytorch Model](https://download.pytorch.org/models/resnet50-19c8e357.pth)
| ResNet-50 ([IN-9L](https://arxiv.org/abs/2006.09994))         | (initial entry)               | 12.0%                     | 96.4%          | [Download](https://www.dropbox.com/s/5bmfiunmlh0hy8n/in9l_resnet50.pt?dl=0)


## Running the Backgrounds Challenge Evaluation
To evaluate your model against adversarial backgrounds, you will need to do the following:

1. Download and unzip the datasets included in the [release](https://github.com/MadryLab/backgrounds_challenge/releases).
2. Run `python challenge_eval.py --checkpoint '/PATH/TO/CHECKPOINT' --data-path '/PATH/TO/DATA'`.

The model checkpoint that the script takes as input must be one of the following.
1. A 1000-class ImageNet classifier.
2. A 9-class IN-9 classifier.

See `python challenge_eval.py -h` for how to toggle between the two.

**Note**: evaluation requires PyTorch to be installed with CUDA support.

## Submitting a Model
We invite any interested researchers to submit models and results by submitting a pull request with your model checkpoint included. The most successful models will be listed in the leaderboard above. We have already included baseline pre-trained models for reference.

# Testing your model on ImageNet-9 and its variations
<img align="center" src="assets/imagenet9_insect.png" width="750">
<sub><sup>All variations of IN-9; each variation contains different amounts of foreground and background signal.</sup></sub>
<br/>
<br/>

ImageNet-9 and its variations can be useful for measuring the impact of backgrounds on model decision making&mdash;see the [paper](https://arxiv.org/abs/2006.09994) for more details. You can test your own models on IN-9 and its variations as follows.

1. Download and unzip the datasets included in the [release](https://github.com/MadryLab/backgrounds_challenge/releases).
2. Run, for example, `python in9_eval.py --eval-dataset 'mixed_same' --checkpoint '/PATH/TO/CHECKPOINT' --data-path '/PATH/TO/DATA'`. You can replace `mixed_same` with whichever variation of IN-9 you are interested in.

Just like in the challenge, the input can either be a 1000-class ImageNet model or a 9-class IN-9 model.

There is no leaderboard or challenge for these datasets, but we encourage researchers to use these datasets to measure the role of image background in their models' decision making. Furthermore, we include a table of results for common pre-trained models and various models discussed in the paper.

## Test Accuracy Results on ImageNet-9
| Model                      | Original        | Mixed-Same      | Mixed-Rand      | BG-Gap          |
|----------------------------|-----------------|-----------------|-----------------|-----------------|
| VGG16-BN                   | 94.3%           | 83.6%           | 73.4%           | 10.2%           |
| ResNet-50                  | 95.6%           | 86.2%           | 78.9%           |  7.3%           |
| ResNet-152                 | 96.7%           | 89.3%           | 83.5%           |  5.8%           |
| ResNet-50 (IN-9L)          | 96.4%           | 89.8%           | 75.6%           | 14.2%           |
| ResNet-50 (IN-9/Mixed-Rand)| 73.3%           | 71.5%           | 71.3%           |  0.2%           |

<sub><sup>The BG-Gap, or the difference between Mixed-Same and Mixed-Rand, measures the impact of background correlations in the presence of correct-labeled foregrounds.</sup></sub>

## Training Data
**Updated June 24, 2020**: We are releasing all training data that we used to train models described in the paper. The download links are as follows:
[IN-9L](https://www.dropbox.com/s/8w29bg9niya19rn/in9l.tar.gz?dl=0),
[Mixed-Next](https://www.dropbox.com/s/4hnkbvxastpcgz2/mixed_next.tar.gz?dl=0),
[Mixed-Rand](https://www.dropbox.com/s/cto15ceadgraur2/mixed_rand.tar.gz?dl=0),
[Mixed-Same](https://www.dropbox.com/s/f2525w5aqq67kk0/mixed_same.tar.gz?dl=0),
[No-FG](https://www.dropbox.com/s/0v6w9k7q7i1ytvr/no_fg.tar.gz?dl=0),
[Only-BG-B](https://www.dropbox.com/s/u1iekdnwail1d9u/only_bg_b.tar.gz?dl=0),
[Only-BG-T](https://www.dropbox.com/s/03lk878q73hyjpi/only_bg_t.tar.gz?dl=0),
[Only-FG](https://www.dropbox.com/s/alrf3jo8yyxzyrn/only_fg.tar.gz?dl=0),
[Original](https://www.dropbox.com/s/0vv2qsc4ywb4z5v/original.tar.gz?dl=0).

Each downloadable dataset contains both training data and validation data generated in the same way as the training data (that is, with no manual cleaning); this validation data can be safely ignored. The test data in the [release](https://github.com/MadryLab/backgrounds_challenge/releases) should be used instead.


## Citation

If you find these datasets useful in your research, please consider citing:

    @article{xiao2020noise,
      title={Noise or Signal: The Role of Image Backgrounds in Object Recognition},
      author={Kai Xiao and Logan Engstrom and Andrew Ilyas and Aleksander Madry},
      journal={ArXiv preprint arXiv:2006.09994},
      year={2020}
    }
