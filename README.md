# Voice-Based Image Captioning System

## Project Overview

This project implements a voice-based image captioning system that generates descriptive captions from images and converts them into speech output. It combines computer vision and natural language processing techniques to deliver an end-to-end multimodal solution for image understanding and accessibility.

## Key Features

* Automatic image caption generation
* Text-to-speech conversion of generated captions
* Integration of deep learning-based captioning models
* Performance evaluation using BLEU score
* Web-based deployment

## Technologies and Concepts Used

* EfficientNetB0 for image feature extraction
* LSTM (Long Short-Term Memory) for sequence modeling
* BLIP (Bootstrapped Language Image Pretraining) for caption generation
* Tokenization for text preprocessing
* Beam search for improved caption prediction
* BLEU score for evaluation
* Flickr30k dataset for training and validation

## Workflow

1. Image input is provided by the user
2. Visual features are extracted using EfficientNetB0
3. Captions are generated using LSTM and BLIP models with tokenization and beam search
4. Generated captions are evaluated using BLEU score
5. Final captions are converted into speech output

## Model Evaluation

The model performance is evaluated using BLEU score (BLEU-1 to BLEU-4), which measures the similarity between generated captions and reference captions.

## Deployment

The application is deployed on Hugging Face Spaces. The deployed version focuses on the BLIP model along with BLEU score evaluation for real-time caption generation.

Live Application Link:
https://huggingface.co/spaces/RanaSrijal/voicebasedimagecaptioning

