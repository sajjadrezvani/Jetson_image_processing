# Jetson_image_processing
In this project, I used several deep learning models and techniqes on GTSRB DataSet including ViT(visual transformers), Resnent using PyTorch and Transfer Learning and Safe Machine Learnig using Tensor Flow to classify traffic sign. 

## Dataset
The GTSRB dataset, which stands for German Traffic Sign Recognition Benchmark, is a collection of images of traffic signs commonly found on roads in Germany. These signs include things like speed limits, yield signs, stop signs, and more. The dataset is often used by researchers and developers to train and test algorithms for recognizing and interpreting traffic signs using computers or artificial intelligence. It's like a big set of pictures of road signs that people use to teach computers how to understand and respond to what the signs mean.

## Visual Transformers (ViT):
Visual Transformers, or ViT, are a type of model that uses transformer architecture to process images. Transformers were initially popular in natural language processing (NLP) tasks but have been adapted for computer vision tasks like image classification. ViT breaks down an image into smaller patches, treats them as tokens (like words in NLP), and processes them using transformer layers to understand the image's content.

### Structure:
The structure of ViT involves breaking down the input image into smaller patches, then flattening them into sequences. These sequences are then processed by transformer layers, which consist of attention mechanisms to learn relationships between different patches. Finally, a classification head is attached to predict the image's label based on the learned features.

### Application and Performance:
ViT has shown promising results in various computer vision tasks, including image classification, object detection, and segmentation. It performs particularly well on tasks where capturing long-range dependencies in images is crucial. With proper training and tuning, ViT can achieve competitive or even state-of-the-art performance on benchmark datasets.

### Comparison with CNN:
Compared to Convolutional Neural Networks (CNNs), which have traditionally been the go-to architecture for computer vision tasks, ViT offers some distinct advantages. ViT can capture global dependencies more effectively due to its self-attention mechanism, whereas CNNs rely on local receptive fields. Additionally, ViT has a more structured and scalable architecture, making it easier to adapt to different tasks and datasets. However, ViT may require more computational resources for training and inference compared to CNNs, and it may struggle with tasks that involve spatial information, where CNNs excel.

## ResNet:
ResNet, short for Residual Network, is a type of deep neural network architecture designed to address the problem of vanishing gradients during training. It introduced the concept of residual learning, which involves learning residual functions with reference to the layer inputs, rather than learning the actual underlying mapping.

### Structure:
The structure of a ResNet consists of several layers organized into blocks, with each block containing multiple convolutional layers. The key innovation in ResNet is the introduction of skip connections, also known as shortcut connections or identity mappings, which allow the network to bypass certain layers. These skip connections enable the gradient flow to bypass the problematic layers, alleviating the vanishing gradient problem and facilitating the training of very deep networks.

### Application and Performance:
ResNet has been widely used in various computer vision tasks, particularly in image classification, object detection, and segmentation. It has consistently demonstrated excellent performance on benchmark datasets like ImageNet. ResNet's ability to effectively train very deep neural networks has made it a cornerstone architecture in the field of deep learning.

### Comparison with other architectures:
Compared to earlier convolutional neural network architectures, ResNet's skip connections enable it to effectively train much deeper networks. This capability has led to improved performance on challenging tasks and datasets. ResNet's success has inspired the development of other architectures with similar skip connection mechanisms, such as DenseNet and Highway Networks.

## Transfer Learning:
Transfer Learning is a technique in machine learning where a model trained on one task is reused or adapted for a different but related task. Instead of starting the learning process from scratch, Transfer Learning leverages the knowledge gained from solving one problem and applies it to a new, similar problem. The idea is that features learned from the original task can be useful for the new task, potentially speeding up training and improving performance, especially when labeled data for the new task is limited. Transfer Learning allows the knowledge gained from solving one machine learning problem to be applied to another related problem, leading to improved performance and efficiency, particularly when data for the new task is limited. It's like taking what you've learned from one situation and applying it to a similar but different situation to solve problems more efficiently.

### Application and Performance:
Transfer Learning has found wide applications across various domains, including computer vision, natural language processing, and speech recognition. For example, in computer vision, a model pretrained on a large dataset like ImageNet, which contains millions of labeled images, can be fine-tuned on a smaller dataset for a specific task like detecting different species of flowers or identifying objects in medical images. This approach often leads to faster convergence during training and better generalization performance, especially when the new dataset is small or similar to the original dataset.

 ## Safe ML:
"Safe ML" refers to the practice of ensuring that machine learning (ML) models are reliable, robust, and trustworthy. It involves implementing techniques and strategies to mitigate potential risks associated with ML systems, such as biases, unfairness, safety concerns, and ethical implications. The goal of Safe ML is to develop models that not only perform well but also behave responsibly and ethically in real-world scenarios. Safe ML practices contribute to building trustworthy and responsible AI systems that prioritize reliability, fairness, transparency, and ethical considerations, ultimately enhancing societal well-being and trust in machine learning technology.

### Structure:
The structure of Safe ML involves several key components and practices:
Data Quality Assurance: Ensuring that training data is representative, diverse, and free from biases that could adversely impact model performance or fairness.
Model Explainability: Implementing methods to interpret and explain model predictions, making them more transparent and understandable to users and stakeholders.
Fairness and Bias Mitigation: Employing techniques to detect and mitigate biases in training data and model predictions, ensuring fairness and equity across different demographic groups.
Robustness and Security: Strengthening models against adversarial attacks, ensuring resilience to input perturbations and malicious inputs.
Ethical Considerations: Addressing ethical concerns and societal implications of ML models, such as privacy preservation, algorithmic transparency, and accountability.

### Application and Performance:
Safe ML practices are applied across various domains and applications where machine learning is deployed, including healthcare, finance, autonomous vehicles, and social media. For example:
In healthcare, Safe ML ensures that predictive models for diagnosis and treatment recommendations are accurate, explainable, and unbiased, safeguarding patient well-being and privacy.
In finance, Safe ML helps prevent algorithmic biases in credit scoring models and fraud detection systems, ensuring fairness and transparency in lending decisions.
In autonomous vehicles, Safe ML enhances the safety and reliability of self-driving systems by addressing robustness, security, and ethical considerations, minimizing the risk of accidents and ensuring user trust.
