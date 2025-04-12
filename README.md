# AAI-590-Ocular-Disease-Recognition

Vision is a paramount aspect of navigating both social and physical infrastructure within this world. Yet, eye disease and disorders are incredibly common, with the World Health Organization estimating 

    "Over over 2.2 billion people have some form of vision impairment or blindness”

The most common of which being cataracts, glaucoma, age-related macular degeneration, and diabetes-related retinopathy (Cleveland Clinic, 2024). Early diagnosis is critical for effective treatment, however, traditional diagnostic methods rely heavily on manual image interpretation by ophthalmologists, which is time-consuming and opens the possibility of human error through misinterpretation. Similarly, one of the methodologies used to effectively diagnose individuals depends on retinal imaging, a medium of information primed for deep learning applications (Cleveland Clinic, 2024). With continuous advancements in artificial intelligence (AI) and deep learning, an automated screening and diagnostic tool becomes an increasingly viable solution in assisting healthcare professionals to more efficiently detect and diagnose ocular diseases. 

This project aims to bring theory into practice, developing a deep learning-based model for automated ocular disease recognition using fundus (eye) images and associated patient metadata. By leveraging various convolutional neural network architectures (CNNs), we manage to accurately and consistently diagnose fundus images into one of eight categories covering diabetes, glaucoma, cataracts, age related macular degeneration, hypertension, pathological myopia, other diseases and abnormalities, or unaffected normal eyes.

## Project Overview
Course: AAI-590

Institution: University of San Diego

## Collaborators:

Sarah Durrani

Franklin Guzman

Hani Jandali

Instructors: Anna Marbut, Roozbeh Sadeghian

GitHub Repository URL: https://github.com/tankyfrank/AAI-590-Ocular-Disease-Recognition/edit/main/README.md

## Files Structures

The Github Repository is organized into three respective folders:

*Initial_Scripts*
 - Contains all initial scripts tackling data cleaning and EDA for our initial project approach

 *Initial_Models* 
 - Contains all initial model architectures and iterations developed during the final term. In our paper, this is refered to as the "initial model"

 *Final_Model*
 - Contains the code and PDF of the final model for submission and grading

In addition to the folders, we have a standard .gitignore and README.md file enabling clear and concise project workflow and presentation. 

## Project Overview 

1. Data Collection, EDA
  - Images and metadata are imported from the ODIR-5K dataset available online, particularly on Kaggle (Andrew, D.). Data is subsequently cleaned, with image data augmented and transformed into tesnors, and patient medical data vectorized through TF-IDF

2. Model Design and Development
  - Two models were developed, the initial being an purely image-based CNN with regularization and dropout techniques. The second model was a hybridized CNN utilizing image and patient metadata with residual connections, CBAM modules, and ASPP for better feature abstraction through channels and spatial abstraction from images.

3. Model Training and Optimization

  - Our initial model trained across 100 epochs on Adam Optimizer set at 0.001, resulting in 47% accuracy. Our final model trained across 30 epochs utilizing AdamW optimizer set at 0.003 with a OneCyclerLR adjustment and a weight decay of 0.0001. It resulted in a total training time of approximately 6.5 minutes achieving a 98% accuracy and validation loss of 0.1019. 


## Key Technologies Used
Programming Language: Python 3.8+

Deep Learning Frameworks: TensorFlow, Keras, PyTorch

Data Processing: Pandas, NumPy

Visualization Tools: Matplotlib, Seaborn

Machine Learning Models: Convolutional Neural Networks


## Future Improvements
Deploying the model via Flask API for real-world usage. 

Expanding model to train on a more diversified dataset with increased samples taken from a diversified population. 

## Acknowledgments
This project was completed as part of AAI-590 at the University of San Diego under the guidance of Professor Anna Marbut and Professor Roozbeh Sadeghian

## References

Andrew, M. V. D. Ocular Disease Recognition (ODIR-5K) (Data set). Kaggle.
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2016). DeepLab:
Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and
Fully Connected CRFs. arXiv. https://arxiv.org/abs/1606.00915v2

Cleveland Clinic. (2024). Eye Diseases. 
https://my.clevelandclinic.org/health/diseases/eye-diseases

Fauw, D. et al. (2018). Clinically Applicable Deep Learning for Diagnosis and Referral in
Retinal Disease. Nature Medicine, 24(9), 1342–1350. 
https://www.nature.com/articles/s41591-018-0107-6

Gulshan, V . et al. (2016). Development and Validation of a Deep Learning Algorithm for 
Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA, 316(22), 
2402–2410. https://pubmed.ncbi.nlm.nih.gov/27898976/

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
IEEE Conference on Computer Vision and Pattern Recognition, 770–778. 
https://doi.org/10.48550/arXiv.1512.03385

IDx Technologies. (2018). IDx-DR: Autonomous AI-based diagnostic system for diabetic
retinopathy (Software). https://www.idxdr.com/

Lian-Hong, P., Lin, C., Ning, K., Fang, J., Zhang, S., Xiao, J., Wei-Jiang, Y., Xiong, Y., Shi, H.,
Zhou, X., Yin, Z. (2012). Prevalence of Eye Diseases and Causes of Visual Impairment in 
School-Aged Children in Western China. Journal of Epidemiology, (22), 37-44.
 https://pmc.ncbi.nlm.nih.gov/articles/PMC3798578/

MengzhangLi. Et. al. (2025). Awesome Medical Dataset. 
https://github.com/openmedlab/Awesome-Medical-Dataset/tree/main

OpenAI. (2023). ChatGPT (Large language model). https://chat.openai.com/

Ting, D., Cheung, C., Lim, G. (2017). Development and Validation of a Deep Learning System 
for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images from 
Multiethnic Populations with Diabetes. JAMA, 318(22), 2211–2223. 
https://jamanetwork.com/journals/jama/fullarticle/2665775

Triveldi, S. (June 12th, 2020). Understanding Attention Modules: CBAM and BAM. Medium.
https://medium.com/visionwizard/understanding-attention-modules-cbam-and-bam-a-quick-Read-ca8678d1c671


Tsang, Sik-Ho. (November 19th, 2018) Review: DeepLabV1, DeepLabV2, Atrous Convolution
& Semantic Segmentation. Medium. https://medium.com/data-science/review-deeplabv1-
Deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d

Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention
Module. European Conference on Computer Vision (ECCV), 3-19.
https://arxiv.org/abs/1807.06521

Zhongwen, L., Chen, W. (2023). Solving Data Quality Issues of Fundus Images in Real-World
Settings by Ophthalmic AI.
https://pmc.ncbi.nlm.nih.gov/articles/PMC9975325/#:~:text=Ocular%20fundus%20diseases%2C%20such%20as,the%20living%20quality%20of%20patients


## License
This project is for educational purposes only. If you use or modify this work, please provide appropriate attribution.


