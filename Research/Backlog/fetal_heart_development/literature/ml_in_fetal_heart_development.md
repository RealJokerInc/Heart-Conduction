# Machine Learning in Fetal Heart Development — Literature Review

**Date compiled:** 2026-03-06
**Source databases:** PubMed, bioRxiv, medRxiv

---

## Table of Contents

1. [Review Papers & Surveys](#1-review-papers--surveys)
2. [CNNs for Fetal Echocardiogram Image Segmentation](#2-cnns-for-fetal-echocardiogram-image-segmentation)
3. [Deep Learning for CHD Detection & Classification](#3-deep-learning-for-chd-detection--classification)
4. [Standard Plane Detection & Quality Assessment](#4-standard-plane-detection--quality-assessment)
5. [Automated Biometrics & Cardiac Measurements](#5-automated-biometrics--cardiac-measurements)
6. [Fetal Heart Rate Analysis & Cardiotocography](#6-fetal-heart-rate-analysis--cardiotocography)
7. [Fetal MRI Reconstruction & Analysis](#7-fetal-mri-reconstruction--analysis)
8. [Fetal ECG & Electrophysiology](#8-fetal-ecg--electrophysiology)
9. [Gene Regulatory Networks & Single-Cell Genomics](#9-gene-regulatory-networks--single-cell-genomics)
10. [Computational Modeling of Cardiac Morphogenesis](#10-computational-modeling-of-cardiac-morphogenesis)
11. [Clinical Protocols & Multicenter Studies](#11-clinical-protocols--multicenter-studies)
12. [Summary: ML Models Used by Application](#12-summary-ml-models-used-by-application)

---

## 1. Review Papers & Surveys

### Zhang et al. (2024) — Advances in the Application of Artificial Intelligence in Fetal Echocardiography
- **PMID:** 38199332 | **DOI:** [10.1016/j.echo.2023.12.013](https://doi.org/10.1016/j.echo.2023.12.013)
- **Journal:** J Am Soc Echocardiogr (2024)
- **ML Models Discussed:** CNNs (VGG, ResNet, Inception), U-Net variants, YOLO, SSD, Faster R-CNN, RNNs/LSTMs for video analysis
- **Summary:** Comprehensive review of AI in fetal echocardiography covering image processing (denoising, enhancement), biometrics (automated measurements), and disease diagnosis. Systematically reviews applications over the years in terms of image processing, automated biometrics, and disease diagnosis. AI has great potential to bridge the skill gap across regions and empower sonographers in time-saving, accurate diagnosis.

### Reddy et al. (2022) — Artificial Intelligence in Perinatal Diagnosis and Management of Congenital Heart Disease
- **PMID:** 35396036 | **DOI:** [10.1016/j.semperi.2022.151588](https://doi.org/10.1016/j.semperi.2022.151588)
- **Journal:** Seminars in Perinatology, 46(4):151588
- **ML Models Discussed:** Deep CNNs, transfer learning, ensemble methods, SVM, random forests
- **Summary:** Reviews how AI can improve prenatal CHD detection rates beyond current ~85% accuracy. Covers the full pipeline: fetal echocardiography → automated analysis → clinical decision support. Discusses challenges including limited training data, class imbalance, and need for large multicenter validation. Outlines potential for AI to improve care coordination, risk stratification, and outcomes.

### Norris et al. (2025) — Artificial Intelligence Aided Ultrasound Imaging of Foetal Congenital Heart Disease: A Scoping Review
- **PMID:** 40961819 | **DOI:** [10.1016/j.radi.2025.103170](https://doi.org/10.1016/j.radi.2025.103170)
- **Journal:** Radiography (2025)
- **ML Models Discussed:** Broad survey across CNN architectures, attention mechanisms, video-based models
- **Summary:** Scoping review of AI-assisted ultrasound imaging for CHD detection. Notes that detection of CHDs can be limited by factors including maternal habitus, fetal position, gestational age, and sonographer expertise. AI offers potential to overcome these barriers. Covers recent advances and remaining barriers to clinical deployment.

### Suha et al. (2025) — The Artificial Intelligence-Enhanced Echocardiographic Detection of Congenital Heart Defects in the Fetus: A Mini-Review
- **PMID:** 40282852 | **DOI:** [10.3390/medicina61040561](https://doi.org/10.3390/medicina61040561)
- **Journal:** Medicina (2025)
- **ML Models Discussed:** Various deep learning architectures for image quality control, function measurement, defect detection, and classification
- **Summary:** Outlines AI technical background in echocardiography and presents clinical applications across image quality control, cardiac function measurements, defect detection, and classifications. Notes rapid progress but need for larger validation studies.

### de Siqueira et al. (2021) — AI Applied to Support Medical Decisions for the Automatic Analysis of Echocardiogram Images: A Systematic Review
- **PMID:** 34629153 | **DOI:** [10.1016/j.artmed.2021.102165](https://doi.org/10.1016/j.artmed.2021.102165)
- **Journal:** Artificial Intelligence in Medicine, 120:102165
- **ML Models Discussed:** CNNs, autoencoders, GANs, SVMs, random forests, recurrent networks
- **Summary:** Systematic review covering AI techniques for automating echocardiogram image analysis. Covers both adult and fetal applications. Notes that the field has matured from traditional ML (SVM, RF) to deep learning approaches, with U-Net and its variants dominating segmentation tasks.

### Jost et al. (2023) — Evolving the Era of 5D Ultrasound? AI Ultrasound Imaging in Obstetrics and Gynecology
- **PMID:** 37959298 | **DOI:** [10.3390/jcm12216833](https://doi.org/10.3390/jcm12216833)
- **Journal:** J Clin Med (2023)
- **ML Models Discussed:** CNNs, 3D CNNs for volumetric data, attention networks, transfer learning
- **Summary:** Systematic literature review covering AI in obstetric/gynecologic ultrasound. Discusses how AI can augment 2D, 3D, and 4D ultrasound toward "5D" (AI-assisted) imaging. Reviews image classification, anomaly detection, and automated measurements.

### Francis et al. (2024) — Machine Learning on Cardiotocography Data to Classify Fetal Outcomes: A Scoping Review
- **PMID:** 38489990 | **DOI:** [10.1016/j.compbiomed.2024.108220](https://doi.org/10.1016/j.compbiomed.2024.108220)
- **Journal:** Computers in Biology and Medicine (2024)
- **ML Models Discussed:** SVM, random forest, XGBoost, neural networks, deep learning (CNNs, LSTMs), ensemble methods
- **Summary:** Comprehensive scoping review of ML applied to cardiotocography (CTG) for fetal outcome classification. Catalogs all ML model types used across studies, performance metrics, and dataset characteristics. Notes that deep learning approaches are increasingly outperforming traditional ML on CTG classification tasks.

### Yi et al. (2020) — Technology Trends and Applications of Deep Learning in Ultrasonography
- **PMID:** 33152846 | **DOI:** [10.14366/usg.20102](https://doi.org/10.14366/usg.20102)
- **Journal:** Ultrasonography (2020)
- **ML Models Discussed:** CNNs (ResNet, VGG, DenseNet), U-Net, GANs, RNNs
- **Summary:** Reviews deep learning architectures for classification, detection, segmentation, and generation in ultrasonography. Covers image quality enhancement, diagnostic support, and workflow efficiency. Provides accessible technical explanations of network architectures.

---

## 2. CNNs for Fetal Echocardiogram Image Segmentation

### Xu et al. (2020) — DW-Net: A Cascaded CNN for Apical Four-Chamber View Segmentation in Fetal Echocardiography
- **PMID:** 31968286 | **DOI:** [10.1016/j.compmedimag.2019.101690](https://doi.org/10.1016/j.compmedimag.2019.101690)
- **Journal:** Computerized Medical Imaging and Graphics (2020)
- **ML Model:** Cascaded CNN — Dilated Convolutional Chain (DCC) + W-Net
- **Performance:** DSC 0.827, Pixel Accuracy 0.933, AUC 0.990
- **Summary:** Proposes an end-to-end DW-Net for segmenting 7 important anatomical structures in the apical four-chamber (A4C) view. The DCC component aggregates multi-scale contextual information, while the W-Net refines segmentation boundaries. Trained on 895 A4C views. Addresses challenges of artifacts, speckle noise, and missing boundaries in fetal ultrasound.

### An et al. (2021) — A Category Attention Instance Segmentation Network for Four Cardiac Chambers in Fetal Echocardiography
- **PMID:** 34610500 | **DOI:** [10.1016/j.compmedimag.2021.101983](https://doi.org/10.1016/j.compmedimag.2021.101983)
- **Journal:** Computerized Medical Imaging and Graphics (2021)
- **ML Model:** Category Attention Instance Segmentation Network (attention-based CNN)
- **Summary:** Introduces category attention mechanism for accurate segmentation of four cardiac chambers. Advances beyond previous work by differentiating overlapping/adjacent structures. Instance segmentation rather than semantic segmentation provides more clinically useful delineation.

### Yu et al. (2017) — Segmentation of Fetal Left Ventricle in Echocardiographic Sequences Based on Dynamic CNNs
- **PMID:** 28113289 | **DOI:** [10.1109/TBME.2016.2628401](https://doi.org/10.1109/TBME.2016.2628401)
- **Journal:** IEEE Trans Biomed Engineering (2017)
- **ML Model:** Dynamic CNN with temporal adaptation for video sequences
- **Summary:** One of the earliest deep learning approaches for fetal LV segmentation. Proposes dynamic CNN that adapts convolutional filters across frames in echocardiographic sequences. Handles fetal random movements and image inhomogeneities. Demonstrated on temporal sequences rather than single frames.

### Hurtado et al. (2024) — Segmentation of Four-Chamber View Images Using a Novel Deep Learning Model Ensemble Method
- **PMID:** 39395344 | **DOI:** [10.1016/j.compbiomed.2024.109188](https://doi.org/10.1016/j.compbiomed.2024.109188)
- **Journal:** Computers in Biology and Medicine (2024)
- **ML Model:** Ensemble of multiple deep learning models (multi-model fusion)
- **Summary:** Introduces novel ensemble approach combining various DL models for anatomical structure segmentation in four-chamber view. Shows that model ensembles can improve robustness over individual models, particularly for fetal images where quality varies.

### Torres et al. (2024) — Deep-DM: Deep-Driven Deformable Model for 3D Image Segmentation Using Limited Data
- **PMID:** 39110559 | **DOI:** [10.1109/JBHI.2024.3440171](https://doi.org/10.1109/JBHI.2024.3440171)
- **Journal:** IEEE J Biomed Health Inform (2024)
- **ML Model:** Hybrid deep learning + deformable model
- **Summary:** Addresses the critical challenge of limited annotated data in medical imaging. Combines deep learning feature extraction with classical deformable models for 3D segmentation. Relevant to fetal cardiac imaging where labeled datasets are scarce.

### Schlemper et al. (2019) — Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images
- **PMID:** 30802813 | **DOI:** [10.1016/j.media.2019.01.012](https://doi.org/10.1016/j.media.2019.01.012)
- **Journal:** Medical Image Analysis (2019)
- **ML Model:** Attention Gate (AG) mechanism integrated with U-Net and VGG-16
- **Performance:** Applied to fetal ultrasound among other domains
- **Summary:** Proposes attention gate models that automatically learn to focus on target structures of varying shapes and sizes, suppressing irrelevant regions. Tested on fetal ultrasound abdominal circumference estimation and cardiac MRI segmentation. AGs are now widely adopted in fetal cardiac imaging pipelines.

---

## 3. Deep Learning for CHD Detection & Classification

### Athalye et al. (2024) — Deep-Learning Model for Prenatal CHD Screening Generalizes to Community Setting and Outperforms Clinical Detection
- **PMID:** 37774040 | **DOI:** [10.1002/uog.27503](https://doi.org/10.1002/uog.27503)
- **Journal:** Ultrasound in Obstetrics & Gynecology (2024)
- **ML Model:** Deep CNN (previously developed, now validated in community settings)
- **Summary:** Landmark study demonstrating that a DL model for CHD screening generalizes from academic training centers to community obstetric practice and outperforms clinical detection rates. Addresses the key challenge of prenatal CHD screening: most cases are missed despite universal screening. Shows real-world generalizability, a critical barrier for clinical adoption.

### Gong et al. (2019) — Fetal CHD Echocardiogram Screening Based on DGACNN
- **PMID:** 31603775 | **DOI:** [10.1109/TMI.2019.2946059](https://doi.org/10.1109/TMI.2019.2946059)
- **Journal:** IEEE Trans Medical Imaging (2019)
- **ML Model:** DGACNN — Adversarial One-Class Classification + Video Transfer Learning
- **Summary:** Novel approach using adversarial one-class classification for CHD screening. Trains only on normal fetal heart videos (does not require labeled pathological examples). Incorporates video transfer learning for temporal analysis of echocardiographic sequences. Addresses the data imbalance problem where normal cases vastly outnumber abnormal ones. FHD birth defect rates in Asia reach 9.3%.

### Li et al. (2024) — Application of AI in VSD Prenatal Diagnosis from Fetal Heart Ultrasound Images
- **PMID:** 39550543 | **DOI:** [10.1186/s12884-024-06916-y](https://doi.org/10.1186/s12884-024-06916-y)
- **Journal:** BMC Pregnancy and Childbirth (2024)
- **ML Model:** Deep CNN for classification (VSD vs normal)
- **Summary:** Developed AI system for detecting ventricular septal defects (VSD) from fetal heart ultrasound. Analyzed 1,451 images from 500 pregnant women (2016–2022). Demonstrates that AI can provide objective, efficient adjunctive diagnosis for the most common CHD subtype.

### Wang et al. (2026) — A Multi-Stage Deep Learning Network for Prenatal Diagnosis of Coarctation of the Aorta
- **PMID:** 41532285 | **DOI:** [10.1002/mp.70230](https://doi.org/10.1002/mp.70230)
- **Journal:** Medical Physics (2026)
- **ML Model:** Multi-stage deep learning pipeline
- **Summary:** Addresses coarctation of the aorta (CoA), a common congenital cardiovascular disorder where severe cases cause neonatal shock or heart failure. Multi-stage approach sequentially performs view classification, feature extraction, and diagnosis. CoA is notoriously difficult to detect prenatally with high false-positive and false-negative rates.

### Hernandez-Cruz et al. (2025) — ML-Based Method for Detection of Dextrocardia in Ultrasound Video Clips
- **PMID:** 40848560 | **DOI:** [10.1016/j.cmpb.2025.109023](https://doi.org/10.1016/j.cmpb.2025.109023)
- **Journal:** Computer Methods and Programs in Biomedicine (2025)
- **ML Model:** SegFormer (Transformer-based segmentation) + geometric reasoning
- **Performance:** Dice 0.968/0.958/0.953/0.949 for chest/spine/stomach/heart; FBCS 0.99
- **Summary:** Uses Transformer-based segmentation model (SegFormer) to segment fetal anatomy (chest, spine, stomach, heart) from ultrasound videos. Then applies geometric reasoning on centroid positions to classify dextrocardia. Three-stage pipeline: segmentation → quality assessment → orientation determination.

### S et al. (2024) — Investigation on Ultrasound Images for Detection of Fetal CHD
- **PMID:** 38781934 | **DOI:** [10.1088/2057-1976/ad4f91](https://doi.org/10.1088/2057-1976/ad4f91)
- **Journal:** Biomedical Physics & Engineering Express (2024)
- **ML Model:** Various CNN architectures compared for CHD classification
- **Summary:** Compares multiple deep learning approaches for CHD detection from fetal ultrasound. Notes that current screening technology has ~60% detection rates. Supplementing with AI could significantly improve detection.

### G et al. (2025) — FHD Deep Learning Prognosis Approach: Early Detection of Fetal Heart Disease Using IROI Combined Multiresolution DCNN
- **PMID:** 40007382 | **DOI:** [10.1177/09287329241310981](https://doi.org/10.1177/09287329241310981)
- **Journal:** Technology and Health Care (2025)
- **ML Model:** IROI (Intelligent Region of Interest) + Multiresolution DCNN
- **Summary:** FHD accounts for 21% of all congenital abnormalities. Proposes a method combining intelligent region-of-interest extraction with multiresolution deep CNN for early detection. Operates on ultrasonography images of four-chamber and blood vessel views. Addresses the problem that clinical diagnosis of abnormality is time-consuming and operator-dependent.

---

## 4. Standard Plane Detection & Quality Assessment

### Sarker et al. (2025) — HarmonicEchoNet: Leveraging Harmonic Convolutions for Automated Standard Plane Detection in Fetal Heart Ultrasound Videos
- **PMID:** 40876099 | **DOI:** [10.1016/j.media.2025.103758](https://doi.org/10.1016/j.media.2025.103758)
- **Journal:** Medical Image Analysis (2025)
- **ML Model:** HarmonicEchoNet — harmonic convolutions for rotation equivariance
- **Summary:** Addresses the problem that manual acquisition of standard heart views is time-consuming. Proposes harmonic convolution-based network that is naturally equivariant to rotations, important for fetal imaging where orientation varies. Automated detection of standard planes enables systematic quality assessment.

### Zhang et al. (2021) — Automatic Quality Assessment for 2D Fetal Sonographic Standard Plane Based on Multitask Learning
- **PMID:** 33530242 | **DOI:** [10.1097/MD.0000000000024427](https://doi.org/10.1097/MD.0000000000024427)
- **Journal:** Medicine (2021)
- **ML Model:** Multi-task learning CNN (joint classification + regression)
- **Summary:** Quality control of fetal sonographic images is essential for correct biometric measurements and anomaly diagnosis. Proposes multi-task learning to simultaneously classify standard plane type and assess image quality. Reduces the labor-intensive manual quality control process.

### Lam-Rachlin et al. (2025) — Use of AI-Based Software to Aid in Identification of Ultrasound Findings Associated with Fetal CHDs
- **PMID:** 41100866 | **DOI:** [10.1097/AOG.0000000000006087](https://doi.org/10.1097/AOG.0000000000006087)
- **Journal:** Obstetrics & Gynecology (2025)
- **ML Model:** AI-based software (commercial system)
- **Summary:** Evaluates whether AI-based software enhances identification of 8 second-trimester ultrasound findings suspicious for CHDs. Tests performance among OB-GYNs and maternal-fetal medicine specialists using 200 fetal ultrasound exams from 11 centers. Multicenter validation of a practical clinical tool.

---

## 5. Automated Biometrics & Cardiac Measurements

### Taksøe-Vester et al. (2024) — Role of AI-Assisted Automated Cardiac Biometrics in Prenatal Screening for Coarctation of Aorta
- **PMID:** 38339776 | **DOI:** [10.1002/uog.27608](https://doi.org/10.1002/uog.27608)
- **Journal:** Ultrasound Obstet Gynecol (2024)
- **ML Model:** AI-automated biometric measurement system
- **Summary:** ~60% of newborns with isolated coarctation of the aorta (CoA) are not identified prior to birth. Automated cardiac biometrics using AI significantly improves prenatal CoA detection. Shows that AI-derived measurements can capture subtle size discrepancies that human operators miss.

### Taksøe-Vester et al. (2025) — Fetal Cardiac Remodeling in Second Trimester: Deep-Learning-Based Approach Using Population-Wide Data
- **PMID:** 41164991 | **DOI:** [10.1002/uog.70123](https://doi.org/10.1002/uog.70123)
- **Journal:** Ultrasound Obstet Gynecol (2025)
- **ML Model:** Deep learning for automated cardiac measurement from population screening data
- **Summary:** Uses DL-based automated measurements to examine fetal cardiac remodeling in pregnancies complicated by pre-eclampsia and fetal growth restriction. Population-wide analysis enabled by automated measurement — would be impossible with manual assessment at scale.

### Yang et al. (2024) — An Intelligent Quantification System for Fetal Heart Rhythm Assessment
- **PMID:** 38266752 | **DOI:** [10.1016/j.hrthm.2024.01.024](https://doi.org/10.1016/j.hrthm.2024.01.024)
- **Journal:** Heart Rhythm (2024)
- **ML Model:** Deep learning for automated Doppler waveform analysis
- **Summary:** Multicenter prospective study. Develops a system to automatically calculate fetal cardiac time intervals (CTIs) from pulsed-wave Doppler spectra. Motion relationships and time intervals are essential for diagnosing fetal arrhythmia. Few prior technologies could automatically compute CTIs.

---

## 6. Fetal Heart Rate Analysis & Cardiotocography

### McCoy et al. (2024) — Intrapartum Electronic Fetal Heart Rate Monitoring to Predict Acidemia at Birth with Deep Learning
- **PMID:** 38663662 | **DOI:** [10.1016/j.ajog.2024.04.022](https://doi.org/10.1016/j.ajog.2024.04.022)
- **Journal:** American Journal of Obstetrics and Gynecology (2024)
- **ML Model:** Deep learning (CNN/LSTM hybrid) on raw FHR tracings
- **Summary:** Electronic fetal monitoring is used in most US hospital births but has significant limitations. Deep learning applied directly to raw FHR tracings to predict neonatal acidemia. Demonstrates that deep learning can improve complex pattern recognition beyond what visual inspection achieves.

### Davis Jones et al. (2026) — Identifying High-Risk Pre-Term Pregnancies Using Fetal Heart Rate and Machine Learning
- **PMID:** 41749742 | **DOI:** [10.3390/bioengineering13020203](https://doi.org/10.3390/bioengineering13020203)
- **Journal:** Bioengineering (2026)
- **ML Model:** Machine learning (ensemble methods) on antepartum FHR recordings
- **Summary:** Analyzed 4,867 antepartum FHR recordings from preterm gestations. Preterm births carry high burden of stillbirth and severe fetal compromise. ML-based approach to identify high-risk pregnancies earlier to justify iatrogenic preterm delivery.

### Fergus et al. (2017) — Classification of Caesarean Section and Normal Vaginal Deliveries Using Foetal Heart Rate Signals and Advanced ML
- **PMID:** 28679415 | **DOI:** [10.1186/s12938-017-0378-z](https://doi.org/10.1186/s12938-017-0378-z)
- **Journal:** BioMedical Engineering OnLine (2017)
- **ML Model:** SVM, decision trees, random forest, artificial neural networks, ensemble methods
- **Summary:** Early ML approach to CTG classification. Compares multiple traditional ML algorithms for predicting delivery mode from FHR signals. Inter- and intra-observer variability in visual CTG interpretation is high (only 30% positive predictive value), motivating automated approaches.

### Liu et al. (2020) — ML Algorithms to Predict Early Pregnancy Loss After IVF-ET with Fetal Heart Rate as a Strong Predictor
- **PMID:** 32623348 | **DOI:** [10.1016/j.cmpb.2020.105624](https://doi.org/10.1016/j.cmpb.2020.105624)
- **Journal:** Computer Methods and Programs in Biomedicine (2020)
- **ML Model:** Multiple ML algorithms compared (logistic regression, SVM, random forest, gradient boosting, neural networks)
- **Summary:** Uses fetal heart rate as a key predictor variable in ML models to predict early pregnancy loss after IVF. Demonstrates FHR as a strong predictive feature across multiple ML algorithms.

### Prachi et al. (2025) — Precision Unveiled in Unborn: A Cutting-Edge Hybrid ML Approach for Fetal Health State Classification
- **PMID:** 40866753 | **DOI:** [10.1007/s13239-025-00800-2](https://doi.org/10.1007/s13239-025-00800-2)
- **Journal:** Cardiovascular Engineering and Technology (2025)
- **ML Model:** Hybrid ML approach (multiple classifiers combined)
- **Summary:** Proposes hybrid ML for classifying fetal health states from CTG data. Fetal cardiac abnormalities (structural or functional) need immediate medical attention. Multi-classifier fusion improves robustness over single models.

---

## 7. Fetal MRI Reconstruction & Analysis

### Uus et al. (2022) — Automated 3D Reconstruction of the Fetal Thorax from Motion-Corrupted MRI Stacks for 21–36 Weeks GA
- **PMID:** 35649314 | **DOI:** [10.1016/j.media.2022.102484](https://doi.org/10.1016/j.media.2022.102484)
- **Journal:** Medical Image Analysis (2022)
- **ML Model:** Deep learning for slice-to-volume registration (SVR) + 3D reconstruction
- **Summary:** Fully automated pipeline for 3D reconstruction of fetal thorax from motion-corrupted MRI stacks. Covers 21–36 weeks gestational age. SVR-based pipelines allow more informed diagnosis of body anomalies including congenital heart defects. Addresses the challenge of constant fetal movement during MRI acquisition.

### Cromb et al. (2023) — Total and Regional Brain Volumes in Fetuses with Congenital Heart Disease
- **PMID:** 37846811 | **DOI:** [10.1002/jmri.29078](https://doi.org/10.1002/jmri.29078)
- **Journal:** J Magnetic Resonance Imaging (2023)
- **ML Model:** Automated brain segmentation (deep learning) applied to fetal MRI
- **Summary:** Uses automated MRI analysis to compare brain volumes in fetuses with CHD versus typically developing fetuses. Tests hypothesis that expected cerebral substrate delivery is associated with total and regional fetal brain volumes. Demonstrates that ML-based volumetric analysis can reveal links between CHD and impaired early brain development.

---

## 8. Fetal ECG & Electrophysiology

### de Vries et al. (2023) — Fetal Electrocardiography and Artificial Intelligence for Prenatal Detection of CHD
- **PMID:** 37563851 | **DOI:** [10.1111/aogs.14623](https://doi.org/10.1111/aogs.14623)
- **Journal:** Acta Obstet Gynecol Scand (2023)
- **ML Model:** Artificial neural network (ANN) trained on non-invasive fetal ECG
- **Summary:** Investigates non-invasive fetal electrocardiography as an alternative/complement to ultrasound for CHD detection. An ANN was trained for identification of CHD using non-invasively obtained fetal ECGs. Novel modality for prenatal CHD screening that could be deployed in settings without expert sonographers.

---

## 9. Gene Regulatory Networks & Single-Cell Genomics

### Sun et al. (2025) — Bioinformatics Combining ML and Single-Cell Sequencing Analysis
- **PMID:** 39897930 | **DOI:** [10.1016/j.heliyon.2025.e41641](https://doi.org/10.1016/j.heliyon.2025.e41641)
- **Journal:** Heliyon (2025)
- **ML Model:** Machine learning classifiers (LASSO, SVM-RFE, random forest) + scRNA-seq analysis
- **Summary:** Combines ML with single-cell sequencing to identify common mechanisms and biomarkers between rheumatoid arthritis and ischemic heart failure. Analyzed RNA sequencing data from five datasets. ML used for feature selection and biomarker identification from high-dimensional omics data.

### Wang et al. (2025) — Epistasis Regulates Genetic Control of Cardiac Hypertrophy
- **PMID:** 40473955 | **DOI:** [10.1038/s44161-025-00656-8](https://doi.org/10.1038/s44161-025-00656-8)
- **Journal:** Nature Cardiovascular Research (2025)
- **ML Model:** Signed iterative random forests + deep learning-derived cardiac measurements
- **Summary:** Develops novel ML approach to uncover epistatic (non-additive) genetic interactions controlling cardiac hypertrophy. Uses deep learning to derive left ventricular mass estimates from imaging data, then applies low-signal iterative random forests to discover gene-gene interactions. Demonstrates how ML can elucidate complex genetic architecture of cardiac traits.

---

## 10. Computational Modeling of Cardiac Morphogenesis

### Sarkar et al. (2022) — 3D Cell Morphology Detection by Association for Embryo Heart Morphogenesis
- **PMID:** 38510433 | **DOI:** [10.1017/S2633903X22000022](https://doi.org/10.1017/S2633903X22000022)
- **Journal:** Biological Imaging (2022)
- **ML Model:** Deep learning for 3D cell segmentation + association-based tracking
- **Summary:** Computational methods to automatize cell segmentation in 3D and deliver accurate quantitative morphology of cardiomyocytes during embryonic development. Advances in tissue engineering for cardiac regenerative medicine require cellular-level understanding of cardiac muscle growth during embryonic stage. Deep learning applied to confocal microscopy images.

### Hong et al. (2022) — Prdm6 Controls Heart Development by Regulating Neural Crest Cell Differentiation and Migration
- **PMID:** 35108221 | **DOI:** [10.1172/jci.insight.156046](https://doi.org/10.1172/jci.insight.156046)
- **Journal:** JCI Insight (2022)
- **ML Model:** Computational analysis of single-cell data for lineage tracing
- **Summary:** Uses computational approaches including single-cell data analysis to identify Prdm6 as an epigenetic modifier regulating neural crest cell fate. Demonstrates how ML-assisted scRNA-seq analysis reveals molecular mechanisms of cardiac development.

---

## 11. Clinical Protocols & Multicenter Studies

### Patey et al. (2025) — CAIFE: Clinical AI in Fetal Echocardiography — International Multicentre Study Protocol
- **PMID:** 40473283 | **DOI:** [10.1136/bmjopen-2025-101263](https://doi.org/10.1136/bmjopen-2025-101263)
- **Journal:** BMJ Open (2025)
- **ML Model:** Deep learning for image and video analysis (protocol for prospective validation)
- **Summary:** Protocol for a major international multicentre, multidisciplinary study to validate DL-based prenatal CHD detection. CHD is a significant, rapidly emerging global problem in child health and a leading cause of neonatal death. Study aims to establish clinical evidence for AI deployment in routine prenatal screening across multiple countries.

### Ungureanu et al. (2023) — LIFE: Learning Deep Architectures for Interpretation of First-Trimester Fetal Echocardiography
- **PMID:** 36631859 | **DOI:** [10.1186/s12884-022-05204-x](https://doi.org/10.1186/s12884-022-05204-x)
- **Journal:** BMC Pregnancy and Childbirth (2023)
- **ML Model:** Deep learning (study protocol for first-trimester screening)
- **Summary:** Study protocol for developing an automated intelligent decision support system for early (first-trimester) fetal echocardiography. Correct prenatal diagnosis of specific cardiac anomalies improves neonatal care, neurologic outcomes, and surgery outcomes. Aims to push CHD detection earlier in pregnancy.

### Tran et al. (2019) — Deep Learning as a Predictive Tool for Fetal Heart Pregnancy Following Time-Lapse Incubation
- **PMID:** 31111884 | **DOI:** [10.1093/humrep/dez064](https://doi.org/10.1093/humrep/dez064)
- **Journal:** Human Reproduction (2019)
- **ML Model:** Deep learning (IVY system) — CNN on time-lapse embryo videos
- **Summary:** Created "IVY," a fully automated DL system that predicts the probability of pregnancy with fetal heart directly from raw time-lapse videos without manual morphokinetic annotation. Applied to IVF — predicts successful implantation leading to detectable fetal heart.

---

## 12. Summary: ML Models Used by Application

### Model Type → Application Matrix

| ML Model / Architecture | Application Domain | Key References |
|---|---|---|
| **CNNs (VGG, ResNet, Inception, DenseNet)** | CHD classification, standard plane detection, image quality assessment | Zhang 2024, Reddy 2022, Athalye 2024, S 2024 |
| **U-Net & variants** | Cardiac structure segmentation (4-chamber, valves, vessels) | Xu 2020, Schlemper 2019, Torres 2024 |
| **Attention mechanisms (AG, Category Attention)** | Focused segmentation, salient feature extraction | Schlemper 2019, An 2021 |
| **Transformer / SegFormer** | Dextrocardia detection, anatomical segmentation | Hernandez-Cruz 2025 |
| **DCNN + Multiresolution** | Early FHD detection from US images | G 2025 |
| **Cascaded / Multi-stage pipelines** | CoA diagnosis, multi-task screening | Wang 2026, Hernandez-Cruz 2025 |
| **Adversarial / One-class classification** | CHD screening (normal-only training) | Gong 2019 |
| **LSTM / RNN (temporal)** | FHR time-series analysis, video-based echo | McCoy 2024, Yu 2017 |
| **SVM, Random Forest, XGBoost** | CTG classification, FHR prediction, biomarker selection | Fergus 2017, Liu 2020, Francis 2024 |
| **Ensemble methods** | Robust classification, fetal health state | Hurtado 2024, Prachi 2025 |
| **Harmonic convolutions** | Rotation-equivariant plane detection | Sarker 2025 |
| **Multi-task learning** | Joint quality assessment + classification | Zhang 2021 |
| **Artificial neural networks (classical)** | Fetal ECG-based CHD detection | de Vries 2023 |
| **Transfer learning** | Domain adaptation from adult to fetal cardiac imaging | Gong 2019, general reviews |
| **Deep learning on time-lapse video** | IVF outcome prediction (fetal heart) | Tran 2019 |
| **3D CNNs / SVR** | Fetal thorax MRI reconstruction | Uus 2022 |
| **LASSO / SVM-RFE / RF (feature selection)** | Genomic biomarker discovery | Sun 2025 |
| **Iterative random forests** | Epistatic gene interaction discovery | Wang 2025 |
| **DL-based cell segmentation** | 3D cardiomyocyte morphology during morphogenesis | Sarkar 2022 |
| **GANs** | Data augmentation, image synthesis (reviewed) | de Siqueira 2021, Yi 2020 |

### Key Trends

1. **Shift from traditional ML to deep learning (2017–2026):** Earlier studies relied on SVMs, random forests, and logistic regression for CTG/FHR classification. Post-2019, deep CNNs dominate for image analysis, with Transformer architectures emerging by 2025.

2. **From single-frame to video/temporal analysis:** Early approaches analyzed individual US frames. Recent work (DGACNN, McCoy 2024, HarmonicEchoNet) processes full video sequences or time-series data.

3. **Attention and Transformer architectures:** Attention mechanisms (2019) and Transformer-based models (2025) improve focus on clinically relevant regions and handle the variable orientations inherent in fetal imaging.

4. **Multi-stage and multi-task pipelines:** State-of-the-art systems combine detection → segmentation → measurement → diagnosis in end-to-end or cascaded pipelines.

5. **Community/population-scale validation:** Recent studies (Athalye 2024, CAIFE 2025, Taksøe-Vester 2025) emphasize generalization from academic centers to community settings and population-wide screening.

6. **Addressing data scarcity:** One-class classification (Gong 2019), hybrid deformable models (Torres 2024), and transfer learning tackle the fundamental challenge of limited annotated pathological fetal cardiac data.

7. **Multi-modal integration:** Combining US, MRI, ECG, and genomic data with ML is an emerging frontier. Single-cell genomics + ML is increasingly used to dissect developmental gene regulatory networks.

---

## Summary Statistics

- **Total papers cataloged:** 38
- **Date range:** 2017 – 2026
- **Key journals:** Medical Image Analysis, IEEE Trans Medical Imaging, Ultrasound Obstet Gynecol, J Am Soc Echocardiogr, Computers in Biology and Medicine, BMJ Open, Am J Obstet Gynecol, Heart Rhythm, Nature Cardiovascular Research

---

*All PubMed references retrieved via PubMed/NCBI. DOI links provided for all entries.*
