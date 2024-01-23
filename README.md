# Project overview
The project belonging to this repository aims to use some Machine Learning (ML) algorithms in order to analyse a simulated Higgs boson deacy into two taus, i.e. to perform an event selection ad a signal-to-background ratio. In particular, the codes presented in this project give the possibility to perform a multivariate analysis both with TMVA (Toolkit for MultiVariate Analysis) and TensorFlow. For each, the following alorithms have been implemented:

- TMVA: Cuts, Fisher discriminant, feed-forward Multilayer Perceptron Bayesian Neural Network (MLPBNN), Boosted decision trees (BDT)
- TensorFlow: MLPBNN

By looking at the results given as output it is possible to decide what is the best suitable method to analyse de data.

# Dataset description
The dataset has been built from official ATLAS full-detector simulation, with H->tautau events mixed with different backgrounds. The signal sample contains events in which Higgs bosons (with a fixed mass of 125 GeV) were produced. The background sample was generated by other known processes that can produce events mimicking signal. The data was divided in two datasets, one for the signal and one for the background, namely:

- atlas-higgs-challenge-2014-v2-sig.root
- atlas-higgs-challenge-2014-v2-bkg.root

The chosen variables to train and test the algorithm are

- DER_mass_MMC: estimated mass of the Higgs boson candidate, obtained through a probabilistic phase space integration
- DER_mass_transverse_met_lep: transverse mass between the missing transverse energy and the lepton
- DER_mass_vis: invariant mass of the hadronic tau and the lepton
- DER_pt_ratio_lep_tau: ratio of the transverse momenta of the lepton and the hadronic tau
- PRI_tau_pt: transverse momentum of the hadronic tau.
- PRI_met: missing transverse energy

More and complete informations about the dataset can be found here: http://opendata.cern.ch/record/328

# Folders (and files) description
- **ML_TMVA**: the folder which contains all the files needed to perform analysis with TMVA, i.e.
  - signal and background .root files
  - ML_TMVA.C : macro for dataset training. It produces the file "TMVA.root" and the folder "dataset" which will be used both by "TMVA_analysis.C"
  - TMVA_analysis.C : macro for testing the training results for each TMVA method
  - dockerfile : contains root environment (useful for users which do not have installed root)

- **ML_tensorflow**: the folder which contains all the files needed to perform analysis with TMVA, i.e.
  - signal and background .root files
  - ML_py.py : code for train and test data with a MLPBNN
  - dockerfile : contains tensorflow environment (useful for users which do not have installed the required libraries)
 
- **output_TMVA**: the folder which contains some selected output/resuls from TMVA training, such as
- **output_tensorflow**: the folder which contains some selected output/resuls from tensorflow analysis, such as

 
# How to run the code(s)
**IF YOU HAVE root AND tensorflow ALREADY INSTALLED**  
$ git clone https://github.com/ilmaestrelli/Software_and_Computing.git  

- _TMVA analysis_    
  $ cd your/path/to/Software_and_Computing/ML_TMVA  
  $ root  
  root [0] .L ML_TMVA.C  
  root [1] ML_TMVA()  

  Note: ML_TMVA() trains all the method at once; you can also train different ML methods separately by insterting the name of one method ("Cuts", "Fisher", "MLPBNN", "BDT") in the      argument of the function (ex. ML_TMVA("BDT").

  root [2] .L TMVA_analysis.C  
  root [3] TMVA_analysis ("method_you_want_to_use")  

- _tensorflow analysis_    
  $ cd your/path/to/Software_and_Computing/ML_tensorflow  
  ($ conda activate apple_tensorflow)  
  $ python3 ML_py.py

---------------------------------------------------------------------------

**IF YOU DO NOT HAVE root AND tensorflow ALREADY INSTALLED**  
$ git clone https://github.com/ilmaestrelli/Software_and_Computing.git  

- _TMVA analysis_  
  $ cd your/path/to/Software_and_Computing/ML_TMVA  
  $ docker build -t your_image .  
  $ docker --rm --it your_image  
  $ root  
  root [0] .L ML_TMVA.C  
  root [1] ML_TMVA()  

  Note: ML_TMVA() trains all the method at once; you can also train different ML methods separately by insterting the name of one method ("Cuts", "Fisher", "MLPBNN", "BDT") in the      argument of the function (ex. ML_TMVA("BDT").

  root [2] .L TMVA_analysis.C  
  root [3] TMVA_analysis ("method_you_want_to_use")  

- _tensorflow analysis_     
  $ cd your/path/to/Software_and_Computing/ML_tensorflow  
  $ docker build -t your_image .  
  $ docker --rm --it your_image  
  $ python3 ML_py.py  
  
**NOTE**: If you have any issues with the two .root files containing the dataset for the signal and the backgroung, you can also download them from this drive folder https://drive.google.com/drive/u/0/folders/1Qjlm0LlU3V8yMXpdPKd_vEr3gy74bmDr?q=sharedwith:public%20parent:1Qjlm0LlU3V8yMXpdPKd_vEr3gy74bmDr 
    






