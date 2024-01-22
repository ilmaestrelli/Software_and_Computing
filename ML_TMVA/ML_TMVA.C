
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int ML_TMVA( TString myMethodList = "" )
{  
   //This loads the libraries
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Cut optimisation
   Use["Cuts"]            = 1;
   
   // Linear Discriminant Analysis
   Use["Fisher"]          = 1;
   
   // Neural Networks (all are feed-forward Multilayer Perceptrons)
   Use["MLPBNN"]          = 1; // Recommended ANN with BFGS training method and bayesian regulator

   // Boosted Decision Trees
   Use["BDT"]             = 1; // uses Adaptive Boost

   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassification" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return 1;
         }
         Use[regMethod] = 1;
      }
   }

   // ----------------------------------------------------------------

   // Read training and test data
   TFile *sig = TFile::Open("atlas-higgs-challenge-2014-v2-sig.root", "READ");
   TFile *bkg = TFile::Open("atlas-higgs-challenge-2014-v2-bkg.root", "READ");

   // Register the training and test trees
   TTree *signalTree     = (TTree*)sig->Get("tree");
   TTree *background     = (TTree*)bkg->Get("tree");

   // Create output root file
   TString outfileName( "TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Declare TMVA Factory object
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   // Declare DataLoader
   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
   
   // Add variables to the dataset
   dataloader->AddVariable( "DER_mass_MMC", 'F' );
   dataloader->AddVariable( "DER_mass_transverse_met_lep", 'F' );
   dataloader->AddVariable( "DER_mass_vis", 'F' );
   dataloader->AddVariable( "DER_pt_ratio_lep_tau", 'F' );
   dataloader->AddVariable( "PRI_tau_pt", 'F' );
   dataloader->AddVariable( "PRI_met", 'F' );


   // You can add an arbitrary number of signal or background trees
   dataloader->AddSignalTree    ( signalTree, 1.0); // signal weight = 1
   dataloader->AddBackgroundTree( background, 1.0); // bkg weight = 1

   // Tell the dataloader how to use the training and testing events
   TCut mycuts, mycutb;
   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb, "nTrain_Signal=10000:nTrain_Background=20000:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

   //----------------------------- BOOKING METHODS -------------------------------
   
   // Cut optimisation
   if (Use["Cuts"])
      factory->BookMethod( dataloader, TMVA::Types::kCuts, "Cuts",
                           "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );

   // Fisher discriminant (same as LD)
   if (Use["Fisher"])
      factory->BookMethod( dataloader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
  
   // TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
   if (Use["MLPBNN"])
      factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=200:HiddenLayers=3:TestRate=5:TrainingMethod=BFGS:UseRegulator" ); // BFGS training with bayesian regulators

   // Boosted Decision Trees
   if (Use["BDT"])  // Adaptive Boost
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDT",
                           "!H:!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

   // -----------------------------------------------------------------------

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << " Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "TMVA classification is done!" << std::endl;

   delete factory;
   delete dataloader;
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

   return 0;
}

