#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>
 
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"

#include "TROOT.h"
#include "TStopwatch.h"
 
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
 
using namespace TMVA;
using namespace std;
 
void TMVA_analysis ( TString myMethodList = "" )
{
   // This loads the library
   TMVA::Tools::Instance();
 
   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Methods
   Use["Cuts"]    = 1; //Cut optimization
   Use["Fisher"]  = 1; //Linear discriminant
   Use["MLPBNN"]  = 1; //feed-forward multilayer-perception - Neural Network
   Use["BDT"]     = 1; //Boosted Decision Trees

   std::cout << std::endl;
   std::cout << "Start TMVA analysis!" << std::endl;
 
   // --------------------------------------------------------------------------------------------------
 
   // Declare the Reader object
   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

   TFile *input = TFile::Open("atlas-higgs-challenge-2014-v2-bkg.root", "READ");
   std::cout << "--- Select signal sample" << std::endl;
   TTree* theTree = (TTree*)input->Get("tree");
 
   //Varaibles declaration
   Float_t DER_mass_MMC;
   Float_t DER_mass_transverse_met_lep;
   Float_t DER_mass_vis;
   Float_t DER_pt_ratio_lep_tau;
   Float_t PRI_tau_pt;
   Float_t PRI_met;

   //Adding variables to the reader   
   reader->AddVariable( "DER_mass_MMC", &DER_mass_MMC );
   reader->AddVariable( "DER_mass_transverse_met_lep", &DER_mass_transverse_met_lep );
   reader->AddVariable( "DER_mass_vis", &DER_mass_vis);
   reader->AddVariable( "DER_pt_ratio_lep_tau", &DER_pt_ratio_lep_tau );
   reader->AddVariable( "PRI_tau_pt", &PRI_tau_pt);
   reader->AddVariable( "PRI_met", &PRI_met);
 
   // Book the MVA methods
   TString dir    = "dataset/weights/";
   TString prefix = "TMVAClassification_";
   TString weightfile = dir + prefix + myMethodList +TString(".weights.xml");
   reader->BookMVA( myMethodList, weightfile);

   // Book output histos
   TH1F *hFisher(0);
   TH1F *hMLPBNN(0);
   TH1F *hBDT(0);

   if (myMethodList == "Fisher") hFisher = new TH1F( "MVA_Fisher", "MVA_Fisher", 1000, -5, 5);
   if (myMethodList == "MLPBNN") hMLPBNN= new TH1F( "MVA_MLPBNN", "MVA_MLPBNN", 1000, -5, 5);
   if (myMethodList == "BDT") hBDT = new TH1F( "MVA_BDT", "MVA_BDT", 1000, -5, 5);

   

   Double_t v1, v2, v3, v4, v5, v6;

   // Setting branch address
   theTree->SetBranchAddress( "DER_mass_MMC", &v1 );
   theTree->SetBranchAddress( "DER_mass_transverse_met_lep", &v2 );
   theTree->SetBranchAddress( "DER_mass_vis", &v3);
   theTree->SetBranchAddress( "DER_pt_ratio_lep_tau", &v4 );
   theTree->SetBranchAddress( "PRI_tau_pt", &v5);
   theTree->SetBranchAddress( "PRI_met", &v6);


   // Efficiency calculator for Cuts method
   Int_t nSelCutsGA = 0;
   Double_t effS = 0.7;
 
   std::vector<Float_t> vecVar(4); // vector for EvaluateMVA tests
 
   std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;

   TStopwatch sw;
   sw.Start();
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0) std::cout << "--- ... Processing event: " << ievt << std::endl;
 
      theTree->GetEntry(ievt);

      // Filling the histos
      if (myMethodList == "Cuts") {
         Bool_t passed = reader->EvaluateMVA("Cuts", effS); //I use signal efficiency (0.7)
         if (passed) nSelCutsGA++;
         }
      
      if (myMethodList == "Fisher") hFisher->Fill(reader->EvaluateMVA("Fisher"));
      if (myMethodList == "MLPBNN") hMLPBNN->Fill(reader->EvaluateMVA("MLPBNN"));
      if (myMethodList == "BDT") hBDT->Fill(reader->EvaluateMVA("BDT"));


   }
 
   // Elapsed time
   sw.Stop();
   std::cout << "--- End of event loop: "; 
   sw.Print();
 
   // Writing histos
   TFile *target  = new TFile( "TMVApp.root","RECREATE" );

   if (myMethodList == "Fisher") hFisher->Write();
   if (myMethodList == "MLPBNN") hMLPBNN->Write();
   if (myMethodList == "BDT") hBDT->Write();
   
   std::cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << std::endl;
   
   target->Close();
   
   delete reader;
 
   std::cout << "TMVA anlysis is done! You can find the histograms in the file TMVApp.root." << std::endl << std::endl;

}
 
 