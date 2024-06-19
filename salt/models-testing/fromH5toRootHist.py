import ROOT
import h5py
import glob
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    #inpath = '../logs/HSGNN_v6_allObj_objCounting_lessParameters_2_noNjets_20240607-T213359/ckpts/'
    #eospath = './HSGNN_v6_allObj_objCounting_lessParameters_2_noNjets'

    #eospath = 'HSGN2_v7_allObj_objCounting_lessParameters' #sys.argv[1]
    #inpath = '/data/gfrattar/hs-ml/training-outputs/Evaluated-HSGN2_v7_allObj_objCounting_lessParameters_20240613-T151944/' #sys.argv[2]
    eospath = 'HSGN2_v8_allObj_baseline'
    inpath = '/data/gfrattar/hs-ml/training-outputs/Evaluated-HSGN2_v8_allObj_baseline_20240617-T153533/'
    #inpath = '/home/gfrattar/hs-ml/training/salt/salt/logs/HSGN2_v7_allObj_baseline_20240613-T150731/ckpts/'
    modelName = eospath

    processes_with_negative_weights = [
        "NNtestFile_0_Sh_2214_Znunu_pTV2_CFilterBVeto",
        "NNtestFile_0_Sh_2214_mumugamma",
        "NNtestFile_0_Sh_2214_Znunu_pTV2_CVetoBVeto",
        "NNtestFile_0_Sh_2214_tautaugamma",
        "NNtestFile_0_Sh_2214_nunugamma",
        "NNtestFile_0_Sh_2214_eegamma",
        "NNtestFile_0_Sh_2214_munugamma",
        "NNtestFile_0_Sh_2214_taunugamma",
        "NNtestFile_0_Sh_2214_enugamma",
        "NNtestFile_0_Sh_2214_Znunu_pTV2_BFilter"
    ]
    
    xval = dict()
    varToTake = dict()
    varToTake['SumPt2'] = 'sumPt2'
    varToTake['HyyNN'] = 'isHggHardScatter'
    varToTake['BDT'] = 'BDT_score'
    varToTake['GNN'] = 'hsvertex'

    efficiencyFilesPath = "results/HSEfficienciesFiles-"+modelName
    if not os.path.isdir(efficiencyFilesPath):
        os.system('mkdir {}'.format(efficiencyFilesPath))
        
    file_paths_labels = []
    listOfFiles = glob.glob(inpath+"*__test_*.h5")

    for file_path in listOfFiles:
        process = (file_path.split("_test_")[-1]).replace('.h5','')
        file_paths_labels.append((file_path,process))

    print("Found = {}".format(file_paths_labels))

    sig_eff_map = dict()
    rej_map = dict()
    sig_eff_map_sumPt2 = dict()
    rej_map_sumPt2 = dict()

    for file_path, label in file_paths_labels:
        if 'NNtestFile_0' not in file_path: continue
        if 'NNtestFile_0_Sh_2214_Znunu_pTV2_CFilterBVeto' not in file_path: continue

        with h5py.File(file_path, 'r') as hdf_file:
            print("Executing {}".format(file_path))

            ds = hdf_file['vertices']

            df = pd.DataFrame({
                'isHggHardScatter': np.array(ds['isHggHardScatter']).transpose(),
                'isHardScatter': np.array(ds['isHardScatter']).transpose(), 
                'BDT_score': np.array(ds['BDT_score']).transpose(), 
                'hsvertex': np.array(ds[modelName+'_phsvertex']).transpose(),
                'sumPt2': np.array(ds['sumPt2']).transpose(),
                'eventNumber': np.array(ds['eventNumber']).transpose(),
                'eventWeight': np.array(ds['eventWeight']).transpose(),
                'actualIntPerXing': np.array(ds['actualIntPerXing']).transpose()
            })
            df = df.dropna()

            is_hs = df["isHardScatter"] == 1

            #Need to get the unique values of the event number, then check in the reduced dataset for each event number if the
            #highest score is the one with the true flag as well
            events = df['eventNumber'].unique()
            print("Number of events: {}".format(len(events)))
            #Get the dataframe for only the fixed event number
            events = events[0:30000]
            NPV = []
            ActualIntPerXing = []
            for eN in events:
                if(len(df[is_hs & (df['eventNumber']==eN)]['actualIntPerXing'] > 0)):
                    thisEvent_mu = df[is_hs & (df['eventNumber']==eN)]['actualIntPerXing'].astype(int).values[0]
                    thisEvent_NPV = len(df[df['eventNumber']==eN])
                else: continue
                NPV.append(thisEvent_NPV)
                ActualIntPerXing.append(thisEvent_mu)

            ews = df['eventWeight'].unique()
            ews_mean = ews.mean()
            ews_std = ews.std()
            ews_selected = df[ abs(df['eventWeight']) < ews_mean + 2*ews_std ]
            ews_un_selected = ews_selected['eventWeight'].unique()

            histoPass = dict()
            histoTotal = dict()
            weighted_histoPass = dict()
            weighted_histoTotal = dict()
            
            variables = ["mu","npv"]
            
            for var in variables:
                histoPass[var] = dict()
                histoTotal[var] = dict()
                weighted_histoPass[var] = dict()
                weighted_histoTotal[var] = dict()
            
            algos = ["SumPt2","HyyNN","GNN","BDT"]
            
            process = label.split('File_0_')[1]
            
            for var in variables:
                for algo in algos:
                    histoPass[var][algo] = ROOT.TH1F("pass_"+algo+"_"+process+"_"+var,"",100,0,100)
                    histoTotal[var][algo] = ROOT.TH1F("total_"+algo+"_"+process+"_"+var,"",100,0,100)
                    weighted_histoPass[var][algo] = ROOT.TH1F("weighted_pass_"+algo+"_"+process+"_"+var,"",100,0,100)
                    weighted_histoTotal[var][algo] = ROOT.TH1F("weighted_total_"+algo+"_"+process+"_"+var,"",100,0,100)

            #unique_mu_values = df['actualIntPerXing'].astype(int).unique()
            for eN in events:
                #print("---NEW EVENT---")
                if not(len(df[is_hs & (df['eventNumber']==eN)]['actualIntPerXing'] > 0)): continue
                
                df_toInvestigate = df[df['eventNumber']==eN]

                if label in processes_with_negative_weights and float(df_toInvestigate['eventWeight'].values[0]) not in ews_un_selected:
                    continue #drop event if event weight is more than 5 sigma away from mean

                NPV = len(df[df['eventNumber']==eN])
                MU = int(df[is_hs & (df['eventNumber']==eN)]['actualIntPerXing'].values[0])
                xval['mu'] = MU
                xval['npv'] = NPV

                weight = float(df_toInvestigate['eventWeight'].values[0])

                df_toInvestigate_trueHS = df[(df['eventNumber']==eN) & (df['isHardScatter']==1)]

                hsvertex_Max = df_toInvestigate['hsvertex'].max()
                sumPt2_Max = df_toInvestigate['sumPt2'].max()

                for algo in algos:
                    histoTotal[var][algo].Fill(xval[var])
                    weighted_histoTotal[var][algo].Fill(xval[var],weight)
                                        
                    if df_toInvestigate[varToTake[algo]].idxmax() == df_toInvestigate['isHardScatter'].idxmax():
                        histoPass[var][algo].Fill(xval[var])
                        weighted_histoPass[var][algo].Fill(xval[var],weight)
    
            ofile = ROOT.TFile('{}/eff_vs_{}_{}.root'.format(efficiencyFilesPath,var,label),'RECREATE')
            for k in histoPass[var].keys():
                histoPass[var][k].Write()
                histoTotal[var][k].Write()
                weighted_histoPass[var][k].Write()
                weighted_histoTotal[var][k].Write()
            ofile.Close()
