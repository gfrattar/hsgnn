"""Produce histogram of discriminant from tagger output and labels."""

import numpy as np
from ftag import Flavours

from puma import Roc, RocPlot
from puma import Histogram, HistogramPlot
from puma.metrics import calc_rej
from puma.utils import get_dummy_2_taggers, logger, get_good_linestyles
import h5py
import pandas as pd
import glob
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

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

    efficiencyFilesPath = "results/HSEfficienciesFiles-"+modelName
    if not os.path.isdir(efficiencyFilesPath):
        os.system('mkdir {}'.format(efficiencyFilesPath))
        
    file_paths_labels = []
    listOfFiles = glob.glob(inpath+"*__test_*.h5")

    for file_path in listOfFiles:
        process = (file_path.split("_test_")[-1]).replace('.h5','')
        file_paths_labels.append((file_path,process))

    sig_eff_map = dict()
    rej_map = dict()
    sig_eff_map_sumPt2 = dict()
    rej_map_sumPt2 = dict()

    for file_path, label in file_paths_labels:
        if 'NNtestFile_0' not in file_path: continue

        with h5py.File(file_path, 'r') as hdf_file:

            print("Executing {}".format(file_path))

            ds = hdf_file['vertices']

            df = pd.DataFrame({
                'isHggHardScatter': np.array(ds['isHggHardScatter']).transpose(),
                'isHardScatter': np.array(ds['isHardScatter']).transpose(), 
                'BDT_score': np.array(ds['BDT_score']).transpose(), 
                'hsvertex': np.array(ds[modelName+'_phsvertex']).transpose(),
                #'ntrk': np.array(ds['ntrk']).transpose(),
                #'sumPt': np.array(ds['sumPt']).transpose(),
                'sumPt2': np.array(ds['sumPt2']).transpose(),
                #'chi2Over_ndf': np.array(ds['chi2Over_ndf']).transpose(),
                #'z_asymmetry': np.array(ds['z_asymmetry']).transpose(),
                #'photon_deltaz': np.array(ds['photon_deltaz']).transpose(),
                'eventNumber': np.array(ds['eventNumber']).transpose(),
                'eventWeight': np.array(ds['eventWeight']).transpose(),
                'actualIntPerXing': np.array(ds['actualIntPerXing']).transpose()
            })
            df = df.dropna()

            # defining boolean arrays to select the different flavour classes
            #is_pu = df["isHardScatter"] == 0
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
            ews_selected = df[ abs(df['eventWeight']) < ews_mean + 3*ews_std ]
            ews_un_selected = ews_selected['eventWeight'].unique()

            linestyles = get_good_linestyles()[:2]
            if label in processes_with_negative_weights: 
                plot_histo = HistogramPlot(
                    n_ratio_panels=0,
                    ylabel="Normalised number of vertices",
                    xlabel="Event weight",
                    logy=True,
                    leg_ncol=1,
                    figsize=(5.5, 4.5),
                    bins=np.linspace(-1000000000, 1000000000, 10000),
                    #y_scale=1.5,
                    atlas_second_tag=label
                )
    
                # Add the histograms
                plot_histo.add(
                    Histogram(
                        df['eventWeight'].unique(),
                        label="All event weights",
                        colour=Flavours["bjets"].colour,
                        linestyle=linestyles[0],
                    ),
                    reference=False,
                )
                plot_histo.add(
                    Histogram(
                        ews_un_selected,
                        label="Selected",
                        colour=Flavours["cjets"].colour,
                        linestyle=linestyles[1],
                    ),
                    reference=False,
                )
                plot_histo.draw()
                plot_histo.savefig("eventWeight_{}.png".format(eospath,label), transparent=False)
                plot_histo.savefig("eventWeight_{}.pdf".format(eospath,label), transparent=False)
            '''
            plot_histo = HistogramPlot(
                n_ratio_panels=0,
                ylabel="Normalised number of vertices",
                xlabel="Actual interactions per bunch crossing, $\\mu$",
                logy=False,
                leg_ncol=1,
                figsize=(5.5, 4.5),
                bins=np.linspace(0, 70, 70),
                #y_scale=1.5,
                atlas_second_tag="$\\sqrt{s}=13.6$ TeV, "+label,
            )
            # Add the histograms
            plot_histo.add(
                Histogram(
                    ActualIntPerXing,
                    label="Test dataset",
                    colour=Flavours["bjets"].colour,
                    linestyle=linestyles[0],
                ),
                reference=False,
            )
            
            plot_histo.draw()
            plot_histo.savefig("{}/ActualIntPerXing_{}.png".format(eospath,label), transparent=False)
            plot_histo.savefig("{}/ActualIntPerXing_{}.pdf".format(eospath,label), transparent=False)
            '''
            isEfficient_mu = dict()
            isEfficient_mu['sumPt2'] = dict()
            isEfficient_mu['GNN'] = dict()
            isEfficient_mu['BDT'] = dict()
            isEfficient_mu['HyyNN'] = dict()
            
            isEfficient_mu_weighted = dict()
            isEfficient_mu_weighted['sumPt2'] = dict()
            isEfficient_mu_weighted['GNN'] = dict()
            isEfficient_mu_weighted['BDT'] = dict()
            isEfficient_mu_weighted['HyyNN'] = dict()

            isEfficient_npv = dict()
            isEfficient_npv['sumPt2'] = dict()
            isEfficient_npv['GNN'] = dict()
            isEfficient_npv['BDT'] = dict()
            isEfficient_npv['HyyNN'] = dict()

            total_npv = dict()
            total_mu = dict()
            total_mu_weighted = dict()
            
            unique_mu_values = df['actualIntPerXing'].astype(int).unique()
            unique_npv_values = [i for i in range(0,100)]
            
            for mu in unique_mu_values:
                isEfficient_mu['sumPt2'][mu] = 0
                isEfficient_mu['GNN'][mu] = 0
                isEfficient_mu['BDT'][mu] = 0
                isEfficient_mu['HyyNN'][mu] = 0
                isEfficient_mu_weighted['sumPt2'][mu] = 0
                isEfficient_mu_weighted['GNN'][mu] = 0
                isEfficient_mu_weighted['BDT'][mu] = 0
                isEfficient_mu_weighted['HyyNN'][mu] = 0
                total_mu[mu] = 0
                total_mu_weighted[mu] = 0
                
            for npv in unique_npv_values:
                isEfficient_npv['sumPt2'][npv] = 0
                isEfficient_npv['GNN'][npv] = 0
                isEfficient_npv['BDT'][npv] = 0
                isEfficient_npv['HyyNN'][npv] = 0
                total_npv[npv] = 0
            

            for eN in events:
                #print("---NEW EVENT---")
                if not(len(df[is_hs & (df['eventNumber']==eN)]['actualIntPerXing'] > 0)): continue
                
                df_toInvestigate = df[df['eventNumber']==eN]

                if label in processes_with_negative_weights and float(df_toInvestigate['eventWeight'].values[0]) not in ews_un_selected:
                    continue #drop event if event weight is more than 5 sigma away from mean

                NPV = len(df[df['eventNumber']==eN])
                MU = int(df[is_hs & (df['eventNumber']==eN)]['actualIntPerXing'].values[0])

                df_toInvestigate_trueHS = df[(df['eventNumber']==eN) & (df['isHardScatter']==1)]
                df_toInvestigate_truePU = df[(df['eventNumber']==eN) & (df['isHardScatter']==0)]


                #Need to ensure there is a true HS in the df_toInvetigate otherwise drop the event      
                hsvertex_Max = df_toInvestigate['hsvertex'].max()
                sumPt2_Max = df_toInvestigate['sumPt2'].max()
                #print('For event {} - hsvertexMax = {} ; sumPt2Max = {}'.format(eN,hsvertex_Max,sumPt2_Max))
                #print(df_toInvestigate)


                isEfficient_mu['GNN'][MU] += (df_toInvestigate['hsvertex'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                isEfficient_mu['BDT'][MU] += (df_toInvestigate['BDT_score'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                isEfficient_mu['HyyNN'][MU] += (df_toInvestigate['isHggHardScatter'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                isEfficient_mu['sumPt2'][MU] += (df_toInvestigate['sumPt2'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())

                isEfficient_mu_weighted['GNN'][MU] += (df_toInvestigate['hsvertex'].idxmax() == df_toInvestigate['isHardScatter'].idxmax()) * float(df_toInvestigate['eventWeight'].values[0])
                isEfficient_mu_weighted['BDT'][MU] += (df_toInvestigate['BDT_score'].idxmax() == df_toInvestigate['isHardScatter'].idxmax()) * float(df_toInvestigate['eventWeight'].values[0])
                isEfficient_mu_weighted['HyyNN'][MU] += (df_toInvestigate['isHggHardScatter'].idxmax() == df_toInvestigate['isHardScatter'].idxmax()) * float(df_toInvestigate['eventWeight'].values[0])
                isEfficient_mu_weighted['sumPt2'][MU] += (df_toInvestigate['sumPt2'].idxmax() == df_toInvestigate['isHardScatter'].idxmax()) * float(df_toInvestigate['eventWeight'].values[0])


                isEfficient_npv['GNN'][NPV] += (df_toInvestigate['hsvertex'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                isEfficient_npv['BDT'][NPV] += (df_toInvestigate['BDT_score'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                isEfficient_npv['HyyNN'][NPV] += (df_toInvestigate['isHggHardScatter'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                isEfficient_npv['sumPt2'][NPV] += (df_toInvestigate['sumPt2'].idxmax() == df_toInvestigate['isHardScatter'].idxmax())
                
                total_mu[MU] += 1
                total_mu_weighted[MU] += float(df_toInvestigate['eventWeight'].values[0])
                total_npv[NPV] += 1
            
            '''
            ofile = open('eff_vs_npv_{}.txt'.format(label),'w')
            for k in total_npv.keys():
                ofile.write("{} {} {} {}\n".format(k,isEfficient_npv['sumPt2'][k],isEfficient_npv['GNN'][k],total_npv[k]))
            ofile.close()

            ofile = open('eff_vs_mu_{}.txt'.format(label),'w')
            for k in total_mu.keys():
                ofile.write("{} {} {} {}\n".format(k,isEfficient_mu['sumPt2'][k],isEfficient_mu['GNN'][k],total_mu[k]))
            ofile.close()
            '''

            ofile = open('{}/eff_vs_npv_{}.txt'.format(efficiencyFilesPath,label),'w')
            for k in total_npv.keys():
                ofile.write("{} {} {} {} {} {}\n".format(k,isEfficient_npv['sumPt2'][k],isEfficient_npv['GNN'][k],isEfficient_npv['HyyNN'][k],isEfficient_npv['BDT'][k],total_npv[k]))
            ofile.close()

            ofile = open('{}/eff_vs_mu_{}.txt'.format(efficiencyFilesPath,label),'w')
            for k in total_mu.keys():
                ofile.write("{} {} {} {} {} {}\n".format(k,isEfficient_mu['sumPt2'][k],isEfficient_mu['GNN'][k],isEfficient_mu['HyyNN'][k],isEfficient_mu['BDT'][k],total_mu[k]))
            ofile.close()
            
            ofile = open('{}/eff_vs_mu_{}_weighted.txt'.format(efficiencyFilesPath,label),'w')
            for k in total_mu.keys():
                ofile.write("{} {} {} {} {} {}\n".format(k,isEfficient_mu_weighted['sumPt2'][k],isEfficient_mu_weighted['GNN'][k],isEfficient_mu_weighted['HyyNN'][k],isEfficient_mu_weighted['BDT'][k],total_mu_weighted[k]))
            ofile.close()
